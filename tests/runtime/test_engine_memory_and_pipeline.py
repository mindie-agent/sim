"""Engine arbitration, bound memory, pipelines and runtime recording.

[Role] Exercise scheduler and memory contracts on Python and C11.
[Invariants] Cohorts preserve service, byte ledgers conserve traffic and tracing
  preserves event times. Directional HBM service honors concurrent R/W capacity.
[Timeouts & Locks] Bounded Tier-0 schedules with an isolated C11 build cache.
[Side-Effect Policy] CPU-only tests using in-memory engines and temporary caches.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from _support.axes import ALL_SKUS
from mindie_sim import compose_case
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.runtime.coupled import coupled_rates
from mindie_sim.prediction.engine import Engine, OccurrenceKey, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.execution.operator_replay import operator_costs
from mindie_sim.prediction.operators.matrix import refill_steady
from mindie_sim.prediction.ownership import unit_rates
import mindie_sim.prediction.program as prog_pkg
from mindie_sim.prediction.program import (
    AddressRun, AddressRunPlan, AdmissionMemoryView, Allocation, BurstTurnRule,
    DirtyWriteback, HbmBusSpec, KernelProgram, MemoryAccessPattern, ParallelProgram,
    PipelineProgram, RecipeBook, RecipeContext, ReuseKey, Stage, VectorInstructionGroup, VisibilityContract,
    coalesce_execution_program, merge_programs, proportional_coupled_placement,
)
from mindie_sim.prediction.runtime.accel_tables import pure_task
from mindie_sim.prediction.runtime.accel_terms import (
    TERMS_ABI_VERSION, MsimAccess, expected_terms_abi_probe,
)
from mindie_sim.prediction.runtime.arbiter_fast_coupled import fast_single_tier_coupled
from _support.prepared_terms_parity import (
    POLICY_FAIR, POLICY_FIFO, PreparedCapacityRow, PreparedDelivery,
    PreparedProgress, PreparedTerm, PreparedTermsRunner,
    allocate_prepared_epoch, integrate_prepared_epoch,
)
from mindie_sim.prediction.runtime.exposed_observer import ServiceGrantObserver
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.runtime.tile_solver import (
    LiveBufferSpec, SubtileWaveSpec, TileCandidate, TileInfeasibleError,
    TilePipelineSpec, TileShape, solve as solve_tiles,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

pytestmark = pytest.mark.tier0
P05_SKUS = tuple(p for p in ALL_SKUS if p.id in {"Ascend910_9362", "Ascend950DT_9582"})

def _ulp_distance(a: float, b: float) -> int:
    return 0 if a == b else int(round(abs(a - b) / max(math.ulp(a), math.ulp(b))))

def _make_context(*, alpha_rw: float = 0.25) -> RecipeContext:
    rates = {
        "hbm@d0": 1_600_000_000.0, "mte2_aic@d0": 800_000_000.0, "mte2_aiv@d0": 800_000_000.0, "mte3@d0": 800_000_000.0,
        "l2@d0": 3_200_000_000.0, "vec_issue@d0": 1_000_000.0, "vec_alu@d0": 1_000_000.0, "vec_sfu@d0": 250_000.0, "vec_reduce@d0": 500_000.0,
    }
    return RecipeContext(
        op=SimpleNamespace(family="elementwise", recipe="direct", metadata={}), work={},
        resource_names={row.removesuffix("@d0"): row for row in rates},
        rates=rates, core_count=20, physical_mechanisms=SimpleNamespace(hbm_alpha_rw=alpha_rw), compute_arch=SimpleNamespace(clock_hz=1e9),
    )

@pytest.mark.parametrize("backend", ["python", "c"])
@pytest.mark.parametrize("hbm_work,cube_work", [(1.0, 10.0), (10.0, 1.0)])
def test_widened_cohort_asymmetric_split_matches_unmerged_across_backends(backend: str, hbm_work: float, cube_work: float):
    def run(merged: bool) -> tuple[float, dict[str, float], int]:
        eng = Engine([Resource("hbm", 1.0, units=1), Resource("cube", 1.0, units=20)], backend=backend)
        progs = tuple(KernelProgram((Stage("run", {"hbm": hbm_work * (29 if c == 19 else 28), "cube": cube_work * (29 if c == 19 else 28)},
                                           caps={"hbm": 1.0, "cube": 1.0}, placement={"cube": (c,)}, coupled=True),)) for c in range(20))
        done = (coalesce_execution_program(ParallelProgram(progs)) if merged else ParallelProgram(progs)).start(eng, "op", eng.submit("start"))
        eng.advance(until=(done,))
        return float(done.time), dict(eng.serviced), eng.accel_status()["native_steps"]
    t_m, s_m, steps_m = run(True)
    t_u, s_u, steps_u = run(False)
    assert abs(t_m - t_u) <= 1e-9 and s_m == pytest.approx(s_u, rel=1e-12, abs=1e-9)
    assert backend != "c" or (steps_m > 0 and steps_u > 0)

@pytest.mark.parametrize("backend", ["python", "c"])
def test_shared_multiunit_and_fifo_cohorts_preserve_semantics(backend: str):
    def run_shared(merged: bool) -> tuple[float, dict[str, float]]:
        eng = Engine([Resource("cube", 2.0, units=2), Resource("hbm", 1.0, units=1)], backend=backend)
        start = eng.submit("start")
        k0 = KernelProgram((Stage("s", {"cube": 2.0, "hbm": 1.0}, placement={"cube": (0, 1)}, coupled=True),))
        k1 = KernelProgram((Stage("s", {"cube": 2.0, "hbm": 1.0}, placement={"cube": (0, 1)}, coupled=True),))
        peer = eng.submit("peer", {"cube": 2.0, "hbm": 1.0}, after=(start,), placement={"cube": (0, 1)}, coupled=True)
        if merged:
            coh = merge_programs([k0, k1])
            assert coh is not None and coh.stages[0].cohort_size == 2 and coh.stages[0].cohort_widened == ()
            done = eng.submit("done", after=(coh.start(eng, "merged", start), peer))
        else:
            done = eng.submit("done", after=(k0.start(eng, "m0", start), k1.start(eng, "m1", start), peer))
        eng.advance(until=(done,))
        assert backend != "c" or eng.accel_status()["native_steps"] > 0
        return float(done.time), dict(eng.serviced)
    t_m, s_m = run_shared(True)
    t_u, s_u = run_shared(False)
    assert abs(t_m - t_u) <= 1e-9 and s_m == pytest.approx(s_u, rel=1e-12)
    assert merge_programs([KernelProgram((Stage("u", {"hbm": 4.0}),)), KernelProgram((Stage("u", {"hbm": 4.0}),))]) is None
    def run_fifo(merged: bool) -> float:
        eng = Engine([Resource("dma", 1.0, policy="fifo", units=1)], backend=backend)
        start = eng.submit("start")
        f0, f1 = KernelProgram((Stage("f", {"dma": 1.0}, coupled=True),)), KernelProgram((Stage("f", {"dma": 1.0}, coupled=True),))
        c_ev = merge_programs([f0, f1]).start(eng, "mf", start) if merged else eng.submit("j", after=(f0.start(eng, "f0", start), f1.start(eng, "f1", start)))
        peer = eng.submit("peer", {"dma": 1.0}, after=(start,))
        done = eng.submit("done", after=(c_ev, eng.submit("succ", {"dma": 10.0}, after=(peer,))))
        eng.advance(until=(done,))
        return float(done.time)
    assert run_fifo(True) == pytest.approx(13.0, abs=1e-9) and run_fifo(False) == pytest.approx(13.0, abs=1e-9)

@pytest.mark.parametrize("backend", ["python", "c"])
def test_midflight_asymmetric_split_runs_native_until_interference_and_publishes_once(backend: str):
    def run(merged: bool) -> tuple[float, dict[str, float], int, int]:
        eng = Engine([Resource("cube", 4.0, units=4), Resource("hbm", 4.0, units=1)], backend=backend)
        start = eng.submit("start")
        members = [KernelProgram((Stage("k", {"cube": 2.0, "hbm": 1.0}, placement={"cube": (i,)}, coupled=True),)) for i in range(4)]
        main_ev = (merge_programs(members).start(eng, "coh", start) if merged
                   else eng.submit("join", after=tuple(m.start(eng, f"m{i}", start) for i, m in enumerate(members))))
        ext = eng.submit("ext", {"cube": 1.0}, release=0.5, placement={"cube": (0,)})
        done = eng.submit("done", after=(main_ev, ext))
        eng.advance(until=(done,))
        return float(done.time), dict(eng.serviced), eng.accel_status()["native_steps"], eng.completed_count
    t_m, s_m, steps_m, comp_m = run(True)
    t_u, s_u, _, _ = run(False)
    assert abs(t_m - t_u) <= 1e-9 and s_m == pytest.approx(s_u, rel=1e-12) and comp_m == 4
    assert backend != "c" or steps_m > 0
    # Symmetric case: cohort activates via dependency at 0.5 ms beside active unit-0 cube + HBM peers.
    e_sym = Engine([Resource("cube", 4.0, units=4), Resource("hbm", 4.0, units=1)], backend=backend)
    pc = e_sym.submit("pc", {"cube": 2.0}, placement={"cube": (0,)})
    ph = e_sym.submit("ph", {"hbm": 4.0})
    dep = e_sym.submit("dep", {"cube": 0.5}, placement={"cube": (3,)})
    coh = e_sym.submit("coh", {"cube": 2.0, "hbm": 2.0}, after=(dep,), placement={"cube": (0, 1)},
                       coupled=True, cohort_size=2, cohort_widened=("cube",), cohort_members={"cube": ((0,), (1,))})
    e_sym.advance(until=(pc, ph, coh), first=False)
    assert float(ph.time) == pytest.approx(17.0 / 14.0, abs=1e-12)
    assert backend != "c" or e_sym.accel_status()["native_steps"] > 0

def test_over_64_link_widening_and_proportional_native_admission():
    n_res = 70
    resources = [Resource(f"r{i}", 4.0, units=2) for i in range(n_res)]
    work, placement, widened = {f"r{i}": 2.0 for i in range(n_res)}, {f"r{i}": (0, 1) for i in range(n_res)}, tuple(f"r{i}" for i in range(n_res))
    times = {}
    for be in ("python", "c"):
        eng = Engine(resources, backend=be)
        ev = eng.submit("wide", work, placement=placement, coupled=True, cohort_size=2, cohort_widened=widened)
        eng.advance(until=(ev,))
        times[be] = float(ev.time)
        assert be != "c" or eng.accel_status()["native_steps"] > 0
    assert abs(times["python"] - times["c"]) <= 1e-12 and times["c"] == pytest.approx(0.5, abs=1e-12)
    assert proportional_coupled_placement((0,), (0, 1, 2), 2, 4) == ((0,), (0, 1))
    skew_eng = Engine([Resource("cube", 2.0, units=2), Resource("vector", 4.0, units=4)], backend="c")
    skew_ev = skew_eng.submit("skew", {"cube": 2.0, "vector": 4.0}, coupled=True, placement={"cube": (0,), "vector": (0, 1, 2)})
    assert not pure_task(skew_eng, skew_eng._tasks[skew_ev.sequence])
    skew_eng.advance(until=(skew_ev,))
    assert skew_eng.accel_status()["native_steps"] == 0

def test_coupled_and_disjoint_uncoupled_parity_zero_ulp_and_peer_invariance():
    res1 = [Resource("a", 26.289056632197735), Resource("b", 36.495015663833634)]
    works1 = [{"a": 7.05924503470419, "b": 8.494354162581535}, {"a": 6.694301119164283, "b": 5.5353786183562095}, {"a": 8.483307144965345, "b": 7.87160639063931}]
    out1 = {}
    for be in ("python", "c"):
        eng = Engine(res1, backend=be)
        evs = [eng.submit(f"t{i}", w, coupled=True) for i, w in enumerate(works1)]
        eng.advance(until=tuple(evs), first=False)
        out1[be] = [float(e.time) for e in evs]
    assert all(_ulp_distance(a, b) == 0 for a, b in zip(out1["python"], out1["c"]))
    res2 = [Resource("cube", 245.76, units=20), Resource("vector", 1.92, units=40)]
    u_specs = (("u0", {"cube": 1.0154159211083948}, {"cube": (10, 15, 17)}),
               ("u1", {"cube": 3.6808735333841436}, {"cube": (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19)}))
    out2 = {}
    for be in ("python", "c"):
        eng_solo = Engine(res2, backend=be)
        evs_solo = [eng_solo.submit(n, w, placement=p) for n, w, p in u_specs]
        eng_solo.advance(until=tuple(evs_solo), first=False)
        eng_peer = Engine(res2, backend=be)
        evs_peer = [eng_peer.submit(n, w, placement=p) for n, w, p in u_specs]
        c0 = eng_peer.submit("c0", {"cube": 500.0, "vector": 100.0}, coupled=True, placement={"cube": (3, 4), "vector": (6, 7, 8, 9)})
        eng_peer.advance(until=(*evs_peer, c0), first=False)
        assert float(evs_peer[0].time) == float(evs_solo[0].time) and float(evs_peer[1].time) == float(evs_solo[1].time) and float(evs_peer[0].time) < float(evs_peer[1].time)
        out2[be] = [float(evs_peer[0].time), float(evs_peer[1].time), float(c0.time)]
    assert all(_ulp_distance(a, b) == 0 for a, b in zip(out2["python"], out2["c"]))
    ur = unit_rates(245.76, 20, {0: math.inf, 1: math.inf}, {0: (10, 15, 17), 1: (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19)})
    assert ur[0] == pytest.approx(ur[1], rel=1e-15)

def test_fast_single_tier_coupled_matches_coupled_rates_on_widened_cohort():
    eng = Engine([Resource("cube", 20.0, units=20), Resource("hbm", 1.0, units=1)], backend="python")
    eng.submit("coh", {"cube": 190.0, "hbm": 19.0}, placement={"cube": tuple(range(19))}, coupled=True, cohort_size=19, cohort_widened=("cube",))
    eng.submit("solo", {"cube": 10.0, "hbm": 1.0}, placement={"cube": (19,)}, coupled=True)
    eng._settle()
    r_fast = fast_single_tier_coupled(eng.resources, eng._tasks, eng._active)
    r_ref = coupled_rates(eng.resources, {k: eng._tasks[k] for k in eng._active})
    assert r_fast is not None and r_fast["hbm"][1] == pytest.approx(19.0 / 20.0, rel=1e-12)
    for rname in ("cube", "hbm"):
        for seq in (1, 2):
            assert r_fast[rname][seq] == pytest.approx(r_ref[rname][seq], rel=1e-12)

@pytest.mark.parametrize("backend", ["python", "c"])
def test_bound_memory_access_through_engine_single_bw_divide_owner_pin_and_deliveries(backend: str):
    bus = HbmBusSpec(die=0, read_bytes_per_ms=1024.0, write_bytes_per_ms=1024.0, alpha_rw=0.25)
    ctx = RecipeContext(
        op=SimpleNamespace(family="elementwise", recipe="direct", metadata={}),
        work={}, resource_names={"hbm": "hbm@d0", "mte2_aiv": "mte2@d0", "cube": "aic_r", "vector": "aiv_r"},
        rates={"hbm@d0": 1024.0, "mte2@d0": 128.0, "aic_r": 2.0, "aiv_r": 4.0},
        resource_units={"mte2_aiv": 2, "cube": 2, "vector": 4}, core_count=2,
        physical_mechanisms=SimpleNamespace(hbm_alpha_rw=0.25),
        stream_owners=SimpleNamespace(isolation="disjoint", aic=(0,), aiv=(0, 1, 2)),
    )
    assert ctx.bind_coupled_stream_placement() == {"aic_r": (0,), "aiv_r": (0, 1)}
    pat = MemoryAccessPattern("x", 1, 0, 1, "read", "mte2_aiv", 128, AddressRunPlan((AddressRun(0, 1, 128, 128),)),
                              reuse_key=ReuseKey(die=0, backing_allocation="buf_x", tensor_version=1))
    bound = ctx.bind_access(pat, hbm_bus=bus)
    assert bound.stage_placement() == {"mte2@d0": (1,)}
    eng = Engine([Resource("mte2@d0", 128.0, unit="bytes", units=2)], backend=backend)
    eng.add_hbm_bus(bus)
    ev = eng.submit_bound_access("cold_rd", bound)
    eng.advance(horizon=0.5)
    assert eng.delivered["hbm_read@d0"] == pytest.approx(32.0, rel=1e-12)
    eng.advance(until=(ev,))
    assert float(ev.time) == pytest.approx(2.0, abs=1e-12) and eng.delivered["hbm_read@d0"] == pytest.approx(128.0, rel=1e-12)
    assert backend != "c" or eng.accel_status()["native_steps"] > 0
    b_miss = ctx.bind_access(pat, hbm_bus=bus, copy_miss_bytes_per_ms=32.0)
    for use_stage in (False, True):
        eng2 = Engine([Resource("mte2@d0", 128.0, unit="bytes", units=2)], backend=backend)
        eng2.add_hbm_bus(bus)
        ev2 = (KernelProgram((b_miss.to_stage("rd_st"),)).start(eng2, "op", eng2.submit("go"))
               if use_stage else eng2.submit_bound_access("rd_sb", b_miss))
        eng2.advance(until=(ev2,))
        assert float(ev2.time) == pytest.approx(4.0, abs=1e-12) and eng2.delivered["hbm_read@d0"] == pytest.approx(128.0, rel=1e-12)
    k0 = KernelProgram((Stage("rd", {"mte2": 1.0}, placement={"mte2": (0,)}, coupled=True, deliveries=(("hbm_read@d0", 128.0),)),))
    k1 = KernelProgram((Stage("rd", {"mte2": 1.0}, placement={"mte2": (1,)}, coupled=True, deliveries=(("hbm_read@d0", 128.0),)),))
    eng3 = Engine([Resource("mte2", 2.0, units=2)], backend=backend)
    ev3 = merge_programs([k0, k1]).start(eng3, "op", eng3.submit("go"))
    eng3.advance(until=(ev3,))
    assert eng3.delivered["hbm_read@d0"] == pytest.approx(256.0, rel=1e-12)

def test_three_ledger_sector_union_turnaround_and_l2_sector_residency_rules():
    ctx = _make_context(alpha_rw=0.25)
    discrete = MemoryAccessPattern("kv_rows", 1, 0, 0, "read", "mte2_aiv", 656 * 2048,
                                   AddressRunPlan((AddressRun(0, 2048, 656, 672),)),
                                   reuse_key=ReuseKey(0, "kv", 1), visibility=VisibilityContract(scope="owner"))
    turn = BurstTurnRule(1, 1024, 1, 1024, 164.0)
    bound = ctx.bind_access(discrete, burst_rules=(turn,))
    assert bound.ledger.dram_sector_bytes == 1_376_256 and bound.ledger.dram_bus_transaction_bytes == pytest.approx(1_376_256 / 0.8)
    mib = 1024 * 1024
    with pytest.raises(ModelingError, match="shared_l2_buffer_requires_one_die_owner"):
        AdmissionMemoryView({0: 128 * mib}, l2_buffer_units_by_die={0: 2})
    view = AdmissionMemoryView({0: 128 * mib}, l2_buffer_units_by_die={0: 1})
    k_range = ReuseKey(0, "buf", 1, reuse_stream="main")
    p_str = MemoryAccessPattern("ts", 1, 0, 0, "read", "mte2_aiv", 64, AddressRunPlan((AddressRun(0, 2, 32, 64),)), reuse_key=k_range)
    p_gap = MemoryAccessPattern("tg", 1, 0, 0, "read", "mte2_aiv", 32, AddressRunPlan((AddressRun(32, 1, 32, 32),)), reuse_key=k_range)
    ctx.bind_access(p_str, state=view)
    assert not view.is_resident(k_range) and view.admit_access(p_str, 64) == (0, 64) and view.admit_access(p_gap, 32) == (0, 32)
    # 128 B cache evicts [0, 128) when admitting disjoint [128, 256).
    v_cap = AdmissionMemoryView({0: 128})
    k_cap = ReuseKey(0, "cap", 1)
    pa = MemoryAccessPattern("a", 1, 0, 0, "read", "mte2_aiv", 128, AddressRunPlan((AddressRun(0, 1, 128, 128),)), reuse_key=k_cap)
    pb = MemoryAccessPattern("b", 1, 0, 0, "read", "mte2_aiv", 128, AddressRunPlan((AddressRun(128, 1, 128, 128),)), reuse_key=k_cap)
    assert v_cap.admit_access(pa, 128) == (0, 128) and v_cap.admit_access(pb, 128) == (0, 128)
    assert v_cap.preview_access(pa, 128) == (0, 128) and v_cap.preview_access(pb, 128) == (128, 0)
    # Admission-time work rebind on dependent 128 B reads (128 B HBM, not 256 B) + 32B hit/32B miss @ 50% burst (64 B).
    bus = HbmBusSpec(die=0, read_bytes_per_ms=1024.0, write_bytes_per_ms=1024.0, alpha_rw=0.25)
    v_dep = AdmissionMemoryView({0: 256})
    b1, b2 = ctx.bind_access(pa, state=v_dep, hbm_bus=bus), ctx.bind_access(pa, state=v_dep, hbm_bus=bus)
    e_dep = Engine([Resource("mte2_aiv@d0", 800_000_000.0), Resource("l2@d0", 3_200_000_000.0)], backend="python")
    e_dep.add_hbm_bus(bus)
    ev1 = e_dep.submit_bound_access("r1", b1, memory_state=v_dep, pattern=pa)
    ev2 = e_dep.submit_bound_access("r2", b2, after=(ev1,), memory_state=v_dep, pattern=pa)
    e_dep.advance(until=(ev2,))
    assert e_dep.delivered["hbm_read@d0"] == pytest.approx(128.0, rel=1e-12)
    assert e_dep.delivered["l2@d0"] == pytest.approx(128.0, rel=1e-12)
    v_b = AdmissionMemoryView({0: 256})
    k_b = ReuseKey(0, "bb", 1)
    assert v_b.admit_access(MemoryAccessPattern("h", 1, 0, 0, "read", "mte2_aiv", 32, AddressRunPlan((AddressRun(0, 1, 32, 32),)), reuse_key=k_b), 32) == (0, 32)
    p_mix = MemoryAccessPattern("m", 1, 0, 0, "read", "mte2_aiv", 64, AddressRunPlan((AddressRun(0, 1, 64, 64),)), reuse_key=k_b)
    b_mix = ctx.bind_access(p_mix, state=v_b, hbm_bus=bus, burst_rules=(BurstTurnRule(1, 65, 0, 65, 32.0),))
    assert b_mix.ledger.l2_hit_bytes == 32 and b_mix.ledger.dram_bus_transaction_bytes == pytest.approx(64.0)
    k_other = ReuseKey(0, "buf2", 1, reuse_stream="aux")
    p_other = MemoryAccessPattern("o", 1, 0, 0, "write", "mte3", 48 * mib, AddressRunPlan((AddressRun(0, 1, 48 * mib, 48 * mib),)),
                                  reuse_key=k_other, visibility=VisibilityContract(scope="owner", private_workspace=True, live_out=False))
    assert view.admit_access(p_other, 48 * mib) == (48 * mib, 0) and view.effective_l2_bytes(0) == 64 * mib
    view.grant_reservation(0, "pin", 120 * mib)
    wbs = view.drain_writebacks(0)
    assert len(wbs) == 1 and isinstance(wbs[0], DirtyWriteback) and wbs[0].bytes == 48 * mib
    view.release_reservation(0, "pin")
    assert not view.is_resident(k_other)

@pytest.mark.parametrize("alpha_rw", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("backend", ["python", "c"])
def test_hbm_two_row_directional_bus_laws_python_and_c(alpha_rw: float, backend: str):
    bus = HbmBusSpec(die=0, read_bytes_per_ms=1000.0, write_bytes_per_ms=1000.0, alpha_rw=alpha_rw)
    t_r, t_w, t_v = 4.0, 2.0, 1.5
    def run_case(mode: str) -> float:
        eng = Engine([Resource("vec", 1.0)], backend=backend)
        eng.add_hbm_bus(bus)
        r_w = {t.row_id: t.ms_per_progress for t in bus.terms_for("read", t_r * 1000.0) if t.ms_per_progress > 0.0}
        w_w = {t.row_id: t.ms_per_progress for t in bus.terms_for("write", t_w * 1000.0) if t.ms_per_progress > 0.0}
        if mode == "solo_r":
            ev = eng.submit("r", r_w, coupled=True)
        elif mode == "solo_w":
            ev = eng.submit("w", w_w, coupled=True)
        elif mode == "serial":
            ev = eng.submit("w", w_w, after=(eng.submit("v", {"vec": t_v}, after=(eng.submit("r", r_w, coupled=True),)),), coupled=True)
        else:
            ev = eng.submit("done", after=(eng.submit("r", r_w, coupled=True), eng.submit("w", w_w, coupled=True)))
        eng.advance(until=(ev,))
        assert backend != "c" or eng.accel_status()["native_steps"] > 0
        return float(ev.time)
    assert run_case("solo_r") == pytest.approx(t_r, rel=1e-12) and run_case("solo_w") == pytest.approx(t_w, rel=1e-12)
    assert run_case("serial") == pytest.approx(t_r + t_v + t_w, rel=1e-12)
    assert run_case("concurrent") == pytest.approx(max(t_r, t_w) + alpha_rw * min(t_r, t_w), rel=1e-12)

def test_prepared_terms_c11_parity_row_invalidation_and_access_validation():
    bus = HbmBusSpec(die=0, read_bytes_per_ms=1_600_000.0, write_bytes_per_ms=1_200_000.0, alpha_rw=0.35)
    rows = [
        PreparedCapacityRow(0, policy=POLICY_FAIR, capacity=1.0),
        PreparedCapacityRow(1, policy=POLICY_FAIR, capacity=1.0),
        PreparedCapacityRow(2, policy=POLICY_FIFO, capacity=1.0),
    ]
    d_r, d_w = 3_200_000.0, 2_400_000.0
    (tr_r, tr_w), (tw_r, tw_w) = bus.terms_for("read", d_r), bus.terms_for("write", d_w)
    with PreparedTermsRunner() as runner:
        c_r, c_w = runner.hbm_terms(bus, "read", d_r)
        assert _ulp_distance(c_r, tr_r.ms_per_progress) <= 1 and _ulp_distance(c_w, tr_w.ms_per_progress) <= 1
        assert runner.validate_access(MsimAccess(1024, 1024, 512, 512, 512, 512.0, 0, 1, 0, 0, 0)) == 0
        assert runner.validate_access(MsimAccess(1024, 1024, 512, 256, 256, 256.0, 0, 1, 0, 0, 0)) != 0
        progs_py = [
            PreparedProgress(1, 1, 0, 0, 0, 1.0, 1.0), PreparedProgress(2, 2, 0, 0, 0, 1.0, 1.0),
            PreparedProgress(3, 3, 1, 0, 0, 1.0, 2.0), PreparedProgress(4, 4, 0, 0, 0, 1.0, 1.0),
        ]
        progs_c = [PreparedProgress(p.sequence, p.task, p.priority, p.placement_id, p.member_map, p.z_left, p.weight, p.max_dz_ms) for p in progs_py]
        terms = [
            PreparedTerm(0, 0, tr_r.ms_per_progress), PreparedTerm(1, 0, tr_w.ms_per_progress),
            PreparedTerm(0, 1, tw_r.ms_per_progress), PreparedTerm(1, 1, tw_w.ms_per_progress),
            PreparedTerm(2, 1, 0.5), PreparedTerm(0, 2, 1.0), PreparedTerm(2, 3, 0.5),
        ]
        dels = [PreparedDelivery(0, 0, d_r), PreparedDelivery(1, 1, d_w), PreparedDelivery(2, 2, 100.0), PreparedDelivery(2, 3, 50.0)]
        runner.config_rows(rows)
        runner.prepare_terms(progs_c, terms, dels, ncounters=3)
        ctrs_py = [0.0, 0.0, 0.0]
        for _ in range(10):
            if all(p.z_left <= 0.0 for p in progs_py):
                break
            dz_py, dz_c = allocate_prepared_epoch(rows, progs_py, terms), runner.allocate_epoch()
            assert all(_ulp_distance(a, b) <= 8 for a, b in zip(dz_py, dz_c))
            dt = min(p.z_left / r for p, r in zip(progs_py, dz_py) if p.z_left > 0.0 and r > 0.0)
            s_py = integrate_prepared_epoch(progs_py, dels, dz_py, dt, ctrs_py)
            s_c = runner.integrate_epoch(dz_c, dt)
            assert all(_ulp_distance(a, b) <= 8 for a, b in zip(s_py, s_c))
        z_c, ctrs_c = runner.read_state()
        assert all(z == 0.0 for z in z_c) and all(_ulp_distance(a, b) <= 8 for a, b in zip(ctrs_py, ctrs_c))
        runner.config_rows([rows[0]])
        assert runner.allocate_epoch() == []
        buf = (ctypes.c_double * 4)()
        assert int(runner.library.msim_allocate_epoch(runner.ctx, buf, 4)) != 0

def test_vector_subpipes_target_eff_and_raw_latency_anchor_parity():
    ctx = _make_context()
    groups = (
        VectorInstructionGroup("exp", "vec_sfu", count=10, issue_slots=1, initiation_cycles=1, result_latency_cycles=20),
        VectorInstructionGroup("sum", "vec_reduce", count=4, issue_slots=1, initiation_cycles=1, result_latency_cycles=10, dependencies=("exp",)),
        VectorInstructionGroup("mul", "vec_alu", count=8, issue_slots=1, initiation_cycles=1, dependencies=("sum",)),
    )
    full, half = ctx.bind_vector(groups, target_eff=1.0), ctx.bind_vector(groups, target_eff=0.5)
    assert full.raw_latency_ms == pytest.approx(half.raw_latency_ms, rel=1e-12)
    assert half.span_ms > full.span_ms and all("vector" not in s.work for s in full.stages)
    assert full.vector_projection == pytest.approx(22.0)
    ctx_2k = RecipeContext(
        op=SimpleNamespace(family="elementwise", recipe="direct", metadata={}),
        work={}, resource_names={"vec_issue": "vi", "vec_alu": "va"},
        rates={"vi": 2.0, "va": 2.0}, core_count=1, compute_arch=SimpleNamespace(clock_hz=1000.0),
    )
    b4 = ctx_2k.bind_vector((VectorInstructionGroup("g", "vec_alu", count=4, result_latency_cycles=3),))
    assert b4.span_ms == pytest.approx(4.5, abs=1e-12)
    eng = Engine([Resource("vi", 2.0, policy="fifo"), Resource("va", 2.0)], backend="python")
    done = KernelProgram(b4.stages).start(eng, "vraw", eng.submit("go"))
    eng.advance(until=(done,))
    assert float(done.time) == pytest.approx(4.5, abs=1e-12)
    # FIFO peer holding vi for [0, 5) ms anchors final instruction's 3 ms RAW to t=5 -> 8.0 ms.
    eng2 = Engine([Resource("vi", 2.0, policy="fifo"), Resource("va", 2.0)], backend="python")
    go2 = eng2.submit("go")
    eng2.submit("blk", {"vi": 10.0}, after=(go2,))
    done2 = KernelProgram(ctx_2k.bind_vector((VectorInstructionGroup("g1", "vec_alu", count=1, result_latency_cycles=3),)).stages).start(eng2, "v", go2)
    eng2.advance(until=(done2,))
    assert float(done2.time) == pytest.approx(8.0, abs=1e-12)

def test_ring_buffer_distance_d_credit_and_abi_v3_wire_roundtrip():
    ring = RingBuffer(occurrence=7, die=0, owner=1, pool="l1", allocation="kv_ring", slot_bytes=4096, depth=2)
    gens = [lower_generation(ring, k, producer="load", consumers=("fast_c", "slow_c"), producer_work={"mte2": 1.0},
                             consumer_works={"fast_c": {"cube": 1.0}, "slow_c": {"vec": 3.0}}) for k in range(4)]
    assert gens[2].credit == gens[0].last_use and gens[3].credit == gens[1].last_use
    with PreparedTermsRunner() as runner:
        probe = runner.terms_abi_probe()
        assert probe[0] == TERMS_ABI_VERSION and probe[0] > 2 and probe == expected_terms_abi_probe()
        assert runner.roundtrip_ring(ring) == ring.to_c_records()
    eng = Engine([Resource("mte2", 1.0), Resource("cube", 1.0), Resource("vec", 1.0)], pools={"l1": 8192.0}, pool_units={"l1": 2}, backend="python")
    done = KernelProgram(ring.to_stages()).start(eng, "ring_op", eng.submit("ready"))
    eng.advance(until=(done,))
    assert float(done.time) >= 6.0 - 1e-12

def test_tile_solver_owner_kind_scaling_zero_item_flops_and_ladder_order():
    cand_1pass = TileCandidate("resident_1pass", TileShape(10, 64, 64, head_dim=4096), 1, (4, 8), 10, 3,
                               (LiveBufferSpec("q", "ub", 48 * 1024, 0, 2), LiveBufferSpec("scores", "ub", 160 * 1024, 1, 2)),
                               logical_flops=2 * 10 * 64 * 64, instruction_block=(16, 16, 16), uses_simt=True)
    cand_online = TileCandidate(
        "online_reduce", TileShape(10, 64, 64, head_dim=4096), 1, (4, 8), 10, 3,
        (LiveBufferSpec("q", "ub", 48 * 1024, 0, 2), LiveBufferSpec("a", "ub", 64 * 1024, 0, 1, alias_group="s"),
         LiveBufferSpec("b", "ub", 64 * 1024, 1, 2, alias_group="s"), LiveBufferSpec("v_l2", "l2", 32 * 1024, 0, 2, owner_kind="aiv")),
        logical_flops=2 * 10 * 64 * 64, instruction_block=(16, 16, 16), uses_simt=True, native_mode="narrow_m",
    )
    spec = TilePipelineSpec("attention", (cand_online, cand_1pass), simt_ub_carveout_bytes=32 * 1024,
                            subtile_waves=(SubtileWaveSpec("narrow_m", 32, 64, 64, independently_issuable_m=16),))
    owners = SimpleNamespace(aic_limit=4, aiv_limit=8)
    with pytest.raises(TileInfeasibleError):
        solve_tiles(spec, None, SimpleNamespace(buffers=(("ub", 192 * 1024), ("l2", 192 * 1024)), compute_units=4, stream_owners=owners))
    hw_ok = SimpleNamespace(buffers=(("ub", 192 * 1024), ("l2", 256 * 1024)), compute_units=4, stream_owners=owners)
    sol = solve_tiles(spec, None, hw_ok)
    assert sol.algorithm == "online_reduce" and sol.waves == 3 and sol.rejections[0].algorithm == "resident_1pass"
    empty_c = TileCandidate("resident_1pass", TileShape(0, 64, 64), 1, (1, 1), 0, 0, (), logical_flops=0, instruction_block=(16, 16, 16))
    assert solve_tiles(TilePipelineSpec("empty", (empty_c,)), None, hw_ok).issued_flops == 0

@pytest.mark.parametrize("backend", ["python", "c"])
def test_service_grants_preserve_unsorted_placement_and_event_times(backend):
    def run(observer):
        engine = Engine([Resource("cube", 4.0, units=4)], backend=backend)
        if observer is not None:
            engine.attach_wave_recorder(observer)
        events = []
        for node, owners in enumerate(((2, 0), (1, 3))):
            if observer is not None:
                observer.current_node_idx = node
            events.append(engine.submit(str(node), {"cube": 2.0}, placement={"cube": owners}, coupled=True))
        engine.advance(until=events, first=False)
        return [event.time for event in events]

    observer = ServiceGrantObserver()
    assert run(None) == run(observer) == [1.0, 1.0]
    grants = observer.to_observed_grants(("a", "b"))
    assert {(g.occurrence_id, g.owner, g.start_ns, g.end_ns, g.delivered) for g in grants} == {
        ("a", 0, 0, 1_000_000, 1.0), ("a", 2, 0, 1_000_000, 1.0),
        ("b", 1, 0, 1_000_000, 1.0), ("b", 3, 0, 1_000_000, 1.0),
    }

@pytest.mark.parametrize("backend", ["python", "c"])
def test_sub_nanosecond_service_completes_before_rounding_grants(backend):
    engine = Engine([Resource("r", 1.0)], backend=backend)
    observer = ServiceGrantObserver()
    engine.attach_wave_recorder(observer)
    observer.current_node_idx = 0
    event = engine.submit("half_ns", {"r": 5e-7})
    engine.advance(until=(event,))
    assert event.time == pytest.approx(5e-7)
    assert observer.to_observed_grants(("half_ns",)) == ()

def test_native_grants_account_for_every_concurrent_task():
    count = 4120
    engine = Engine([Resource("r", float(count))], backend="c")
    observer = ServiceGrantObserver()
    engine.attach_wave_recorder(observer)
    names = tuple(f"t{i}" for i in range(count))
    for node, name in enumerate(names):
        observer.current_node_idx = node
        engine.submit(name, {"r": 1.0})
    engine.advance()
    grants = observer.to_observed_grants(names)
    assert len(grants) == count and {g.occurrence_id for g in grants} == set(names)
    assert all(g.delivered == 1.0 for g in grants)
    assert engine.accel_status()["native_steps"] > 0

@pytest.mark.parametrize("backend", ["python", "c"])
@pytest.mark.parametrize("two_predecessors", [False, True])
def test_dependent_service_starts_when_the_last_predecessor_finishes(backend, two_predecessors):
    engine = Engine([Resource(name, 1.0) for name in ("cube", "mte2", "vector")], backend=backend, trace_limit=3)
    predecessors = [engine.submit("load", {"mte2": 1.0})]
    if two_predecessors:
        predecessors.append(engine.submit("compute", {"cube": 0.3}))
    child = engine.submit("consume", {"vector": 0.5}, after=predecessors)
    engine.advance(until=(child,))
    start = next(row["start_ms"] for row in engine.trace if row["task"] == "consume")
    assert start == max(event.time for event in predecessors) == pytest.approx(1.0)
    assert child.time == pytest.approx(start + 0.5)

@pytest.mark.parametrize("period,repeats,pattern,expected_ms", [
    (2.0, 100_000, (("cube", 0.0, 1.0), ("vector", 1.0, 2.0)), 200_000.0),
    (2.0, 5, (("cube", 0.0, 1.0), ("vector", 0.0, 1.0)), 5.0),
    (1.5e-6, 10, (("cube", 0.0, 1.5e-6),), 15e-6),
])
def test_periodic_busy_union_conserves_service_across_repeats(period, repeats, pattern, expected_ms):
    from mindie_sim.onboarding.whitebox_runner import _BusyUnionObserver

    observer = _BusyUnionObserver()
    observer.on_period(0.0, period, repeats, service_pattern=pattern)
    assert observer.total_busy_ms() == pytest.approx(expected_ms)

def test_serial_allocations_preserve_occurrence_identity_and_release_the_slot():
    engine = Engine([Resource("mte2", 1.0)], pools={"l1": 64.0}, pool_units={"l1": 1},
                    backend="python", trace_limit=2)
    keys = [OccurrenceKey(runtime_uid="rt", launch=i) for i in (1, 2)]
    for key in keys:
        alloc = Allocation(f"buf{key.launch}", "l1", 64.0, (0,))
        engine.submit(alloc.name, {"mte2": 1.0}, acquire=(alloc,), free=(alloc,), metadata={"occurrence": key})
    engine.advance()
    assert [row["metadata"]["occurrence"] for row in engine.trace] == keys
    assert [(row["name"], row["action"], row["time"]) for row in engine.allocation_trace] == [
        ("buf1", "acquire", 0.0), ("buf1", "release", 1.0),
        ("buf2", "acquire", 1.0), ("buf2", "release", 2.0)]

def test_program_facade_reexports_and_stage_builtin_metadata_guards():
    for name in """
        Allocation DirtyWriteback Engine Event GENERIC_MATRIX_PROGRAM HostStaging KernelProgram ModelingError
        NATIVE_WORK_UNITS NativeProgram PIPELINE_COHORT_WINDOW PIPELINE_FOLD_WINDOW PIPELINE_UNROLL_LIMIT
        ParallelProgram PipelinePhase PipelineProgram PipelineTemplate PipelineTemplateProgram Program
        ProgramFragment Recipe RecipeBook RecipeContext RecurrentScan ResourceEnvelope Stage TransferRecipe
        coalesce_execution_program coalesce_programs fold_interior_pipeline indexed_pipeline_consumer_available
        interior_fold_plan is_pure_static_program merge_programs program_signature
        proportional_coupled_placement scope_allocations split_cohort_kernel stage_from_fragment unit_group
    """.split():
        assert hasattr(prog_pkg, name) and name in prog_pkg.__all__
    st = Stage("meta_ok", work={"r": 1.0}, metadata={"tag": "x", "k": 2, "flag": True, "scale": 0.5})
    kp_tag = KernelProgram((Stage("acq", acquire=(Allocation("b", "l1", 64.0, (0,)),), metadata={"stage_id": 7}),
                            Stage("wk", {"r": 1.0}, after=("acq",), free=(Allocation("b", "l1", 64.0, (0,)),))))
    assert any(s.metadata.get("stage_id") == 7 for s in coalesce_execution_program(kp_tag).stages)
    eng = Engine([Resource("r", 1.0)], backend="python", trace_limit=8)
    ev = KernelProgram((st,)).start(eng, "op", eng.submit("go"))
    eng.advance(until=(ev,))
    assert any(row["metadata"] == {"tag": "x", "k": 2, "flag": True, "scale": 0.5} for row in eng.trace)
    for metadata in ({"occurrence": 1}, *({"nested": v} for v in ((1, 2), b"bytes", None, [1, 2], Fraction(1, 8)))):
        with pytest.raises(ModelingError, match="stage_metadata_invalid"):
            Stage("bad_metadata", metadata=metadata)
    with pytest.raises(ModelingError, match="stage_cohort_widened_invalid"):
        Stage("bad_w", cohort_widened=(123,))  # type: ignore[arg-type]

@pytest.mark.parametrize("tails,count,load,latency,cube,expected", [
    ((1, 7), 32, 0.25, 0, 0, 32), ((1, 3), 20, 0.365, 0, 0, 29.2), ((1, 1), 32, 0.25, 0.125, 0.5, 32),
    pytest.param((1,) * 32, 32, 0.25, 0, 2, 320, id="shared_hbm_burst_then_private_compute")])
def test_refill_peers_match_literal_service(tails, count, load, latency, cube, expected):
    peers = len(tails)
    context = RecipeContext(None, {}, {"hbm": "hbm", "cube": "cube"}, {"hbm": 1, "cube": peers}, peers, resource_units={"cube": peers})
    def tile(index, peer, tail):
        read = Stage("load", {"hbm": load}, latency=latency if peer and index == count - 1 else 0)
        if cube:
            return read, Stage("compute", {"cube": cube}, after=("load",), placement={"cube": (peer,)})
        if index < count - tail:
            return read, Stage("compute", after=("load",), latency=2 * load)
        return read, Stage("store", after=("load",), latency=load), Stage("drain", after=("store",), latency=load)
    def run(programs):
        engine = Engine([Resource("hbm", 1), Resource("cube", peers, units=peers)], backend="python")
        ready = engine.submit("ready")
        ends = tuple(pipe.start(engine, f"peer{idx}", ready) for idx, pipe in enumerate(programs))
        dependent = engine.submit("dependent", after=(ends[-1],), latency=10)
        engine.advance(until=(*ends, dependent), first=False)
        return max(end.time for end in ends), dependent.time, engine.serviced["hbm"]
    specs = tuple(refill_steady.PeriodicPipelineSpec(count, 1, lambda j, i=i: tile(j, i, tails[i]), tail=tails[i]) for i in range(peers))
    literal = run(spec.literal() for spec in specs)
    assert literal == pytest.approx((expected, expected + 10, peers * count * load))
    group = refill_steady.compact_periodic_peer_group(context, specs)
    individual = tuple(refill_steady.compact_periodic_pipeline(
        context, count, spec.slots, spec.tile_fn, tail=spec.tail, peer_counts=(count,) * peers, peer_tails=tails,
    ) for spec in specs)
    assert group == individual == tuple(spec.literal() for spec in specs)
    assert run(group) == literal

@pytest.mark.parametrize("sku", P05_SKUS)
def test_live_admission_uses_sku_curve_past_capacity(sku):
    case = parse_case(compose_case("qwen3-32b", sku, tp=1, dp=1, pp=1, layers=1))
    execution = Execution(case, CostModel(case))
    view = execution.engine.admission_memory_view
    capacity = view.effective_l2_bytes(0)
    assert view._hit_fraction(capacity, capacity) == 1
    assert 0.99 < view._hit_fraction(capacity + 1, capacity) < 1
    execution.reset_engine()
    assert execution.engine.admission_memory_view is execution.admission_memory_view is not view
    assert execution.admission_memory_view.residency_fingerprint() == view.residency_fingerprint()
    assert AdmissionMemoryView({0: capacity})._hit_fraction(capacity + 1, capacity) == 0

def test_isolated_read_replay_rebinds_prepared_recipe_to_cold_cache():
    case = parse_case(compose_case("qwen3-32b", "Ascend950DT_9582", tp=1, dp=1, pp=1, layers=1))
    size, sector = 672, 32
    runs = AddressRunPlan((AddressRun(0, 1, size, size),), sector_bytes=sector,
                          l2_line_bytes=sector, l2_sector_bytes=sector, hbm_burst_granule_bytes=sector)
    access = MemoryAccessPattern("copy", 0, 0, 0, "read", "mte2_aiv", size, runs, ReuseKey(0, "copy", 0))
    traffic = []
    @dataclass
    class Read:
        view: AdmissionMemoryView | None = None
        def prepare(self, op, geometry):
            return op, replace(self, view=geometry.admission_memory_view)
        def lower(self, context):
            bound = context.bind_access(access, self.view)
            def start(engine, name, ready, *, owner=""):
                engine.add_hbm_bus(bound.bind_config[0])
                return engine.submit_bound_access(name, bound, after=(ready,), memory_state=self.view, pattern=access,
                                                  on_delivery=lambda ledger, *_: traffic.append(
                                                      (ledger.dram_sector_bytes, ledger.l2_hit_bytes)))
            return SimpleNamespace(start=start)
    recipes = RecipeBook({"cache_probe": Read()})
    shape = (size,)
    op = OpSpec("first", "elementwise", "prefill", 0, 0, (), shape, shape, shape, shape,
                recipe="cache_probe", hbm_bytes=size)
    execution = Execution(case, CostModel(case), recipes=recipes)
    for reset in (False, False, True):
        if reset:
            execution.reset_engine()
        prepared, recipe = execution.prepare(op)
        program = execution.lower_program(execution._context(prepared, 0, "none", recipe=recipe), recipe)
        engine = execution.engine
        engine.advance(until=(program.start(engine, "probe", engine.submit("ready")),))
    assert traffic == [(672, 0), (0, 672), (672, 0)]
    traffic.clear()
    operator_costs(case, (op, replace(op, op_id="second", hbm_bytes=size + 1)), recipes=recipes)
    assert traffic == [(672, 0), (672, 0)]


def test_service_grant_observer_conserves_skipped_periodic_cycles_and_rejects_silent_drop():
    """Skipped periodic cycles conserve observed grants in O(1) entries and fail loud if empty."""
    from mindie_sim.prediction.engine import ExecutionError
    from mindie_sim.prediction.program import PipelinePhase, PipelineTemplate
    from mindie_sim.prediction.runtime.exposed_observer import ObservedInterval, attribute_stream0_intervals
    from mindie_sim.prediction.scheduling.templates import start_pipeline_template

    eng = Engine([Resource("cube", 1)], backend="python")
    obs = ServiceGrantObserver()
    obs.current_node_idx = 0
    eng.attach_wave_recorder(obs)
    phase = PipelinePhase("body", (Stage("tile", {"cube": 1}),), 1000)
    done = start_pipeline_template(eng, PipelineTemplate(0, 1, (phase,)), "op", eng.submit("ready"))
    eng.advance(until=(done,))
    assert eng._periodic_detector.stats()["skipped_cycles"] == 997
    grants = obs.to_observed_grants(["op"])
    assert sum(g.delivered for g in grants) == pytest.approx(1000.0)
    assert sum(amt for *_, amt in obs.grants) == pytest.approx(1000.0)
    assert len(obs._grant_acc) <= 10
    exp = attribute_stream0_intervals(
        [ObservedInterval("op", "main", 0, 0, round(done.time * 1e6))],
        start_ns=0, terminal_ns=round(done.time * 1e6), grants=grants,
    )
    assert exp.exposed_ns_by_op == {"op": 1_000_000_000} and exp.conservation_delta_ns == 0

    class _EmptyPatternRecorder:
        def register_pipe(self, _name: str) -> int:
            return 0
        def on_submit(self, *_a, **_kw) -> None:
            return None
        def on_admit(self, *_a, **_kw) -> None:
            return None
        def on_service(self, *_a, **_kw) -> None:
            return None
        def allocate_slot_generation(self, *_a, **_kw) -> tuple[int, int]:
            return 0, 1
        def release_slot_generation(self, *_a, **_kw) -> tuple[int, int]:
            return 0, 1
        def on_allocation(self, *_a, **_kw) -> None:
            return None
        def extract_cycle_pattern(self, *_a, **_kw):
            return (), ()
        def on_period(self, *_a, **_kw) -> None:
            return None

    eng_bad = Engine([Resource("cube", 1)], backend="python")
    eng_bad.attach_wave_recorder(_EmptyPatternRecorder())
    done_bad = start_pipeline_template(eng_bad, PipelineTemplate(0, 1, (phase,)), "op", eng_bad.submit("ready"))
    with pytest.raises(ExecutionError, match="periodic_skip_missing_proved_cycle_pattern"):
        eng_bad.advance(until=(done_bad,))
    # Rejection must be atomic: clock stays aligned with actual serviced work and resumes cleanly.
    assert eng_bad.now == pytest.approx(eng_bad.serviced["cube"])
    eng_bad.detach_wave_recorder()
    eng_bad.advance(until=(done_bad,))
    assert done_bad.time == pytest.approx(1000.0)

    # Alternating p->q / q->p periodic cycles preserve disjoint within-cycle phases (zero false contention).
    eng_alt = Engine([Resource("p", 1), Resource("q", 1)], backend="python")
    obs_alt = ServiceGrantObserver()
    eng_alt.attach_wave_recorder(obs_alt)
    r_alt = eng_alt.submit("ready")
    m_st = (Stage("mp", {"p": 1}), Stage("mq", {"q": 1}, after=("mp",)))
    a_st = (Stage("aq", {"q": 1}), Stage("ap", {"p": 1}, after=("aq",)))
    obs_alt.current_node_idx = 0
    d_main = start_pipeline_template(eng_alt, PipelineTemplate(0, 1, (PipelinePhase("m", m_st, 20),)), "main", r_alt)
    obs_alt.current_node_idx = 1
    d_aux = start_pipeline_template(eng_alt, PipelineTemplate(0, 1, (PipelinePhase("a", a_st, 20),)), "aux", r_alt)
    eng_alt.advance(until=(d_main, d_aux), first=False)
    assert eng_alt._periodic_detector.stats()["skipped_cycles"] >= 15
    alt_exp = attribute_stream0_intervals(
        [
            ObservedInterval("main", "main", 0, 0, round(d_main.time * 1e6)),
            ObservedInterval("aux", "aux", 1, 0, round(d_aux.time * 1e6)),
        ],
        start_ns=0, terminal_ns=round(eng_alt.now * 1e6),
        grants=obs_alt.to_observed_grants(("main", "aux")),
    )
    assert alt_exp.exposed_ns_by_op == {"main": 40_000_000, "aux": 0}


@pytest.mark.parametrize("backend", ["python", "c"])
def test_implicit_full_width_placement_matches_explicit_owners_across_backends(backend: str):
    """Implicit placement distributes service across active owners matching explicit placement & capacity."""
    from mindie_sim.prediction.runtime.exposed_observer import ObservedInterval, attribute_stream0_intervals

    results = []
    for wide_place in (None, {"aiv": (0, 1)}):
        eng = Engine([Resource("aiv", 2, units=2)], backend=backend)
        obs = ServiceGrantObserver()
        eng.attach_wave_recorder(obs)
        ids, events = ("wide", "narrow"), []
        for idx, place in enumerate((wide_place, {"aiv": (1,)})):
            obs.current_node_idx = idx
            events.append(eng.submit(ids[idx], {"aiv": 1}, placement=place))
        eng.advance()
        if backend == "c":
            assert eng.accel_status()["native_steps"] > 0
        intervals = [
            ObservedInterval(ids[i], ("main", "aux")[i], i, 0, round(ev.time * 1e6))
            for i, ev in enumerate(events)
        ]
        grants = obs.to_observed_grants(ids)
        exposure = attribute_stream0_intervals(
            intervals, start_ns=0, terminal_ns=round(eng.now * 1e6), grants=grants,
        )
        wide_owners = sorted(
            (g.owner, round(g.delivered, 6)) for g in grants if g.occurrence_id == "wide"
        )
        results.append((round(eng.now, 6), exposure.exposed_ns_by_op, wide_owners))
    assert results[0] == results[1] == (1.5, {"wide": 500_000, "narrow": 1_000_000}, [(0, 0.5), (1, 0.5)])

    # Lower-priority unplaced work must emit on residual owner 0, never over-subscribing saturated owner 1.
    if backend == "python":
        eng_t = Engine([Resource("aiv", 4, units=2)], backend="python")
        obs_t = ServiceGrantObserver()
        eng_t.attach_wave_recorder(obs_t)
        obs_t.current_node_idx = 0
        eng_t.submit("hi", {"aiv": 2}, priority=0, placement={"aiv": (1,)})
        obs_t.current_node_idx = 1
        eng_t.submit("lo", {"aiv": 2}, priority=1)
        eng_t.advance()
        totals: dict[int | None, float] = {}
        for gr in obs_t.to_observed_grants(("hi", "lo")):
            totals[gr.owner] = totals.get(gr.owner, 0.0) + gr.delivered
        assert totals == {0: 2.0, 1: 2.0}


def test_coalesce_preserves_cross_tile_previous_cohort_and_fifo_boundaries():
    """Coalescing must preserve cross-tile previous targets, cohort barriers, and FIFO boundaries."""
    tiles = (
        (Stage("convert", {"vector": 4}), Stage("pack", {"vector": 4}, after=("convert",))),
        (Stage("store", {"hbm": 8}, previous=("pack",)),),
    )
    pipe = PipelineProgram(2, 2, tiles.__getitem__)
    for body in (pipe, coalesce_execution_program(pipe), coalesce_execution_program(pipe, fair_resources={"vector"})):
        eng = Engine([Resource("vector", 1), Resource("hbm", 1)], backend="python")
        done = body.start(eng, "kernel", eng.submit("ready"))
        eng.advance(until=(done,))
        assert done.time == pytest.approx(16.0) and eng.serviced == {"vector": 8.0, "hbm": 8.0}

    for policy, cohort, release, expected in (("fair", 4, 4, 22.0), ("fifo", 1, 1, 20.0)):
        prog = KernelProgram((
            Stage("head", {"vector": 4}),
            Stage("tail", {"vector": 8}, after=("head",), cohort_size=cohort),
        ))
        variants = (prog, coalesce_execution_program(prog)) + (
            (coalesce_execution_program(prog, fair_resources={"vector"}),) if policy == "fair" else ()
        )
        for body in variants:
            eng = Engine([Resource("vector", 1, policy=policy), Resource("hbm", 1)], backend="python")
            done = body.start(eng, "kernel", eng.submit("ready"))
            peer = eng.submit("peer", {"vector": 8}, release=release)
            copy = eng.submit("copy", {"hbm": 8}, after=(done if policy == "fair" else peer,))
            end = eng.submit("end", after=(done, peer, copy))
            eng.advance(until=(end,))
            assert end.time == pytest.approx(expected)

    # Identical cohort_size > 1 stages still carry an inter-stage barrier and must never merge.
    coh_prog = KernelProgram((
        Stage("s1", {"u": 2}, placement={"u": (0, 1)}, cohort_size=2,
              cohort_widened=("u",), cohort_members={"u": ((0,), (1,))}),
        Stage("s2", {"u": 2}, after=("s1",), placement={"u": (0, 1)}, cohort_size=2,
              cohort_widened=("u",), cohort_members={"u": ((0,), (1,))}),
    ))
    for body in (coh_prog, coalesce_execution_program(coh_prog, fair_resources={"u"})):
        eng_c = Engine([Resource("u", 2, units=2)], backend="python")
        done_c = body.start(eng_c, "k", eng_c.submit("r"))
        eng_c.submit("p0", {"u": 1}, placement={"u": (0,)}, release=0)
        eng_c.submit("p1", {"u": 1}, placement={"u": (1,)}, release=2)
        eng_c.advance(until=(done_c,))
        assert done_c.time == pytest.approx(4.0)


def test_interior_fold_rejects_multistage_pipeline_and_preserves_dma_overlap():
    """Interior folding only batches single-stage non-cohort fair tiles and is fail-closed."""
    def multi_tile(i):
        n = (4, 4, 1, 2)[i]
        return Stage("load", {"hbm": n}), Stage("compute", {"vector": n}, after=("load",))

    assert prog_pkg.interior_fold_plan(
        PipelineProgram(4, 1, multi_tile, interior_fold=True), fair_resources={"hbm", "vector"},
    ) is None
    for fold in (False, True):
        eng = Engine([Resource("hbm", 1), Resource("vector", 1)], backend="python")
        done = PipelineProgram(4, 1, multi_tile, interior_fold=fold).start(eng, "kernel", eng.submit("ready"))
        eng.submit("dma", {"hbm": 4}, release=4)
        eng.advance(until=(done,))
        assert done.time == pytest.approx(22.0) and eng.serviced == {"hbm": 15.0, "vector": 11.0}

    # Direct calls without fair_resources fail closed (protecting FIFO); verified fair single-stage still folds.
    def single_tile(i):
        n = (3, 3, 1, 2)[i]
        return (Stage("vec", {"vector": n}),)

    single_prog = PipelineProgram(4, 1, single_tile, interior_fold=True)
    assert prog_pkg.fold_interior_pipeline(single_prog) is None
    assert prog_pkg.interior_fold_plan(single_prog, fair_resources={"vector"}) is not None
    times = []
    for fold in (False, True):
        eng = Engine([Resource("vector", 1)], backend="python")
        done = PipelineProgram(4, 1, single_tile, interior_fold=fold).start(eng, "kernel", eng.submit("ready"))
        eng.submit("peer", {"vector": 4}, release=2)
        eng.advance(until=(done,))
        times.append(done.time)
    assert times[0] == pytest.approx(times[1])

    # Cohort tiles (cohort_size > 1) keep inter-tile synchronization barriers unmerged.
    def coh_tile(i):
        if i < 2:
            return (Stage("s", {"u": 2}, placement={"u": (0, 1)}, cohort_size=2,
                          cohort_widened=("u",), cohort_members={"u": ((0,), (1,))}),)
        return (Stage("t", {"v": 1 if i == 2 else 2}),)

    assert prog_pkg.interior_fold_plan(
        PipelineProgram(4, 1, coh_tile, interior_fold=True), fair_resources={"u", "v"},
    ) is None
    for fold in (False, True):
        eng_cf = Engine([Resource("u", 2, units=2), Resource("v", 1)], backend="python")
        done_cf = PipelineProgram(4, 1, coh_tile, interior_fold=fold).start(eng_cf, "p", eng_cf.submit("r"))
        eng_cf.submit("p0", {"u": 1}, placement={"u": (0,)}, release=0)
        eng_cf.submit("p1", {"u": 1}, placement={"u": (1,)}, release=2)
        eng_cf.advance(until=(done_cf,))
        assert done_cf.time == pytest.approx(7.0)


def test_refill_steady_compaction_matches_literal_replay_after_heavy_prologue():
    """Periodic compaction proves full prologue drain without retaining probe trace caches."""
    from _support.matrix_cases import matrix_setup

    assert not hasattr(refill_steady._periodic_probe, "cache_info")
    _, ctx = matrix_setup(cores=1)

    def tile(idx):
        load = (1 if idx == 0 else 1.25) * ctx.rates["mte2"]
        cube = (20 if idx == 0 else 1) * ctx.rates["cube"]
        return (
            Stage("load", {"mte2": load}),
            Stage("compute", {"cube": cube}, previous=("load",)),
        )

    def span(program):
        eng, _ = matrix_setup(cores=1)
        done = program.start(eng, "pipe", eng.submit("ready"))
        eng.advance(until=(done,))
        return round(done.time, 6)

    compacted = refill_steady.compact_periodic_pipeline(ctx, 100, 4, tile)
    literal = PipelineProgram(100, 4, tile)
    assert span(compacted) == span(literal) == 140.0
