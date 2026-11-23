"""Spine prevention locks for primitive wiring (Lock 1) and non-degenerate bounds/search (Lock 5).

[Role] Enforce RC-1 single-source primitive rate wiring across U/M/FW/DSL vs
  dual bounds and RC-4 non-zero lower bounds, K2/K3 auto-generation,
  compatible prefetch+streams, search funnel disclosure, and O(N log N) sweep.
[Invariants] Tier-0 in-memory execution with zero device contexts or host I/O.
[Timeouts & Locks] Bounded 1-layer replays and synthetic micro-graphs; no locks.
[Side-Effect Policy] Pure read-only simulation assertions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import pytest

from mindie_sim import api
from mindie_sim.compose import compose_case
from mindie_sim.optimization import (
    CandidateSearchStats,
    SemanticWorkItem,
    build_dual_inputs_from_scenario,
    certify_lower_bound,
    select_catlass_candidates,
    select_dsl_candidates,
)
from mindie_sim.prediction.operators.vector import TiledPointwise
from mindie_sim.prediction.program import RecipeBook, Stage
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.runtime.engine import Engine
from mindie_sim.prediction.runtime.engine_types import Allocation, Resource
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, AddressRunPlan, AdmissionMemoryView, HbmBusSpec, PeriodicSectorRun, SectorCoverage,
    _measure_periodic_combo,
)
from mindie_sim.prediction.runtime.periodic_runs import _coverage_overlap, _prune_periodic_subruns
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.streams import MAIN_STREAM, StreamCoreBudget, StreamLane
from mindie_sim.prediction.structure import parse_case
from mindie_sim.primitives.loader import load_primitives, resolve_sim_rate_overlay
from mindie_sim.primitives.schema import compute_envelope_sha256, parse_primitive_table
from mindie_sim.rewrite.oracle import OracleCache, ScenarioRequest, finalize_plan, graph_costs, prepare_scenario
from mindie_sim.rewrite.schema import Edit, OpRef, PortRef, StreamSchedule, WeightPrefetch
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Device
from mindie_sim.schema.hardware_links import Link
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.results import LoweringRequest
from mindie_sim.templates import catlass_lower, compile_fuse_region, registry as templates_registry
from mindie_sim.templates.catlass_catalog import CatlassProblem, enumerate_catlass_specs
from _support.axes import REPRESENTATIVE_SKUS
from _support.collective_cases import case as col_case, exchange
from _support.linear_cases import recipe_context, sku_planning_context
from scripts._arch_governance_runtime import verify_ag006_ownership
from scripts.check_arch_governance import scan_repository

pytestmark = pytest.mark.tier0

SKU_A5 = "Ascend950DT_9582"
DECODE_CTX = 64
HALF_SCALE = 0.5
DOUBLE_SCALE = 2.0
SWEEP_INTERVAL_COUNT = 2000
SWEEP_STRIDE = 128
SWEEP_WIDTH = 64
SWEEP_MAX_SECONDS = 0.5
UNIT_RATE_PER_S = 10**9
UNIT_PEAK_FLOPS = 10**12
UNIT_LINK_BPMS = 10**6
WORK_UNITS = 10**9
LINK_BYTES = 10**6
COPY_BYTES = 10**6
MEM_CAPACITY = 16 * 1024 * 1024
PREFETCH_BYTES = 1024


def test_recipe_ownership_checks_factories_and_shipped_selections(monkeypatch):
    class FlatRecipe:
        def lower(self, context):
            raise AssertionError("The ownership gate should need only selection")

    original_factories = RecipeBook._mapping_factories
    with monkeypatch.context() as patch:
        patch.setattr(RecipeBook, "_mapping_factories", staticmethod(
            lambda: {**original_factories(), "tiled_pointwise": FlatRecipe},
        ))
        assert any(v.fingerprint == "AG006:factory:tiled_pointwise"
                   for v in verify_ag006_ownership())

    original_select = RecipeBook.select

    def select(self, op):
        binding = original_select(self, op)
        return FlatRecipe() if isinstance(binding, TiledPointwise) else binding

    monkeypatch.setattr(RecipeBook, "select", select)
    findings = verify_ag006_ownership()
    assert {v.path.as_posix() for v in findings} == {"examples/cases/qwen38-a5.json"}
    assert {v.fingerprint.split(":")[2] for v in findings} == {"prefill", "decode"}


def test_primitive_concepts_reject_sibling_module_definitions(tmp_path):
    primitives = tmp_path / "mindie_sim" / "primitives"
    primitives.mkdir(parents=True)
    concepts = {"loader": "CalibratedRateOverlay", "schema": "PrimitiveTable"}
    for module, name in concepts.items():
        (primitives / f"{module}.py").write_text(f"class {name}:\n    pass\n")
    assert scan_repository(tmp_path) == ()

    shadow = primitives / "shadow.py"
    shadow.write_text("".join(f"class {name}:\n    pass\n" for name in concepts.values()))
    assert {v.fingerprint for v in scan_repository(tmp_path)} == {
        f"AG008:duplicate_concept:mindie_sim/primitives/shadow.py:{name}"
        for name in concepts.values()
    }


def _scale_param_curves(params: dict, scale: float) -> None:
    if not isinstance(params, dict) or scale == 1.0:
        return
    for key, val in list(params.items()):
        if isinstance(val, (int, float)) and not isinstance(val, bool) and (
            key.endswith("_bps") or key.endswith("_bps_p50") or key.endswith("_bytes_per_s")
        ):
            params[key] = float(val) * scale
        elif isinstance(val, dict) and (
            key.endswith("_bps") or key.endswith("_bytes_per_s") or "flops_per_s" in key
        ):
            for k, v in val.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    val[k] = float(v) * scale
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and isinstance(item.get("bps_p50"), (int, float)):
                    item["bps_p50"] = float(item["bps_p50"]) * scale


def _resigned_table(base_table, *, p50_scale: float = 1.0, cap_scale: float = 1.0):
    raw = deepcopy(base_table.to_envelope_dict())
    for pid, cell in raw["primitives"].items():
        if pid not in {"P01_datacopy", "P05_l2_residency_reuse", "P06_mmad", "P07_vec_alu"}:
            continue
        _scale_param_curves(cell.get("parameters"), p50_scale)
        for metric, band in cell.get("metrics", {}).items():
            if metric == "hbm_rw_alpha" or not isinstance(band, dict):
                continue
            if cap_scale != 1.0 and band.get("certified_upper_capacity"):
                band["certified_upper_capacity"] = float(band["certified_upper_capacity"]) * cap_scale
            if p50_scale != 1.0 and band.get("p50") is not None:
                for key in ("p05", "p50", "p95", "p99", "trimmed_mean"):
                    if band.get(key) is not None:
                        band[key] = float(band[key]) * p50_scale
        for mode_metrics in cell.get("modes", {}).values():
            if not isinstance(mode_metrics, dict):
                continue
            for band in mode_metrics.values():
                if not isinstance(band, dict):
                    continue
                if cap_scale != 1.0 and band.get("certified_upper_capacity"):
                    band["certified_upper_capacity"] = float(band["certified_upper_capacity"]) * cap_scale
                if p50_scale != 1.0 and band.get("p50") is not None:
                    for key in ("p05", "p50", "p95", "p99", "trimmed_mean"):
                        if band.get(key) is not None:
                            band[key] = float(band[key]) * p50_scale
    raw["envelope_sha256"] = compute_envelope_sha256(raw)
    return parse_primitive_table(raw)


def test_lock1_sim_rate_perturbation_moves_all_four_rungs_preserving_bounds():
    base_table = load_primitives(SKU_A5)
    half_sim_table = _resigned_table(base_table, p50_scale=HALF_SCALE)
    case = compose_case("qwen3-32b", hardware=SKU_A5, tp=1, layers=1)
    step = StepRequest("decode", context=DECODE_CTX)

    base = api.optimize_ladder(case, step, primitive_table=base_table, max_trials=4)
    slow = api.optimize_ladder(case, step, primitive_table=half_sim_table, max_trials=4)

    assert slow.ladder.u_unmerge_ns > base.ladder.u_unmerge_ns
    assert slow.ladder.m_mainline_ns > base.ladder.m_mainline_ns
    assert slow.ladder.a_star_framework_ns > base.ladder.a_star_framework_ns
    assert slow.ladder.a_star_dsl_ns > base.ladder.a_star_dsl_ns
    assert slow.ladder.l_star_lean_ns == base.ladder.l_star_lean_ns
    assert slow.ladder.l_star_hw_ns == base.ladder.l_star_hw_ns
    assert (
        0 < slow.ladder.l_star_hw_ns <= slow.ladder.l_star_lean_ns
        <= slow.ladder.a_star_dsl_ns <= slow.ladder.a_star_framework_ns
        <= min(slow.ladder.m_mainline_ns, slow.ladder.u_unmerge_ns)
    )


def test_lock1_certified_capacity_perturbation_moves_hw_bound_preserving_sim():
    base_table = load_primitives(SKU_A5)
    relaxed_table = _resigned_table(base_table, cap_scale=DOUBLE_SCALE)
    case = compose_case("qwen3-32b", hardware=SKU_A5, tp=1, layers=1)
    step = StepRequest("decode", context=DECODE_CTX)

    base = api.optimize_ladder(case, step, primitive_table=base_table, max_trials=0)
    relaxed = api.optimize_ladder(case, step, primitive_table=relaxed_table, max_trials=0)

    assert relaxed.ladder.u_unmerge_ns == base.ladder.u_unmerge_ns
    assert relaxed.ladder.m_mainline_ns == base.ladder.m_mainline_ns
    assert relaxed.ladder.a_star_framework_ns == base.ladder.a_star_framework_ns
    assert relaxed.ladder.a_star_dsl_ns == base.ladder.a_star_dsl_ns
    assert relaxed.ladder.l_star_lean_ns == base.ladder.l_star_lean_ns
    assert relaxed.ladder.l_star_hw_ns is not None and base.ladder.l_star_hw_ns is not None
    assert relaxed.ladder.l_star_hw_ns < base.ladder.l_star_hw_ns


def test_lock1_default_cube_peak_geometry_guessing_is_deleted():
    assert not hasattr(templates_registry, "default_cube_peak")
    case = Case.from_mapping(compose_case("qwen3-32b", hardware=SKU_A5, tp=1, layers=1))
    die = case.machine.dies[0]
    dp = datapath_for_die(die)
    assert dp.cube_peak("bf16") == pytest.approx(die.peak("bf16"))


def test_lock5_dual_bound_nonzero_across_all_work_classes_and_partial_annotations():
    die0 = Device(
        rank=0, compute_units=1, memory_bytes=MEM_CAPACITY,
        rates=(("hbm", UNIT_RATE_PER_S), ("mte2", UNIT_RATE_PER_S), ("vector", UNIT_RATE_PER_S)),
        compute_peaks=(("bf16", UNIT_PEAK_FLOPS),),
    )
    die1 = Device(
        rank=1, compute_units=1, memory_bytes=MEM_CAPACITY,
        rates=(("hbm", UNIT_RATE_PER_S), ("mte2", UNIT_RATE_PER_S), ("vector", UNIT_RATE_PER_S)),
        compute_peaks=(("bf16", UNIT_PEAK_FLOPS),),
    )
    link = Link(name="fabric", capacity_bytes_per_ms=UNIT_LINK_BPMS, members=(0, 1))
    machine = SimpleNamespace(dies=(die0, die1), links=(link,))
    shape = (1, 16)

    cube_op = OpSpec("cube0", "gemm", "decode", 0, 0, (), shape, shape, shape, shape,
                     dtype="bf16", flops=WORK_UNITS)
    vec_op = OpSpec("vec0", "rmsnorm", "decode", 0, 0, (), shape, shape, shape, shape,
                    dtype="bf16", vector_ops=WORK_UNITS)
    link_op = OpSpec("comm0", "allreduce", "decode", 0, 0, (), shape, shape, shape, shape,
                     dtype="bf16", link_bytes=LINK_BYTES)
    copy_op = OpSpec("copy0", "tensor_copy", "decode", 0, 0, (), shape, shape, shape, shape,
                     dtype="bf16", mte2_bytes=COPY_BYTES)

    for label, op in (("cube", cube_op), ("vector", vec_op), ("link", link_op), ("copy", copy_op)):
        scenario = SimpleNamespace(machine=machine, atomic_ops=(op,))
        prog, dag, _ = build_dual_inputs_from_scenario(scenario)
        cert, _ = certify_lower_bound(prog, dag)
        assert cert.l_star_lean_ns > 0, label

    # Partial semantic_work on op 0 must NOT zero out inferred work on unannotated op 1.
    annotated = replace(cube_op, metadata=(
        ("semantic_work", (SemanticWorkItem("explicit_mac", "matrix_mac", "bf16", WORK_UNITS, ("cube",)),)),
    ))
    partial = SimpleNamespace(machine=machine, atomic_ops=(annotated, vec_op))
    prog_p, dag_p, names_p = build_dual_inputs_from_scenario(partial)
    cert_p, _ = certify_lower_bound(prog_p, dag_p)
    assert any(name.startswith("vector@") for name in names_p)
    assert cert_p.l_star_lean_ns > 0


def test_lock5_k2_k3_autogeneration_and_search_stats_disclosure():
    base_table = load_primitives(SKU_A5)
    case_ds = compose_case("deepseek-v4-flash", hardware=SKU_A5, tp=1, layers=1)
    report = api.optimize_ladder(case_ds, StepRequest("decode", context=DECODE_CTX),
                                 primitive_table=base_table, max_trials=2)
    assert isinstance(report.search_stats, CandidateSearchStats)
    assert report.search_stats.eligible_sites > 0
    assert report.search_stats.generated_candidates > 0
    assert report.search_stats.replayed_candidates > 0
    assert (
        report.search_stats.replayed_candidates + report.search_stats.rejected_candidates
        == report.search_stats.generated_candidates
    )
    assert not report.search_stats.rejected_candidates or bool(report.search_stats.rejection_reasons)
    assert isinstance(report.primitive_coverage_gaps, tuple)
    unmeasured_table = load_primitives("Ascend950PR_9579")
    case_gap = compose_case("deepseek-v4-flash", hardware="Ascend950PR_9579", tp=1, layers=1)
    gap_report = api.optimize_ladder(case_gap, StepRequest("decode", context=DECODE_CTX),
                                     primitive_table=unmeasured_table, max_trials=1)
    assert isinstance(gap_report.primitive_coverage_gaps, tuple) and gap_report.primitive_coverage_gaps

    spine = PredictionSpine()
    case_qwen = compose_case("qwen3-32b", hardware=SKU_A5, tp=4, layers=1, quant="bf16")
    case_obj = Case.from_mapping(case_qwen)
    dp = datapath_for_die(case_obj.machine.dies[0])
    prep_q = spine.lower(case_obj, LoweringRequest(step=StepRequest("decode", context=DECODE_CTX),
                                                   tier="unmerge", apply_body_defaults=False))
    qkv = next(op for op in prep_q.graph.executable[0].ops if op.op_id.endswith(".qkv"))
    assert len(select_catlass_candidates(qkv, dp, dp.aic_count, dp.aiv_count)) > 0

    mla_op = OpSpec(
        "s0.l0.mla_prolog", "mla_prolog", "decode", 0, 0, (), (4, 7168), (4, 7168), (4, 7168), (4, 7168),
        dtype="bf16", recipe="mla_prolog_v4",
    )
    dsl_cands = select_dsl_candidates((mla_op,), dp, dp.aic_count, dp.aiv_count)
    assert any(spec.kernel_id.startswith("dsl.mla_prolog_v4") for spec, _ in dsl_cands)


def test_lock5_prefetch_and_streams_compose_and_periodic_sweep_is_fast():
    case = compose_case("qwen3-32b", hardware=SKU_A5, tp=1, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", context=DECODE_CTX)))
    ops = dossier.handle.scenario.atomic_ops[0].ops
    anchor = ops[0].op_id
    target = next(op for op in ops if op.op_id.endswith(".qkv") and op.op_id != anchor)
    lanes = (
        StreamLane(MAIN_STREAM, StreamCoreBudget(None, None, "shared_cap")),
        StreamLane("aux", StreamCoreBudget(None, None, "shared_cap")),
    )
    edits = (
        Edit("sched", operation=StreamSchedule(lanes, (), ())),
        Edit("pref", after=("sched",), operation=WeightPrefetch(
            PortRef(OpRef("atom", target.op_id), "weight", 0),
            PortRef(OpRef("atom", anchor), "output", 0),
            max_bytes=PREFETCH_BYTES,
        )),
    )
    combined = replace(dossier.unmerge_plan, edits=edits)
    full = finalize_plan(dossier.handle, combined)
    assert full.wall_ns > 0 and full.applied.prefetch_leases and full.applied.streams

    intervals = tuple((i * SWEEP_STRIDE, i * SWEEP_STRIDE + SWEEP_WIDTH) for i in range(SWEEP_INTERVAL_COUNT))
    t0 = perf_counter()
    measured = _measure_periodic_combo((), (), intersect=False, extra_left_ivs=intervals)
    elapsed = perf_counter() - t0
    assert measured == SWEEP_INTERVAL_COUNT * SWEEP_WIDTH
    assert elapsed < SWEEP_MAX_SECONDS


def test_rc14_dual_bound_soundness_for_onchip_copy_per_pair_link_and_pure_vector():
    c_dt = compose_case("qwen3-32b", hardware=SKU_A5, tp=1, layers=1)
    dos_a5 = prepare_scenario(ScenarioRequest(c_dt, StepRequest("decode", context=1)))
    base_g = dos_a5.handle.scenario.atomic_ops[0]
    for res_mode in ("hbm_cold", "l2_warm"):
        op_copy = OpSpec("copy0", "tensor_copy", "decode", 0, 0, (), (1, 16), (1, 16), (1, 16), (1, 16),
                         dtype="bf16", mte2_bytes=131072, hbm_bytes=0, vector_ops=0, flops=0,
                         metadata=(("initial_residency", res_mode),))
        sc1 = replace(dos_a5.handle.scenario, atomic_ops=(replace(base_g, ops=(op_copy,)),))
        cert1, _ = certify_lower_bound(*build_dual_inputs_from_scenario(sc1)[:2])
        wall1_ns = round(graph_costs(sc1.case, {0: (op_copy,)})["duration_ms"] * 1_000_000)
        assert 0 < cert1.l_star_lean_ns <= wall1_ns

    op_vec = OpSpec("vec0", "elementwise", "decode", 0, 0, (), (1, 16), (1, 16), (1, 16), (1, 16),
                    dtype="bf16", vector_ops=65536, hbm_bytes=0, mte2_bytes=0, mte3_bytes=0, flops=0)
    sc3 = replace(dos_a5.handle.scenario, atomic_ops=(replace(base_g, ops=(op_vec,)),))
    cert3, _ = certify_lower_bound(*build_dual_inputs_from_scenario(sc3)[:2])
    wall3_ns = round(graph_costs(sc3.case, {0: (op_vec,)})["duration_ms"] * 1_000_000)
    assert 0 < cert3.l_star_lean_ns <= wall3_ns

    raw_pp = col_case(tp=4)
    raw_pp["machine"]["links"][0].update(semantics="per_pair", capacity_bytes_per_ms=1000, latency_us=0)
    case_pp = parse_case(raw_pp)
    ex = exchange(30)
    sc2 = SimpleNamespace(machine=case_pp.machine, atomic_ops=(replace(base_g, replica=0, ops=(ex,)),))
    cert2, _ = certify_lower_bound(*build_dual_inputs_from_scenario(sc2)[:2])
    wall2_ns = round(graph_costs(case_pp, {0: (ex,)})["duration_ms"] * 1_000_000)
    assert 0 < cert2.l_star_lean_ns <= wall2_ns


def test_rc14_catlass_exact_panel_causality_l0a_hold_l2_and_splitk():
    geo, _ = sku_planning_context(SKU_A5)
    dp = datapath_for_die(parse_case(compose_case("qwen3-32b", hardware=SKU_A5, tp=1, layers=1)).machine.dies[0])
    p1 = CatlassProblem(m=1, n=32, k=32800, dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32")
    s1 = next(s for s in enumerate_catlass_specs(p1, dp, l1_tiles=((32, 32, 32),), l0_tiles=((32, 32, 32),), aic_budget=2, aiv_budget=2)
              if s.scheduler == "dataparallel" and s.epilogue_kind == "none" and s.dispatch_policy == "MmadPingpong")
    ctx1 = recipe_context(geo, p1)
    r1 = catlass_lower.lower_catlass_spec(s1, p1, dp, context=ctx1)
    pools = dict(dp.buffers)
    resources = [Resource(role, rate, units=ctx1.resource_units.get(role, 1)) for role, rate in ctx1.rates.items()]
    punits = {pool: 1 if pool == "l2" else dp.aiv_count if pool == "ub" else dp.aic_count for pool in pools}
    eng1 = Engine(resources, pools=pools, pool_units=punits, trace_limit=200000)
    done1 = r1.program.start(eng1, "k1", eng1.submit("start"))
    eng1.advance(until=(done1,))
    b_end = max(row["end_ms"] for row in eng1.trace if ":gm_b" in row["task"])
    mma_end = max(row["end_ms"] for row in eng1.trace if ":mma" in row["task"])
    assert mma_end >= b_end
    eng1h = Engine(resources, pools=pools, pool_units=punits)
    hold = eng1h.submit("hold_l0a", latency=1.0,
                        acquire=(Allocation("hold_l0a", "l0a", pools["l0a"], units=(0,)),),
                        free=(Allocation("hold_l0a", "l0a", pools["l0a"], units=(0,)),))
    done1h = catlass_lower.lower_catlass_spec(s1, p1, dp, context=recipe_context(geo, p1)).program.start(eng1h, "k1h", eng1h.submit("start"))
    eng1h.advance(until=(done1h, hold), first=False)
    assert done1h.time >= done1.time + 0.99

    p2 = CatlassProblem(m=32, n=2080, k=512, dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32")
    s2 = next(s for s in enumerate_catlass_specs(p2, dp, l1_tiles=((32, 32, 32),), l0_tiles=((32, 32, 32),), aic_budget=4, aiv_budget=4)
              if s.scheduler == "dataparallel" and s.epilogue_kind == "none" and s.dispatch_policy == "MmadPingpong")
    r2 = catlass_lower.lower_catlass_spec(s2, p2, dp, memory_view=AdmissionMemoryView({0: 65536}, granule_bytes=32))
    assert (r2.l2_hit_bytes, r2.summary.dram_sector_bytes) == (1_572_864, 2_820_096)

    p3 = CatlassProblem(m=32, n=32, k=8224, dtype_a="bf16", dtype_b="bf16", dtype_c="bf16", dtype_acc="fp32")
    s3_sk = next(s for s in enumerate_catlass_specs(p3, dp, l1_tiles=((32, 32, 32),), l0_tiles=((32, 32, 32),), aic_budget=2, aiv_budget=2)
                 if s.scheduler == "multi_core_splitk" and s.epilogue_kind == "none" and s.splitk_factor == 2)
    r_sk = catlass_lower.lower_catlass_spec(s3_sk, p3, dp)
    assert sum(st.work.get("vec_reduce", 0) for st in r_sk.program.stages) == 1024


def test_rc14_full_duplex_hbm_shipping_stages_python_and_c11():
    bus = HbmBusSpec(0, 1024.0, 1024.0, 0.0, row_r="hbm:r", row_w="hbm:w")
    for backend in ("python", "c"):
        eng = Engine([Resource("hbm", 1024.0), Resource("mte2_aic", 8192.0), Resource("mte3", 8192.0)], backend=backend)
        eng.add_hbm_bus(bus)
        ev_r = eng.submit("ship_read", {"hbm": 8192.0, "mte2_aic": 8192.0}, deliveries=(("hbm_r", 8192.0),))
        ev_w = eng.submit("ship_write", {"hbm": 8192.0, "mte3": 8192.0})
        eng.advance(horizon=1.0)
        assert eng.delivered["hbm_r"] == pytest.approx(1024.0)
        assert eng.serviced["hbm"] == pytest.approx(2048.0)
        assert eng.service_of(ev_r)["hbm"] == pytest.approx(1024.0)
        eng.advance(until=(ev_r, ev_w), first=False)
        assert eng.now == pytest.approx(8.0) and eng.serviced["hbm"] == pytest.approx(16384.0)
        assert eng.delivered["hbm_r"] == pytest.approx(8192.0)
        ev_r2 = eng.submit("ship_read_2", {"hbm": 4096.0, "mte2_aic": 4096.0}, after=(ev_r, ev_w))
        ev_w2 = eng.submit("ship_write_2", {"hbm": 4096.0, "mte3": 4096.0}, after=(ev_r, ev_w))
        eng.advance(until=(ev_r2, ev_w2), first=False)
        assert eng.now == pytest.approx(12.0) and eng.serviced["hbm"] == pytest.approx(24576.0)
        eng_ret = Engine([Resource("hbm", 1024.0), Resource("mte2_aic", 8192.0), Resource("mte3", 8192.0)], backend=backend)
        r_ret = eng_ret.submit("ship_read", {"hbm": 8192.0, "mte2_aic": 8192.0})
        w_ret = eng_ret.submit("ship_write", {"hbm": 8192.0, "mte3": 8192.0})
        eng_ret.advance(horizon=2.0)
        eng_ret.add_hbm_bus(bus)
        eng_ret.advance(until=(r_ret, w_ret), first=False)
        assert eng_ret.now == pytest.approx(9.0) and eng_ret.serviced["hbm"] == pytest.approx(16384.0)
        eng_re = Engine([Resource("hbm", 1024.0), Resource("mte2_aic", 8192.0)], backend=backend)
        ev_re = eng_re.submit("ship_read", {"hbm": 8192.0}, deliveries=(("hbm_b", 8192.0),))
        eng_re.advance(horizon=2.0)
        eng_re.add_hbm_bus(bus)
        eng_re.bind_hbm_bus(HbmBusSpec(0, 512.0, 512.0, 0.0, row_r="hbm:r", row_w="hbm:w"))
        eng_re.advance(until=(ev_re,))
        assert eng_re.now == pytest.approx(14.0) and eng_re.serviced["hbm"] == pytest.approx(8192.0)
        assert eng_re.delivered["hbm_b"] == pytest.approx(8192.0)


def test_rc14_periodic_streaming_coprime_and_disjoint_pruning():
    plan = AddressRunPlan(tuple(AddressRun(0, 2**22, 32, stride) for stride in (4097, 4099, 4103)))
    t0 = perf_counter()
    measured = plan.sector_coverage().measure_bytes()
    assert measured == 780_607_488 and (perf_counter() - t0) < 5.0
    contained = AddressRunPlan((AddressRun(1024, 4097, 960, 4096), AddressRun(0, 4097, 2048, 4096)))
    assert contained.sector_coverage().measure_bytes() == 8_390_656
    disjoint = tuple(PeriodicSectorRun(i * 10**8, 8192, 32, 4097, 32) for i in range(SWEEP_INTERVAL_COUNT))
    t1 = perf_counter()
    assert len(_prune_periodic_subruns(disjoint)) == SWEEP_INTERVAL_COUNT
    assert _coverage_overlap(SectorCoverage((), disjoint[:1000]), SectorCoverage((), disjoint[1000:])) == 0
    assert (perf_counter() - t1) < SWEEP_MAX_SECONDS


def test_rc14_primitive_cache_isolation_sku_compatibility_and_dtype_aliases():
    dt_table = load_primitives(SKU_A5)
    raw_slow = deepcopy(dt_table.to_envelope_dict())
    band = raw_slow["primitives"]["P01_datacopy"]["metrics"]["hbm_read_bytes_per_s"]
    for key in ("p05", "p50", "p95", "p99", "trimmed_mean"):
        if band.get(key) is not None:
            band[key] = float(band[key]) * 0.125
    raw_slow["envelope_sha256"] = compute_envelope_sha256(raw_slow)
    mutated = replace(dt_table, primitives=parse_primitive_table(raw_slow).primitives)
    c_dt = compose_case("qwen3-32b", hardware=SKU_A5, tp=1, layers=1)
    cache = OracleCache()
    d_base = prepare_scenario(ScenarioRequest(c_dt, StepRequest("decode", context=DECODE_CTX), primitive_table=dt_table), cache=cache)
    d_mut = prepare_scenario(ScenarioRequest(c_dt, StepRequest("decode", context=DECODE_CTX), primitive_table=mutated), cache=cache)
    d_fresh = prepare_scenario(ScenarioRequest(c_dt, StepRequest("decode", context=DECODE_CTX), primitive_table=mutated))
    assert d_mut.mainline_eval.span_ms > d_base.mainline_eval.span_ms
    assert d_mut.mainline_eval.span_ms == pytest.approx(d_fresh.mainline_eval.span_ms)

    c_pr = compose_case("qwen3-32b", hardware="Ascend950PR_9579", tp=1, layers=1)
    dev_pr = parse_case(c_pr).machine.dies[0]
    with pytest.raises(ModelingError, match="primitive_table_sku_mismatch"):
        resolve_sim_rate_overlay(dt_table, datapath_for_die(dev_pr), sku=dev_pr.sku)
    with pytest.raises(ModelingError, match="primitive_table_sku_mismatch"):
        prepare_scenario(ScenarioRequest(c_pr, StepRequest("decode", context=DECODE_CTX), primitive_table=dt_table))
    dos_pr = prepare_scenario(ScenarioRequest(c_pr, StepRequest("decode", context=DECODE_CTX)))
    with pytest.raises(ModelingError, match="primitive_table_sku_mismatch"):
        build_dual_inputs_from_scenario(dos_pr.handle.scenario, primitive_table=dt_table)

    for gen in ("a2", "a3", "a5"):
        dev = parse_case(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS[gen], tp=1, layers=1)).machine.dies[0]
        dp = datapath_for_die(dev)
        for dt in ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"):
            assert dp.cube_peak(dt) == pytest.approx(dev.peak(dt))
        for dt in ("bf16", "bfloat16", "float16", "float32"):
            _, res = compile_fuse_region(
                "fused", ("compute",), dp, dependencies={"compute": ()},
                op_stages={"compute": Stage("compute", {"cube": 2 * 64 * 64 * 64}, metadata={"cube_dtype": dt})},
                edge_routes=(), edge_bytes={},
            )
            assert res.stages_count > 0
