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
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.runtime.memory_physics import _measure_periodic_combo
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.streams import MAIN_STREAM, StreamCoreBudget, StreamLane
from mindie_sim.primitives.loader import load_primitives
from mindie_sim.primitives.schema import compute_envelope_sha256, parse_primitive_table
from mindie_sim.rewrite.oracle import ScenarioRequest, finalize_plan, prepare_scenario
from mindie_sim.rewrite.schema import Edit, OpRef, PortRef, StreamSchedule, WeightPrefetch
from mindie_sim.schema.case import Case
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Device
from mindie_sim.schema.hardware_links import Link
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import LoweringRequest
from mindie_sim.templates import registry as templates_registry

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


def _resigned_table(base_table, *, p50_scale: float = 1.0, cap_scale: float = 1.0):
    raw = deepcopy(base_table.to_envelope_dict())
    for pid, cell in raw["primitives"].items():
        if pid not in {"P01_datacopy", "P05_l2_residency_reuse", "P06_mmad", "P07_vec_alu"}:
            continue
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
        rates=(("hbm", UNIT_RATE_PER_S), ("vector", UNIT_RATE_PER_S)),
        compute_peaks=(("bf16", UNIT_PEAK_FLOPS),),
    )
    die1 = Device(
        rank=1, compute_units=1, memory_bytes=MEM_CAPACITY,
        rates=(("hbm", UNIT_RATE_PER_S), ("vector", UNIT_RATE_PER_S)),
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
    case_qwen = compose_case("qwen3-32b", hardware=SKU_A5, tp=1, layers=1)
    report = api.optimize_ladder(case_qwen, StepRequest("decode", context=DECODE_CTX),
                                 primitive_table=base_table, max_trials=6)
    assert isinstance(report.search_stats, CandidateSearchStats)
    assert report.search_stats.eligible_sites > 0
    assert report.search_stats.generated_candidates > 0
    assert isinstance(report.primitive_coverage_gaps, tuple) and report.primitive_coverage_gaps

    spine = PredictionSpine()
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
