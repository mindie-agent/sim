"""Public certified replay, wall attribution, and physical efficiency parity.

[Role] Verify the API tiers, full ladder, wall attribution, and efficiency exports.
[Invariants] The bound is admissible and every nanosecond is attributed once.
[Timeouts & Locks] Small in-memory cases and bounded constructive search.
[Side-Effect Policy] No device work, external stores, or configuration writes.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from _support.axes import PHASES, REPRESENTATIVE_SKUS, SKUS
from mindie_sim import api
from mindie_sim.compose import compose_case
from mindie_sim.feature_waterfall import compare_feature_tracks
from mindie_sim.optimization.framework_solver import _TrialSearch
from mindie_sim.optimization.ladder import FiveLevelReport, _shipping_regression, partition_step_wall_ns
from mindie_sim.prediction import efficiency
from mindie_sim.prediction.efficiency import analysis, peaks, sla
from mindie_sim.prediction.results import OpOccurrence, OperatorSequenceRow, StepEvaluation, StepRequest
from mindie_sim.prediction.roofline_graph import RooflineTargets, critical_path_roofline
from mindie_sim.prediction.structure import parse_case
from mindie_sim.rewrite import CollectiveTransform, Edit, OpRef, ScenarioRequest, finalize_plan, prepare_scenario

pytestmark = pytest.mark.tier0

MODEL = "qwen3-32b"
NS_PER_MS = 1_000_000


def _case(hardware, **kwargs):
    return compose_case(MODEL, hardware, dp=1, pp=1, layers=1, **kwargs)


@pytest.mark.parametrize("hardware", SKUS)
@pytest.mark.parametrize("phase", PHASES)
def test_public_replay_matches_certified_integer_ladder(hardware, phase):
    case = _case(hardware, tp=1)
    step = StepRequest(phase, tokens=16 if phase == "prefill" else 1, context=64)
    report = api.optimize_ladder(case, step)
    tiers = api.evaluate_three_tiers(case, step)
    assert isinstance(report, FiveLevelReport) and report.schema == "mindie-sim.five-level-report/1"
    ladder = report.ladder
    lower, dsl, fw, main, unmerge = (
        ladder.l_star_lean_ns, ladder.a_star_dsl_ns, ladder.a_star_framework_ns,
        ladder.m_mainline_ns, ladder.u_unmerge_ns,
    )
    assert 0 < lower <= dsl <= fw <= min(main, unmerge)
    deltas = (ladder.delta_shipping_ns, ladder.delta_framework_ns,
              ladder.delta_operator_dsl_ns, ladder.delta_optimality_gap_ns)
    assert all(type(value) is int for value in (lower, dsl, fw, main, unmerge, *deltas))
    assert deltas == (unmerge - main, main - fw, fw - dsl, dsl - lower)
    assert sum(deltas) == unmerge - lower
    assert tiers.certified_lower_bound_ms == lower / NS_PER_MS
    assert tiers.certified_lower_bound_ms <= tiers.should_achieve.span_ms <= min(
        tiers.mainline.span_ms, tiers.unmerge.span_ms,
    )
    assert (tiers.unmerge.span_ms, tiers.mainline.span_ms, tiers.should_achieve.span_ms) == (
        report.unmerge_eval.span_ms, report.mainline_eval.span_ms, report.dsl_eval.span_ms,
    )
    assert report.verification.passed and report.verification.l_star_ns == lower

    waterfall = report.waterfall
    ordered = waterfall.ordered_telescope
    assert ordered and ordered[0].before_ns == main and ordered[-1].after_ns == dsl
    assert len({item.stage_id for item in ordered}) == len(ordered)
    assert all(left.after_ns == right.before_ns for left, right in zip(ordered, ordered[1:]))
    assert all(item.delta_ns == item.before_ns - item.after_ns for item in ordered)
    assert sum(item.delta_ns for item in ordered) == main - dsl
    edits = {edit.edit_id: type(edit.operation).__name__ for edit in report.dsl_plan.edits}
    assert all(edits[impact.edit_id] == impact.opcode for impact in waterfall.loo_impacts)
    regression = waterfall.shipping_regression
    if main > unmerge + 1000:
        assert (regression.delta_ns, regression.baseline_u_ns, regression.baseline_m_ns) == (
            main - unmerge, unmerge, main,
        )
    else:
        assert regression is None
    assert len({region.region_id for region in waterfall.by_region}) == len(waterfall.by_region)
    for region_field, measured, residual in zip(
        ("u_wall_ns", "m_wall_ns", "fw_wall_ns", "dsl_wall_ns"), (unmerge, main, fw, dsl),
        waterfall.reconciliation_ns, strict=True,
    ):
        parts = [getattr(region, region_field) for region in waterfall.by_region]
        assert all(value >= 0 for value in parts)
        assert sum(parts) - measured == residual and abs(residual) <= 1000


def test_parallel_completion_partitions_one_wall_clock():
    rows = tuple(OperatorSequenceRow(
        OpOccurrence(rank, 0, rank, name), rank, "vector", "test", (), stream,
        (1,), "bf16", (), 0, 0, 10,
    ) for rank, (name, stream) in enumerate((("first", "main"), ("second", "aux"))))
    step = StepEvaluation("parallel", StepRequest("decode"), "mainline", 10, rows, ())
    assert sum(row.end_ms - row.start_ms for row in rows) == 20
    assert dict(partition_step_wall_ns(step)) == {"first": 10 * NS_PER_MS}

    tail = replace(rows[0], occurrence=OpOccurrence(0, 0, 0, "tail"), ordinal=2,
                   predecessors=(rows[1].occurrence,), ready_ms=10, start_ms=11, end_ms=13)
    parts = dict(partition_step_wall_ns(replace(step, span_ms=13, operators=(*rows, tail))))
    assert parts == {"second": 10 * NS_PER_MS, "step:wait": NS_PER_MS, "tail": 2 * NS_PER_MS}


def test_shipping_regression_reports_measured_leave_one_out_culprit():
    real = prepare_scenario(ScenarioRequest(
        _case(REPRESENTATIVE_SKUS["a5"], tp=4), StepRequest("prefill", tokens=64),
    ))
    handle = real.handle
    rank = handle.scenario.atomic_ops[0]
    collective = next(op for op in rank.ops if op.link_bytes)
    route, = handle.scenario.case.machine.links
    edit = Edit("shipping_collective", operation=CollectiveTransform(
        OpRef("atom", collective.op_id), "ring", route.name,
    ))
    shipping_plan = replace(real.unmerge_plan, edits=(edit,))
    unmerge = finalize_plan(handle, real.unmerge_plan)
    shipping = finalize_plan(handle, shipping_plan)
    assert unmerge.verification.passed and shipping.verification.passed
    delta = shipping.wall_ns - unmerge.wall_ns
    assert delta > 1000
    handle = replace(handle, full_u_ns=unmerge.wall_ns, full_m_ns=shipping.wall_ns)
    shipping = finalize_plan(handle, shipping_plan)
    dossier = SimpleNamespace(handle=handle, unmerge_plan=real.unmerge_plan,
                              mainline_plan=shipping_plan, mainline_eval=shipping.step_eval)
    search = _TrialSearch(dossier, max_trials=2)
    assert search.loo(shipping_plan, shipping.wall_ns, 1) == ((edit.edit_id, -delta),)
    result = _shipping_regression(dossier, shipping.verification, delta, search)
    assert result.delta_ns == delta
    assert result.loo_culprits == ((edit.edit_id, delta),)
    assert (result.baseline_u_ns, result.baseline_m_ns) == (unmerge.wall_ns, shipping.wall_ns)
    assert _shipping_regression(dossier, shipping.verification, 1000, search) is None


def test_target_efficiency_scaling_is_retired():
    retired = ("_scale_case_efficiency", "DEFAULT_TARGET_EFFICIENCY", "evaluate_target_efficiency",
               "fit_efficiency_targets", "target_0.8x_mbu")
    for module in (efficiency, analysis, sla):
        assert all(not hasattr(module, name) for name in retired)
    assert all("target_mbu" not in scenario and scenario["id"] != "target_0.8x_mbu"
               for scenario in efficiency.DEFAULT_SLA_SCENARIOS)
    case = _case(REPRESENTATIVE_SKUS["a5"], tp=1)
    with pytest.raises(ValueError, match="^target_efficiency_scaling_retired:"):
        efficiency.solve_sla_sizing(case, scenarios=({"id": "custom", "target_mbu": 0.8},))


def test_efficiency_exports_preserve_physical_work_and_five_sla_scenarios():
    case = _case(REPRESENTATIVE_SKUS["a5"], tp=1)
    hardware = efficiency.extract_hardware_peaks(case)
    die, = parse_case(case).machine.dies
    assert hardware == peaks.extract_hardware_peaks(case)
    assert hardware["hbm_peak_bytes_per_s"] == dict(die.rates)["hbm"]
    assert hardware["link_peak_bytes_per_s"] == 0
    baseline = efficiency.analyze_network_efficiency(case, context=64)
    assert efficiency.analyze_network_efficiency is analysis.analyze_network_efficiency
    assert baseline["schema"] == "mindie-sim.efficiency/1"
    assert baseline["total_hbm_bytes"] > 0
    assert baseline["network_mbu"] == pytest.approx(
        baseline["total_hbm_ideal_ms"] / baseline["graph_duration_ms"],
    )
    fit = efficiency.fit_case_roofline_regimes(case, baseline_analysis=baseline)
    assert fit == analysis.fit_case_roofline_regimes(case, baseline_analysis=baseline)
    assert fit["geometry"] == baseline["geometry"]
    assert {row["op_id"]: row["duration_ms"] for row in fit["operators"]} == {
        row["op_id"]: row["duration_ms"] for row in baseline["operators"]
    }
    assert all(not row["bottleneck_identified"] for row in fit["fitted_regimes"])
    assert efficiency.solve_sla_sizing is sla.solve_sla_sizing
    solved = efficiency.solve_sla_sizing(case, context=64, sla_ms=30, max_sequences_per_dp=2,
                                         b1_analysis=baseline)
    scenarios = {scenario["scenario_id"]: scenario for scenario in solved["scenarios"]}
    assert len(scenarios) == 5
    for name, hbm, cube, vector in (
        ("baseline", 1, 1, 1), ("2x_hbm", 2, 1, 1), ("2x_compute", 1, 2, 2),
        ("2x_cube_only", 1, 2, 1), ("2x_hbm_2x_compute", 2, 2, 2),
    ):
        row = scenarios[name]
        assert row["scales"] == {"hbm_bw_scale": hbm, "cube_flops_scale": cube,
                                  "vector_ops_scale": vector, "link_bw_scale": 1, "hbm_capacity_scale": 1}
        assert 0 < row["b_sla"] <= 2
        assert 0 < row["step_ms_at_sla"] <= solved["sla_ms"]
        assert row["tps_at_sla"] == pytest.approx(
            1000 * row["total_concurrency"] * row["emitted_tokens_per_seq"] / row["step_ms_at_sla"],
        )


def test_feature_waterfall_uses_unit_rate_occurrence_roofline():
    sku = REPRESENTATIVE_SKUS["a5"]
    result = compare_feature_tracks(MODEL, sku, tp=1, dp=1, pp=1, layers=1,
                                    context=64, sequences=1, ablate_prs=False)
    case = _case(sku, tp=1, feature_track="expected_ideal", _stack_phase="decode",
                 runtime_stack="vllm-ascend")
    baseline = efficiency.analyze_network_efficiency(case, context=64, sequences=1, _retain_graph=True)
    graph = baseline["_occurrence_graph"]
    physical = critical_path_roofline(graph, RooflineTargets())
    reference = result["efficiency_reference"]
    assert reference["basis"] == "physical_1p0x_critical_path_roofline"
    assert reference["physical_ideal_ms"] == round(physical.critical_path_ideal_ms, 6)
    floor = result["waterfall"]["hardware_roofline"]
    assert floor["physical_ideal_1p0x_ms"] == reference["physical_ideal_ms"]
    assert 0 < reference["physical_ideal_ms"] <= baseline["graph_duration_ms"]


def test_soundness_guards_cyclic_dag_periodic_union_and_aliased_contraction():
    from types import SimpleNamespace

    from mindie_sim.optimization import (
        BilinearContractionSpec, DagEdge, EventDagSpec, FarkasCertificate,
        OptimalityCertificate, ResourceChoiceProgram, TensorRangeObligation,
        build_dual_inputs_from_scenario, certify_lower_bound,
        check_optimality_certificate, compute_compulsory_hbm_cut,
    )
    from mindie_sim.prediction.runtime.engine import Engine, Resource
    from mindie_sim.prediction.runtime.memory_physics import AddressRun
    from mindie_sim.schema.hardware import Device

    # Cyclic EventDagSpec & repeated-node witness must be rejected.
    empty_prog = ResourceChoiceProgram(0, (), ())
    cyclic_dag = EventDagSpec((10,), (DagEdge(0, 0, 0),))
    forged = OptimalityCertificate(FarkasCertificate(1, (), (), 0), (0,) * 10, 0, 110)
    assert check_optimality_certificate(empty_prog, cyclic_dag, forged) is False

    # Disjoint >4096-row periodic AddressRuns must compute exact multi-run union.
    even_run = AddressRun(1, 4097, 1, 64)
    odd_run = AddressRun(33, 4097, 1, 64)
    ob = TensorRangeObligation(0, "a", 0, (), "hbm_cold", "owner", (even_run, odd_run))
    cut = compute_compulsory_hbm_cut((SimpleNamespace(read_obligations=(ob,), write_obligations=()),))[0]
    assert cut.read_bytes == 2 * 4097 * 32

    # Aliased self-contraction (distinct_operands=False) must never invert actual Engine wall time.
    n = 64
    dev = Device(rank=0, compute_units=1, memory_bytes=1024,
                 rates=(("hbm", 10**9),), compute_peaks=(("bf16", 10**12),))
    machine = SimpleNamespace(dies=(dev,), links=())
    obx = TensorRangeObligation(0, "x", 0, ((0, n),), "hbm_cold", "owner")
    region = SimpleNamespace(op_ids=(), semantic_work=(), read_obligations=(obx,), write_obligations=())
    spec = BilinearContractionSpec("x_dot_x", 0, 1, 1, n, 8, 8, 32, 132, write_free=True)
    prog, dag, _ = build_dual_inputs_from_scenario(
        SimpleNamespace(machine=machine, bilinear_contractions=(spec,)), region_summaries=(region,),
    )
    cert, _ = certify_lower_bound(prog, dag)
    eng = Engine((Resource("hbm", 10**6), Resource("cube", 10**9)), backend="python")
    h = eng.submit("read_x", {"hbm": n})
    f = eng.submit("dot_x_x", {"cube": 2 * n}, after=(h,))
    eng.advance(until=(f,))
    assert cert.l_star_lean_ns <= round(eng.now * 10**6)


def test_pointwise_nd_shape_and_gate4_rejects_matrix_recipe_downgrade():
    from dataclasses import replace

    from mindie_sim.rewrite import (
        Edit, FuseRegion, KernelBoundary, OpRef, ScenarioRequest, SelectTemplate,
        evaluate_rewrites, finalize_plan, prepare_scenario,
    )

    case = _case(REPRESENTATIVE_SKUS["a5"], tp=1)
    case["runtime"].setdefault("kernels", {}).setdefault("recipes", {})["pointwise"] = {
        "kind": "tiled_pointwise", "input_bytes_per_element": 2, "outputs": [{"bytes_per_element": 2}],
    }
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", tokens=4, context=64)))
    refs = (OpRef("atom", "s0.l0.norm1"), OpRef("atom", "s0.l0.qkv"))
    bad_fuse = replace(dossier.unmerge_plan, edits=(
        Edit("bad_fuse", operation=FuseRegion(refs, (), (KernelBoundary("pointwise", refs),))),
    ))
    bad_select = replace(dossier.unmerge_plan, edits=(
        Edit("bad_sel", operation=SelectTemplate(OpRef("atom", "s0.l0.qkv"), "K3", "pointwise")),
    ))
    for plan in (bad_fuse, bad_select):
        full = finalize_plan(dossier.handle, plan)
        assert not full.verification.passed
        assert any(g.reason == "WORK_CONSERVATION_MISMATCH" for g in full.verification.gates)
    batch = evaluate_rewrites(dossier.handle, (bad_fuse, bad_select))
    assert all(r.status == "gate_failed" and r.reason == "WORK_CONSERVATION_MISMATCH" for r in batch.results)
