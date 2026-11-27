"""Public certified replay, wall attribution, and physical efficiency parity.

[Role] Verify the API tiers, full ladder, wall attribution, and efficiency exports.
[Invariants] The bound is admissible and every nanosecond is attributed once.
[Timeouts & Locks] Small in-memory cases and bounded constructive search.
[Side-Effect Policy] No device work, external stores, or configuration writes.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import ulp
from types import SimpleNamespace

import pytest

from _support.axes import PHASES, REPRESENTATIVE_SKUS, SKUS
from mindie_sim import api, feature_mechanisms as feature_waterfall
from mindie_sim.compose import compose_case
from mindie_sim.feature_mechanisms import compare_feature_tracks
from mindie_sim.optimization.framework_solver import _TrialSearch
from mindie_sim.optimization import (
    PrimitiveCapacityEntry, PrimitiveCapacityTable, TensorRangeObligation, build_dual_inputs_from_scenario,
    certify_lower_bound, inward_floor_ns,
)
from mindie_sim.prediction.runtime.reload_bounds import compute_compulsory_hbm_cut
from mindie_sim.optimization.ladder import (
    SHIPPING_OVERLAY, SHIPPING_REGRESSION_TOLERANCE_NS, FiveLevelReport, _shipping_regression,
    partition_step_wall_ns,
)
from mindie_sim.prediction import efficiency
from mindie_sim.prediction.efficiency import analysis, peaks, sla
from mindie_sim.prediction.operators.vector import TiledPointwise
from mindie_sim.prediction.results import OpOccurrence, OperatorSequenceRow, StepEvaluation, StepRequest
from mindie_sim.prediction.efficiency.roofline_graph import PhysicalWorkGraph, PhysicalWorkNode, RooflineTargets, critical_path_roofline
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.runtime.memory_physics import AddressRun
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.rewrite import (
    CollectiveTransform, Edit, FuseRegion, KernelBoundary, OpRef, PortRef, ScenarioRequest,
    SelectTemplate, WeightPrefetch, evaluate_rewrites, finalize_plan, prepare_scenario,
)
from mindie_sim.prediction.results import LoweringRequest
from mindie_sim.primitives import load_primitives
from mindie_sim.rewrite import oracle
from mindie_sim.schema.hardware import Device
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

MODEL = "qwen3-32b"
NS_PER_MS = 1_000_000

def _case(hardware, **kwargs):
    return compose_case(MODEL, hardware, dp=1, pp=1, layers=1, **kwargs)

@pytest.mark.parametrize("prepare", (True, False))
def test_mixed_iteration_exports_draft_prefill_when_requested(prepare):
    case = parse_case(compose_case("deepseek-v4-flash", REPRESENTATIVE_SKUS["a5"],
                                   tp=4, ep=4, layers=1, mtp_proposal_tokens=1))
    plan = IterationPlan((RequestView("decode", 1, 64, True, "decode"),
                          RequestView("prefill", 2, 0, True, "prefill", draft_prepare=prepare)))
    request = LoweringRequest(iterations=((0, plan),))
    payload = PredictionSpine().lower(case, request).structure_payload
    assert payload["execution"]["phase"] == "mixed"
    assert payload["execution"].get("draft_prepare", False) is prepare
    draft_nodes = [node for node in payload["nodes"] if node["phase"] == "draft_prefill"]
    assert bool(draft_nodes) is prepare
    assert all(node["deps"] for node in draft_nodes)

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
    shipping = [impact for impact in waterfall.loo_impacts if impact.edit_id == SHIPPING_OVERLAY]
    dossier = prepare_scenario(ScenarioRequest(case, step))
    expected_shipping = []
    if main != unmerge and report.dsl_plan.base_graph_sha256 == dossier.mainline_plan.base_graph_sha256:
        without = finalize_plan(dossier.handle, replace(dossier.unmerge_plan, edits=report.dsl_plan.edits))
        if without.verification.passed:
            expected_shipping = [("ShippingOverlay", without.wall_ns - dsl)]
    assert [(impact.opcode, impact.marginal_delta_ns) for impact in shipping] == expected_shipping
    assert all(edits[impact.edit_id] == impact.opcode for impact in waterfall.loo_impacts if impact not in shipping)
    regression = waterfall.shipping_regression
    if main > unmerge + SHIPPING_REGRESSION_TOLERANCE_NS:
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
    assert parts == {"first": 10 * NS_PER_MS, "step:wait": NS_PER_MS, "tail": 2 * NS_PER_MS}

def test_shipping_regression_reports_measured_leave_one_out_culprit():
    case, step = _case(REPRESENTATIVE_SKUS["a5"], tp=4), StepRequest("prefill", tokens=64)
    case["machine"]["transfer_channels"]["d2d_intra_node"]["alpha_us"] = 24.0  # Exercise a slower ring.
    real = prepare_scenario(ScenarioRequest(case, step))
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
    assert delta > SHIPPING_REGRESSION_TOLERANCE_NS
    handle = replace(handle, full_u_ns=unmerge.wall_ns, full_m_ns=shipping.wall_ns)
    shipping = finalize_plan(handle, shipping_plan)
    dossier = SimpleNamespace(handle=handle, unmerge_plan=real.unmerge_plan,
                              mainline_plan=shipping_plan, mainline_eval=shipping.step_eval)
    search = _TrialSearch(dossier, max_trials=2)
    assert search.loo(shipping_plan, shipping.wall_ns, 1) == ((edit.edit_id, -delta),)
    result = _shipping_regression(shipping.verification, unmerge.wall_ns, shipping.wall_ns)
    assert result.delta_ns == delta and result.loo_culprits == ((SHIPPING_OVERLAY, -delta),)
    assert (result.baseline_u_ns, result.baseline_m_ns) == (unmerge.wall_ns, shipping.wall_ns)
    assert _shipping_regression(shipping.verification, unmerge.wall_ns,
                                unmerge.wall_ns + SHIPPING_REGRESSION_TOLERANCE_NS) is None

    report = api.optimize_ladder(_case(REPRESENTATIVE_SKUS["a2"], tp=4), step, max_trials=2)
    ladder = report.ladder
    regression = report.waterfall.shipping_regression
    assert ladder.m_mainline_ns - ladder.u_unmerge_ns > SHIPPING_REGRESSION_TOLERANCE_NS
    assert regression == report.verification.shipping_regression
    assert (regression.baseline_u_ns, regression.baseline_m_ns) == (ladder.u_unmerge_ns, ladder.m_mainline_ns)
    assert regression.loo_culprits == ((SHIPPING_OVERLAY, ladder.delta_shipping_ns),)
    assert report.waterfall.loo_impacts == ()

def test_mainline_seed_screens_against_its_own_faster_wall():
    case = _case(REPRESENTATIVE_SKUS["a2"], tp=1, quant="bf16")
    step = StepRequest("prefill", tokens=16, context=64)
    dossier = prepare_scenario(ScenarioRequest(case, step))
    assert dossier.m_wall_ns < dossier.u_wall_ns
    batch = evaluate_rewrites(dossier.handle, (dossier.unmerge_plan, dossier.mainline_plan))
    unmerge, mainline = batch.results
    assert unmerge.accepted and mainline.accepted
    assert (unmerge.estimated_wall_ns, mainline.estimated_wall_ns) == (dossier.u_wall_ns, dossier.m_wall_ns)
    assert unmerge.plan_sha256 != mainline.plan_sha256
    report = api.optimize_ladder(case, step)
    assert report.dsl_plan == dossier.mainline_plan
    impact, = report.waterfall.loo_impacts
    assert impact.edit_id == SHIPPING_OVERLAY
    assert impact.marginal_delta_ns == dossier.u_wall_ns - dossier.m_wall_ns == 2500

def test_unmerge_winner_has_empty_leave_one_out_despite_shipping_regression():
    case = _case(REPRESENTATIVE_SKUS["a2"], tp=1)
    step = StepRequest("prefill", tokens=64)
    dossier = prepare_scenario(ScenarioRequest(case, step))
    report = api.optimize_ladder(case, step, max_trials=0)
    assert report.dsl_plan == dossier.unmerge_plan and not report.dsl_plan.edits
    assert report.waterfall.loo_impacts == ()
    regression = report.waterfall.shipping_regression
    assert regression == report.verification.shipping_regression
    assert regression.delta_ns == dossier.m_wall_ns - dossier.u_wall_ns == 8712
    assert report.ladder.delta_shipping_ns == -regression.delta_ns
    assert regression.loo_culprits == ((SHIPPING_OVERLAY, -regression.delta_ns),)

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
    case = _case(sku, tp=1, feature_track="target", _stack_phase="decode",
                 runtime_stack="vllm-ascend")
    baseline = efficiency.analyze_network_efficiency(
        case, context=64, sequences=1, scope="model", _retain_graph=True,
    )
    graph = baseline["_occurrence_graph"]
    physical = critical_path_roofline(graph, RooflineTargets())
    reference = result["efficiency_reference"]
    assert reference["basis"] == "physical_1p0x_critical_path_roofline"
    assert reference["physical_ideal_ms"] == round(physical.physical_ideal_ms, 6)
    floor = result["waterfall"]["hardware_roofline"]
    assert floor["physical_ideal_1p0x_ms"] == reference["physical_ideal_ms"]
    assert 0 < reference["physical_ideal_ms"] <= baseline["graph_duration_ms"]

@pytest.mark.parametrize("shared_die", (True, False), ids=("resource-bound", "path-bound"))
def test_feature_waterfall_checks_both_limits(monkeypatch, shared_die):
    graph = PhysicalWorkGraph(
        nodes=tuple(PhysicalWorkNode(node_id=f"gemm-{i}", cube_ms=10,
                                     resource_ids=(("cube", f"die{0 if shared_die else i}"),))
                    for i in range(2)),
        edges=() if shared_die else (("gemm-0", "gemm-1"),),
    )
    physical = critical_path_roofline(graph, RooflineTargets())
    critical, cut = (10, 20) if shared_die else (20, 10)
    assert (physical.critical_path_ideal_ms, max(physical.resource_cut_bounds.values())) == (critical, cut)

    def analyze(*args, **kwargs):
        return dict(graph_duration_ms=20, operator_sum_ms=20, total_hbm_bytes=0,
                    critical_path_roofline_ms=physical.critical_path_ideal_ms,
                    physical_resource_cut_ms=max(physical.resource_cut_bounds.values()), network_mbu=0,
                    network_cube_mfu=1, network_vec_mfu=0, network_link_util=0,
                    network_roofline_efficiency=1)

    monkeypatch.setattr(efficiency, "analyze_network_efficiency", analyze)
    report = compare_feature_tracks(MODEL, REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1, ablate_prs=False)
    assert report["waterfall"]["hardware_roofline"]["physical_ideal_1p0x_ms"] == 20

def test_efficiency_analysis_uses_package_report_and_attribution_hooks(monkeypatch):
    calls = []
    for name in ("_fast_operator_and_graph_reports", "attribute_program"):
        original = getattr(efficiency, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(efficiency, name, record)

    report = efficiency.analyze_network_efficiency(_case(REPRESENTATIVE_SKUS["a5"], tp=1), context=64)
    assert calls == ["_fast_operator_and_graph_reports", "attribute_program"]
    assert report["graph_duration_ms"] > 0 and report["operators"]

def test_compulsory_cut_unions_disjoint_periodic_runs():
    runs = (AddressRun(1, 4097, 1, 101), AddressRun(33, 4097, 1, 101))
    sectors = [{(run.base_byte + row * run.stride_bytes) // 32 for row in range(run.rows)}
               for run in runs]
    assert sectors[0].isdisjoint(sectors[1])
    read = TensorRangeObligation(0, "interleaved", 0, (), "hbm_cold", "owner", runs)
    summary = SimpleNamespace(read_obligations=(read,), write_obligations=())
    assert compute_compulsory_hbm_cut((summary,))[0].read_bytes == 32 * len(set.union(*sectors)) == 262208

@pytest.mark.parametrize("identity_key,allocations,residency,expected_bytes", (
    ("allocation_id", ("shared", "shared"), "hbm_cold", 4096),
    ("allocation_id", ("left", "right"), "hbm_cold", 8192),
    ("parameter_name", ("left", "right"), "hbm_cold", 0),
    ("parameter_name", ("left", "right"), None, 0),
    ("allocation_id", ("shared", "shared"), None, 0),
))
def test_compulsory_cut_counts_physical_weight_allocation_once(identity_key, allocations, residency, expected_bytes):
    die = Device(rank=0, compute_units=1, memory_bytes=16384, rates=(("hbm", 10**9),),
                 compute_peaks=(("bf16", 10**13),))
    shape = (16, 16, 16)
    ops = tuple(OpSpec(
        f"gemm-{i}", "gemm", "decode", 0, 0, (), shape, shape, shape, shape,
        dtype="bf16", flops=8192, mte2_bytes=4096,
        metadata=(("weight_bytes", 4096), (identity_key, identity))
        + (("initial_residency", residency),) * (residency is not None),
    ) for i, identity in enumerate(allocations))
    scenario = SimpleNamespace(machine=SimpleNamespace(dies=(die,), links=()), atomic_ops=ops)
    program, dag, names = build_dual_inputs_from_scenario(scenario)
    certificate, _ = certify_lower_bound(program, dag)
    demand = dict(zip(names, program.global_work, strict=True))
    if expected_bytes:
        # At one byte/ns, the duration before the inward guard equals the byte count.
        assert demand["hbm_bus_r@r0.p0.d0"] == expected_bytes - 3
        assert expected_bytes - 3 <= certificate.l_star_lean_ns <= expected_bytes + 1
        if allocations[0] == allocations[1]:
            assert certificate.l_star_lean_ns <= 4097.64
    else:
        assert not any(name.startswith("hbm_bus") for name in names)
        assert certificate.l_star_lean_ns <= 1

def test_pointwise_nd_shape_and_gate4_rejects_matrix_recipe_downgrade():
    case = _case(REPRESENTATIVE_SKUS["a5"], tp=1)
    case["runtime"].setdefault("kernels", {}).setdefault("recipes", {})["pointwise"] = {
        "kind": "tiled_pointwise", "input_bytes_per_element": 2, "outputs": [{"bytes_per_element": 2}],
    }
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", tokens=4, context=64)))
    refs = (OpRef("atom", "s0.l0.norm1"), OpRef("atom", "s0.l0.qkv"))
    bad_fuse = replace(dossier.unmerge_plan, edits=(
        Edit("bad_fuse", operation=FuseRegion(refs, (), (KernelBoundary("pointwise", refs),), fusion_tier="K3_dsl_pipeline")),
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

def test_work_gate_rejects_unary_swiglu_recipe_in_screening_and_replay(monkeypatch):
    case = _case(REPRESENTATIVE_SKUS["a5"], tp=1)
    case["runtime"].setdefault("kernels", {}).setdefault("recipes", {})["unary"] = {
        "kind": "tiled_pointwise", "input_bytes_per_element": 2, "outputs": [{"bytes_per_element": 2}],
    }
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", context=64)))
    swiglu = next(op for op in dossier.handle.scenario.atomic_ops[0].ops if op.family == "swiglu")
    anchor, target, *_ = dossier.handle.scenario.atomic_ops[0].ops
    select = Edit("unary", operation=SelectTemplate(OpRef("atom", swiglu.op_id), "K3", "unary"))
    prefetch = Edit("empty_prefetch", operation=WeightPrefetch(
        PortRef(OpRef("atom", target.op_id), "weight", 0),
        PortRef(OpRef("atom", anchor.op_id), "output", 0), max_bytes=0,
    ))
    plan = replace(dossier.unmerge_plan, edits=(select, prefetch))
    unary_read_bytes = []
    lower = TiledPointwise.lower

    def record_loads(recipe, context):
        program = lower(recipe, context)
        if context.op.op_id == swiglu.op_id:
            pipes = program.stages[-1].program.programs
            hbm = context.resource_names["hbm"]
            unary_read_bytes.append(sum(pipe.tile(i)[0].work[hbm] for pipe in pipes for i in range(pipe.count)))
        return program

    monkeypatch.setattr(TiledPointwise, "lower", record_loads)
    baseline, rebound = (finalize_plan(dossier.handle, p) for p in (dossier.unmerge_plan, plan))
    assert baseline.verification.passed
    assert not rebound.applied.prefetch_leases
    row = next(r for r in baseline.step_eval.operators if r.occurrence.op_id == swiglu.op_id)
    assert sum(item.amount for item in row.work if item.resource == "mte2") == 102400
    assert unary_read_bytes and set(unary_read_bytes) == {51200}
    screened, = evaluate_rewrites(dossier.handle, (plan,)).results
    for receipt in (rebound.verification, screened.verification):
        assert all(gate.passed for gate in receipt.gates[:3])
        gate = receipt.gates[3]
        assert not gate.passed and gate.reason == "WORK_CONSERVATION_MISMATCH"
        assert gate.detail == "executed_mte2_service_dropped"

    # Equal aggregate service can hide a shortfall on an individual operator.
    from_report = oracle._evaluation

    def match_aggregate_service(*args, **kwargs):
        replay = from_report(*args, **kwargs)
        evidence = {**replay.execution_evidence, "serviced": baseline.step_eval.execution_evidence["serviced"]}
        return replace(replay, execution_evidence=evidence)

    monkeypatch.setattr(oracle, "_evaluation", match_aggregate_service)
    dossier.handle._finalized.clear()
    gate = finalize_plan(dossier.handle, plan).verification.gates[3]
    assert not gate.passed and gate.reason == "WORK_CONSERVATION_MISMATCH"
    assert gate.detail == f"{swiglu.op_id}:executed_mte2"

def test_strided_run_rounds_to_exact_sector_union():
    run = AddressRun(base_byte=13, rows=4097, row_bytes=39, stride_bytes=101)
    sectors = {address // 32 for row in range(run.rows)
               for address in range(13 + 101 * row, 52 + 101 * row)}
    assert run.sector_union_bytes(32) == 32 * len(sectors) < run.rows * run.stride_bytes

def test_integer_lower_bound_has_inward_guard_at_float_precision_boundary():
    for ns in (2**52, 2**53, 2**54):
        guard = ns - inward_floor_ns(Fraction(ns)) - 2
        assert guard == 8 * ulp(float(ns)) >= 4 * ulp(float(ns))
    assert inward_floor_ns(Fraction(1, 2)) == 0


def test_certified_hbm_capacity_prices_bytes_per_millisecond():
    sku = REPRESENTATIVE_SKUS["a5"]
    case = _case(sku, tp=1)
    table = PrimitiveCapacityTable(sku, (PrimitiveCapacityEntry("hbm_bus_r", 8e9, 8e9, "B/ms"),))
    assert table.upper_capacity_per_second("hbm_bus_r") == 8 * 10**12
    volume = 8_000_000
    physical = TensorRangeObligation(0, "weight", 0, ((0, volume),), "hbm_cold", "owner")
    region = SimpleNamespace(op_ids=(), semantic_work=(), read_obligations=(physical,), write_obligations=())
    machine = parse_case(case).machine
    scenario = SimpleNamespace(machine=machine)
    program, dag, _ = build_dual_inputs_from_scenario(
        scenario, region_summaries=(region,), primitive_table=table, rate_mode="certified_upper_capacity",
    )
    certified, _ = certify_lower_bound(program, dag)
    engine_program, engine_dag, _ = build_dual_inputs_from_scenario(scenario, region_summaries=(region,))
    engine_bound, _ = certify_lower_bound(engine_program, engine_dag)

    rate_per_ms = dict(machine.dies[0].rates)["hbm"] / 1000
    engine = Engine((Resource("hbm", rate_per_ms),), backend="python")
    transferred = engine.submit("cold-weight", {"hbm": volume})
    engine.advance(until=(transferred,))
    assert certified.l_star_lean_ns == 997  # 8 MB / 8 TB/s = 1000 ns, rounded inwards.
    assert certified.l_star_lean_ns <= engine_bound.l_star_lean_ns <= engine.now * NS_PER_MS
    ladder = api.optimize_ladder(case, StepRequest("decode", context=64), primitive_table=table).ladder
    assert 0 <= ladder.l_star_hw_ns <= ladder.l_star_lean_ns <= ladder.a_star_dsl_ns


def test_ladder_certifies_shipped_primitive_capacities():
    table = load_primitives("Ascend950DT_9582")
    case = _case(table.sku, tp=1)
    step = StepRequest("decode", context=64)
    report = api.optimize_ladder(case, step, primitive_table=table, max_trials=0)
    ladder = report.ladder
    assert report.verification.passed
    assert 0 < ladder.l_star_hw_ns <= ladder.l_star_lean_ns <= ladder.a_star_dsl_ns

    dossier = prepare_scenario(ScenarioRequest(case, step))
    result = finalize_plan(dossier.handle, report.dsl_plan, primitive_table=table,
                           rate_mode="certified_upper_capacity")
    assert result.verification.passed and result.verification.l_star_ns == ladder.l_star_hw_ns


def test_certified_primitive_table_prices_directional_hbm_bytes():
    table = load_primitives("Ascend950DT_9582")
    machine = parse_case(_case(table.sku, tp=1)).machine
    scenario = SimpleNamespace(machine=machine)
    volume = 8_000_000
    read = TensorRangeObligation(0, "input", 0, ((0, volume),), "hbm_cold", "owner")
    write = TensorRangeObligation(0, "output", 1, ((0, volume // 2),), "onchip_resident", "device")
    summary = SimpleNamespace(op_ids=(), semantic_work=(), read_obligations=(read,), write_obligations=(write,))
    program, dag, names = build_dual_inputs_from_scenario(
        scenario, region_summaries=(summary,), primitive_table=table, rate_mode="certified_upper_capacity",
    )
    assert dict(zip(names, program.global_work, strict=True)) == {
        "hbm_bus_r@r0.p0.d0": 1997, "hbm_bus_w@r0.p0.d0": 997,
    }
    cert, _ = certify_lower_bound(program, dag)
    assert cert.l_star_lean_ns == 1997


def _graph_spine(*ops):
    from mindie_sim.prediction.spine import PredictionSpine

    class _Spine(PredictionSpine):
        def lower(self, case, request, **kwargs):
            prepared = super().lower(case, request, **kwargs)
            rank, = prepared.graph.executable
            return replace(prepared, graph=replace(prepared.graph, executable=(replace(rank, ops=ops),), streams=()))

    return _Spine()


def _unit_hbm_case():
    base = parse_case(_case(REPRESENTATIVE_SKUS["a5"], tp=1))
    die, = base.machine.dies
    rate = 1024 * 1000
    die = replace(die, hbm_read_bytes_per_s=rate, hbm_write_bytes_per_s=rate, hbm_rw_alpha=0.0,
                  rates=tuple(dict(die.rates, hbm=rate).items()))
    return replace(base, machine=replace(base.machine, dies=(die,))), die


def test_verify_plan_rejects_fabricated_traffic_and_disambiguates_explicit_k3():
    from _support.linear_cases import tile_edge
    from mindie_sim.catalog.ops import vector_op
    from mindie_sim.rewrite.catalog_replay import _dsl_proposals
    from mindie_sim.prediction.program.stages import Stage
    from mindie_sim.rewrite.schema import decode_rewrite_plan, encode_rewrite_plan, plan_sha256
    from mindie_sim.rewrite.verify import verify_plan
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.templates.contracts import SemanticWorkItem

    case, die = _unit_hbm_case()
    pw = lambda oid, deps=(): vector_op(op_id=oid, deps=deps, requests=(), stage=0, layer=0, elements=4096,
                                        ops_per_element=1, dtype="bf16", family="elementwise", phase="decode", recipe="elementwise")
    make, take = pw("make"), pw("take", ("make",))
    decls = tuple(dict(
        numerically_equivalent=True, region_id="same_id", op_ids=("make", "take"), edge_routes=(edg,), backend="dsl_vec",
        compiler_options=dict(edge_bytes={edg.view_id: 8192}, l2_reservation_bytes=int(dict(die.buffers)["l2"]),
                              op_stages={op.op_id: Stage(op.op_id, {"vector": op.vector_ops}) for op in (make, take)},
                              semantic_work=tuple(SemanticWorkItem(op.op_id, "vector_alu", op.dtype, int(op.vector_ops), ("aiv_simd",))
                                                  for op in (make, take))),
    ) for edg in (replace(tile_edge("ub->ub", ("aiv", "aiv")), representation=make.dtype),
                  replace(tile_edge("gm_spill", ("aiv", "aiv")), representation=make.dtype)))
    take_d = replace(take, metadata=(*take.metadata, ("fuse_regions", decls)))
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode")), spine=_graph_spine(make, take_d))
    handle, dp = dossier.handle, dossier.handle.scenario.datapath
    props = list(_dsl_proposals((make, take_d), dp, dp.aic_count, dp.aiv_count))
    p_ub, p_gm = (replace(dossier.unmerge_plan, edits=(Edit(eid, operation=pr[0][0]),))
                  for eid, pr in zip(("f0", "f1"), props, strict=True))
    assert plan_sha256(p_ub) != plan_sha256(p_gm) and decode_rewrite_plan(encode_rewrite_plan(p_gm)) == p_gm
    f_ub, f_gm = finalize_plan(handle, p_ub), finalize_plan(handle, p_gm)
    assert f_ub.verification.passed and f_gm.verification.passed and f_ub.region_summaries[0].dram_sector_bytes == 0
    assert f_gm.region_summaries[0].dram_sector_bytes == 16384 and f_gm.step_eval.span_ms >= 15.9
    refs = (OpRef("atom", "make"), OpRef("atom", "take"))
    ambig = replace(dossier.unmerge_plan, edits=(Edit("a", operation=FuseRegion(
        refs, (), (KernelBoundary("same_id", refs),), fusion_tier="K3_dsl_pipeline")),))
    with pytest.raises(ModelingError, match="rewrite_template_recipe_unbound"):
        finalize_plan(handle, ambig)
    real_s, = f_ub.region_summaries
    fake_b = 8_000_000
    tampered = replace(real_s, read_obligations=(*real_s.read_obligations, TensorRangeObligation(0, "fake", 0, ((0, fake_b),), "hbm_cold", "device")),
                       logical_transfer_bytes=real_s.logical_transfer_bytes + fake_b, issued_copy_bytes=real_s.issued_copy_bytes + fake_b,
                       dram_sector_bytes=real_s.dram_sector_bytes + fake_b, turnaround_service_bytes=real_s.turnaround_service_bytes + fake_b)
    prog, dag, _ = build_dual_inputs_from_scenario(handle.scenario, region_summaries=(tampered,))
    cert, _ = certify_lower_bound(prog, dag)
    receipt = verify_plan(handle.scenario, f_ub.applied, step_eval=f_ub.step_eval, wall_ns=f_ub.wall_ns,
                          region_summaries=(tampered,), dual_certificate=cert, dual_program=prog, event_dag=dag)
    assert not receipt.passed and receipt.l_star_ns is None
    assert (receipt.gates[3].reason, receipt.gates[4].reason) == ("LEDGER_RECONCILIATION_ERROR", "LEAN_CERTIFICATE_INVALID")


def test_catalog_and_shipping_hbm_arbiter_and_nonzero_weight_ranges():
    from mindie_sim.catalog.ops import gemm_op
    from mindie_sim.catalog.types import QuantSpec
    from mindie_sim.rewrite.catalog_replay import _catlass_proposals
    from mindie_sim.prediction.streams import StreamCoreBudget, StreamLane
    from mindie_sim.rewrite.schema import StreamSchedule

    case, _ = _unit_hbm_case()
    mk = lambda oid, md: gemm_op(op_id=oid, deps=(), requests=(), stage=0, layer=0, m=1, n=64, k=64,
                                 quant=QuantSpec("bf16"), category="dense", phase="decode", metadata=md)
    g0 = mk("g0", (("weight_allocation_id", "W0"), ("weight_initial_residency", "hbm_cold"), ("weight_ranges", ((8192, 16384),))))
    g1 = mk("g1", (("weight_allocation_id", "W1"), ("weight_initial_residency", "hbm_cold"), ("weight_bytes", 8192)))
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode")), spine=_graph_spine(g0, g1))
    dp = dossier.handle.scenario.datapath
    ha, hv = dp.aic_count // 2, dp.aiv_count // 2
    lanes = (StreamLane("main", StreamCoreBudget(ha, hv, "disjoint")),
             StreamLane("aux", StreamCoreBudget(dp.aic_count - ha, dp.aiv_count - hv, "disjoint")))
    for target in (g0, g1):
        (cat_edit,), _ = next(_catlass_proposals((target,), dp, ha, hv))
        plan = replace(dossier.unmerge_plan, edits=(
            Edit("s", operation=StreamSchedule(lanes, ((OpRef("atom", "g0"), "main"), (OpRef("atom", "g1"), "aux")), ())),
            Edit("c", operation=cat_edit)))
        full = finalize_plan(dossier.handle, plan)
        assert full.verification.passed and full.step_eval.span_ms >= 16.25 and full.verification.l_star_ns == 16_124_997 <= full.wall_ns
        if target is g0:
            w_obs = [o for o in full.region_summaries[0].read_obligations if o.allocation_id == "W0"]
            assert len(w_obs) == 1 and w_obs[0].ranges == ((8192, 16384),)
