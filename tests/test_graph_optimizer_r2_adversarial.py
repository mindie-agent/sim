"""NR-S2-FIX2: adversarial regressions for every Round-2 open finding.

[Role] Reproduce each counterexample from ``astra_review_r2.md`` through the
  canonical selected-``OpSpec`` bridge, the real ``Engine`` pools and the typed
  graph, not through source-string assertions.
[Invariants] Every assertion is an executed work/timeline/edge fact; rejected
  trials cannot mutate accepted books; tile, pool and priority identities are
  exact.
[Timeouts & Locks] Pure in-process simulation; no I/O, clocks or locks.
[Side-Effect Policy] No filesystem or environment writes.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim import compose_case
from mindie_sim.prediction.engine_plan import (
    HardwareProfile, apply_fusion, baseline_plan, effective_edges, evaluate_plan,
)
from mindie_sim.prediction.fusion_bridge import (
    add_prefetch, add_recipe, apply_rewrite, canonical_plan, canonical_stream_move,
    insert_stream_events, project_graph, rewrite_fusion, timeline_by_op,
)
from mindie_sim.prediction.fusion_contract import (
    CubeResources, ReductionMerge, _contract, tile_bytes,
)
from mindie_sim.prediction.fusion_feasibility import (
    analyze_region, region_legality, ub_worksheet,
)
from mindie_sim.prediction.fusion_graph import (
    FeasibilityError, OpNode, TensorValue, TypedGraph,
)
from mindie_sim.prediction.fusion_pool import PoolLedger, declared_l1_holders
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_cube_epilogue, synthesize_cube_prologue,
    synthesize_vector,
)
from mindie_sim.prediction.graph_optimizer import run_optimizer
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.optimizer_report import Budget
from mindie_sim.prediction.plan_model import PlanNode
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.runtime.arbiter import component_rates
from mindie_sim.prediction.runtime.engine import Resource
from mindie_sim.prediction.coupled import coupled_rates
from mindie_sim.prediction.weight_prefetch import PrefetchPhysics, plan_prefetch
from mindie_sim.schema.operators import OpSpec

from graph_opt_fixtures import (
    canonical_mlapo, canonical_qwen, cast_chain_graph, hw, quant_graph,
    scalar_cube_graph, state_graph, val,
)

_CASE = None

def case():
    global _CASE
    if _CASE is None:
        _CASE = compose_case("Qwen3.8-27B", hardware="a3", tp=1, layers=1)
    return _CASE


def _matmul_op(op_id="mm", *, shape=(1, 2048, 2048), hbm=None, deps=(),
               metadata=()):
    m, n, k = shape
    return OpSpec(
        op_id=op_id, family="gemm", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(m, n, k), local_shape=(m, n, k),
        padded_shape=(m, n, k), launch_shape=(m, n, k),
        flops=2.0 * m * n * k,
        hbm_bytes=float(hbm if hbm is not None else m * k * 2 + n * k * 2 + m * n * 2),
        dtype="bf16", deps=tuple(deps), metadata=tuple(metadata))


def _add_op(op_id, deps=(), *, vector=2.0, hbm=1_000_000.0, shape=(1, 4),
            metadata=()):
    return OpSpec(op_id=op_id, family="add", phase="decode", stage=0, layer=0,
                  requests=(), semantic_shape=shape, local_shape=shape,
                  padded_shape=shape, launch_shape=shape, hbm_bytes=hbm,
                  vector_ops=vector, dtype="bf16", deps=tuple(deps),
                  metadata=tuple(metadata))


def _two_pair_plan():
    ops = (_add_op("a", ("analysis_input",), vector=200_000_000.0),
           _add_op("b", ("a",), vector=2.0),
           _add_op("c", ("analysis_input",), vector=2.0),
           _add_op("d", ("c",), vector=2.0))
    return canonical_plan(case(), ops)


def _fuse_first(plan, pass_name="vector_fusion"):
    from mindie_sim.prediction.graph_optimizer_engine import _candidate_specs
    specs, _ = _candidate_specs(pass_name, plan.graph, plan.hardware,
                                Budget(per_pass=32, total=64, wall_s=30))
    for spec in specs:
        if not isinstance(spec, RejectedFusion):
            return apply_rewrite(plan, rewrite_fusion(plan, spec)), spec
    raise AssertionError(f"no admissible {pass_name} candidate")


# ---------------------------------------------------------------------------
# New P1: private books and occurrence-scoped identities
# ---------------------------------------------------------------------------
def test_p1_trial_recipes_are_private_and_occurrence_scoped():
    plan = _two_pair_plan()
    first, spec_a = _fuse_first(plan)
    fused_a = next(op for op in first.ops if op.family == "native_fusion")
    assert int(fused_a.vector_ops) == 200_000_002
    vector_before = first.evaluate()["execution"]["work"]["vector_ops"]
    assert int(vector_before) == 200_000_006
    # The baseline book was not mutated by registering the accepted trial.
    assert spec_a.recipe_id not in plan.book.recipes
    assert spec_a.recipe_id in first.book.recipes
    second, spec_b = _fuse_first(first)
    assert spec_a.recipe_id != spec_b.recipe_id
    ids = [dict(op.metadata).get("device_kernel_group", {}).get("id")
           for op in second.ops if op.family == "native_fusion"]
    assert all(ids) and len(set(ids)) == len(ids)
    both = second.evaluate()
    assert int(both["execution"]["work"]["vector_ops"]) == 200_000_006
    # Re-evaluating the earlier accepted plan cannot see the later trial.
    replay = first.evaluate()
    assert int(replay["execution"]["work"]["vector_ops"]) == int(vector_before)
    assert replay["duration_ms"] == pytest.approx(first.evaluate()["duration_ms"])
    assert spec_b.recipe_id not in first.book.recipes

    book = plan.book
    marker = object()
    clone = add_recipe(book, "r2:private", marker)
    assert "r2:private" not in book.recipes
    assert clone.recipes["r2:private"] is marker


# ---------------------------------------------------------------------------
# R01: opaque source planners, roots, dtype and control boundaries
# ---------------------------------------------------------------------------
def test_r01_source_kernel_is_opaque_and_keeps_declared_work():
    source = _add_op("src", ("analysis_input",), vector=1_000_000_000.0)
    source = replace(source, family="native_fusion", source_boundary="kernel")
    graph = project_graph(case(), (source, _add_op("tail", ("src",))),
                          opaque_ops=("src",))
    assert graph.ops["src"].opaque is True
    assert graph.ops["src"].source_boundary == "kernel"
    assert graph.ops["src"].work_dict()["vector"] == 1_000_000_000.0
    legality = region_legality(graph, ("src", "tail"))
    assert legality.ok is False
    assert "source_kernel_boundary_opaque" in legality.reasons
    candidate = synthesize_vector(graph, ("src", "tail"), tile_rows=1,
                                  capacity_per_owner=1 << 20)
    assert isinstance(candidate, RejectedFusion)
    assert "source_kernel_boundary_opaque" in candidate.reasons
    automatic = canonical_plan(case(), (source, _add_op("tail", ("src",))))
    assert automatic.opaque_ops == ("src",)
    def build(op, geometry):
        def program(context):
            names = context.resource_names
            return KernelProgram((Stage("body", {
                names["vector"]: 1_000_000_000.0}),))
        return OperatorPlan(Work(resources={"vector": 1_000_000_000.0}), program)
    from mindie_sim.prediction.canonical_optimizer import _run_canonical
    from mindie_sim.prediction.fusion_pool import register_recipe
    source_kernel = replace(
        _add_op("src", ("analysis_input",)), family="native_fusion",
        source_boundary="kernel", recipe="src_recipe",
        vector_ops=1_000_000_000.0)
    tail = _add_op("vec", ("src",), vector=1000.0)
    plan = canonical_plan(case(), (source_kernel, tail))
    plan = replace(plan, book=register_recipe(
        plan.book, "src_recipe", PlannedRecipe(build)))
    report = _run_canonical(plan, evaluation_budget=8, per_pass=4,
                            max_streams=None, wall_budget_s=10.0)
    assert int(report.final.vector_ops) == 1_000_001_000
    assert not any("src" in item["members"] for item in report.applied)


def test_r01_live_out_roots_and_cast_dtype_are_not_fabricated():
    plan = canonical_plan(case(), (
        _add_op("first", ("analysis_input",)),
        _add_op("second", ("first",)),
        _matmul_op("cast_source", shape=(1, 8, 8), deps=("second",)),
    ))
    assert plan.graph.public_outputs == ("cast_source:out",)
    raw_mla, ops_mla = canonical_mlapo()
    assert canonical_plan(raw_mla, ops_mla).graph.public_outputs == (
        "s0.l0.ffn.residual:out",)
    # A BF16 producer consumed by an FP16 Cast keeps BF16; only the Cast output
    # changes dtype.
    cast = OpSpec(op_id="c", family="cast", phase="decode", stage=0, layer=0,
                  requests=(), semantic_shape=(1, 4), local_shape=(1, 4),
                  padded_shape=(1, 4), launch_shape=(1, 4), hbm_bytes=32.0,
                  dtype="fp16", deps=("p",),
                  metadata=(("rounding", "rne"),))
    producer = _add_op("p", ("analysis_input",))
    graph = project_graph(case(), (producer, cast))
    assert graph.values["p:out"].dtype == "bf16"
    assert graph.values["c:out"].dtype == "fp16"
    assert graph.values["p:out"].dtype == "bf16"


def test_r01_control_dependency_never_becomes_a_tensor():
    producer = _add_op("p", ("analysis_input",))
    consumer = _add_op("q", ("p", "barrier:only"),
                       metadata=(("control_deps", ("barrier:only",)),))
    graph = project_graph(case(), (producer, consumer))
    assert graph.ops["q"].inputs == ("p:out",)
    assert "barrier:only:source" not in graph.values
    assert "barrier:only" in graph.external_inputs
    assert graph.ops["q"].semantic_complete is True
    unknown = _add_op("u", ("mystery",))
    graph = project_graph(case(), (unknown,), external_data=("analysis_input",))
    assert graph.ops["u"].semantic_complete is False
    assert "mystery:source" not in graph.values


# ---------------------------------------------------------------------------
# R07 / New P1: canonical declared-work conservation
# ---------------------------------------------------------------------------
def test_r07_canonical_fusion_conserves_declared_hbm_and_resource_work():
    plan = _two_pair_plan()
    fused_plan, spec = _fuse_first(plan)
    fused = next(op for op in fused_plan.ops if op.family == "native_fusion")
    metadata = dict(fused.metadata)
    assert metadata["declared_hbm_before"] == 2_000_000
    assert metadata["proven_eliminated_hbm"] == 16
    assert fused.hbm_bytes == 2_000_000 - 16
    assert int(fused.vector_ops) == 200_000_002
    report = fused_plan.evaluate()
    baseline_hbm = plan.evaluate()["execution"]["work"]["hbm_bytes"]
    assert int(report["execution"]["work"]["hbm_bytes"]) == int(baseline_hbm) - 16
    # A stream-only acceptance may never change physical HBM work.
    moved = canonical_stream_move(fused_plan, ("b",), fused_plan.hardware, "aux")
    streamed = replace(fused_plan, ops=moved,
                       graph=project_graph(fused_plan.case, moved,
                                           opaque_ops=fused_plan.opaque_ops,
                                           external_data=fused_plan.external_data))
    if streamed.hardware.stream_switch_ms == 0:
        assert streamed.evaluate()["execution"]["work"]["hbm_bytes"] == \
            pytest.approx(baseline_hbm - 16.0)

    declared = OpSpec(op_id="svc", family="add", phase="decode", stage=0,
                      layer=0, requests=(), semantic_shape=(1, 4),
                      local_shape=(1, 4), padded_shape=(1, 4),
                      launch_shape=(1, 4), hbm_bytes=8.0, vector_ops=8.0,
                      matrix_flops=((("bf16"), 16.0),), dtype="bf16",
                      metadata=(("resource_work", {"mte1": 1234.0, "host": 0.5}),))
    graph = project_graph(case(), (declared,))
    work = graph.ops["svc"].work_dict()
    assert work["mte1"] == 1234.0 and work["host"] == 0.5
    assert graph.ops["svc"].param("matrix_flops") == {"bf16": 16.0}


def test_r07_qwen_waterfall_conserves_physical_work():
    raw, ops = canonical_qwen()
    report = run_optimizer(raw, ops, evaluation_budget=64,
                           max_candidates_per_pass=32)
    assert report.final.hbm_bytes <= report.baseline.hbm_bytes
    for row in report.passes:
        if row.stage == "+multistream_overlap":
            assert row.after_hbm <= row.before_hbm + 1e-6
    for item in report.applied:
        if item["pass"] == "streams":
            assert item["hbm_before"] == pytest.approx(item["hbm_after"])
        if item["pass"] == "vector_fusion":
            assert int(item["hbm_before"]) - int(item["hbm_after"]) == 163840


# ---------------------------------------------------------------------------
# R02 / R13: exact multi-axis and tail tile sizing
# ---------------------------------------------------------------------------
def test_r02_r13_tile_stride_tail_and_peak_ub_reservation():
    value = TensorValue("v", "bf16", (3, 2, 4))
    two = tile_bytes(value, 2, 3)
    one = tile_bytes(value, 1, 3, start_row=2)
    assert (two, one, two + one) == (32, 16, 48)
    assert tile_bytes(value, 1, 3) != 24

    values, ops = cast_chain_graph("t")
    values = {key: replace(item, shape=(3, 1), storage_bytes=97)
              for key, item in values.items()}
    graph = TypedGraph(3, values, ops, ("tz",))
    worksheet = ub_worksheet(graph, ("ta", "tb"), tile_rows=1, channel_width=1,
                             capacity_per_owner=200)
    assert worksheet.ub_tile_bytes == 384
    assert worksheet.tail_tile_bytes == 384
    assert worksheet.fits is False
    assert max(a.bytes_per_instance for a in worksheet.allocations) == 64


# ---------------------------------------------------------------------------
# R03 / R05: real L1 grants, existing holders, exact weight identity
# ---------------------------------------------------------------------------
def test_r03_existing_l1_holders_reject_an_overlapping_prefetch():
    holder = OpSpec(
        op_id="holder", family="gemm", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, 1, 1), local_shape=(1, 1, 1),
        padded_shape=(1, 1, 1), launch_shape=(1, 1, 1), flops=2.0,
        hbm_bytes=8.0, dtype="bf16", deps=("analysis_input",),
        metadata=(("device_kernel_group", {
            "id": "r2_l1_holder", "kernel": "L1Holder", "entry": True,
            "exit": True, "buffers": ({"id": "w", "pool": "l1",
                                       "bytes": 524288,
                                       "units": tuple(range(1, 24))},)}),
                  # M2 lowers the GEMM body through a physical matrix recipe
                  # whose first owner needs L1 unit 0.  Reserve that unit as a
                  # declared optimizer hold so the holder remains executable
                  # while the planner still sees all 24 units held.
                  ("optimizer_l1_hold", {"bytes": 524288, "units": (0,)})),
    )
    plan = canonical_plan(case(), (holder,))
    assert declared_l1_holders(plan) == 12 * 1024 * 1024
    decision = plan_prefetch(
        plan.graph, "holder", window_ms=10.0, foreground_hbm_bytes=0.0,
        physics=PrefetchPhysics(plan.hardware.hbm_bytes_per_ms,
                                plan.hardware.onchip_bytes_per_ms,
                                plan.hardware.prefetch_capacity_bytes),
        capacity_bytes=plan.hardware.prefetch_capacity_bytes,
        reserved_bytes=declared_l1_holders(plan))
    assert decision.allowed is False
    assert decision.reason == "prefetch_pool_saturated"

    pre = OpSpec(op_id="pre", family="cast", phase="decode", stage=0, layer=0,
                 requests=(), semantic_shape=(1,), local_shape=(1,),
                 padded_shape=(1,), launch_shape=(1,), hbm_bytes=8.0,
                 dtype="bf16", deps=("analysis_input",),
                 metadata=(("resource_work", {"host": 10.0}),))
    held = replace(holder, deps=("pre",))
    from mindie_sim.prediction.canonical_optimizer import _run_canonical
    report = _run_canonical(canonical_plan(case(), (pre, held)),
                            evaluation_budget=8, per_pass=4, max_streams=None,
                            wall_budget_s=10.0)
    assert not any(item["pass"] == "weight_prefetch" for item in report.applied)
    assert any("prefetch_pool_saturated" in rejection.reasons
               for rejection in report.rejections
               if rejection.pass_name == "weight_prefetch")


def test_r03_canonical_prefetch_acquires_at_start_and_frees_at_consumer():
    plan = canonical_plan(case(), (_matmul_op(),))
    decision = plan_prefetch(
        plan.graph, "mm", window_ms=50.0, foreground_hbm_bytes=0.0,
        physics=PrefetchPhysics(plan.hardware.hbm_bytes_per_ms,
                                plan.hardware.onchip_bytes_per_ms,
                                plan.hardware.prefetch_capacity_bytes),
        capacity_bytes=plan.hardware.prefetch_capacity_bytes)
    assert decision.allowed
    trial, prefetch_id = add_prefetch(plan, decision, plan.hardware)
    result = trial.evaluate(audit_allocations=True)
    trace = [row for row in result.get("allocation_trace", ())
             if str(row.get("pool", "")).startswith("l1")]
    # The matrix program also grants/releases per-tile L1 panels; isolate the
    # prefetch grant so the grant-lifetime assertion is not the first tile.
    acquires = [row for row in trace if row["action"] == "acquire"
                and prefetch_id in row["name"]]
    releases = [row for row in trace if row["action"] == "release"
                and prefetch_id in row["name"]]
    assert len(acquires) == len(releases) == 1
    rows = timeline_by_op(result)
    assert acquires[0]["time"] == pytest.approx(rows[prefetch_id]["ready_ms"])
    # The grant is held until both the consumer kernel and the producer copy
    # retire; a delayed copy must never lose its destination storage early.
    release_at = max(rows["mm"]["complete_ms"],
                     rows[prefetch_id]["complete_ms"])
    assert releases[0]["time"] == pytest.approx(release_at)
    assert len(acquires[0]["units"]) == plan.hardware.prefetch_units
    ledger = PoolLedger()
    ledger.add(rows[prefetch_id]["ready_ms"], release_at,
               decision.target_bytes)
    assert ledger.peak() == decision.target_bytes


def test_r05_weight_bytes_are_declared_identity_not_hbm_minus_input():
    weight = 2048 * 2048 * 2
    op = _matmul_op(hbm=weight + 8192,
                    metadata=(("packed_weight_bytes", float(weight)),
                              ("weight_dtype", "bf16")))
    plan = canonical_plan(case(), (op,))
    weight_ids = [v for v in plan.graph.ops["mm"].inputs
                  if plan.graph.values[v].role == "weight"]
    assert len(weight_ids) == 1
    assert plan.graph.values[weight_ids[0]].storage_bytes == weight
    decision = plan_prefetch(
        plan.graph, "mm", window_ms=10.0, foreground_hbm_bytes=0.0,
        physics=PrefetchPhysics(plan.hardware.hbm_bytes_per_ms,
                                plan.hardware.onchip_bytes_per_ms,
                                plan.hardware.prefetch_capacity_bytes),
        capacity_bytes=plan.hardware.prefetch_capacity_bytes)
    assert decision.weight_bytes == weight
    assert decision.bytes_prefetched + decision.hbm_read_bytes == weight
    assert op.hbm_bytes - decision.bytes_prefetched == 8192.0


# ---------------------------------------------------------------------------
# R04: strict tiers for placed/coupled contenders and per-consumer gates
# ---------------------------------------------------------------------------
def _task(priority=0, placed=None, coupled=False, remaining=None, start=0.0):
    from types import SimpleNamespace
    return SimpleNamespace(priority=priority, placement=placed or {},
                           coupled=coupled, remaining=remaining or {},
                           caps={}, start=start, cohort_size=1)


def test_r04_placed_and_coupled_foreground_outrank_prefetch():
    resources = {"hbm": Resource("hbm", 1024, "bytes", units=1),
                 "vector": Resource("vector", 1e9, "ops")}
    foreground = _task(0, {"hbm": (0,)}, remaining={"hbm": 1024})
    prefetch = _task(10, remaining={"hbm": 1024}, start=1.0)
    rates = component_rates(resources, {0: foreground, 1: prefetch}, [0, 1],
                            coupled_active=0, placed_active=1)
    assert rates["hbm"][0] == pytest.approx(1024.0)
    assert rates["hbm"].get(1, 0.0) == pytest.approx(0.0)

    coupled = _task(0, coupled=True,
                    remaining={"hbm": 1024, "vector": 1024})
    rates = component_rates(resources, {0: coupled, 1: prefetch}, [0, 1],
                            coupled_solver=coupled_rates, coupled_active=1,
                            placed_active=0)
    assert rates["hbm"][0] == pytest.approx(1024.0)
    assert rates["hbm"].get(1, 0.0) == pytest.approx(0.0)

    unrelated = _task(0, coupled=True, remaining={"vector": 1e6})
    rates = component_rates(resources, {0: foreground, 1: unrelated, 2: prefetch},
                            [0, 1, 2], coupled_solver=coupled_rates,
                            coupled_active=1, placed_active=0)
    assert rates["hbm"][0] == pytest.approx(1024.0)
    assert rates["hbm"].get(2, 0.0) == pytest.approx(0.0)


def test_r04_prefetch_gate_is_measured_at_each_consumer_start():
    hardware = HardwareProfile(
        512, 1e9, 1e9, 4096, 65536, 1 << 20, max_streams=2, launch_ms=0.0,
        device_min_ms=0.0, capabilities=(("aclgraph_prefetch", True),),
        cube_capacities=CubeResources(65536, 65536, 65536, 65536))
    plan = (
        PlanNode("d1", (("host", 1.0),), kind="kernel", stream="s1"),
        PlanNode("c1", (("vector", 0.0),), deps=("d1",), kind="kernel",
                 stream="s1"),
        PlanNode("d2", (("host", 29.0),), kind="kernel", stream="s2"),
        PlanNode("c2", (("vector", 0.0),), deps=("d2",), kind="kernel",
                 stream="s2"),
        PlanNode("prefetch_W_1", (("hbm", 8192.0),), kind="prefetch",
                 prefetch_for="c1", stream="prefetch", priority=10),
        PlanNode("prefetch_W_2", (("hbm", 1024.0),), kind="prefetch",
                 prefetch_for="c2", stream="prefetch", priority=10),
    )
    measured = evaluate_plan(plan, hardware)
    assert measured.status == "completed"
    assert measured.start("c1") == pytest.approx(1.0)
    assert measured.start("c2") == pytest.approx(30.0)
    assert measured.prefetched("prefetch_W_1") < 8192.0
    assert measured.prefetched("prefetch_W_1") == pytest.approx(256.0)
    assert measured.prefetched("prefetch_W_2") == pytest.approx(1024.0)


# ---------------------------------------------------------------------------
# R05/R06: exact causal phases and true hardware paths
# ---------------------------------------------------------------------------
def test_r06_causal_phases_acquisitions_and_path_capabilities():
    graph = scalar_cube_graph()
    hardware = hw(vector_ops_per_ms=0.1, cube_flops_per_ms=0.2,
                  hbm_bytes_per_ms=1e9, onchip_bytes_per_ms=1e9,
                  ub_bytes_per_aiv=65536, max_streams=1)
    candidate = synthesize_cube_epilogue(
        graph, "mm", ("cast",), tile_rows=1, capacity_per_owner=65536,
        capability=True, cube_resources=hardware.cube_capacities)
    assert not isinstance(candidate, RejectedFusion)
    phases = tuple(phase for phase, _ in candidate.analysis.phase_work)
    assert phases == ("operand_read", "cube", "handoff", "epilogue",
                      "output_write")
    assert dict(candidate.analysis.phase_work)["handoff"]["mte1"] > 0
    work = dict(candidate.analysis.phase_work)
    assert work["operand_read"]["hbm"] == 4.0
    assert work["output_write"]["hbm"] == 2.0

    copies = TypedGraph(1, dict(graph.values), {
        "mm": replace(graph.ops["mm"],
                      work=(("cube", 2.0), ("mte2", 8.0), ("mte3", 4.0))),
        "cast": graph.ops["cast"],
    }, ("c",))
    copied = synthesize_cube_epilogue(
        copies, "mm", ("cast",), tile_rows=1, capacity_per_owner=65536,
        capability=True, cube_resources=hardware.cube_capacities)
    assert not isinstance(copied, RejectedFusion)
    copied_work = dict(copied.analysis.phase_work)
    assert copied_work["operand_read"]["mte2"] == 8.0
    assert copied_work["output_write"]["mte3"] == 4.0
    assert "mte2" not in copied_work["cube"]
    assert "mte3" not in copied_work["epilogue"]
    pools = {a.pool for a in candidate.analysis.worksheet.cube_allocations}
    assert {"l1", "l0a", "l0b", "l0c"} <= pools
    plan = apply_fusion(baseline_plan(graph, hardware), candidate)
    measured = evaluate_plan(plan, hardware)
    assert measured.status == "completed", measured.reason
    assert measured.duration_ms == pytest.approx(20.0, abs=1e-6)

    values = {
        "x": val("x", "bf16", (1, 1), storage=2),
        "w": val("w", "bf16", (1, 1), role="weight", storage=2),
        "h": val("h", "bf16", (1, 1), producer="pre", storage=2),
        "p": val("p", "bf16", (1, 1), producer="mm", storage=2),
    }
    prologue_graph = TypedGraph(1, values, {
        "pre": OpNode("pre", "add", "vector", ("x",), ("h",),
                      work=(("vector", 1.0),)),
        "mm": OpNode("mm", "matmul", "cube", ("h", "w"), ("p",),
                     work=(("cube", 2.0),)),
    }, ("p",))
    prologue = synthesize_cube_prologue(
        prologue_graph, "pre", "mm", tile_rows=1, capacity_per_owner=65536,
        capability=True, cube_resources=hardware.cube_capacities)
    assert not isinstance(prologue, RejectedFusion)
    assert tuple(phase for phase, _ in prologue.analysis.phase_work) == (
        "operand_read", "prologue", "prologue_handoff", "cube", "output_write")

    raw, ops = canonical_qwen()
    profile = canonical_plan(raw, ops).hardware
    assert profile.capability("cube_prologue") is False
    assert profile.capability("fixpipe_epilogue") is False
    assert profile.capability("ub_to_l1") is False
    assert profile.capability("l0c_to_ub") is False
    assert profile.capability("weight_prefetch") is True


# ---------------------------------------------------------------------------
# R08: state DAG reachability, versions and canonical ports
# ---------------------------------------------------------------------------
def test_r08_state_chain_is_valid_and_versions_are_monotone():
    state_graph().validate()
    values = {
        "s2": val("s2", "fp32", (1,), role="state", producer="first",
                  storage=4, version=2, state_key="k"),
        "s3": val("s3", "fp32", (1,), role="state", producer="second",
                  storage=4, version=1, state_key="k"),
        "x": val("x", "bf16", (1,), storage=2),
    }
    ops = {
        "first": OpNode("first", "state_update", "vector", ("x",), ("s2",),
                        state_effect="write", params=(("state_key", "k"),)),
        "second": OpNode("second", "state_update", "vector", ("x",), ("s3",),
                         deps=("first",), state_effect="write",
                         params=(("state_key", "k"),)),
    }
    with pytest.raises(FeasibilityError, match="state_version_not_monotone"):
        TypedGraph(1, values, ops, ("s2", "s3")).validate()

    raw, canonical_ops = canonical_qwen()
    plan = canonical_plan(raw, canonical_ops)
    state_values = [value for value in plan.graph.values.values()
                    if value.state_key == "s0.l0.conv_state[analysis0:0]"]
    assert state_values
    assert {value.state_version for value in state_values} == {16}


# ---------------------------------------------------------------------------
# R09 / R12: numerical contracts and priced reduction merges
# ---------------------------------------------------------------------------
def test_r09_quant_shape_enums_and_explicit_relaxation():
    graph = quant_graph()
    values = dict(graph.values)
    values["s"] = replace(values["s"], shape=(999, 1))
    bad_scale = TypedGraph(16, values, graph.ops, ("y",))
    assert "quant_scale_shape_mismatch" in _contract(
        bad_scale, bad_scale.ops["quant"]).reasons

    params = dict(graph.ops["quant"].params)
    params["rounding"], params["saturation"] = "BOGUS", "BOGUS"
    bad_modes = TypedGraph(16, graph.values, {
        "quant": replace(graph.ops["quant"], params=tuple(sorted(params.items()))),
        "cast": graph.ops["cast"]}, ("y",))
    reasons = _contract(bad_modes, bad_modes.ops["quant"]).reasons
    assert "quant_rounding_unknown" in reasons
    assert "quant_saturation_unknown" in reasons

    relaxed_cast = replace(graph.ops["cast"], dtype="bf16",
                           params=(("target_dtype", "bf16"),))
    relaxed = TypedGraph(16, graph.values,
                         {**graph.ops, "cast": relaxed_cast}, ("y",))
    denied = region_legality(relaxed, ("quant", "cast"))
    assert "relaxed_numerics_not_allowed" in denied.reasons
    allowed = region_legality(relaxed, ("quant", "cast"),
                              allow_relaxed_numerics=True)
    assert "relaxed_numerics_not_allowed" not in allowed.reasons


def test_r12_dynamic_quant_cut_and_priced_merge_work():
    graph = quant_graph()
    cut = analyze_region(graph, ("quant", "cast"), tile_rows=1,
                         channel_width=1, capacity_per_owner=100000)
    assert "whole_width_reduction_requires_merge" in cut.legality.reasons
    incoherent = analyze_region(graph, ("quant", "cast"), tile_rows=1,
                                channel_width=1, capacity_per_owner=100000,
                                reduction_merge=ReductionMerge(128, 0, 2))
    assert incoherent.legality.ok is False
    assert "reduction_merge_tiles_mismatch" in incoherent.legality.reasons
    assert "reduction_merge_work_missing" in incoherent.legality.reasons
    merged = analyze_region(graph, ("quant", "cast"), tile_rows=1,
                            channel_width=1, capacity_per_owner=100000,
                            reduction_merge=ReductionMerge(128, 1_000_000, 64))
    assert merged.legality.ok is True
    assert merged.worksheet.reduction_scratch_bytes >= 128
    assert merged.traffic.candidate_bytes == 4096 + 256
    assert "reduction_merge" in {v.value_id for v in merged.traffic.values}

    heavy = analyze_region(graph, ("quant", "cast"), tile_rows=1,
                           channel_width=1, capacity_per_owner=100000,
                           reduction_merge=ReductionMerge(128, 1_000_000, 64))
    merge_work = dict(heavy.phase_work)["merge"]
    assert merge_work["vector"] == 1_000_000
    assert merge_work["mte1"] == 256.0


# ---------------------------------------------------------------------------
# R10 / R11: canonical capture barriers and idempotent event insertion
# ---------------------------------------------------------------------------
def test_r10_canonical_cache_store_blocks_capture():
    store = OpSpec(
        op_id="cs", family="cache_store", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, 4), local_shape=(1, 4),
        padded_shape=(1, 4), launch_shape=(1, 4), hbm_bytes=64.0,
        dtype="bf16", deps=("analysis_input",),
        metadata=(("gdn_state_effects", ({"target": "cache[k]",
                                          "action": "write",
                                          "version": 1},)),))
    report = run_optimizer(case(), (store,), evaluation_budget=4,
                           max_candidates_per_pass=2)
    assert report.capture["status"] == "blocked"
    assert report.capture["evaluated"] is False
    blockers = tuple(report.capture["blockers"])
    assert any(item.startswith("state_write") for item in blockers)
    assert any(item.startswith("capture_boundary_state") for item in blockers)


def test_r11_stream_event_insertion_is_idempotent():
    raw, ops = canonical_qwen()
    plan = canonical_plan(raw, ops)
    switched = replace(plan.hardware, stream_switch_ms=0.5)
    region = ("s0.l0.beta_contiguous", "s0.l0.delta_gates")
    once = canonical_stream_move(plan, region, switched, "aux")
    twice = insert_stream_events(once, switched)
    thrice = insert_stream_events(twice, switched)
    assert len(once) == len(twice) == len(thrice)
    event_count = sum(op.family == "event" for op in once)
    assert event_count == 6  # three crossing edges, two events per edge
    assert sum(op.family == "event" for op in thrice) == event_count
    images = []
    for opset in (twice, thrice):
        image = replace(plan, ops=opset,
                        graph=project_graph(plan.case, opset,
                                            opaque_ops=plan.opaque_ops,
                                            external_data=plan.external_data))
        images.append(image.evaluate()["duration_ms"])
    assert images[0] == pytest.approx(images[1])


# ---------------------------------------------------------------------------
# R15 / R16: effective DAG order, releases, and bounded evaluation
# ---------------------------------------------------------------------------
def test_r15_exported_stream_order_matches_execution_order():
    hardware = HardwareProfile(
        1e9, 1.0, 1e9, 4e9, 65536, 0, max_streams=2, launch_ms=0.0,
        device_min_ms=0.0, cube_capacities=CubeResources(65536, 65536, 65536, 65536))
    plan = (
        PlanNode("a", (("vector", 1.0),), kind="kernel"),
        PlanNode("b", (("vector", 1.0),), deps=("a",), kind="kernel"),
        PlanNode("c", (("vector", 1.0),), kind="kernel"),
    )
    measured = evaluate_plan(plan, hardware)
    assert measured.start("c") == pytest.approx(1.0)
    assert measured.start("b") == pytest.approx(2.0)
    stream = [(edge["from"], edge["to"]) for edge in effective_edges(plan)
              if edge["kind"] == "stream_order"]
    assert ("c", "b") in stream
    assert ("b", "c") not in stream

    raw, ops = canonical_qwen()
    canonical = canonical_plan(raw, ops)
    target = next(op for op in canonical.ops
                  if op.op_id == "s0.l0.linear_beta_alpha")
    decision = plan_prefetch(
        canonical.graph, target.op_id, window_ms=10.0,
        foreground_hbm_bytes=0.0,
        physics=PrefetchPhysics(canonical.hardware.hbm_bytes_per_ms,
                                canonical.hardware.onchip_bytes_per_ms,
                                canonical.hardware.prefetch_capacity_bytes),
        capacity_bytes=canonical.hardware.prefetch_capacity_bytes)
    assert decision.allowed
    trial, prefetch_id = add_prefetch(canonical, decision, canonical.hardware)
    from mindie_sim.prediction.optimizer_report import (
        _canonical_dag, _canonical_nodes,
    )
    nodes = _canonical_nodes(trial)
    edges = _canonical_dag(trial)
    prefetch_nodes = [node for node in nodes if node.kind == "prefetch"]
    release_nodes = [node for node in nodes if node.kind == "release"]
    assert prefetch_nodes and prefetch_nodes[0].priority == decision.priority
    assert prefetch_nodes[0].caps
    assert release_nodes and release_nodes[0].free_alloc is not None
    assert any(edge["kind"] == "prefetch_join" for edge in edges)
    assert any(edge["kind"] == "reservation_release" for edge in edges)
    assert any(edge["kind"] == "pool_release" for edge in edges)
    assert all(node.priority >= 0 for node in nodes)


def test_r16_every_engine_run_is_charged_and_discovery_is_bounded():
    from graph_opt_fixtures import prefetch_graph
    graph = prefetch_graph(weight_bytes=8192, predecessor=10.0)
    hardware = hw(hbm_bytes_per_ms=1024, onchip_bytes_per_ms=4096,
                  vector_ops_per_ms=1.0, cube_flops_per_ms=1e9,
                  prefetch_capacity_bytes=1 << 20, max_streams=1)
    report = run_optimizer(graph, hardware, evaluation_budget=2,
                           max_candidates_per_pass=1)
    assert report.evaluations <= 2
    wall = run_optimizer(graph, hardware, evaluation_budget=16,
                         max_candidates_per_pass=8, wall_budget_s=0.0)
    assert wall.evaluations == 1
    assert wall.budget_exhausted is True
