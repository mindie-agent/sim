"""General N-stream core control (多流 + 控核), C11 safety, and spine invariants.

[Role] Verify general N-stream ``StreamPlan`` + ``StreamCoreBudget`` owner
  binding (including DSv4-CSA-style 3-stream schedules with 0-AIC pure-Vector
  and 0-AIC/0-AIV comm lanes), transitive cross-stream event reachability,
  C11 ``_engine_accel`` coupled-placement proportionality, destructive removal
  of whole-graph DBO, and single-pass ``PredictionSpine`` lowering parity.
[Invariants] ``ExecutionBindings.create == 1``, ``CostModel.__init__ == 1``,
  and ``effective_body_declaration == 1`` per fixed step. ``graph_id`` matches
  ``structure.export_structure``. C11 native backend runs with
  ``native_steps > 0`` on proportional coupled placements.
[Timeouts & Locks] Fast Tier-0/Tier-1 suite (<5s); isolated temp cache only.
[Side-Effect Policy] Compiles C11 kernel only into isolated cache env.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from _support.axes import STREAM_CORE_CASES

import mindie_sim.api as sdk_api
import mindie_sim.prediction.body_defaults as body_defaults_mod
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.evaluation import ExecutionBindings
from mindie_sim.prediction.frontend import Frontend
import mindie_sim.prediction.operator_analysis as op_analysis_mod
from mindie_sim.prediction.operators.matrix.matrix import TiledMatmul, _TiledGeometry
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelinePhase, PipelineTemplate,
    PipelineTemplateProgram, RecipeContext, Stage,
)
from mindie_sim.prediction.results import ReplayOptions, StepRequest, StructureRequest
from mindie_sim.prediction.runtime.accel_build import build_accel
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.streams import (
    MAIN_STREAM,
    StreamCoreBudget,
    StreamEvent,
    StreamLane,
    StreamPlan,
    apply_stream_owners_to_program,
    bind_stage_to_stream,
    bind_stream_owners,
    lower_stream_edges,
    proportional_coupled_placement,
    stream_quota_resources,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import LoweringRequest, RankOps
import mindie_sim.structure as structure_mod

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_NAMES = {"cube": "cube@0", "vector": "vector@0"}
CORE_RATES = {"cube@0": 24.0, "vector@0": 48.0}
CORE_UNITS = {"cube": 24, "vector": 48}
CORE_BINDING = dict(resource_names=CORE_NAMES, rates=CORE_RATES, resource_units=CORE_UNITS)
SHARED_PLAN = StreamPlan(rank=0, lanes=(
    StreamLane(MAIN_STREAM, StreamCoreBudget(aic=12, aiv=24, isolation="shared_cap")),
    StreamLane("aux", StreamCoreBudget(aic=12, aiv=24, isolation="shared_cap")),
))


def _case_for(sku: str = "Ascend910B2", *, layers: int = 1, tp: int = 1, dp: int = 1):
    return sdk_api.compose_case(
        "qwen3-32b", hardware=sku, options={"dp": dp, "tp": tp, "pp": 1, "layers": layers}
    )


def _main_quotas(owners, datapath):
    resources, names = stream_quota_resources(owners, datapath, resource_names=CORE_NAMES, rates=CORE_RATES)
    return resources, {physical: quota for (stream, physical), quota in names.items() if stream == MAIN_STREAM}


def _rotary_op(op_id: str, *, deps: tuple[str, ...] = ()) -> OpSpec:
    shape = (8, 16, 64)
    return OpSpec(
        op_id=op_id, layer=0, stage=0, family="rotary", recipe="rotary_position_embedding",
        phase="decode", requests=("r0",), dtype="bf16", deps=deps, stream=MAIN_STREAM,
        local_shape=shape, semantic_shape=shape, padded_shape=shape, launch_shape=shape,
    )


def _canonical_cube_op(case, op_id: str) -> OpSpec:
    ops = op_analysis_mod.target_graphs(
        case, phase="decode", tokens=1, context=64, sequences=1, scope="body"
    )[0]
    proj = next(op for op in ops if op.family == "qkv_projection")
    return replace(proj, op_id=op_id, deps=(), stream=MAIN_STREAM)


def test_graph_level_dbo_removed_destructively():
    assert not (REPO_ROOT / "mindie_sim" / "prediction" / "dbo_ledger.py").exists()
    assert not (REPO_ROOT / "mindie_sim" / "prediction" / "overlap_plan.py").exists()
    assert not hasattr(Frontend, "dbo_partition")


@pytest.mark.parametrize(
    "sku,aic_total,aiv_total,main_aic,main_aiv,csa_aiv",
    STREAM_CORE_CASES,
)
def test_three_stream_core_control_binding_and_transitive_events(
    sku, aic_total, aiv_total, main_aic, main_aiv, csa_aiv
):
    case = _case_for(sku, layers=1)
    dp = datapath_for_die(case.machine.dies[0])
    assert dp.aic_count == aic_total and dp.aiv_count == aiv_total

    plan_3 = StreamPlan(
        rank=0,
        lanes=(
            StreamLane(MAIN_STREAM, StreamCoreBudget(aic=main_aic, aiv=main_aiv, isolation="disjoint"), priority=0),
            StreamLane("csa", StreamCoreBudget(aic=0, aiv=csa_aiv, isolation="disjoint"), priority=1),
            StreamLane("comm", StreamCoreBudget(aic=0, aiv=0, isolation="disjoint"), priority=2),
        ),
        assignments=(
            ("op_main", MAIN_STREAM),
            ("op_csa", "csa"),
            ("op_comm", "comm"),
        ),
        events=(
            StreamEvent("ev_main_to_csa", MAIN_STREAM, "op_main", "csa", ("op_csa",)),
            StreamEvent("ev_csa_to_comm", "csa", "op_csa", "comm", ("op_comm",)),
        ),
    )

    owners = bind_stream_owners(plan_3, dp)
    assert len(owners[MAIN_STREAM].aic) == main_aic and owners[MAIN_STREAM].aic_limit == main_aic
    assert len(owners[MAIN_STREAM].aiv) == main_aiv and owners[MAIN_STREAM].aiv_limit == main_aiv
    assert owners["csa"].aic == () and owners["csa"].aic_limit == 0
    assert len(owners["csa"].aiv) == csa_aiv and owners["csa"].aiv_limit == csa_aiv
    assert owners["comm"].aic == () and owners["comm"].aiv == ()
    assert owners["comm"].aic_limit == 0 and owners["comm"].aiv_limit == 0
    assert set(owners[MAIN_STREAM].aiv).isdisjoint(set(owners["csa"].aiv))

    over_plan = StreamPlan(
        rank=0,
        lanes=(
            StreamLane(MAIN_STREAM, StreamCoreBudget(aic=aic_total, aiv=aiv_total)),
            StreamLane("csa", StreamCoreBudget(aic=1, aiv=0)),
        ),
    )
    with pytest.raises(ModelingError, match="stream_core_budget_aic_exceeded"):
        bind_stream_owners(over_plan, dp)

    op_main = _canonical_cube_op(case, "op_main")
    op_csa = _rotary_op("op_csa", deps=("op_main",))
    op_comm = OpSpec(
        op_id="op_comm", layer=0, stage=0, family="barrier", recipe="barrier",
        phase="decode", requests=("r0",), dtype="bf16",
        local_shape=(1,), semantic_shape=(1,), padded_shape=(1,), launch_shape=(1,),
        deps=("op_main", "op_csa"),
    )

    # Transitive 3-stream event chain (main -> csa -> comm) covers op_main -> op_comm.
    lowered = lower_stream_edges((op_main, op_csa, op_comm), plan_3)
    assert [op.stream for op in lowered] == [MAIN_STREAM, "csa", "comm"]

    # Dropping second hop breaks transitive reachability for op_main -> op_comm.
    with pytest.raises(ModelingError, match="stream_cross_dependency_missing_event"):
        lower_stream_edges((op_main, op_csa, op_comm), replace(plan_3, events=(plan_3.events[0],)))

    # Reverse 3-stream cycle rejected.
    back_event = StreamEvent("ev_back", "comm", "op_comm", MAIN_STREAM, ("op_main",))
    plan_cycle = replace(plan_3, events=(*plan_3.events, back_event))
    with pytest.raises(ModelingError, match="stream_event_cycle"):
        lower_stream_edges((op_main, op_csa, op_comm), plan_cycle)

    # Live 3-stream execution through PredictionSpine preserves all 3 stream IDs.
    spine = PredictionSpine()
    request = LoweringRequest(declared_ops=(RankOps(0, (op_main, op_csa, op_comm)),))
    step = spine.lower(case, request, streams=(plan_3,))
    ev = spine.run_step(step)
    assert ev.span_ms > 0.0
    streams_seen = {row.occurrence.op_id: row.stream for row in ev.operators}
    assert streams_seen["op_main"] == MAIN_STREAM
    assert streams_seen["op_csa"] == "csa"
    assert streams_seen["op_comm"] == "comm"

    # Cube op assigned to a 0-AIC stream fails explicitly with stream_zero_aic_budget.
    bad_assign = replace(
        plan_3,
        assignments=(("op_main", "csa"), ("op_csa", "csa"), ("op_comm", "comm")),
        events=(StreamEvent("ev_c", "csa", "op_csa", "comm", ("op_comm",)),),
    )
    step_bad = spine.lower(case, request, streams=(bad_assign,))
    with pytest.raises(ModelingError, match="stream_zero_aic_budget"):
        spine.run_step(step_bad)


def test_coupled_placement_c11_proportionality_and_shared_cap_wave_quantization():
    build = build_accel()
    assert build["ok"] is True and build["backend"] == "c_native"

    case22 = _case_for("Ascend910B2", layers=1)
    dp22 = datapath_for_die(case22.machine.dies[0])

    # Non-proportional (16 AIC, 24 AIV) on a (24 AIC, 48 AIV) die trims to (12, 24) for C11.
    skewed_plan = StreamPlan(
        rank=0,
        lanes=(
            StreamLane(MAIN_STREAM, StreamCoreBudget(aic=16, aiv=24, isolation="disjoint")),
            StreamLane("csa", StreamCoreBudget(aic=8, aiv=24, isolation="disjoint")),
        ),
    )
    owners = bind_stream_owners(skewed_plan, dp22)
    aic_slice, aiv_slice = proportional_coupled_placement(
        owners[MAIN_STREAM], dp22, stream_id=MAIN_STREAM
    )
    assert len(aic_slice) == 12 and len(aiv_slice) == 24
    assert len(aic_slice) * dp22.aiv_count == len(aiv_slice) * dp22.aic_count

    engine = Engine(
        [
            Resource("cube@0", 24.0, "ms", "fair", units=24),
            Resource("vector@0", 48.0, "ms", "fair", units=48),
        ],
        backend="c",
    )
    coupled_stage = Stage("coupled_cv", work={"cube@0": 12.0, "vector@0": 24.0}, coupled=True)
    bound_stage = bind_stage_to_stream(coupled_stage, owners[MAIN_STREAM], dp22,
                                       stream_id=MAIN_STREAM, **CORE_BINDING)
    assert len(bound_stage.placement["cube@0"]) == 12
    assert len(bound_stage.placement["vector@0"]) == 24
    start_ev = engine.submit("start")
    end_ev = KernelProgram((bound_stage,)).start(engine, "coupled_c11", start_ev)
    engine.advance(until=(end_ev,))
    status = engine.accel_status()
    assert status["accel_backend"] == "c_native"
    assert status["native_steps"] > 0
    assert end_ev.time == pytest.approx(1.0)

    # RecipeContext wave_count honors 控核 core budgets honestly.
    ctx_full = RecipeContext(
        op=None, work={}, resource_names={}, rates={},
        core_count=24, datapath=dp22, stream_owners=None,
    )
    ctx_capped = replace(ctx_full, stream_owners=owners[MAIN_STREAM])
    assert ctx_full.effective_aic_cores == 24 and ctx_full.wave_count(48, kind="aic") == 2
    assert ctx_capped.effective_aic_cores == 16 and ctx_capped.wave_count(48, kind="aic") == 3

    # Shared-cap (`aclrtSetStreamResLimit`) caps stage rate proportionally on C11 Engine.
    shared_plan = StreamPlan(
        rank=0,
        lanes=(
            StreamLane(MAIN_STREAM, StreamCoreBudget(aic=12, aiv=24, isolation="shared_cap")),
            StreamLane("csa", StreamCoreBudget(aic=12, aiv=24, isolation="shared_cap")),
        ),
    )
    shared_owners = bind_stream_owners(shared_plan, dp22)
    assert shared_owners[MAIN_STREAM].aic == () and shared_owners[MAIN_STREAM].aic_limit == 12
    shared_stage = bind_stage_to_stream(Stage("shared_cube", work={"cube@0": 12.0}),
                                        shared_owners[MAIN_STREAM], dp22,
                                        stream_id=MAIN_STREAM, **CORE_BINDING)
    assert shared_stage.caps["cube@0"] == pytest.approx(12.0)


def test_spine_single_pass_construction_and_structure_graph_id_parity(monkeypatch):
    case = _case_for("Ascend910B2", layers=2)
    req = StepRequest(phase="decode", tokens=1, context=128, sequences=1, scope="body")

    counts = {"bindings": 0, "cost_models": 0, "body_defaults": 0}
    orig_create = ExecutionBindings.create
    orig_cost_init = CostModel.__init__
    orig_effective = body_defaults_mod.effective_body_declaration

    def counted_create(cls, *args, **kwargs):
        counts["bindings"] += 1
        return orig_create(*args, **kwargs)

    def counted_cost(self, *args, **kwargs):
        counts["cost_models"] += 1
        return orig_cost_init(self, *args, **kwargs)

    def counted_body(*args, **kwargs):
        counts["body_defaults"] += 1
        return orig_effective(*args, **kwargs)

    monkeypatch.setattr(ExecutionBindings, "create", classmethod(counted_create))
    monkeypatch.setattr(CostModel, "__init__", counted_cost)
    monkeypatch.setattr(body_defaults_mod, "effective_body_declaration", counted_body)
    monkeypatch.setattr(op_analysis_mod, "effective_body_declaration", counted_body)

    step_eval = sdk_api.evaluate_step(case, req)
    assert counts == {"bindings": 1, "cost_models": 1, "body_defaults": 1}
    assert step_eval.span_ms > 0.0
    assert len(step_eval.operators) > 0
    assert step_eval.wave_breakdown is None

    struct_export = sdk_api.export_structure(case, StructureRequest(step=req))
    legacy_struct = structure_mod.export_structure(
        case, phase="decode", tokens=1, context=128, sequences=1, scope="body"
    )
    assert step_eval.graph_id == struct_export.graph_id == legacy_struct["graph_id"]

    # Isolated replay reuses the single EvaluationContext (1 CostModel, 1 ExecutionBindings).
    counts = {"bindings": 0, "cost_models": 0, "body_defaults": 0}
    iso_eval = sdk_api.evaluate_step(case, req, replay=sdk_api.ReplayOptions(isolated=True))
    assert counts == {"bindings": 1, "cost_models": 1, "body_defaults": 1}
    assert any(row.isolated_ms is not None and row.isolated_ms > 0.0 for row in iso_eval.operators)


def test_stream_pipeline_template_shared_quota_narrow_tile_and_pp_fifo():
    import pickle
    from mindie_sim.prediction.physical_recipe_base import _tile_count
    from mindie_sim.public.errors import UnmodeledCapabilityError

    build = build_accel()
    assert build["ok"] is True and build["backend"] == "c_native"

    case22 = _case_for("Ascend910B2", layers=1)
    dp22 = datapath_for_die(case22.machine.dies[0])

    # 1. PipelineTemplateProgram binds .template (.setup/.drain/.phases/.motif_phases) cleanly.
    disjoint_plan = StreamPlan(
        rank=0,
        lanes=(
            StreamLane(MAIN_STREAM, StreamCoreBudget(aic=16, aiv=32, isolation="disjoint")),
            StreamLane("tail", StreamCoreBudget(aic=8, aiv=16, isolation="disjoint")),
        ),
    )
    disjoint_owners = bind_stream_owners(disjoint_plan, dp22)
    tmpl = PipelineTemplate(
        owner="tile",
        setup=(Stage("setup", work={"cube@0": 1.0}, placement={"cube@0": (0,)}),),
        drain=(Stage("drain", work={"vector@0": 1.0}, placement={"vector@0": (0,)}),),
        phases=(
            PipelinePhase(
                kind="body",
                count=2,
                stages=(Stage("body", work={"cube@0": 2.0}, placement={"cube@0": (0, 1)}),),
            ),
        ),
        slots=2,
    )
    bound_tmpl_prog = apply_stream_owners_to_program(
        PipelineTemplateProgram(tmpl), disjoint_owners["tail"], dp22, stream_id="tail", **CORE_BINDING,
    )
    assert isinstance(bound_tmpl_prog, PipelineTemplateProgram)
    assert set(bound_tmpl_prog.template.setup[0].placement["cube@0"]).issubset(set(disjoint_owners["tail"].aic))
    assert set(bound_tmpl_prog.template.drain[0].placement["vector@0"]).issubset(set(disjoint_owners["tail"].aiv))
    assert set(bound_tmpl_prog.template.phases[0].stages[0].placement["cube@0"]).issubset(set(disjoint_owners["tail"].aic))
    assert len(bound_tmpl_prog.template.phases[0].stages[0].placement["cube@0"]) == 2

    # 2. shared_cap aggregate quota caps 2 parallel children on a 12-of-24 lane to >= 2.0 ms on C11.
    shared_owners = bind_stream_owners(SHARED_PLAN, dp22)
    q_res, q_for_main = _main_quotas(shared_owners, dp22)
    two_par = ParallelProgram((
        KernelProgram((Stage("child0", work={"cube@0": 12.0}),)),
        KernelProgram((Stage("child1", work={"cube@0": 12.0}),)),
    ))
    bound_par = apply_stream_owners_to_program(
        two_par, shared_owners[MAIN_STREAM], dp22,
        stream_id=MAIN_STREAM, quota_resources=q_for_main, **CORE_BINDING,
    )
    eng_q = Engine([Resource("cube@0", 24.0, "ms", "fair", units=24), *q_res], backend="c")
    ev_q0 = eng_q.submit("start")
    ev_q1 = bound_par.start(eng_q, "two_shared", ev_q0)
    eng_q.advance(until=(ev_q1,))
    assert ev_q1.time == pytest.approx(2.0)
    assert eng_q.accel_status()["native_steps"] > 0

    # 3. Narrow coupled tile width (1 AIC, 2 AIV) + private buffer remapping + coupled_has_unit_res.
    narrow_stage = Stage(
        "narrow_tile",
        work={"cube@0": 1.0, "vector@0": 2.0},
        placement={"cube@0": (0,), "vector@0": (0, 1)},
        coupled=True,
        acquire=(Allocation("k_l1", "l1@0", 1024, units=(0,)), Allocation("k_ub", "ub@0", 1024, units=(0, 1))),
        free=(Allocation("k_l1", "l1@0", 1024, units=(0,)), Allocation("k_ub", "ub@0", 1024, units=(0, 1))),
    )
    bound_narrow = bind_stage_to_stream(
        narrow_stage, disjoint_owners["tail"], dp22,
        stream_id="tail",
        pool_names={"l1": "l1@0", "ub": "ub@0"},
        pool_units={"l1": 24, "ub": 48},
        **CORE_BINDING,
    )
    assert len(bound_narrow.placement["cube@0"]) == 1
    assert len(bound_narrow.placement["vector@0"]) == 2
    assert set(bound_narrow.placement["cube@0"]).issubset(set(disjoint_owners["tail"].aic))
    assert set(bound_narrow.placement["vector@0"]).issubset(set(disjoint_owners["tail"].aiv))
    assert bound_narrow.acquire[0].units == bound_narrow.placement["cube@0"]
    assert bound_narrow.acquire[1].units == bound_narrow.placement["vector@0"]

    unit_coupled = Stage(
        "with_l2",
        work={"cube@0": 1.0, "vector@0": 2.0, "l2@0": 1.0},
        placement={"cube@0": (0,), "vector@0": (0, 1)},
        coupled=True,
    )
    bound_unit = bind_stage_to_stream(
        unit_coupled, disjoint_owners["tail"], dp22,
        resource_names=dict(CORE_NAMES, l2="l2@0"),
        resource_units=dict(CORE_UNITS, l2=1),
        rates={**CORE_RATES, "l2@0": 100.0},
        stream_id="tail",
    )
    assert "cube@0" not in bound_unit.placement and "vector@0" not in bound_unit.placement
    assert bound_unit.caps["cube@0"] == pytest.approx(1.0)
    assert bound_unit.caps["vector@0"] == pytest.approx(2.0)
    eng_u = Engine(
        [
            Resource("cube@0", 24.0, "ms", "fair", units=24),
            Resource("vector@0", 48.0, "ms", "fair", units=48),
            Resource("l2@0", 100.0, "ms", "fair", units=1),
        ],
        backend="c",
    )
    u0 = eng_u.submit("start")
    u1 = KernelProgram((bound_unit,)).start(eng_u, "u_c11", u0)
    eng_u.advance(until=(u1,))
    assert u1.time == pytest.approx(1.0)
    assert eng_u.accel_status()["native_steps"] > 0

    # 4. Unmatched StreamPlan.rank under tp=2, dp=1 must raise ModelingError.
    tp2_case = _case_for(tp=2)
    bad_rank_plan = StreamPlan(rank=1, lanes=(StreamLane(MAIN_STREAM),))
    spine = PredictionSpine()
    with pytest.raises(ModelingError, match="stream_plan_unmatched_rank"):
        spine.lower(
            tp2_case,
            LoweringRequest(step=StepRequest(phase="decode", tokens=1, context=64, sequences=1)),
            streams=(bad_rank_plan,),
        )

    # 5. Multi-stage PP FIFO keys by (stage, stream_id) without false cross-stage serialization.
    op_s0_a = replace(_rotary_op("s0_a"), stage=0)
    op_s1_b = replace(_rotary_op("s1_b", deps=("s0_a",)), stage=1)
    op_s0_c = replace(_rotary_op("s0_c"), stage=0)
    pp_plan = StreamPlan(
        rank=0,
        lanes=(StreamLane(MAIN_STREAM), StreamLane("aux")),
        assignments=(("s0_a", MAIN_STREAM), ("s1_b", "aux"), ("s0_c", MAIN_STREAM)),
    )
    # Cross-stage dep s0_a -> s1_b is NOT same-stage cross-stream; lowers cleanly without spurious event error.
    pp_lowered = lower_stream_edges((op_s0_a, op_s1_b, op_s0_c), pp_plan)
    assert len(pp_lowered) == 3

    # 6. Isolated replay honors StreamPlan core budgets and preserves picklability.
    op_r = _rotary_op("iso_rope")
    plan_full = StreamPlan(rank=0, lanes=(StreamLane(MAIN_STREAM, StreamCoreBudget(aic=24, aiv=48)),))
    plan_quarter = StreamPlan(
        rank=0,
        lanes=(
            StreamLane(MAIN_STREAM, StreamCoreBudget(aic=12, aiv=12)),
            StreamLane("rest", StreamCoreBudget(aic=12, aiv=36)),
        ),
    )
    request = LoweringRequest(declared_ops=(RankOps(0, (op_r,)),))
    step_full = spine.lower(case22, request, streams=(plan_full,))
    step_qtr = spine.lower(case22, request, streams=(plan_quarter,))
    ev_full = spine.run_step(step_full, ReplayOptions(isolated=True))
    ev_qtr = spine.run_step(step_qtr, ReplayOptions(isolated=True))
    assert ev_qtr.operators[0].isolated_ms is not None and ev_full.operators[0].isolated_ms is not None
    assert ev_qtr.operators[0].isolated_ms >= ev_full.operators[0].isolated_ms
    assert pickle.loads(pickle.dumps(ev_full)) == ev_full

    # 7. Matrix/AIV tile builders honor effective_aic_cores / effective_aiv_cores (控核).
    cube_op = _canonical_cube_op(case22, "cube_ctrl")
    ctx_aiv_cap = RecipeContext(
        op=op_r, work={}, resource_names={}, rates={},
        core_count=24, resource_units={"vector": 48},
        datapath=dp22, stream_owners=disjoint_owners["tail"],
    )
    assert _tile_count(ctx_aiv_cap) <= 16
    matmul = TiledMatmul()
    matrix_context = RecipeContext(
        op=cube_op, work={}, resource_names={"hbm": "hbm@0", "mte2": "mte2@0", "mte3": "mte3@0"},
        rates={}, core_count=24, datapath=dp22, stream_owners=None,
    )
    layout_full = _TiledGeometry(matmul, matrix_context)
    layout_tail = _TiledGeometry(matmul, replace(matrix_context, stream_owners=disjoint_owners["tail"]))
    assert layout_full.cores == 24 and layout_tail.cores == 8

    # 8. Explicit enable_dbo=True fails loudly with UnmodeledCapabilityError until Stage 3 in-op DBO.
    dbo_case = replace(case22, runtime=replace(case22.runtime, enable_dbo=True))
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        sdk_api.evaluate_step(dbo_case, StepRequest(phase="decode", tokens=1, context=64, sequences=1))


def test_stream_equal_budget_parallelism_nondividing_matrix_quota_and_validation():
    from mindie_sim.prediction.graph_analysis import graph_costs
    from mindie_sim.prediction.operator_replay import operator_costs

    build = build_accel()
    assert build["ok"] is True and build["backend"] == "c_native"

    case22 = _case_for("Ascend910B2", layers=1)
    dp22 = datapath_for_die(case22.machine.dies[0])

    # 1. Two equal-budget shared_cap lanes (24/48 AIV each) must NOT collide on main's quota.
    op_a = replace(
        _rotary_op("rope_a"),
        hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0, vector_ops=4e10, stream=MAIN_STREAM,
    )
    op_b = replace(op_a, op_id="rope_b", stream="aux")
    eq_plan = replace(SHARED_PLAN, assignments=(("rope_a", MAIN_STREAM), ("rope_b", "aux")))
    single_rep = graph_costs(case22, (op_a,), stream_plans=(eq_plan,), isolated=True)
    pair_rep = graph_costs(case22, (op_a, op_b), stream_plans=(eq_plan,), isolated=True)
    assert pair_rep["duration_ms"] == pytest.approx(single_rep["duration_ms"], rel=0.05)
    # No synthetic stream_quota@ keys may leak into execution/isolated hardware accounting.
    assert not any("stream_quota@" in k for k in pair_rep["execution"]["serviced"])
    assert not any("stream_quota@" in k for k in pair_rep["execution"]["resource_busy_ms"])
    iso_rep = operator_costs(case22, (op_a,), stream_plans=(eq_plan,))
    assert not any("stream_quota@" in k for k in iso_rep["operators"][0]["serviced"])
    assert not any("stream_quota@" in k for k in iso_rep["operators"][0]["service_delta"])

    # 2. Non-dividing AIC lane budget (5 of 24 AICs) lowers TiledMatmul without crashing.
    plan_5 = StreamPlan(
        rank=0,
        lanes=(
            StreamLane(MAIN_STREAM, StreamCoreBudget(aic=5, aiv=10, isolation="disjoint")),
            StreamLane("rest", StreamCoreBudget(aic=19, aiv=38, isolation="disjoint")),
        ),
    )
    owners_5 = bind_stream_owners(plan_5, dp22)
    cube_op = _canonical_cube_op(case22, "cube_5aic")
    ctx_5 = RecipeContext(
        op=cube_op,
        work={"cube@0": 1.0},
        resource_names={"cube": "cube@0", "hbm": "hbm@0", "mte2": "mte2@0", "mte3": "mte3@0", "vector": "vector@0"},
        rates={"cube@0": 24.0, "hbm@0": 100.0, "mte2@0": 100.0, "mte3@0": 100.0, "vector@0": 48.0},
        core_count=24,
        resource_units={"cube": 24, "vector": 48, "mte2": 24, "mte3": 48},
        datapath=dp22,
        stream_owners=owners_5[MAIN_STREAM],
    )
    lowered_5 = TiledMatmul().lower(ctx_5)
    assert lowered_5 is not None

    # 3. Pure shared_cap narrow 1-core tile preserves 1-core rate (12 ms, not 1 ms) on C11.
    sh_owners = bind_stream_owners(SHARED_PLAN, dp22)
    q_res, q_main = _main_quotas(sh_owners, dp22)
    narrow_1core = Stage("tile_1c", work={"cube@0": 12.0}, placement={"cube@0": (0,)})
    bound_1core = bind_stage_to_stream(
        narrow_1core, sh_owners[MAIN_STREAM], dp22,
        stream_id=MAIN_STREAM, quota_resources=q_main, **CORE_BINDING,
    )
    eng_n = Engine([Resource("cube@0", 24.0, "ms", "fair", units=24), *q_res], backend="c")
    n0 = eng_n.submit("start")
    n1 = KernelProgram((bound_1core,)).start(eng_n, "narrow_sh", n0)
    eng_n.advance(until=(n1,))
    assert n1.time == pytest.approx(12.0)
    assert eng_n.accel_status()["native_steps"] > 0

    # 4. Split-stage ordering runs stage.work BEFORE stage.program, and mixed shared/disjoint remaps buffers & non-quota work.
    mixed_plan = StreamPlan(
        rank=0,
        lanes=(
            StreamLane("disj", StreamCoreBudget(aic=8, aiv=24, isolation="disjoint")),
            StreamLane(MAIN_STREAM, StreamCoreBudget(aic=8, aiv=24, isolation="shared_cap")),
        ),
    )
    mix_owners = bind_stream_owners(mixed_plan, dp22)
    mq_res, mq_main = _main_quotas(mix_owners, dp22)
    nested_stage = Stage(
        "split_seq",
        work={"cube@0": 8.0, "vector@0": 24.0},
        acquire=(Allocation("k_l1", "l1@0", 512, units=(0,)),),
        free=(Allocation("k_l1", "l1@0", 512, units=(0,)),),
        program=KernelProgram((Stage("inner", latency=3.0),)),
    )
    bound_mix = bind_stage_to_stream(
        nested_stage, mix_owners[MAIN_STREAM], dp22,
        stream_id=MAIN_STREAM, quota_resources=mq_main,
        pool_names={"l1": "l1@0"}, pool_units={"l1": 24},
        **CORE_BINDING,
    )
    assert set(bound_mix.acquire[0].units).issubset(set(mix_owners[MAIN_STREAM].pool_aic))
    assert isinstance(bound_mix.program, KernelProgram)
    assert len(bound_mix.program.stages) == 2
    # Work takes 1.0 ms (8/8 cube, 24/24 vector), then nested latency 3.0 ms -> total 4.0 ms (sequential!).
    eng_m = Engine(
        [Resource("cube@0", 24.0, "ms", "fair", units=24), Resource("vector@0", 48.0, "ms", "fair", units=48), *mq_res],
        pools={"l1@0": 4096}, pool_units={"l1@0": 24},
    )
    m0 = eng_m.submit("start")
    m1 = KernelProgram((bound_mix,)).start(eng_m, "mix_seq", m0)
    eng_m.advance(until=(m1,))
    assert m1.time == pytest.approx(4.0)

    # 5. Standalone operator_costs & partial-replica graph_costs reject unmatched StreamPlan.rank.
    bad_plan = StreamPlan(rank=1, lanes=(StreamLane(MAIN_STREAM),))
    with pytest.raises(ModelingError, match="stream_plan_unmatched_rank:1"):
        operator_costs(case22, (op_a,), replica=0, stream_plans=(bad_plan,))
    dp2_case = _case_for(dp=2)
    with pytest.raises(ModelingError, match="stream_plan_unmatched_rank:1"):
        graph_costs(dp2_case, {0: (op_a,)}, stream_plans=(bad_plan,))
