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

import mindie_sim.api as sdk_api
import mindie_sim.prediction.body_defaults as body_defaults_mod
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.evaluation import ExecutionBindings
from mindie_sim.prediction.frontend import Frontend
import mindie_sim.prediction.operator_analysis as op_analysis_mod
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.results import StepRequest, StructureRequest
from mindie_sim.prediction.runtime.accel_build import build_accel
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.streams import (
    MAIN_STREAM,
    StreamCoreBudget,
    StreamEvent,
    StreamLane,
    StreamPlan,
    bind_stage_to_stream,
    bind_stream_owners,
    lower_stream_edges,
    proportional_coupled_placement,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import LoweringRequest, RankOps
import mindie_sim.structure as structure_mod

REPO_ROOT = Path(__file__).resolve().parents[2]


def _case_for(sku: str = "Ascend910B2", *, layers: int = 1):
    return sdk_api.compose_case(
        "qwen3-32b", hardware=sku, options={"dp": 1, "tp": 1, "pp": 1, "layers": layers}
    )


def _rotary_op(op_id: str, *, deps: tuple[str, ...] = ()) -> OpSpec:
    shape = (8, 16, 64)
    return OpSpec(
        op_id=op_id,
        layer=0,
        stage=0,
        family="rotary",
        recipe="rotary_position_embedding",
        phase="decode",
        requests=("r0",),
        dtype="bf16",
        local_shape=shape,
        semantic_shape=shape,
        padded_shape=shape,
        launch_shape=shape,
        deps=deps,
        stream=MAIN_STREAM,
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
    (
        ("Ascend910B2", 24, 48, 16, 32, 16),
        ("Ascend910_9382", 20, 40, 12, 24, 16),
        ("Ascend950DT_9581", 32, 64, 24, 48, 16),
    ),
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
    plan_cycle = replace(
        plan_3,
        events=(
            *plan_3.events,
            StreamEvent("ev_back", "comm", "op_comm", MAIN_STREAM, ("op_main",)),
        ),
    )
    with pytest.raises(ModelingError, match="stream_event_cycle"):
        lower_stream_edges((op_main, op_csa, op_comm), plan_cycle)

    # Live 3-stream execution through PredictionSpine preserves all 3 stream IDs.
    spine = PredictionSpine()
    step = spine.lower(
        case,
        LoweringRequest(declared_ops=(RankOps(0, (op_main, op_csa, op_comm)),)),
        streams=(plan_3,),
    )
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
    step_bad = spine.lower(
        case,
        LoweringRequest(declared_ops=(RankOps(0, (op_main, op_csa, op_comm)),)),
        streams=(bad_assign,),
    )
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
    bound_stage = bind_stage_to_stream(
        coupled_stage,
        owners[MAIN_STREAM],
        dp22,
        resource_names={"cube": "cube@0", "vector": "vector@0"},
        resource_units={"cube": 24, "vector": 48},
        rates={"cube@0": 24.0, "vector@0": 48.0},
        stream_id=MAIN_STREAM,
    )
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
    shared_stage = bind_stage_to_stream(
        Stage("shared_cube", work={"cube@0": 12.0}),
        shared_owners[MAIN_STREAM],
        dp22,
        resource_names={"cube": "cube@0", "vector": "vector@0"},
        resource_units={"cube": 24, "vector": 48},
        rates={"cube@0": 24.0, "vector@0": 48.0},
        stream_id=MAIN_STREAM,
    )
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
    assert step_eval.wave_breakdown is not None
    assert step_eval.wave_breakdown.cube_ms > 0.0

    struct_export = sdk_api.export_structure(case, StructureRequest(step=req))
    legacy_struct = structure_mod.export_structure(
        case, phase="decode", tokens=1, context=128, sequences=1, scope="body"
    )
    assert step_eval.graph_id == struct_export.graph_id == legacy_struct["graph_id"]
