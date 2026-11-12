"""Idle graph rows are physical capture rows, not logical requests."""
from __future__ import annotations

from tests.test_joint_collective_observation import cohort_case
from mindie_sim.prediction.operator_analysis import target_graphs


def test_idle_capture_rows_launch_same_physical_kernels_and_keep_logical_zero():
    raw = cohort_case(2)
    raw["runtime"]["graph_mode"] = "full"
    raw["runtime"]["graph_capture_mode"] = "full"
    raw["runtime"]["graph_capture_keys"] = [1, 2, 4]
    graphs = target_graphs(
        raw, phase="decode", tokens=1, context=4, sequences=[1, 0],
        scope="body")
    active = next(op for op in graphs[0] if op.family == "attention")
    idle = next(op for op in graphs[1] if op.family == "attention")
    assert active.launch_shape[0] == 1
    assert idle.launch_shape[0] == 1
    assert active.requests == ("analysis0:0",)
    assert idle.requests == ("analysis1:padding",)
    assert dict(idle.metadata)["graph_active_query_rows"] == 0
    assert dict(idle.metadata)["graph_physical_query_rows"] == 1
    assert dict(active.metadata)["graph_active_query_rows"] == 1
    assert len(graphs[1]) == len(graphs[0])


def test_explicit_padded_idle_never_exposes_physical_rows_as_active():
    raw = cohort_case(2)
    raw["runtime"]["graph_mode"] = "full"
    raw["runtime"]["graph_capture_mode"] = "full"
    raw["runtime"]["graph_capture_keys"] = [1, 2, 4]
    graphs = target_graphs(
        raw, phase="decode", tokens=1, context=4, sequences=[1, 0],
        scope="body", padded_idle=True)
    idle = next(op for op in graphs[1] if op.family == "attention")
    active = next(op for op in graphs[0] if op.family == "attention")
    assert dict(idle.metadata)["graph_active_query_rows"] == 0
    assert dict(idle.metadata)["graph_physical_query_rows"] == 1
    assert dict(active.metadata)["graph_active_query_rows"] == 1
    assert idle.launch_shape[0] == 1


def test_physical_global_rank_scope_preserves_tp_owner_identity():
    from types import SimpleNamespace
    from mindie_sim.prediction.body import physical_global_ranks
    from mindie_sim.schema.parallel import ParallelSpec
    scope = physical_global_ranks(
        SimpleNamespace(parallel=ParallelSpec(dp=4, tp=4)), (1, 3))
    assert scope[3] == (12, 13, 14, 15)
    assert scope[1] == (4, 5, 6, 7)
    # R31 actual source roles: DP3/TP0=12 active, DP1/TP2=6 idle.
    assert scope[3][0] == 12
    assert scope[1][2] == 6


def test_body_geometry_exposes_physical_global_rank_scope_without_relabeling():
    from mindie_sim.prediction.body import body_forward
    raw = cohort_case(2)
    raw["runtime"]["graph_mode"] = "full"
    raw["runtime"]["graph_capture_mode"] = "full"
    raw["runtime"]["graph_capture_keys"] = [1, 2, 4]
    out = body_forward(raw, phase="decode", tokens=1, context=4,
                       sequences=[1, 0], event_limit=0, compact=True)
    assert out["geometry"]["valid_query_tokens_by_replica"] == {0: 1, 1: 0}
    assert out["geometry"]["graph_query_tokens_by_replica"] == {0: 1, 1: 1}
    assert out["geometry"]["physical_global_rank_basis"] == (
        "dp_replica * tp + tp_rank")
    assert out["geometry"]["physical_global_ranks_by_replica"][1] == (1,)
    assert out["profile"]["by_rank"][1]["operator_events"] > 0
