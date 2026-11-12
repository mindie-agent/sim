"""Ordinary graph/body execution consumes C7 context without using its clock."""
import pytest
from mindie_sim import graph_costs
from mindie_sim.catalog.ops import comm_op
from test_graph_analysis import op
from test_joint_collective_execution import cohort_case
from test_prediction_system import case


def _report():
    calls = []
    for rank in (0, 1):
        calls.append({
            "rank_id": f"rank_{rank}", "rank_index": rank,
            "operation": {"dtype": "BFP16", "typed_bytes": 10240,
                          "collective": "all_gather"},
            "entry": {"start_ns": 1_000_000 + rank},
            "completion": {"end_ns": 2_000_000 + rank},
            "observed_context": {"entry_domain": "rank_local",
                                 "cross_rank_aligned": False},
            "clock": {"domain": "rank_local", "cross_rank_aligned": False},
            "dependencies": (), "waits": (),
            "unknowns": ("cross_rank_clock_not_established",),
        })
    return {"schema": "mindie-profiler/collective-readiness/1", "calls": calls}


def _raw():
    raw = case(dp=1, tp=2)
    raw["runtime"]["metadata"] = {"collective_readiness": _report()}
    return raw


def _gather():
    return comm_op(
        op_id="ag", deps=("producer",), stage=0, layer=0, requests=(),
        family="allgather", phase="prefill", recipe="ring_allgather",
        link_bytes=10240, shape=(2, 5120),
        metadata=(("collective", "allgather"), ("group_axis", "tp")))


def test_actual_graph_predecessor_controls_collective_not_c7_time():
    baseline = graph_costs(
        _raw(), {0: (op("producer", flops=1e6), _gather())},
        external_inputs={"input": 0.0}, include_submission=False)
    delayed = graph_costs(
        _raw(), {0: (op("producer", flops=1e6, deps=("input",)), _gather())},
        external_inputs={"input": 5.0}, include_submission=False)
    b = {row["id"]: row for row in baseline["timeline"]}
    d = {row["id"]: row for row in delayed["timeline"]}
    assert d["r0:ag"]["ready_ms"] == b["r0:ag"]["ready_ms"] + 5.0
    assert d["r0:ag"]["complete_ms"] == b["r0:ag"]["complete_ms"] + 5.0
    assert baseline["execution"]["serviced"].get("link") == delayed["execution"]["serviced"].get("link")
    observations = d["r0:ag"].get("collective_readiness")
    assert observations is None
    execution = delayed["execution"]
    assert len(execution["collective_readiness_observations"]) == 1
    item = execution["collective_readiness_observations"][0]
    assert item["op_id"] == "ag"
    assert item["context"][0]["readiness_basis"] == "pipeline_engine_event"
    assert item["context"][0]["entry_ns"] == 1_000_000


def test_cross_dp_distinct_participant_ready_is_retained():
    raw = cohort_case(2)
    raw["runtime"]["kernels"] = {"families": {"allgather": {"kind": "ring_collective"}}}
    raw["runtime"]["metadata"] = {"collective_readiness": _report()}

    def gather():
        return comm_op(
            op_id="ag", deps=("producer",), stage=0, layer=0, requests=(),
            family="allgather", phase="prefill", recipe="ring_allgather",
            link_bytes=10240, shape=(2, 5120),
            metadata=(("collective", "allgather"), ("group_axis", "ep")))

    report = graph_costs(
        raw,
        {0: (op("producer", flops=1e6), gather()),
         1: (op("producer", flops=1e6, deps=("input",)), gather())},
        external_inputs={"input": 5.0}, include_submission=False)
    rows = {row["id"]: row for row in report["timeline"]}
    assert rows["r0:ag"]["ready_ms"] == 1.0
    assert rows["r1:ag"]["ready_ms"] == 6.0
    assert len(report["execution"]["collective_readiness_observations"]) == 1
    assert report["execution"]["collective_readiness_unresolved"] == []
    context = report["execution"]["collective_readiness_observations"][0]["context"]
    assert sorted(context) == [0, 1]


def test_c7_native_dtype_and_logical_input_bytes_are_checked():
    from mindie_sim.prediction.collective_readiness import bind_pipeline_ready
    from mindie_sim.prediction.engine import Engine, Resource
    engine = Engine([Resource("wire", 1_000_000_000, "bytes")], trace_limit=8)
    ready = {0: engine.submit("r0"), 1: engine.submit("r1")}
    bind_pipeline_ready(_report(), ready, expected_collective="all_gather",
                        expected_bytes=10240, expected_dtype="bf16",
                        expected_group_size=2)
    bad = _report()
    bad["calls"][0]["operation"]["dtype"] = "F32"
    with pytest.raises(Exception, match="collective_readiness_dtype_mismatch"):
        bind_pipeline_ready(bad, ready, expected_collective="all_gather",
                            expected_bytes=10240, expected_dtype="bf16")
    bad = _report()
    bad["calls"][0]["operation"]["typed_bytes"] = 20480
    with pytest.raises(Exception, match="collective_readiness_bytes_mismatch"):
        bind_pipeline_ready(bad, ready, expected_collective="all_gather",
                            expected_bytes=10240, expected_dtype="bf16")


def test_wire_to_logical_input_factors_are_primitive_and_group_specific():
    from mindie_sim.prediction.execution import _collective_native_basis
    ag = comm_op(
        op_id="ag4", deps=(), stage=0, layer=0, requests=(),
        family="allgather", phase="prefill", recipe="ring_allgather",
        link_bytes=30720, shape=(2, 5120),
        metadata=(("collective", "allgather"), ("group_axis", "tp"),
                  ("payload_dtype", "bf16")))
    rs = comm_op(
        op_id="rs2", deps=(), stage=0, layer=0, requests=(),
        family="reduce_scatter", phase="prefill", recipe="ring_reduce_scatter",
        link_bytes=5120, shape=(2, 5120),
        metadata=(("collective", "reduce_scatter"), ("group_axis", "tp"),
                  ("payload_dtype", "bf16")))
    assert _collective_native_basis(ag, 4)[1] == 10240
    assert _collective_native_basis(rs, 2)[1] == 10240
    assert _collective_native_basis(ag, 4)[2] == "bf16"
