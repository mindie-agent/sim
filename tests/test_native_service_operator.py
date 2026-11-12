"""Native wave curve binds through the existing sim ServiceCurve/OpSpec path."""
import json
from dataclasses import replace
from pathlib import Path

from mindie_sim.catalog.ops import comm_op
from mindie_sim.observations.a5_comm_calibration import collect_native_service
from mindie_sim.prediction.structure import parse_case

QWEN = Path("/Volumes/data/code/mindie-release-20261014/profiler/examples/"
            "structure/cases/a5-qwen38-tp2-b4-caller-fused.json")


def _case_op():
    case = parse_case(json.loads(QWEN.read_text()))
    local_rows = (1024 + case.runtime.parallel.tp - 1) // case.runtime.parallel.tp
    logical = local_rows * int(case.model.hidden_size) * 2
    op = replace(comm_op(
        op_id="ag", deps=(), stage=0, layer=0, requests=(),
        family="allgather", phase="prefill", recipe="ring_allgather",
        link_bytes=logical, shape=(1024, int(case.model.hidden_size)),
        metadata=(("collective", "allgather"), ("group_axis", "tp"),
                  ("payload_dtype", "bf16"))), dtype="bf16")
    return case, op


def test_native_curve_drives_qwen_operator_and_rejects_off_support():
    case, op = _case_op()
    bundle = collect_native_service("/Volumes/data/mindie-sim-data/calibration-20261107/a5/hccl-native/raw/parsed/native-summary.json", stack="rack08", cann="9.2.0", execution_core="aicpu", placement="intra_node_ub", deployment_image_id="sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")
    curve = bundle.bind_service_curve(
        case, op, expansion_mode="aicpu", stack="rack08", cann="9.2.0",
        placement_scope="intra_node_ub")
    a5 = bundle.lookup_curve(
        "all_gather", 2, placement_scope="intra_node_ub", dtype="bf16",
        expansion_mode="aicpu", stack="rack08", cann="9.2.0")
    assert a5.identity["world_size"] == 2
    assert a5.identity["expansion_regime"] == "aicpu"
    assert curve.provenance["bound_a5_curve"] == a5.model_id
    estimate = curve.estimate(op.link_bytes)
    assert estimate is not None and estimate.duration_ms > 0
    assert estimate.basis in {"anchor", "interpolated"}
    assert curve.estimate(1) is None
    assert curve.estimate(200_000_000) is None


def test_group4_curve_is_not_used_for_qwen_group2():
    case, op = _case_op()
    bundle = collect_native_service("/Volumes/data/mindie-sim-data/calibration-20261107/a5/hccl-native/raw/parsed/native-summary.json", stack="rack08", cann="9.2.0", execution_core="aicpu", placement="intra_node_ub", deployment_image_id="sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")
    g2 = bundle.lookup_curve(
        "all_gather", 2, placement_scope="intra_node_ub", dtype="bf16",
        expansion_mode="aicpu", stack="rack08", cann="9.2.0")
    g4 = bundle.lookup_curve(
        "all_gather", 4, placement_scope="intra_node_ub", dtype="bf16",
        expansion_mode="aicpu", stack="rack08", cann="9.2.0")
    assert g2.identity["world_size"] == 2
    assert g4.identity["world_size"] == 4
    assert g2.knots != g4.knots
