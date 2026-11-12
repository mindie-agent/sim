"""Real A5 native-wave active service becomes reusable ServiceCurve data."""
import math

import pytest

from mindie_sim.observations.a5_comm_calibration import collect_native_service


def test_native_collector_fits_active_aicpu_not_communication_envelope():
    bundle = collect_native_service("/Volumes/data/mindie-sim-data/calibration-20261107/a5/hccl-native/raw/parsed/native-summary.json", stack="rack08", cann="9.2.0", execution_core="aicpu", placement="intra_node_ub", deployment_image_id="sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")
    curve = bundle.lookup_curve(
        "all_gather", 2, placement_scope="intra_node_ub", dtype="bf16",
        expansion_mode="aicpu", stack="rack08", cann="9.2.0")
    assert curve.identity["expansion_regime"] == "aicpu"
    assert curve.provenance["stack"] == "rack08"
    assert curve.provenance["cann"] == "9.2.0"
    # Active AI_CPU child, not the wider COMMUNICATION envelope.
    assert math.isclose(curve.knots[0][1], 0.041113, rel_tol=1e-9)
    assert curve.knots[0][1] < 0.069943
    assert curve.estimate(786432) is not None
    assert curve.estimate(32768) is not None
    assert curve.estimate(1) is None
    assert curve.estimate(200_000_000) is None
    assert bundle.holdout


def test_native_collector_keeps_groups_separate():
    bundle = collect_native_service("/Volumes/data/mindie-sim-data/calibration-20261107/a5/hccl-native/raw/parsed/native-summary.json", stack="rack08", cann="9.2.0", execution_core="aicpu", placement="intra_node_ub", deployment_image_id="sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")
    g2 = bundle.lookup_curve("all_gather", 2, placement_scope="intra_node_ub",
                             dtype="bf16", expansion_mode="aicpu",
                             stack="rack08", cann="9.2.0")
    g4 = bundle.lookup_curve("all_gather", 4, placement_scope="intra_node_ub",
                             dtype="bf16", expansion_mode="aicpu",
                             stack="rack08", cann="9.2.0")
    assert g2.knots[0][1] != g4.knots[0][1]
    assert g2.identity["world_size"] == 2
    assert g4.identity["world_size"] == 4


SOURCE = "/Volumes/data/mindie-sim-data/calibration-20261107/a5/hccl-native/raw/parsed/native-summary.json"
IMAGE = "sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06"


def test_native_collector_rejects_rewritten_rack_or_image():
    for change in ({"stack": "rack06"},
                   {"deployment_image_id": "sha256:" + "0" * 64}):
        args = dict(stack="rack08", cann="9.2.0", execution_core="aicpu",
                    placement="intra_node_ub", deployment_image_id=IMAGE)
        args.update(change)
        with pytest.raises(ValueError):
            collect_native_service(SOURCE, **args)


def test_native_collector_cache_reuses_source_revision():
    from mindie_sim.observations import a5_comm_calibration as calibration
    calibration.collect_native_service(SOURCE, stack="rack08", cann="9.2.0",
                                       execution_core="aicpu", placement="intra_node_ub",
                                       deployment_image_id=IMAGE)
    before = calibration._collect_native_service_cached.cache_info()
    for _ in range(4):
        calibration.collect_native_service(SOURCE, stack="rack08", cann="9.2.0",
                                           execution_core="aicpu", placement="intra_node_ub",
                                           deployment_image_id=IMAGE)
    after = calibration._collect_native_service_cached.cache_info()
    assert after.hits >= before.hits + 4
