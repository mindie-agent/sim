"""A5 native group2 standalone TP service curve is typed and bounded."""
from mindie_sim.observations.a5_comm_golden import (
    A5_NATIVE_GROUP2, a5_native_group2_curve_ms,
)


def test_native_group2_curve_interpolates_only_measured_support():
    assert A5_NATIVE_GROUP2["stack"] == "rack08"
    assert A5_NATIVE_GROUP2["cann"] == "9.2.0"
    assert A5_NATIVE_GROUP2["mode"] == "aicpu"
    assert abs(a5_native_group2_curve_ms(
        "all_gather", 32768, "communication") - 0.069943) < 1e-12
    assert abs(a5_native_group2_curve_ms("all_gather", 32768) - 0.041113) < 1e-12
    mid = a5_native_group2_curve_ms("all_gather", 10_485_760)
    assert abs(mid - 0.059318) < 1e-12
    assert a5_native_group2_curve_ms("all_gather", 10240) is None
    assert a5_native_group2_curve_ms("all_gather", 200_000_000) is None
    assert a5_native_group2_curve_ms("all_reduce", 32768) is None
    assert a5_native_group2_curve_ms("all_to_all", 32768) is None

from types import SimpleNamespace

from mindie_sim.prediction.subgraphs.instantiate import _native_group2_declared

R35 = "sha256:5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06"


def _ctx(*, tp=2, dtype="bf16", image="repo:cann9.2.0-r35", image_id=R35,
         declaration=None):
    if declaration is None:
        declaration = {
            "schema": "mindie-sim/a5-native-group2/1",
            "group_size": 2, "dtype": "bf16", "execution_core": "aicpu",
            "stack": "rack08", "cann": "9.2.0", "image_id": R35,
            "primitives": ["all_gather", "reduce_scatter"],
        }
    meta = {"image": image, "image_id": image_id,
            "a5_native_group2": declaration}
    return SimpleNamespace(tp=tp, features=SimpleNamespace(
        activation_dtype=dtype, metadata=tuple(meta.items())))


def test_only_structured_aicpu_bf16_ag_rs_opt_in_admits_curve():
    assert _native_group2_declared(_ctx())
    assert not _native_group2_declared(_ctx(
        declaration={"HCCL_mode": "conditional_declared_template_and_payload",
                     "template_correction": {
                         "conditional_selection": "explicit AICPU_TS Mesh OneShot",
                         "reason": "AICPU mesh is only valid for AG/RS"}}))
    assert not _native_group2_declared(_ctx(dtype="fp8"))
    assert not _native_group2_declared(_ctx(tp=4))
    assert not _native_group2_declared(_ctx(
        declaration={"schema": "mindie-sim/a5-native-group2/1", "group_size": 2,
                     "dtype": "bf16", "execution_core": "aiv", "stack": "rack08",
                     "cann": "9.2.0", "image_id": R35,
                     "primitives": ["all_gather", "reduce_scatter"]}))
    assert not _native_group2_declared(_ctx(image="repo:cann9.1.0-r35"))
    assert not _native_group2_declared(_ctx(image_id="sha256:" + "0" * 64))
    assert not _native_group2_declared(_ctx(
        declaration={"schema": "mindie-sim/a5-native-group2/1", "group_size": 2,
                     "dtype": "bf16", "execution_core": "aicpu", "stack": "rack08",
                     "cann": "9.2.0", "image_id": R35,
                     "primitives": ["all_gather"]}))
