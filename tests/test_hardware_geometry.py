"""SKU geometry must reach scheduling, including per-device overrides."""
import pytest

from mindie_sim.backends.ascend import profiles
from mindie_sim.catalog.ops import vector_op
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.pointwise import TiledPointwise
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.hardware import Hardware
from test_mechanism_contract import hardware_case


@pytest.mark.parametrize("sku", profiles())
def test_profile_vector_units_determine_finite_pointwise_waves(sku):
    raw = hardware_case()
    raw["machine"] = {"backend": "ascend", "sku": sku, "num_devices": 1}
    declaration = profiles()[sku]["source"]["declaration"]
    vector_units = declaration["aiv_count_per_die"]
    machine = Hardware.from_mapping(raw["machine"])
    device = machine.dies[0]
    assert device.compute_units == declaration["aic_count_per_die"]
    # One row per unit, followed by a tail row: the source loop executes two
    # waves, including masked arithmetic on the tail wave's inactive lanes.
    rows = vector_units + 1
    op = vector_op(op_id="tail", deps=(), stage=0, layer=0, requests=(),
                   elements=rows, ops_per_element=1, dtype="bf16",
                   family="pointwise", phase="prefill", recipe="unit_rows",
                   shape=(rows, 1), read_bytes_override=0, write_bytes_override=0)
    recipes = RecipeBook(recipes={
        "unit_rows": TiledPointwise(0, (), tile_rows=1, tile_cols=1),
    })
    result = operator_costs(raw, (op,), recipes=recipes)["operators"][0]
    expected_work = 2 * vector_units
    assert result["serviced"]["vector@r0:s0"] == expected_work
    minimum = dict(device.kernel_invocations).get("vector")
    assert result["duration_ms"] == pytest.approx(max(
        1000 * expected_work / dict(device.rates)["vector"],
        minimum.minimum_ms if minimum else 0))
    assert Hardware.from_mapping(machine.to_dict()) == machine


@pytest.mark.parametrize("override,expected", [
    ({}, 64),
    ({"aiv_count": 11}, 11),
    ({"resource_units": {"vector": 7}}, 7),
    ({"aiv_count": 11, "resource_units": {"vector": 7, "mte2": 3}}, 7),
])
def test_native_and_neutral_unit_overrides_preserve_role_ownership(override, expected):
    machine = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950DT_9581", "device": override,
    })
    device = machine.dies[0]
    assert device.units("vector") == expected
    assert device.compute_units == 32
    if "resource_units" in override and "mte2" in override["resource_units"]:
        assert device.units("mte2") == 3


def test_device_records_override_inherited_units_without_losing_other_roles():
    machine = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950DT_9581", "num_devices": 3,
        "device": {"aiv_count": 11, "resource_units": {"vector": 7, "mte2": 3}},
        "devices": [
            {"aiv_count": 5},
            {"aiv_count": 5, "resource_units": {"vector": 2}},
            {"resource_units": {"mte3": 4}},
        ],
    })
    assert [device.units("vector") for device in machine.dies] == [5, 2, 7]
    assert [device.units("mte2") for device in machine.dies] == [3, 3, 3]
    assert machine.dies[2].units("mte3") == 4
    assert Hardware.from_mapping(machine.to_dict()) == machine
