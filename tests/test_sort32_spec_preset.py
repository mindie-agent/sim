"""Theoretical API capacity executes on the existing owner-local vector pipe."""
from copy import deepcopy

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.instructions import (
    CONTROL_INSTRUCTIONS, InstructionCount, InstructionInventory, PrimitiveInventory,
)
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.schema.hardware import Hardware


def machine(sku="Ascend950DT_9581", clock=None, owners=4):
    arch = {"presets": ["ascend_sort32_spec"]}
    if clock is not None:
        arch["clock_hz"] = clock
    return {"backend": "ascend", "sku": sku, "num_devices": 1,
            "device": {"compute_arch": arch, "resource_units": {"vector": owners}}}


def execute(device, owners=(0,), vector_work=0):
    rate = dict(device.rates)["vector"] / 1000
    units = device.units("vector")
    engine = Engine([Resource("vector", rate, units=units)])
    context = RecipeContext(None, {}, {"vector": "vector"}, {"vector": rate},
        device.compute_units, resource_units={"vector": units},
        service_modes=dict(device.service_modes))
    counts = InstructionInventory({"VBitSort32": InstructionCount(1600, 51200)})
    work = counts.capacity_work(CONTROL_INSTRUCTIONS)
    priced, unpriced = PrimitiveInventory(work.resources).partition(
        Geometry.from_devices((device,)).resource_units)
    assert not unpriced
    demand = context.bind(priced)
    stages = [Stage(str(index), demand,
                    placement=context.owned(owner, units, ("vbitsort32",)))
              for index, owner in enumerate(owners)]
    if vector_work:
        stages.append(Stage("other_vector", context.bind(Work(resources={"vector": vector_work})),
                            placement=context.owned(0, units, ("vector",))))
    ready = engine.submit("external_input", latency=7)
    done = KernelProgram(tuple(stages)).start(engine, "sort", ready)
    engine.advance(until=(done,))
    assert engine.serviced["vector"] == pytest.approx(
        len(owners) * demand["vector"] + vector_work)
    assert work.resources == {"vbitsort32": 1600}
    return done.time - ready.time


@pytest.mark.parametrize("sku,clock", [
    ("Ascend910B4", 1.5e9), ("Ascend910_9391", 1.85e9),
    ("Ascend950DT_9581", 1.65e9),
])
def test_sku_declaration_drives_opt_in_capacity_and_shares_vector_owners(sku, clock):
    baseline = Hardware.from_mapping({"backend": "ascend", "sku": sku}).dies[0]
    assert "vbitsort32" not in Geometry.from_devices((baseline,)).resource_units
    device = Hardware.from_mapping(machine(sku)).dies[0]
    expected = 1600 * 16 / clock * 1000
    assert execute(device) == pytest.approx(expected)
    assert execute(device, (0, 1)) == pytest.approx(expected)
    assert execute(device, (0, 0)) == pytest.approx(2 * expected)
    # A vector task with equal service demand must compete on the same AIV;
    # the Sort32 capacity is not a parallel resource with a copied owner count.
    equal_vector = dict(device.rates)["vector"] / device.units("vector") * expected / 1000
    assert execute(device, vector_work=equal_vector) == pytest.approx(2 * expected)
    assert "vbitsort32" not in dict(device.rates)


def test_clock_and_owner_scans_survive_roundtrip_without_idle_core_acceleration():
    for owners in (2, 8):
        for scale in (.5, 1, 2):
            hardware = Hardware.from_mapping(machine(clock=1.6e9 * scale, owners=owners))
            assert execute(hardware.dies[0]) == pytest.approx(.016 / scale)
            assert Hardware.from_mapping(hardware.to_dict()) == hardware
            raw = hardware.to_dict()
            service = raw["devices"][0]["compute_arch"]["instruction_services"]["vbitsort32"]
            service["clock_hz"] *= 2
            assert execute(Hardware.from_mapping(raw).dies[0]) == pytest.approx(.008 / scale)
    # A throughput assumption for generic vector operations is not a clock
    # measurement. Sort capacity stays fixed while conversion to pipe work changes.
    for vector_rate in (1e12, 2e12, 4e12):
        raw = machine(clock=1.6e9)
        raw["device"]["rates"] = {"vector": vector_rate}
        assert execute(Hardware.from_mapping(raw).dies[0]) == pytest.approx(.016)


def test_explicit_service_and_mode_supplies_override_preset_without_detaching_pipe():
    raw = machine(clock=1.6e9)
    raw["device"]["compute_arch"]["instruction_services"] = {
        "vbitsort32": {"per_owner_rate": 2e8}}
    declared = Hardware.from_mapping(raw)
    assert execute(declared.dies[0]) == pytest.approx(.008)
    # A direct mode override remains a rate declaration even if the clock
    # bound is changed after normalized hardware is exported.
    raw = machine(clock=1.6e9)
    raw["device"]["service_modes"] = {"vbitsort32": {"rate": 4 * 4e8}}
    explicit = Hardware.from_mapping(raw)
    assert execute(explicit.dies[0]) == pytest.approx(.004)
    exported = explicit.to_dict()
    exported["devices"][0]["compute_arch"]["instruction_services"]["vbitsort32"]["clock_hz"] *= 2
    assert execute(Hardware.from_mapping(exported).dies[0]) == pytest.approx(.004)


def test_per_device_clock_is_resolved_after_inheritance_and_macro_work_stays_unknown():
    raw = machine(clock=1.6e9)
    raw["num_devices"] = 2
    raw["devices"] = [{}, {"compute_arch": {"clock_hz": 3.2e9},
                           "resource_units": {"vector": 8}}]
    saved = deepcopy(raw)
    devices = Hardware.from_mapping(raw).dies
    assert raw == saved
    assert [execute(device) for device in devices] == pytest.approx([.016, .008])
    inventory = PrimitiveInventory({"vbitsort32": 7, "vmrgsort4v2": 9,
                                    "source_sort32_blocks": 128})
    priced, unknown = inventory.partition(Geometry.from_devices(devices).resource_units)
    assert priced.resources == {"vbitsort32": 7}
    assert unknown == {"vmrgsort4v2": 9, "source_sort32_blocks": 128}


def test_resource_only_mode_declaration_does_not_freeze_derived_clock_supply():
    raw = machine(clock=1.6e9)
    raw["device"]["service_modes"] = {"vbitsort32": {"resource": "vector"}}
    # This partial binding has no explicit capacity to rescale during per-device
    # inheritance; the clock-based service supplies it after that merge.
    raw["devices"] = [{"rates": {"vector": 2e12}}]
    hardware = Hardware.from_mapping(raw)
    assert execute(hardware.dies[0]) == pytest.approx(.016)
    exported = hardware.to_dict()
    exported["devices"][0]["compute_arch"]["instruction_services"]["vbitsort32"]["clock_hz"] *= 2
    assert execute(Hardware.from_mapping(exported).dies[0]) == pytest.approx(.008)


def test_clock_based_independent_service_preserves_owner_partition_after_export():
    raw = {"device": {"memory_bytes": 1000, "compute_peaks": {"bf16": 1000},
        "rates": {"hbm": 1000, "vector": 1000}, "resource_units": {"vector": 4},
        "compute_arch": {"instruction_services": {"counted_service": {
            "unit": "blocks", "owner_axis": "vector", "cycles_per_unit": 16,
            "clock_hz": 1.6e9}}}}}
    exported = Hardware.from_mapping(raw).to_dict()
    for owners in (4, 8):
        exported["devices"][0]["resource_units"]["vector"] = owners
        device = Hardware.from_mapping(exported).dies[0]
        role = "counted_service"
        engine = Engine([Resource(role, dict(device.rates)[role] / 1000,
                                  units=device.units(role))])
        done = engine.submit("blocks", {role: 1600}, placement={role: (0,)})
        engine.advance(until=(done,))
        assert done.time == pytest.approx(.016)


def test_one_device_can_explicitly_disable_an_inherited_optional_preset():
    raw = machine()
    raw["num_devices"] = 2
    raw["devices"] = [{}, {"compute_arch": {"presets": []}}]
    first, second = Hardware.from_mapping(raw).dies
    assert execute(first) == pytest.approx(1600 * 16 / 1.65e9 * 1000)
    work = PrimitiveInventory({"vbitsort32": 1600})
    priced, unknown = work.partition(Geometry.from_devices((second,)).resource_units)
    assert not priced.resources
    assert unknown == {"vbitsort32": 1600}


def test_resource_only_rebinding_retains_the_competing_queue_after_export():
    # This is an explicit queue-override control, not the source Sort32 preset.
    raw = {"device": {"memory_bytes": 1000, "compute_peaks": {"bf16": 1000},
        "rates": {"hbm": 1000, "vector": 1e12, "mte2_aiv": 2e12},
        "resource_units": {"vector": 4, "mte2_aiv": 4},
        "compute_arch": {"instruction_services": {"counted_service": {
            "unit": "blocks", "owner_axis": "vector", "resource": "vector",
            "cycles_per_unit": 16, "clock_hz": 1.6e9}}},
        "service_modes": {"counted_service": {"resource": "mte2_aiv"}}}}
    hardware = Hardware.from_mapping(raw)
    restored = Hardware.from_mapping(hardware.to_dict())
    assert restored == hardware
    for machine_ in (hardware, restored):
        device = machine_.dies[0]
        rates = {role: value / 1000 for role, value in device.rates}
        engine = Engine([Resource(role, rate, units=device.units(role))
                         for role, rate in rates.items()])
        context = RecipeContext(None, {}, {r: r for r in rates}, rates,
            device.compute_units, resource_units=dict(device.resource_units),
            service_modes=dict(device.service_modes))
        demand = context.bind(Work(resources={"counted_service": 1600}))
        placement = context.owned(0, 4, ("counted_service",))
        first = engine.submit("counted", demand, placement=placement)
        second = engine.submit("copy_competitor", {"mte2_aiv": demand["mte2_aiv"]},
                               placement={"mte2_aiv": (0,)})
        engine.advance(until=(first, second))
        assert first.time == second.time == pytest.approx(.032)
