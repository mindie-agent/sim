"""Count-to-capacity controls with explicit owners and observable units."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.instructions import (
    CONTROL_INSTRUCTIONS, InstructionCount, InstructionInventory, PrimitiveInventory,
)
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.schema.hardware import Hardware


def hardware(scale=1):
    return Hardware.from_mapping({"num_devices": 1, "device": {
        "memory_bytes": 1000, "compute_peaks": {"bf16": 1000},
        "rates": {"hbm": 1000, "vector": 1000}, "compute_units": 2,
        "resource_units": {"vector": 4},
        "compute_arch": {"instruction_services": {
            "mte2_descriptor_issue": {"unit": "descriptors", "owner_axis": "vector",
                "issue_ns": 1000 / scale, "provenance": {"kind": "test_control"}},
            "vbitsort32": {"unit": "blocks32", "owner_axis": "vector",
                "per_owner_rate": 1e6 * scale},
        }}}})


def execute(scale, owners):
    device = hardware(scale).dies[0]
    roles = ("mte2_descriptor_issue", "vbitsort32")
    rates = {role: dict(device.rates)[role] / 1000 for role in roles}
    units = {role: device.units(role) for role in roles}
    engine = Engine([Resource(role, rates[role], units=units[role]) for role in roles])
    context = RecipeContext(None, {}, {r: r for r in roles}, rates, 2, resource_units=units)
    inventory = InstructionInventory({"DataCopyRead": InstructionCount(calls=8),
                                      "VBitSort32": InstructionCount(calls=4)})
    demand = context.bind(inventory.capacity_work(CONTROL_INSTRUCTIONS))
    ready = engine.submit("late_input", latency=7)
    done = KernelProgram(tuple(Stage(str(i), demand,
        placement=context.owned(owner, 4, roles)) for i, owner in enumerate(owners))).start(
            engine, "calls", ready)
    engine.advance(until=(done,))
    return done.time - ready.time, engine.serviced


def test_owner_issue_capacity_and_rate_scans_preserve_operation_counts():
    # Eight descriptors on one owner cost 8us. An idle sibling cannot lend
    # its issue capacity; independent owners overlap, same-owner calls contend.
    for scale in (.5, 1, 2):
        one, work = execute(scale, (0,))
        assert one == pytest.approx(.008 / scale)
        assert work == {"mte2_descriptor_issue": 8, "vbitsort32": 4}
        assert execute(scale, (0, 1))[0] == pytest.approx(one)
        assert execute(scale, (0, 0))[0] == pytest.approx(2 * one)


def test_roundtrip_and_explicit_supply_override_are_not_lost():
    machine = hardware()
    assert Hardware.from_mapping(machine.to_dict()) == machine
    raw = machine.to_dict()
    raw["devices"][0]["rates"]["mte2_descriptor_issue"] *= 2
    restored = Hardware.from_mapping(raw).dies[0]
    assert dict(restored.rates)["mte2_descriptor_issue"] == 8e6
    assert restored.units("mte2_descriptor_issue") == 4
    assert dict(restored.work_units)["mte2_descriptor_issue"] == "descriptors"


def test_unknown_primitive_count_survives_partition_and_sort_is_not_simd_lanes():
    counts = InstructionInventory({"VBitSort32": InstructionCount(3, 96),
                                    "VMrgSort4V2": InstructionCount(2, 128)})
    work = counts.capacity_work(CONTROL_INSTRUCTIONS)
    assert work.resources == {"vbitsort32": 3, "vmrgsort4v2": 2}
    view = PrimitiveInventory(work.resources).to_dict({"vbitsort32"})
    assert view["priced_service_work"] == {"vbitsort32": 3}
    assert view["unpriced_service_work"] == {"vmrgsort4v2": 2}


def linked_machine(scale=1, cores=4, explicit=None):
    raw = {"num_devices": 1, "device": {
        "memory_bytes": 1000, "compute_peaks": {"bf16": 1000},
        "rates": {"hbm": 1000, "vector": 1000, "mte2_aiv": cores * 100e9 * scale},
        "compute_units": 2, "resource_units": {"vector": cores, "mte2_aiv": cores},
        "compute_arch": {"instruction_services": {
            "mte2_descriptor_issue": {"unit": "descriptors", "owner_axis": "vector",
                "issue_ns": 1000, "scale_with": {
                    "resource": "mte2_aiv", "reference_per_owner_rate": 100e9},
                "provenance": {"kind": "explicit_test_transfer_assumption"}}}}}}
    if explicit is not None:
        raw["device"]["rates"]["mte2_descriptor_issue"] = explicit
    return Hardware.from_mapping(raw)


def linked_execution(machine):
    device = machine.dies[0]
    role = "mte2_descriptor_issue"
    engine = Engine([Resource(role, dict(device.rates)[role] / 1000,
                              units=device.units(role))])
    ready = engine.submit("input", latency=7)
    event = engine.submit("descriptors", {role: 8}, after=(ready,), placement={role: (0,)})
    engine.advance(until=(event,))
    return event.time - ready.time, engine.serviced


def test_explicit_capacity_transfer_scales_one_owner_without_idle_core_acceleration():
    for scale in (.5, 1, 2):
        for cores in (4, 8):
            machine = linked_machine(scale, cores)
            elapsed, work = linked_execution(machine)
            assert elapsed == pytest.approx(.008 / scale)
            assert work == {"mte2_descriptor_issue": 8}
            assert Hardware.from_mapping(machine.to_dict()) == machine
    # A direct capacity declaration remains authoritative over the link.
    assert linked_execution(linked_machine(.5, explicit=16e6))[0] == pytest.approx(.002)


def test_unlinked_service_stays_independent_of_byte_throughput():
    raw = linked_machine().to_dict()
    device = raw["devices"][0]
    device["rates"].pop("mte2_descriptor_issue", None)
    del device["compute_arch"]["instruction_services"]["mte2_descriptor_issue"]["scale_with"]
    for scale in (.5, 1, 2):
        device["rates"]["mte2_aiv"] = 400e9 * scale
        assert linked_execution(Hardware.from_mapping(raw))[0] == pytest.approx(.008)


def test_linked_supply_roundtrip_preserves_sweep_and_real_override_priority():
    raw = linked_machine().to_dict()
    raw["devices"][0]["rates"]["mte2_aiv"] *= 2
    assert linked_execution(Hardware.from_mapping(raw))[0] == pytest.approx(.004)
    explicit = linked_machine(explicit=16e6)
    override = explicit.to_dict()
    override["devices"][0]["rates"]["mte2_aiv"] *= 2
    assert linked_execution(Hardware.from_mapping(override))[0] == pytest.approx(.002)
