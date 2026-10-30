"""Capacity and physical-owner controls independent of model identities."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.instructions import (
    InstructionClass, InstructionCount, InstructionInventory,
)
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage


def execute(inventory, classes, *, vector_rate=100, owners=(0,)):
    # Two vector owners: two issue slots, two Reg writes and one store per
    # owner/ms. A changed base unit cannot change typed issue capacity.
    engine = Engine([
        Resource("vector", vector_rate, units=2),
        Resource("writes", 4, units=2),
        Resource("ub", 4, units=2),
        Resource("stores", 2, units=2),
    ])
    roles = {"vector": "vector", "reg_data_writes": "writes",
             "reg_ub_accesses": "ub", "reg_ub_stores": "stores"}
    context = RecipeContext(
        None, {}, roles, {name: value.rate for name, value in engine.resources.items()}, 1,
        resource_units={role: 2 for role in roles},
        service_modes={"reg_issue_slots": ("vector", 4000)},
    )
    work = context.bind(inventory.capacity_work(classes))
    stages = tuple(Stage(str(owner), work,
                         placement=context.owned(owner, 2, tuple(roles))) for owner in owners)
    end = KernelProgram(stages).start(engine, "registers", engine.submit("ready"))
    engine.advance(until=(end,))
    return end.time, engine.serviced


def test_single_issue_shuffle_consumes_both_slots_and_two_register_writes():
    # Four actual calls, each producing two registers. Counting two outputs
    # as two calls would incorrectly double the four-cycle issue demand.
    inventory = InstructionInventory({"shuffle": InstructionCount(4, 256)})
    elapsed, work = execute(inventory, {"shuffle": InstructionClass(2, 2)})
    assert elapsed == pytest.approx(4)
    assert work["writes"] == 8
    assert work["vector"] == 200


def test_masked_calls_do_not_disappear_and_typed_service_ignores_base_vector_unit():
    classes = {"arithmetic": InstructionClass(1, 1)}
    inventory = InstructionInventory({"arithmetic": InstructionCount(8, 8)})
    assert execute(inventory, classes)[0] == 4
    assert execute(inventory, classes, vector_rate=400)[0] == 4
    assert execute(inventory, classes, owners=(0, 1))[0] == 4
    # The same owner cannot borrow its idle sibling's issue/write slots.
    assert execute(inventory, classes, owners=(0, 0))[0] == 8


def test_ub_ports_constrain_calls_independently_of_payload_size():
    inventory = InstructionInventory({
        "broadcast_load": InstructionCount(4, 256, read_bytes=16),
        "scalar_store": InstructionCount(4, 4, write_bytes=16),
    })
    classes = {"broadcast_load": InstructionClass(1, 1, ub_read=True),
               "scalar_store": InstructionClass(2, ub_write=True)}
    elapsed, work = execute(inventory, classes)
    assert elapsed == 6
    assert work["ub"] == 8
    assert work["stores"] == 4
    assert inventory.to_dict()["instructions"]["broadcast_load"]["read_bytes"] == 16
