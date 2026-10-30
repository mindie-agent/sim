"""Synthetic declared Sort32 count/capacity control.

[Role] Verify installed instruction capacity, ownership and shared-pipe contention.
[Invariants] Synthetic instruction inventory; no source-caller or latency claim.
[Timeouts & Locks] Nine finite in-memory Engine controls; no locks or children.
[Side-Effect Policy] Read installed profiles and emit JSON only.
"""
import json
import math

import mindie_sim
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.instructions import CONTROL_INSTRUCTIONS, InstructionCount, InstructionInventory
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.schema.hardware import Hardware


def run(clock, owners, tasks=(0,), vector_competitor=False):
    hardware = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950DT_9581",
        "device": {"resource_units": {"vector": owners}, "compute_arch": {
            "presets": ["ascend_sort32_spec"], "clock_hz": clock}}})
    device = hardware.dies[0]
    rate = dict(device.rates)["vector"] / 1000
    engine = Engine([Resource("vector", rate, units=owners)])
    context = RecipeContext(None, {}, {"vector": "vector"}, {"vector": rate},
        device.compute_units, resource_units={"vector": owners},
        service_modes=dict(device.service_modes))
    inventory = InstructionInventory({"VBitSort32": InstructionCount(1600, 51200)})
    native = inventory.capacity_work(CONTROL_INSTRUCTIONS)
    demand = context.bind(native)
    stages = [Stage(str(i), demand, placement=context.owned(owner, owners, ("vbitsort32",)))
              for i, owner in enumerate(tasks)]
    if vector_competitor:
        stages.append(Stage("vector_work", {"vector": demand["vector"]},
                            placement={"vector": (0,)}))
    ready = engine.submit("input", latency=7)
    done = KernelProgram(tuple(stages)).start(engine, "sort32", ready)
    engine.advance(until=(done,))
    expected = 1600 * 16 / clock * 1000 * (2 if tasks == (0, 0) or vector_competitor else 1)
    assert math.isclose(done.time - ready.time, expected, rel_tol=1e-10)
    assert math.isclose(engine.serviced["vector"],
                        demand["vector"] * (len(tasks) + vector_competitor), rel_tol=1e-12)
    assert native.resources == {"vbitsort32": 1600}
    assert Hardware.from_mapping(hardware.to_dict()) == hardware
    return {"clock_hz": clock, "owners": owners, "tasks": tasks,
            "equal_vector_competitor": vector_competitor,
            "duration_ms": done.time - ready.time,
            "blocks32_per_task": 1600, "cycles_per_block": 16}


results = [run(1.6e9 * scale, owners) for scale in (.5, 1, 2) for owners in (2, 8)]
results.extend([run(1.6e9, 4, (0, 1)), run(1.6e9, 4, (0, 0)),
                run(1.6e9, 4, vector_competitor=True)])
print(json.dumps({"package": mindie_sim.__file__,
    "boundary": "synthetic_instruction_inventory_engine_capacity_control",
    "source_caller": None, "results": results}, indent=2))
