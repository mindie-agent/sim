"""Independent split-copy contention controls and source-plan conservation."""
import pytest

from mindie_sim import Case
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.instructions import REGISTER_RESOURCES
from mindie_sim.prediction.mechanism import PlannedRecipe, Work
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.operators import gdn_output, gdn_prepare, gdn_state
from mindie_sim.prediction.operators.mixed_core import phase_stages
from mindie_sim.prediction.program import KernelProgram, RecipeBook, RecipeContext


@pytest.mark.parametrize("memory_rate,expected", [(1e9, 1), (100, 3)])
def test_aic_and_both_aiv_reads_use_independent_pipes_but_shared_memory(memory_rate, expected):
    engine = Engine([
        Resource("memory", memory_rate),
        Resource("aic_read", 100, units=1),
        Resource("aiv_read", 200, units=2),
    ], trace_limit=100)
    context = RecipeContext(
        None, {}, {"hbm": "memory", "mte2_aic": "aic_read", "mte2_aiv": "aiv_read"},
        {name: resource.rate for name, resource in engine.resources.items()}, 1,
        resource_units={"mte2_aic": 1, "mte2_aiv": 2},
    )
    stages = []
    for name, lane in (("cube", None), ("vec0", 0), ("vec1", 1)):
        stages.extend(phase_stages(
            name, Work(resources={"hbm": 100, "mte2": 100}), context, 0, aiv=lane))
    end = KernelProgram(tuple(stages)).start(engine, "mixed", engine.submit("ready"))
    engine.advance(until=(end,))
    assert end.time == pytest.approx(expected)
    assert engine.serviced == {"memory": 300, "aic_read": 100, "aiv_read": 200}
    loads = {row["task"]: row for row in engine.trace if row["task"].endswith(":load")}
    assert loads["mixed:cube:load"]["placement"] == {"aic_read": (0,)}
    assert loads["mixed:vec0:load"]["placement"] == {"aiv_read": (0,)}
    assert loads["mixed:vec1:load"]["placement"] == {"aiv_read": (1,)}


@pytest.mark.parametrize("cores", [1, 4])
@pytest.mark.parametrize("role", ["reg_issue_slots", "ub_copy"])
def test_source_vector_modes_cannot_borrow_other_aiv_owners(cores, role):
    engine = Engine([Resource("source_pipe", 200 * cores, units=2 * cores)], trace_limit=20)
    context = RecipeContext(
        None, {}, {"custom_vector_pipe": "source_pipe"},
        {"source_pipe": 200 * cores}, cores,
        resource_units={"custom_vector_pipe": 2 * cores},
        service_modes={role: ("custom_vector_pipe", 200000 * cores)},
    )
    program = KernelProgram(phase_stages(
        "vector", Work(resources={role: 100}), context, 0, aiv=1))
    done = program.start(engine, "mixed", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == pytest.approx(1)
    assert engine.serviced == {"source_pipe": pytest.approx(100)}
    compute = next(row for row in engine.trace if row["task"] == "mixed:vector:compute")
    assert compute["placement"] == {"source_pipe": (1,)}


@pytest.mark.parametrize("module", [gdn_prepare, gdn_state, gdn_output])
def test_native_source_plan_accounts_for_every_split_read_without_generic_pipe(module):
    case = Case.from_mapping({
        "machine": {"device": {
            "compute_units": 2, "memory_bytes": 2**30,
            "compute_peaks": {"bf16": 1e12, "fp32": 1e11},
            "rates": {"hbm": 1e11, "vector": 1e11, "mte1": 1e11, "fixpipe": 1e11,
                      "mte2_aic": 1e11, "mte2_aiv": 1e11, "mte3": 1e11,
                      **{role: 1e11 for role in REGISTER_RESOURCES}},
            "resource_units": {"vector": 4, "mte2_aic": 2, "mte2_aiv": 4, "mte3": 4,
                               **{role: 4 for role in REGISTER_RESOURCES}},
            "service_modes": {
                "ub_copy": {"resource": "vector", "rate": 1e11},
                "reg_issue_slots": {"resource": "vector", "rate": 1e11}},
        }},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8, "num_heads": 1,
                  "num_kv_heads": 1, "head_dim": 8, "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"prompt_tokens": 1, "output_tokens": 1}]},
    })
    op = module.make_op({
        "lengths": [73], "heads": 3, "key_heads": 1, "dk": 128, "dv": 128, "chunk": 64,
        "dtype": "bf16", "gate_dtype": "fp32", "use_exp2": True,
    })
    row = operator_costs(case, (op,), recipes=RecipeBook(
        {op.recipe: PlannedRecipe(module.build)}))["operators"][0]
    for role in ("mte2_aic", "mte2_aiv"):
        key = role + "@r0:s0"
        assert row["declared_service"][key] > 0
        assert row["serviced"][key] == pytest.approx(row["declared_service"][key])
    assert "mte2@r0:s0" not in row["serviced"]
    assert row["live_tasks"] == 0
