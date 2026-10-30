"""Mixed-precision shared service and independent arithmetic counterexamples."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Work, OperatorPlan, PlannedRecipe
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView


def hardware_case():
    return {
        "machine": {"name": "arbitrary", "num_devices": 1,
                    "device": {"memory_bytes": 1024**3, "compute_units": 4,
                               "compute_peaks": {"bf16": 2e6, "fp32": 1e6},
                               "rates": {"memory": 1e6, "vector": 1e6}}},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8, "num_heads": 1,
                  "num_kv_heads": 1, "head_dim": 8, "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def operator():
    # Deliberately stale analytical envelope; a plan must replace all its work.
    return OpSpec("mixed", "fused", "prefill", 0, 0, ("r",),
                  (8, 8), (8, 8), (8, 8), (8, 8), flops=999,
                  hbm_bytes=999, dtype="bf16", recipe="two_products",
                  metadata=(("resource_work", {"vector": 999}),))


def two_products(op, geometry):
    load = Work(resources={"hbm": 1000, "mte2": 1000})
    bf16 = Work(matrix={"bf16": 2000})
    fp32 = Work(matrix={"fp32": 2000})
    store = Work(resources={"hbm": 1000, "mte3": 1000})

    def program(context):
        return KernelProgram((
            Stage("load", context.bind(load)),
            Stage("bf16", context.bind(bf16), after=("load",)),
            Stage("fp32", context.bind(fp32), after=("load",)),
            Stage("store", context.bind(store), after=("bf16", "fp32")),
        ))
    return OperatorPlan(load + bf16 + fp32 + store, program)


def test_mixed_precisions_share_matrix_capacity_and_publish_the_same_physical_work():
    case = hardware_case()
    result = verify_operator(case, operator(), two_products)
    row = result["result"]
    # Load 1ms + (BF16 1ms + FP32 2ms on ONE pipe) + store 1ms.
    assert row["duration_ms"] == 5
    assert row["operator"]["flops"] == 4000
    assert row["operator"]["matrix_flops"] == {"bf16": 2000, "fp32": 2000}
    assert row["operator"]["hbm_bytes"] == 2000
    assert row["operator"]["vector_ops"] == 0
    assert row["declared_service"] == row["serviced"] == {
        "matrix@r0:s0": 3, "memory@r0:s0": 2000,
    }
    assert result["status"] == "passed"
    case["machine"]["device"]["compute_peaks"]["fp32"] = 2e6
    faster = verify_operator(case, operator(), two_products)["result"]
    assert faster["duration_ms"] == 4
    assert faster["operator"] == row["operator"]


def test_plan_is_shared_by_iteration_and_isolated_operator_paths():
    parsed = parse_case(hardware_case())
    recipes = RecipeBook(recipes={"two_products": PlannedRecipe(two_products)})
    op = operator()
    execution = Execution(parsed, CostModel(parsed), recipes=recipes)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one")
    result = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(result.completion,))
    row = operator_costs(parsed, (op,), recipes=recipes)["operators"][0]
    assert result.completion.time == row["duration_ms"]
    assert execution.work["flops"] == row["operator"]["flops"]
    assert execution.engine.serviced["matrix@r0:s0"] == row["serviced"]["matrix@r0:s0"]
    renamed = replace(op, op_id="different", layer=99, requests=("unseen",))
    assert verify_operator(parsed, renamed, two_products)["result"]["duration_ms"] == 5


def test_contract_catches_a_missing_workspace_roundtrip_even_when_timing_is_finite():
    def broken(op, geometry):
        plan = two_products(op, geometry)
        return replace(plan, work=plan.work + Work(resources={"hbm": 128}))
    result = verify_operator(hardware_case(), operator(), broken)
    assert result["finite_completion"]
    assert result["status"] == "failed"
    assert result["work_mismatches"] == {
        "memory@r0:s0": {"declared": 2128, "executed": 2000},
    }


def test_contract_catches_prefetch_that_bypasses_a_parent_input_dependency():
    class BrokenPrefetch:
        def start(self, engine, name, ready, *, owner=""):
            early = engine.submit(name + ":load", {"memory@r0:s0": 1000})
            return engine.submit(name + ":done", after=(early, ready))

    def build(op, geometry):
        return OperatorPlan(Work(resources={"hbm": 1000}), lambda _: BrokenPrefetch())

    zero = verify_operator(hardware_case(), operator(), build)
    delayed = verify_operator(hardware_case(), operator(), build, input_ready_ms=2)
    assert zero["status"] == "passed"
    assert delayed["finite_completion"] and not delayed["work_mismatches"]
    assert delayed["result"]["service_before_input"] == {"memory@r0:s0": 1000}
    assert delayed["status"] == "failed"
    correct = verify_operator(hardware_case(), operator(), two_products, input_ready_ms=2)
    assert correct["status"] == "passed"
    assert correct["result"]["duration_ms"] == 5


def test_plan_requires_a_declared_precision_rate_instead_of_bf16_scaling():
    case = hardware_case()
    del case["machine"]["device"]["compute_peaks"]["fp32"]
    with pytest.raises(ValueError, match="compute_precision_rate_required:fp32"):
        verify_operator(case, operator(), two_products)


def test_source_kernel_expansion_preserves_eager_launches_and_graph_capture():
    def kernels(op):
        return (replace(op, op_id="first", deps=()),
                replace(op, op_id="second", deps=("first",)))

    raw = hardware_case()
    raw["machine"]["device"]["launch_us"] = 50
    case = parse_case(raw)
    recipe = RecipeBook(recipes={"two_products": PlannedRecipe(two_products, kernels)})
    op = operator()
    isolated = operator_costs(case, (op,), recipes=recipe)
    assert isolated["operator_occurrences"] == 2
    assert isolated["operator_sum_ms"] == 10
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    consumer = replace(op, op_id="consumer", deps=(op.op_id,), recipe="direct")
    recipe.recipes["direct"] = PlannedRecipe(two_products)
    for scope, launches in (("none", 3), ("full", 1)):
        execution = Execution(case, CostModel(case), recipes=recipe)
        iteration = IterationWork((op, consumer), plan, 1, 1, 1, None, "prefill", "one",
                                  replay_scope=scope)
        result = execution.add(iteration, 0, 0)
        execution.engine.advance(until=(result.completion,))
        assert execution.engine.serviced["host@r0:s0"] == pytest.approx(launches * .05)
        assert result.completion.time == pytest.approx(15.05)
        assert execution.work["flops"] == 12000
    check = verify_operator(case, op, two_products, kernels=kernels)
    assert check["status"] == "passed"
    assert check["result"]["duration_ms"] == 10


def test_instruction_modes_share_one_physical_pipe_without_changing_byte_counts():
    case = hardware_case()
    device = case["machine"]["device"]
    device["rates"]["copy"] = 4e6
    device["service_modes"] = {"copy_strided": {"resource": "copy", "rate": 2e6}}
    regular = Work(resources={"copy": 1000})
    strided = Work(resources={"copy_strided": 1000})

    def build(op, geometry):
        def program(context):
            owned = context.owned(0, context.core_count, ("copy",))
            return KernelProgram((
                Stage("regular", context.bind(regular), placement=owned),
                Stage("strided", context.bind(strided), placement=owned),
            ))
        return OperatorPlan(regular + strided, program)

    # Each unit supplies 1e6 B/s regular or .5e6 B/s strided. Both
    # instructions target unit0, so it needs1+2ms, not independent1/2ms pipes.
    row = verify_operator(case, operator(), build)
    assert row["status"] == "passed"
    assert row["result"]["duration_ms"] == 3
    assert row["result"]["operator"]["metadata"]["resource_work"] == {
        "copy": 1000, "copy_strided": 1000,
    }
    device["service_modes"]["copy_strided"]["rate"] = 4e6
    faster = verify_operator(case, operator(), build)
    assert faster["result"]["duration_ms"] == 2
    assert faster["result"]["operator"] == row["result"]["operator"]


@pytest.mark.parametrize("use_explicit_placement", (False, True))
def test_mode_ownership_retains_fixed_per_owner_capacity(use_explicit_placement):
    work = Work(resources={"copy_strided": 1000})

    def build(op, geometry):
        def program(context):
            owned = (context.placement("copy_strided", (0,)) if use_explicit_placement
                     else context.owned(0, geometry.compute_units, ("copy_strided",)))
            return KernelProgram((Stage("copy", context.bind(work), placement=owned),))
        return OperatorPlan(work, program)

    for units in (1, 4, 8):
        raw = hardware_case()
        device = raw["machine"]["device"]
        device["compute_units"] = units
        device["rates"]["copy"] = units * 1e6
        device["service_modes"] = {
            "copy_strided": {"resource": "copy", "rate": units * .5e6}}
        result = verify_operator(raw, operator(), build, input_ready_ms=7)
        assert result["status"] == "passed"
        assert result["result"]["duration_ms"] == pytest.approx(2)


def test_composite_contract_cannot_cancel_opposite_child_errors():
    def kernels(op):
        return (replace(op, op_id="first", metadata=(("part", 1),)),
                replace(op, op_id="second", deps=("first",), metadata=(("part", 2),)))

    def build(op, geometry):
        declared = Work(resources={"hbm": 1000})
        executed = Work(resources={"hbm": 0 if dict(op.metadata)["part"] == 1 else 2000})
        return OperatorPlan(declared, lambda context: KernelProgram((
            Stage("copy", context.bind(executed)),)))

    result = verify_operator(hardware_case(), operator(), build, kernels=kernels)
    assert result["result"]["declared_service"] == result["result"]["serviced"]
    assert result["status"] == "failed"
    assert result["work_mismatches"] == {
        "mixed/first": {"memory@r0:s0": {"declared": 1000, "executed": 0}},
        "mixed/second": {"memory@r0:s0": {"declared": 1000, "executed": 2000}},
    }


def test_representative_rank_does_not_combine_one_ranks_rate_with_anothers_units():
    raw = hardware_case()
    raw["model"].update(hidden_size=16, num_heads=2, num_kv_heads=2)
    raw["runtime"]["tp"] = 2
    raw["machine"]["num_devices"] = 2
    raw["machine"]["devices"] = [
        {"compute_units": 1, "rates": {"memory": 1e9, "vector": 1e9, "copy": 1e6},
         "service_modes": {"slow": {"resource": "copy", "rate": 1e6}}},
        {"compute_units": 4, "rates": {"memory": 1e9, "vector": 1e9, "copy": 2e6},
         "service_modes": {"slow": {"resource": "copy", "rate": 2e6}}},
    ]
    work = Work(resources={"slow": 1000})

    def build(op, geometry):
        return OperatorPlan(work, lambda context: KernelProgram((Stage(
            "copy", context.bind(work),
            placement=context.owned(0, geometry.compute_units, ("slow",))),)))

    # Ranks need1ms and2ms per owner; min(total)/min(owners) wrongly gives1ms.
    with pytest.raises(ValueError, match="heterogeneous_owners_require_per_rank_execution"):
        verify_operator(raw, operator(), build)
    # Equal owner topology with different rates is a conservative envelope.
    first = raw["machine"]["devices"][0]
    first["compute_units"] = 4
    first["rates"]["copy"] = 4e6
    first["service_modes"]["slow"]["rate"] = 4e6
    result = verify_operator(raw, operator(), build)
    assert result["status"] == "passed"
    assert result["result"]["duration_ms"] == 2
