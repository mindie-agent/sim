"""Source operators cannot silently execute an analytical family envelope."""
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.program import RecipeBook, ResourceEnvelope
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization, norm_quant, residual_norm
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView
from tools.run_operator_cases import control_case


ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = (
    "cuda_norm",
    "cuda_quant",
    "attention_online",
    "native_causal_conv_prefill",
    "grouped_gemm",
    "moe_align",
    "index_topk",
    "mla_decode",
    "moe_combine",
    "causal_conv_prefill",
    "gdn_prepare",
    "gdn_state",
    "gdn_output",
)


def source_case(name):
    module = importlib.import_module("mindie_sim.prediction.operators." + name)
    rows = json.loads((ROOT / "tests/fixtures/operator_cases" / (name + ".json")).read_text())
    op = replace(module.make_op(rows["cases"][0]["params"]), requests=("r",))
    raw = control_case(units=4, service_roles=getattr(module, "SERVICE_ROLES", {}))
    return module, op, parse_case(raw)


def iteration(op):
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    return IterationWork((op,), plan, 1, 1, 1, None, "prefill", "source")


@pytest.mark.parametrize("name", EXTENSIONS)
def test_default_isolated_and_iteration_paths_reject_unbound_source_plans(name):
    _, op, case = source_case(name)
    with pytest.raises(ModelingError, match="source_plan_binding_required:" + op.recipe):
        operator_costs(case, (op,))
    execution = Execution(case, CostModel(case))
    with pytest.raises(ModelingError, match="source_plan_binding_required:" + op.recipe):
        execution.add(iteration(op), 0, 0)
    assert not any(execution.engine.serviced.values())


@pytest.mark.parametrize("binding", ("family", "program", "envelope"))
def test_family_or_program_defaults_cannot_replace_an_explicit_source_binding(binding):
    module, op, case = source_case("cuda_norm")
    recipe = PlannedRecipe(module.build)
    op = replace(op, metadata=op.metadata + (("program_kind", "matmul"),))
    books = {
        "family": RecipeBook(families={op.family: recipe}),
        "program": RecipeBook(programs={"matmul": recipe}),
        "envelope": RecipeBook({op.recipe: ResourceEnvelope()}),
    }
    with pytest.raises(ModelingError, match="source_plan_binding_required"):
        operator_costs(case, (op,), recipes=books[binding])


@pytest.mark.parametrize("name", EXTENSIONS[:3] + ("mla_decode",))
def test_bound_source_plans_keep_public_isolated_and_iteration_work_identical(name):
    module, op, case = source_case(name)
    recipes = RecipeBook({op.recipe: PlannedRecipe(module.build, getattr(module, "kernels", None))})
    isolated = operator_costs(case, (op,), recipes=recipes)
    execution = Execution(case, CostModel(case), recipes=recipes)
    scheduled = execution.add(iteration(op), 0, 0)
    execution.engine.advance(until=(scheduled.completion,))
    assert isolated["operator_sum_ms"] > 0
    assert scheduled.completion.time == pytest.approx(isolated["operator_sum_ms"])
    expected = {}
    for row in isolated["operators"]:
        assert row["program"] == "OperatorPlan"
        for resource, amount in row["serviced"].items():
            expected[resource] = expected.get(resource, 0) + amount * row["occurrences"]
    actual = {key: value for key, value in execution.engine.serviced.items() if value}
    assert actual == pytest.approx({key: value for key, value in expected.items() if value})


def test_generic_norm_quant_fusion_retains_source_norm_and_its_typed_execution():
    from mindie_sim.catalog.ops import vector_op

    module, source, case = source_case("cuda_norm")
    quant = vector_op(
        op_id="quant", deps=(source.op_id,), stage=0, layer=0, requests=("r",),
        elements=5120, ops_per_element=4, dtype="bf16", family="quantize",
        phase="prefill", recipe="dynamic_quantize", shape=source.local_shape)
    ops = norm_quant((source, quant), "bf16")
    book = RecipeBook({source.recipe: PlannedRecipe(module.build)})
    before = operator_costs(case, (source, quant), recipes=book)
    after = operator_costs(case, ops, recipes=book)
    assert after == before
    assert after["operators"][0]["program"] == "OperatorPlan"
    with pytest.raises(ModelingError, match="source_plan_binding_required"):
        operator_costs(case, ops)


def test_generic_residual_fusion_preserves_a_source_norm_kernel():
    from mindie_sim.catalog.ops import vector_op

    module, source, case = source_case("cuda_norm")
    residual = vector_op(
        op_id="add", deps=(), stage=0, layer=0, requests=("r",),
        elements=5120, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="prefill", recipe="residual_add", shape=source.local_shape)
    source = replace(source, deps=("add",))
    book = RecipeBook({source.recipe: PlannedRecipe(module.build)})
    before = operator_costs(case, (residual, source), recipes=book)
    fused = residual_norm((residual, source), "bf16")
    assert operator_costs(case, fused, recipes=book) == before


def test_generic_quantization_does_not_rewrite_source_owned_matmul_inputs():
    from mindie_sim.catalog.ops import gemm_op
    from mindie_sim.catalog.quant import quant_preset

    # An explicit source planner may already include input conversion. The
    # semantic pass must not insert a second producer ahead of that boundary.
    op = gemm_op(
        op_id="source", deps=(), stage=0, layer=0, requests=("r",),
        m=2, n=8, k=8, quant=quant_preset("w8a8_int8"),
        category="dense_mlp", phase="prefill", recipe="declared_source")
    op = replace(op, source_boundary="kernel")
    assert activation_quantization((op,), "bf16") == [op]


@pytest.mark.parametrize("name", (
    "mla_decode",
    "index_topk",
    "moe_align",
))
def test_source_calls_require_expansion_and_expanded_kernels_are_not_expanded_again(name):
    module, op, case = source_case(name)
    unexpanded = RecipeBook({op.recipe: PlannedRecipe(module.build)})
    with pytest.raises(ModelingError, match="source_kernel_expansion_required"):
        operator_costs(case, (op,), recipes=unexpanded)
    execution = Execution(case, CostModel(case), recipes=unexpanded)
    with pytest.raises(ModelingError, match="source_kernel_expansion_required"):
        execution.add(iteration(op), 0, 0)
    book = RecipeBook({op.recipe: PlannedRecipe(module.build, module.kernels)})
    with pytest.raises(ModelingError, match="source_call_requires_kernel_expansion"):
        book.prepare(op, Geometry(4))
    children = module.kernels(op)
    expanded = book.expand((op,))
    assert len(expanded) == len(children) + 1
    assert book.expand(expanded) == expanded
    for child in children:
        # Direct analysis of one physical kernel must not execute its wrapper.
        row = operator_costs(case, (child,), recipes=book)
        assert row["operator_occurrences"] == 1
        assert row["operators"][0]["program"] == "OperatorPlan"
