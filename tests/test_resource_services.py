"""Independent service, composition and shared-capacity numerical controls."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import predict
from mindie_sim.observations.service_fitting import fit_service_curve
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.services import ResourceService, ServiceCurve, service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_mechanism_contract import hardware_case


def copy_op(work=2000, **changes):
    return replace(OpSpec(
        "copy", "copy", "prefill", 0, 0, ("r",),
        (work // 2,), (work // 2,), (work // 2,), (work // 2,),
        hbm_bytes=work, dtype="bf16", layout="contiguous", recipe="copy",
    ), **changes)


def curve(case=None, op=None, **changes):
    case = case or parse_case(hardware_case())
    op = op or copy_op()
    values = dict(
        resource="hbm", work_unit="bytes", identity=service_identity(case, op),
        boundary="device_kernel", active_units=4, startup_ms=1,
        knots=((1000, 2), (3000, 4)),
    )
    return ServiceCurve(**{**values, **changes})


def iteration(ops):
    return IterationWork(tuple(ops), IterationPlan((RequestView("r", 1, 0, True, "prefill"),)),
                         1, 1, 1, None, "prefill", "resource_service_control")


def execute(ops, recipe, raw=None, release_ms=0):
    parsed = parse_case(raw or hardware_case())
    recipes = RecipeBook({"copy": recipe})
    execution = Execution(parsed, CostModel(parsed), recipes=recipes, trace_limit=200)
    step = execution.add(iteration(ops), 0, release_ms)
    execution.engine.advance(until=(step.completion,))
    return step, execution


def packet():
    model = curve()
    return {
        **model.to_dict(),
        "samples": [
            {"work": work, "duration_ms": duration, "run_id": run, "repeat_index": 0,
             "source": f"oracle:{run}:{work}", "split": split}
            for work, duration, run, split in (
                (1000, 2, "a", "train"), (1000, 2, "b", "train"),
                (3000, 4, "a", "train"), (3000, 4, "b", "train"),
                (2000, 3.1, "c", "heldout"), (5000, 6.2, "c", "heldout"),
            )
        ],
    }


def test_fit_keeps_heldout_out_of_knots_startup_and_model_identity():
    source = packet()
    fitted = fit_service_curve(source)
    changed = deepcopy(source)
    for sample in changed["samples"]:
        if sample["split"] == "heldout":
            sample["duration_ms"] *= 100
    second = fit_service_curve(changed)
    assert fitted["model"] == second["model"]
    assert fitted["model_id"] == second["model_id"]
    assert fitted["model"]["startup_ms"] == 1
    assert fitted["heldout"][0]["predicted_ms"] == 3
    assert fitted["heldout"][1]["predicted_ms"] is None
    assert fitted["heldout"][0]["relative_error"] != second["heldout"][0]["relative_error"]
    assert ServiceCurve.from_mapping(json.loads(json.dumps(fitted["model"]))).model_id == fitted["model_id"]


def test_runs_have_equal_weight_and_reimport_does_not_change_a_fit():
    source = packet()
    # Repeating one fast run many times must not outweigh another independent run.
    source["samples"] = [
        {"work": work, "duration_ms": duration, "run_id": run, "repeat_index": repeat,
         "source": f"oracle:{run}:{work}:{repeat}", "split": "train"}
        for work, base in ((1000, 1), (3000, 3))
        for run, duration, count in (("a", base + 1, 1), ("b", base + 3, 19))
        for repeat in range(count)
    ]
    fitted = fit_service_curve(source)
    assert fitted["model"]["knots"] == ((1000, 3), (3000, 5))
    assert fitted["model"]["startup_ms"] == 2
    source["samples"] += deepcopy(source["samples"])
    assert fit_service_curve(source) == fitted
    source["samples"][-1]["split"] = "heldout"
    with pytest.raises(ValueError, match="conflicting_primitive_repeat_or_split"):
        fit_service_curve(source)


def test_unseen_work_interpolates_duration_through_both_public_execution_paths():
    model = curve(startup_ms=1, knots=((1000, 3), (3000, 9), (5000, 17)))
    op = copy_op(4000)
    recipe = ResourceService(model)
    row = operator_costs(hardware_case(), (op,), recipes=RecipeBook({"copy": recipe}),
                         input_ready_ms=7)["operators"][0]
    step, execution = execute((op,), recipe, release_ms=7)
    # Independent interpolation: 4000 is halfway between9ms and17ms.
    assert row["duration_ms"] == 13
    assert step.completion.time == 20
    assert row["service_before_input"] == {}
    assert row["declared_service"] == row["serviced"] == {"memory@r0:s0": 4000}
    assert execution.work["hbm_bytes"] == 4000
    assert row["basis"]["service_models"][0]["queries"] == {"interpolated": 1}
    assert all(event["start_ms"] >= 7 for event in execution.engine.trace if event["demand"])


def test_parallel_invocations_share_capacity_and_each_pays_startup_once():
    left = copy_op(stream="left")
    right = copy_op(op_id="right", stream="right")
    parallel, execution = execute((left, right), ResourceService(curve()))
    serial, _ = execute((left, replace(right, deps=(left.op_id,))), ResourceService(curve()))
    # Both start for1ms;4000 bytes on one1000 B/ms pipe take another4ms.
    assert parallel.completion.time == 5
    assert serial.completion.time == 6
    assert execution.engine.serviced["memory@r0:s0"] == 4000
    assert execution.cost.summary()["service_models"][0]["queries"] == {"interpolated": 2}


def test_capped_service_releases_unused_bandwidth_to_another_ready_operator():
    slow = copy_op(1000, stream="slow")
    fast = copy_op(3000, op_id="fast", stream="fast", recipe="analytic")
    model = curve(startup_ms=0, knots=((1000, 4), (3000, 12)))
    step, execution = execute((slow, fast), ResourceService(model))
    # Slow producer supplies250 B/ms; the remaining750 serves the3000-byte op.
    assert step.completion.time == 4
    ends = {event["task"].split(":")[2]: event["end_ms"]
            for event in execution.engine.trace if event["task"].endswith(":service")}
    assert ends == {"copy": 4, "fast": 4}
    assert execution.engine.serviced["memory@r0:s0"] == 4000


def test_fixed_owner_cannot_borrow_idle_owners_when_curve_exceeds_local_supply():
    model = curve(active_units=1, startup_ms=.5, knots=((1000, 1.5), (3000, 3.5)))
    recipe = ResourceService(model, units=(0,))
    row = operator_costs(hardware_case(), (copy_op(),), recipes=RecipeBook({"copy": recipe}))["operators"][0]
    # One of four owners has250 B/ms;2000 bytes need8ms, plus.5ms startup.
    assert row["duration_ms"] == 8.5
    assert row["serviced"] == {"memory@r0:s0": 2000}
    assert row["basis"]["service_models"][0]["capacity_limited_queries"] == 1
    with pytest.raises(ValueError, match="service_active_units_mismatch"):
        execute((copy_op(),), ResourceService(model))


@pytest.mark.parametrize("work", (500, 4000))
def test_outside_domain_uses_declared_physics_explicitly_without_a_latency_floor(work):
    row = operator_costs(
        hardware_case(), (copy_op(work),), recipes=RecipeBook({"copy": ResourceService(curve())}),
    )["operators"][0]
    assert row["duration_ms"] == work / 1000
    assert row["basis"]["service_models"][0]["queries"] == {"analytic_outside_domain": 1}
    with pytest.raises(ValueError, match="service_work_outside_domain"):
        execute((copy_op(work),), ResourceService(curve(), outside_domain="error"))


@pytest.mark.parametrize("changes", (
    {"dtype": "fp32"}, {"layout": "strided"},
    {"metadata": (("kernel", "another_kernel"),)},
    {"metadata": (("implementation", {"block_threads": 128}),)},
    {"metadata": (("service_axes", {"stride_bytes": 128}),)},
))
def test_different_implementation_conditions_cannot_reuse_the_curve(changes):
    with pytest.raises(ValueError, match="service_identity_mismatch"):
        execute((copy_op(**changes),), ResourceService(curve()))


def test_runtime_revision_and_graph_scope_are_conditions_but_model_names_are_not():
    raw = hardware_case()
    raw["model"]["name"] = "unseen-model"
    step, _ = execute((copy_op(op_id="new", layer=17, requests=("new-request",)),),
                      ResourceService(curve()), raw)
    assert step.completion.time == 3
    raw["runtime"]["vllm_version"] = "different-runtime"
    with pytest.raises(ValueError, match="service_identity_mismatch"):
        execute((copy_op(),), ResourceService(curve()), raw)
    with pytest.raises(ValueError, match="service_identity_mismatch"):
        operator_costs(hardware_case(), (copy_op(),),
                       recipes=RecipeBook({"copy": ResourceService(curve())}), replay_scope="full")
    op = copy_op(metadata=(("service_axes", {"tile": (16, 32)}),))
    model = curve(op=op)
    # JSON configurations and Python planners express the same categorical axes.
    step, _ = execute((op,), ResourceService(json.loads(json.dumps(model.to_dict()))))
    assert step.completion.time == 3


def test_unit_binding_and_mixed_resource_rejection_prevent_false_occupancy():
    with pytest.raises(ValueError, match="service_work_unit_mismatch"):
        execute((copy_op(),), ResourceService(curve(work_unit="ops")))
    with pytest.raises(ValueError, match="service_curve_requires_one_explicit_resource_phase"):
        execute((copy_op(vector_ops=200),), ResourceService(curve()))
    raw = hardware_case()
    raw["machine"]["device"]["service_modes"] = {"alias": {"resource": "hbm", "rate": 1e6}}
    with pytest.raises(ValueError, match="service_curve_requires_one_explicit_resource_phase"):
        execute((copy_op(metadata=(("resource_work", {"alias": 1000}),)),),
                ResourceService(curve()), raw)


def test_typed_matrix_and_byte_mode_curves_keep_their_native_work():
    raw = hardware_case()
    raw["machine"]["device"].update(
        service_modes={"strided": {"resource": "vector", "rate": 5e5}},
        work_units={"strided": "bytes"},
    )
    case = parse_case(raw)
    op = copy_op(hbm_bytes=0, flops=2000, matrix_flops=(("bf16", 2000),))
    matrix = curve(case, op, resource="cube", work_unit="flops")
    row = operator_costs(case, (op,), recipes=RecipeBook({"copy": ResourceService(matrix)}))["operators"][0]
    assert row["duration_ms"] == 3
    assert row["operator"]["matrix_flops"] == {"bf16": 2000}
    assert row["serviced"] == {"matrix@r0:s0": 1}
    # Equal precision peaks do not turn two instruction modes into one primitive.
    mixed = replace(op, matrix_flops=(("bf16", 1000), ("fp32", 1000)))
    with pytest.raises(ValueError, match="service_curve_requires_one_compute_precision"):
        operator_costs(case, (mixed,), recipes=RecipeBook({"copy": ResourceService(matrix)}))
    op = copy_op(hbm_bytes=0, metadata=(("resource_work", {"strided": 2000}),))
    mode = curve(case, op, resource="strided", work_unit="bytes", startup_ms=1,
                 knots=((1000, 5), (3000, 13)))
    row = operator_costs(case, (op,), recipes=RecipeBook({"copy": ResourceService(mode)}))["operators"][0]
    assert row["duration_ms"] == 9
    assert row["operator"]["metadata"]["resource_work"] == {"strided": 2000}
    assert row["declared_service"] == row["serviced"] == {"vector@r0:s0": 4000}


def test_phase_calibration_composes_with_analytic_work_and_preserves_source_binding():
    op = copy_op(source_boundary="kernel")
    identity = {**curve().identity, "kernel": "load"}
    model = curve(identity=identity, boundary="resource_phase")
    service = ResourceService(model)
    load = Work(resources={"hbm": 1000})
    compute = Work(resources={"vector": 500})
    store = Work(resources={"hbm": 1000})

    def build(op, geometry):
        def program(context):
            return KernelProgram((
                Stage("load", program=service.lower(context.for_phase(load, kernel="load"))),
                Stage("compute", context.bind(compute), after=("load",)),
                Stage("store", context.bind(store), after=("compute",)),
            ))
        return OperatorPlan(load + compute + store, program)

    result = verify_operator(hardware_case(), op, build, input_ready_ms=7)
    assert result["status"] == "passed"
    assert result["result"]["duration_ms"] == 3.5
    assert result["result"]["serviced"] == {"memory@r0:s0": 2000, "vector@r0:s0": 500}
    step, _ = execute((op,), PlannedRecipe(build), release_ms=7)
    assert step.completion.time == 10.5
    with pytest.raises(ValueError, match="source_plan_binding_required"):
        execute((op,), ResourceService(curve()))
    # A measured full invocation is not an isolated inner phase.
    wrong = ResourceService(curve(identity=identity))
    execution = Execution(parse_case(hardware_case()), CostModel(parse_case(hardware_case())))
    phase = execution._context(copy_op(), 0, "none").for_phase(load, kernel="load")
    with pytest.raises(ValueError, match="service_boundary_mismatch"):
        wrong.lower(phase)
    # The outer invocation owns its setup and completion constraint.
    plan = build(op, None)
    context = execution._context(plan.work.apply(op), 0, "none")
    context = replace(context, setup_ms=.25, minimum_ms=7)
    done = plan.lower(context).start(execution.engine, "wrapped", execution.engine.submit("ready"))
    execution.engine.advance(until=(done,))
    assert done.time == 7


def test_serialized_recipe_reaches_public_predict_and_conserves_work_across_iterations():
    model = curve()
    raw = hardware_case()
    raw["workload"]["requests"][0].update(prompt_tokens=2, output_tokens=3)
    raw["runtime"]["kernels"] = {"recipes": {"copy": {
        "kind": "resource_service", "model": model.to_dict(), "outside_domain": "analytic",
    }}}

    def lower(plan, model, runtime, machine):
        op = copy_op(plan.valid_tokens * 1000)
        return IterationWork((op,), plan, 1, plan.valid_tokens, plan.valid_tokens,
                             None, "prefill", "service_control")

    result = predict(json.loads(json.dumps(raw)), lower=lower)
    assert result["requests"][0]["ttft_ms"] == 3
    assert result["requests"][0]["finish_ms"] == 7
    assert result["execution"]["work"]["hbm_bytes"] == 4000
    assert result["execution"]["serviced"]["memory@r0:s0"] == 4000


def test_retained_gpu_measurements_generalize_inside_domain_on_the_public_engine():
    from tools.measure.replay_memory_service import replay

    source = Path(__file__).parents[1] / "data/measurements/gpu/sim-probe-20261027-234735.stdout"
    raw = json.loads(source.read_text())
    _, result = replay(raw, source.name, memory_bytes_per_second=1e12)
    samples = raw["samples"]
    train = {work: sorted(s["duration_ms"] for s in samples if s["work_bytes"] == work)[2]
             for work in (201326592, 268435456)}
    independent_midpoint = sum(train.values()) / 2
    score = result["scores_by_work"][0]
    assert score["engine_predicted_ms"] == pytest.approx(independent_midpoint)
    assert abs(score["median_relative_error"]) < .004
    assert len(result["heldout_engine"]) == 5
    assert all(row["declared_bytes"] == row["serviced_bytes"] == 234881024
               for row in result["heldout_engine"])
    assert all(not row["service_before_input"] for row in result["calibrated_operator_costs"]["operators"])
    control = result["composition_controls"][0]
    assert control["parallel"]["duration_ms"] == pytest.approx(2 * 234881024 / 1e9)
    assert control["serial"]["duration_ms"] == pytest.approx(2 * independent_midpoint)
    assert control["parallel"]["execution"]["serviced"]["memory@r0:s0"] == 2 * 234881024
    # The scalar response stays fixed above the physical floor. Changing the
    # declared roof changes concurrent service, not the training observations.
    _, faster_roof = replay(raw, source.name, memory_bytes_per_second=1.2e12)
    assert faster_roof["calibration"]["model"] == result["calibration"]["model"]
    assert faster_roof["scores_by_work"] == result["scores_by_work"]
    assert faster_roof["composition_controls"][0]["parallel"]["duration_ms"] < control["parallel"]["duration_ms"]
