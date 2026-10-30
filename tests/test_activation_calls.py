"""Independent complete-call structure and retained protocol alignment."""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs
from mindie_sim.observations.call_bindings_activation import bind_activation_condition
from mindie_sim.observations.captures import iter_conditions
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.calls_activation import ActivationCallSpec, build_activation_call
from mindie_sim.schema.common import ModelingError
from test_capture_adapters import activation_protocol
from test_prediction_system import case


def test_fused_and_separate_closures_conserve_the_actual_materialized_tensors():
    n = 2 * 7
    fused = build_activation_call(ActivationCallSpec("swiglu_fused", 2, 7))
    separate = build_activation_call(ActivationCallSpec("swiglu_separate", 2, 7))
    assert [(op.mte2_bytes, op.mte3_bytes) for op in fused.ops] == [(2 * n * 2, n * 2)]
    assert [(op.mte2_bytes, op.mte3_bytes) for op in separate.ops] == [
        (n * 2, n * 2), (2 * n * 2, n * 2)]
    assert separate.ops[1].deps == (separate.ops[0].op_id,)
    assert separate.coverage["intermediate"]["bytes"] == n * 2
    assert fused.coverage["intermediate"] is None
    assert sum(op.vector_ops for op in fused.ops) == sum(op.vector_ops for op in separate.ops) == 5 * n
    fused_cost = graph_costs(case(), fused.ops, recipes=fused.recipes)
    separate_cost = graph_costs(case(), separate.ops, recipes=separate.recipes)
    # Declared rates are both 1e9/s: every API envelope is memory-bound here.
    assert fused_cost["duration_ms"] == pytest.approx(6 * n / 1e6)
    assert separate_cost["duration_ms"] == pytest.approx(10 * n / 1e6)
    assert separate_cost["timeline"][1]["ready_ms"] == separate_cost["timeline"][0]["complete_ms"]


def test_standalone_and_tanh_gated_variants_keep_expression_and_api_boundaries():
    n = 3 * 11
    silu = build_activation_call(ActivationCallSpec("silu_standalone", 3, 11))
    gelu = build_activation_call(ActivationCallSpec("gelu_standalone", 3, 11))
    gated = build_activation_call(ActivationCallSpec("geglu_separate", 3, 11))
    assert len(silu.ops) == len(gelu.ops) == 1
    assert len(gated.ops) == 2
    assert silu.coverage["hbm_bytes"] == gelu.coverage["hbm_bytes"] == 4 * n
    assert gated.coverage["hbm_bytes"] == 10 * n
    assert silu.coverage["vector_lane_operations"] == 4 * n
    assert gelu.coverage["vector_lane_operations"] == 9 * n
    assert gated.coverage["vector_lane_operations"] == 10 * n
    assert "approximate=tanh" in gated.coverage["api_sequence"][0]


def test_retained_single_and_burst_bind_to_one_closure_without_relabelling_predictions():
    conditions = list(iter_conditions(activation_protocol()))
    single = bind_activation_condition(conditions[0])
    burst = bind_activation_condition(conditions[1])
    separate = bind_activation_condition(conditions[4])
    assert single.call.ops == burst.call.ops
    assert single.invocations == 1 and burst.invocations == 100
    assert single.call.spec.tokens == 1 and single.call.spec.intermediate == 17408
    assert len(separate.call.ops) == 2
    assert single.target_identity["boundary"] == "npu_event_call"
    assert burst.target_identity["boundary"] == "npu_event_burst"
    graph = graph_costs(case(), single.call.ops, recipes=single.call.recipes)
    assert graph["boundary"] == "device_graph"
    assert single.ledger["strict_identity_equivalence"] is False
    assert single.call.coverage["native_kernel_count"] is None
    assert all(sample["split"] == "unassigned" for condition in conditions for sample in condition["samples"])


def test_observed_durations_context_labels_and_roles_cannot_change_the_call():
    condition = next(iter_conditions(activation_protocol()))
    first = bind_activation_condition(condition)
    changed = deepcopy(condition)
    changed["samples"] = [{"duration_ms": 1e20, "split": "heldout"}]
    changed["metadata"] = {"context": {"hidden": 999999}, "fitted_duration_ms": 1e30}
    second = bind_activation_condition(changed)
    assert first.call.ops == second.call.ops
    assert first.call.coverage == second.call.coverage
    assert first.ledger == second.ledger
    assert second.ledger["measured_values_consumed"] is False
    a = graph_costs(case(), first.call.ops, recipes=first.call.recipes)["duration_ms"]
    b = graph_costs(case(), second.call.ops, recipes=second.call.recipes)["duration_ms"]
    assert a == b
    slower = case()
    slower["machine"]["device"]["rates"]["memory"] /= 2
    assert graph_costs(slower, second.call.ops, recipes=second.call.recipes)["duration_ms"] == 2 * b


def test_retained_burst_runs_full_protocol_and_preserves_nominal_host_overlap():
    conditions = list(iter_conditions(activation_protocol()))
    single = bind_activation_condition(conditions[0])
    burst = bind_activation_condition(conditions[1])
    raw = case()
    raw["machine"]["device"]["launch_us"] = 1000
    first = call_costs(raw, single.call.ops, recipes=single.call.recipes)
    repeated = call_costs(raw, burst.call.ops, recipes=burst.call.recipes,
                          invocations=burst.invocations)
    assert repeated["duration_ms"] == pytest.approx(first["duration_ms"] * 100)
    assert repeated["execution"]["work"]["hbm_bytes"] == first["execution"]["work"]["hbm_bytes"] * 100
    assert repeated["operator_occurrences"] == 100
    submitted = call_costs(raw, burst.call.ops, recipes=burst.call.recipes,
                           invocations=burst.invocations, include_submission=True)
    # The hypothetical 1ms API submission queues overlap prior device work.
    assert submitted["duration_ms"] == pytest.approx(100 + first["duration_ms"])
    assert submitted["boundary"] == "submitted_device_graph"
    assert submitted["duration_ms"] != pytest.approx(100 * (1 + first["duration_ms"]))
    assert submitted["execution"]["live_tasks"] == submitted["execution"]["live_programs"] == 0


def test_unknown_native_protocol_is_not_silently_bound_to_the_audited_closure():
    condition = next(iter_conditions(activation_protocol()))
    changed = deepcopy(condition)
    changed["identity"]["implementation"]["api"] = ["another_kernel"]
    with pytest.raises(ModelingError, match="api_or_layout"):
        bind_activation_condition(changed)
    changed = deepcopy(condition)
    changed["identity"]["execution"]["mode"] = "graph"
    with pytest.raises(ModelingError, match="execution_differs"):
        bind_activation_condition(changed)
