"""Actual frontend branch and public execution contract for the native GDN path."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.instructions import REGISTER_RESOURCES
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_operator_mechanisms import recurrent_case


def native_case():
    case = recurrent_case()
    layer = replace(case.model.layer_plans[0], linear_key_heads=1, linear_value_heads=3,
                    linear_heads=3, linear_key_head_dim=128, linear_head_dim=128, conv_kernel=4)
    device = replace(case.machine.dies[0], resource_units=tuple(
                         {"vector": 8, **{role: 8 for role in REGISTER_RESOURCES}}.items()),
                     rates=tuple({**dict(case.machine.dies[0].rates),
                                  "mte1": 1e12, "fixpipe": 1e12,
                                  **{role: 1e12 for role in REGISTER_RESOURCES}}.items()),
                     service_modes=(("ub_copy", ("vector", 1e12)),
                                    ("reg_issue_slots", ("vector", 1e12))))
    return replace(
        case, model=replace(case.model, layer_plans=(layer,)),
        machine=replace(case.machine, dies=(device,)),
        runtime=replace(case.runtime, kernels={"lowerings": {"gdn_prefill": "gdn_prefill_arch35"}},
                        tensor_layouts={"linear_output_norm_gate": {"gate": "contiguous"}}))


def test_real_prefill_dispatch_includes_internal_normalization_and_distinct_kernel_boundaries():
    case = native_case()
    views = (RequestView("new", 64, 0, False, "prefill"),
             RequestView("continued", 17, 32, False, "prefill"))
    ops, _ = Frontend().body(case.model, case.runtime, views, mode="prefill")
    by_name = {op.recipe: op for op in ops}
    assert "qk_l2norm" not in by_name
    assert not any(op.family in {"recurrent_local", "recurrent_scan", "recurrent_output",
                                "triangular_solve"} for op in ops)
    prepare, state, output = [next(op for op in ops if op.family == family)
                              for family in ("gdn_prepare", "gdn_state", "gdn_output")]
    dispatch = dict(prepare.metadata)["dispatch"]
    assert dispatch["gate_dtype"] == "fp32"
    assert dispatch["use_qk_l2norm_in_kernel"] is True
    assert dispatch["use_beta_sigmoid_in_kernel"] is False
    assert dispatch["use_exp2"] is True
    assert dict(state.metadata)["state_v_first"] is True
    assert dict(state.metadata)["use_initial_state"] is True
    assert dict(state.metadata)["lengths"] == (64, 17)
    assert state.op_id in output.deps
    assert by_name["beta_fp32"].mte2_bytes == 81 * 3 * 2
    assert by_name["beta_fp32"].mte3_bytes == 81 * 3 * 4
    assert by_name["qkv_concatenate"].hbm_bytes == 2 * 81 * (2 + 3) * 128 * 2
    # 48 CTAs/sequence; new prompts read only flags, resumed prompts read
    # flags, int32 indices and their actual FP32 state payload.
    assert by_name["gather_recurrent_state"].mte2_bytes == 3 * 128 * 128 * 4 + 48 * (2 + 4)
    assert by_name["gather_recurrent_state"].mte3_bytes == 2 * 3 * 128 * 128 * 4
    copied = by_name["core_output_copy"]
    assert copied.mte2_bytes == copied.mte3_bytes == 81 * 3 * 128 * 2
    assert output.op_id in copied.deps
    # A size-one head axis retains contiguous storage after the transpose.
    assert not any(name in by_name for name in ("q_hat_export", "k_hat_export"))
    # Separate primitive plans are really selected by the same runtime binding.
    rows = operator_costs(case, (prepare, state, output))["operators"]
    assert len(rows) == 3
    assert all(row["program"] == "OperatorPlan" for row in rows)
    assert all(row["operator"]["flops"] > 0 for row in rows)
    for row in rows:
        assert row["serviced"] == pytest.approx(row["declared_service"])


def test_mainline_execution_uses_the_same_native_work_as_isolated_operator_analysis():
    case = native_case()
    view = RequestView("r", 16, 0, True, "prefill", draft_prepare=False)
    work = Frontend()(IterationPlan((view,)), case.model, case.runtime, case.machine)
    result = operator_costs(case, work.ops)
    execution = Execution(case, CostModel(case))
    submitted = execution.add(work, 0, 0)
    execution.engine.advance(until=(submitted.completion,))
    for resource in ("matrix@r0:s0", "memory@r0:s0", "vector@r0:s0"):
        expected = sum(row["serviced"].get(resource, 0) * row["occurrences"]
                       for row in result["operators"])
        assert execution.engine.serviced[resource] == pytest.approx(expected)
    assert all(not size for size in execution.engine.allocated.values())
