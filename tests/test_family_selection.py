"""Model-to-source selection, physical execution and declared-domain controls."""
from dataclasses import replace

import pytest

from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction import predict
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.operators import attention_online, cuda_norm
from mindie_sim.prediction.program import RecipeBook, ResourceEnvelope
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def declared_case(*, norm=False, attention=False, dtype="bf16"):
    raw = case(prompt=4, output=2)
    raw["model"].update(layers=1, activation_dtype=dtype, weight_dtype=dtype,
                        compute_dtype=dtype, kv_dtype=dtype)
    device = raw["machine"]["device"]
    roles = {**attention_online.SERVICE_ROLES, **cuda_norm.SERVICE_ROLES}
    device["rates"]["shared_memory"] = 1e9
    device["compute_peaks"]["fp16"] = 1e9
    device["service_modes"] = {
        role: {"resource": resource, "rate": 1e9} for role, resource in roles.items()}
    bindings = {}
    if attention:
        bindings["gqa_attention"] = {
            "implementation": attention_online.IMPLEMENTATION_ID,
            "options": {"compute_capability": [8, 9]},
        }
    if norm:
        bindings["rmsnorm"] = {
            "implementation": cuda_norm.IMPLEMENTATION, "recipes": ["rmsnorm"]}
    raw["runtime"]["kernels"] = {"lowerings": bindings}
    return raw


def work_for(parsed, views=None):
    return Frontend(attention_policy=AttentionPolicy(**parsed.runtime.attention_policy))(
        IterationPlan(views or (RequestView("r", 4, 0, True, "prefill"),)),
        parsed.model, parsed.runtime, parsed.machine)


def test_runtime_declaration_reaches_real_model_and_shared_execution():
    parsed = parse_case(declared_case(norm=True, attention=True))
    work = work_for(parsed)
    source = [op for op in work.ops if op.plan_required]
    attention = next(op for op in source if op.family == "attention")
    metadata = dict(attention.metadata)
    # Four causal queries, four heads and eight channels are declared by the
    # model. Source padded pair slots stay separate from mathematical pairs.
    assert metadata["semantic_pairs"] == 4 * (1 + 2 + 3 + 4)
    assert attention.flops > 4 * 8 * metadata["semantic_pairs"]
    assert metadata["source_params"]["page_block_size"] == parsed.runtime.kv_block_size
    assert attention.deps == ("s0.l0.cache",)
    assert {op.family for op in source} == {"attention", "rmsnorm"}
    isolated = operator_costs(parsed, work.ops, input_ready_ms=3)
    execution = Execution(parsed, CostModel(parsed))
    submission = execution.add(work, 0, 0)
    execution.engine.advance(until=(submission.completion,))
    expected = {}
    for row in isolated["operators"]:
        assert not row["service_before_input"]
        for resource, amount in row["serviced"].items():
            expected[resource] = expected.get(resource, 0) + amount * row["occurrences"]
    assert {key: value for key, value in execution.engine.serviced.items() if value} == pytest.approx(expected)
    assert not any(execution.engine.allocated.values())
    result = predict(parsed)
    assert result["requests"][0]["generated_tokens"] == 2
    assert result["parameters"]["operator_occurrences"]["measured_primitive"] == 0


def test_source_selection_keeps_unselected_analytic_model_usable():
    raw = declared_case()
    parsed = parse_case(raw)
    work = work_for(parsed)
    attention = next(op for op in work.ops if op.family == "attention")
    assert isinstance(RecipeBook.from_mapping(parsed.runtime.kernels).select(attention), ResourceEnvelope)
    assert attention.flops == 4 * 8 * 4 * (1 + 2 + 3 + 4)
    raw["machine"]["name"] = "unseen-device-label"
    raw["model"]["name"] = "unseen-model-label"
    assert predict(raw)["metrics"] == predict(parsed)["metrics"]


def test_unknown_binding_is_a_visible_modeling_failure_in_both_public_paths():
    raw = declared_case(norm=True)
    raw["runtime"]["kernels"]["lowerings"]["rmsnorm"] = "unregistered_norm"
    error = "implementation_not_registered:rmsnorm:unregistered_norm"
    with pytest.raises(ModelingError, match=error):
        target_operators(raw, phase="prefill", tokens=3)
    with pytest.raises(ModelingError, match=error):
        predict(raw)


def test_source_option_typo_cannot_be_ignored_or_abort_other_study_candidates():
    from mindie_sim import study

    raw = declared_case(norm=True)
    result = study([
        {"id": "aligned", "case": raw},
        {"id": "typo", "case": raw, "overrides": {
            "runtime": {"kernels": {"lowerings": {
                "rmsnorm": {"options": {"aligend": False}}}}}}},
    ], [{"id": "prefill", "phase": "prefill", "tokens": 3}])
    assert result["status"] == "partial"
    accepted, rejected = result["evaluations"]
    assert accepted["status"] == "modeled" and accepted["duration_ms"] > 0
    assert rejected["error"] == "implementation_options_unknown:rmsnorm:aligend"
    assert "duration_ms" not in rejected
    assert [row["candidate"] for row in result["rankings"][0]["candidates"]] == ["aligned"]


def test_source_attention_uses_declared_call_batching_without_fabricating_launches():
    parsed = parse_case(declared_case(attention=True))
    views = (RequestView("a", 3, 2, False, "prefill"),
             RequestView("b", 5, 0, False, "prefill"))
    with pytest.raises(ModelingError, match="one_sequence_launch"):
        work_for(parsed, views)
    parsed = replace(parsed, runtime=replace(parsed.runtime, attention_policy={"fused_batch": False}))
    work = work_for(parsed, views)
    calls = [op for op in work.ops if op.family == "attention"]
    assert len(calls) == 2
    assert [dict(op.metadata)["source_params"]["query"] for op in calls] == [3, 5]
    assert [op.requests for op in calls] == [("a",), ("b",)]
    assert work.ops[-1].deps  # The model still consumes the separate attention outputs.
    report = operator_costs(parsed, calls)
    assert report["operator_occurrences"] == 2
    assert all(row["program"] == "OperatorPlan" for row in report["operators"])


def test_source_attention_rates_change_time_without_rewriting_work():
    raw = declared_case(attention=True)
    parsed = parse_case(raw)
    ops = [op for op in target_operators(parsed, phase="prefill", tokens=7)
           if op.family == "attention"]
    fast = operator_costs(parsed, ops)["operators"][0]
    raw["machine"]["device"]["service_modes"]["fp32_exp"]["rate"] = 1e3
    slow = operator_costs(parse_case(raw), ops)["operators"][0]
    assert slow["duration_ms"] > fast["duration_ms"]
    assert slow["operator"]["metadata"]["resource_work"] == fast["operator"]["metadata"]["resource_work"]
    assert slow["operator"]["flops"] == fast["operator"]["flops"]


def test_attention_binding_uses_local_heads_and_declared_source_tile_dispatch():
    raw = declared_case(attention=True)
    raw["model"].update(hidden_size=1024, head_dim=256)
    raw["machine"]["num_devices"] = 2
    raw["machine"]["links"][0]["members"] = [0, 1]
    raw["runtime"]["tp"] = 2
    first = next(op for op in target_operators(raw, phase="prefill", tokens=7)
                 if op.family == "attention")
    binding = raw["runtime"]["kernels"]["lowerings"]["gqa_attention"]
    binding["options"]["compute_capability"] = [10, 0]
    second = next(op for op in target_operators(raw, phase="prefill", tokens=7)
                  if op.family == "attention")
    a, b = dict(first.metadata), dict(second.metadata)
    assert a["source_params"]["heads"] == 2
    assert a["source_params"]["key_heads"] == 1
    assert a["semantic_pairs"] == b["semantic_pairs"] == 2 * 7 * 8 // 2
    assert (a["dispatch"]["tile_size"], b["dispatch"]["tile_size"]) == (32, 128)
    assert first.flops < second.flops  # Larger source tiles execute padded work.


def test_source_attention_domain_does_not_silently_erase_window_or_kv_quantization():
    parsed = parse_case(declared_case(attention=True))
    layer = replace(parsed.model.layer_plans[0], window=2)
    windowed = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)))
    with pytest.raises(ModelingError, match="unwindowed_unquantized"):
        work_for(windowed)
    analytic = replace(windowed, runtime=replace(windowed.runtime, kernels={}))
    attention = next(op for op in work_for(analytic).ops if op.family == "attention")
    assert attention.flops == 4 * 8 * 4 * (1 + 2 + 2 + 2)
    quantized = replace(parsed, model=replace(
        parsed.model, quant=replace(parsed.model.quant, kv_dtype="int8")))
    with pytest.raises(ModelingError, match="unwindowed_unquantized"):
        work_for(quantized)


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_model_norm_preserves_precision_and_source_traffic(dtype):
    parsed = parse_case(declared_case(norm=True, dtype=dtype))
    ops = target_operators(parsed, phase="prefill", tokens=3)
    norm = next(op for op in ops if op.op_id == "s0.l0.norm1")
    assert norm.dtype == dtype
    row = operator_costs(parsed, (norm,))["operators"][0]
    # Two input passes, one gamma read and one output write, without cache
    # credit. This differs from the semantic envelope's shared-weight bytes.
    size = 4 if dtype == "fp32" else 2
    assert row["operator"]["hbm_bytes"] == 3 * 32 * size * 4
    assert row["operator"]["metadata"]["dispatch"]["width"] == 32
    assert row["program"] == "OperatorPlan"


def test_explicit_norm_binding_survives_generic_fusion_and_keeps_quantization_separate():
    raw = declared_case(norm=True)
    raw["runtime"]["lowering_passes"] = [
        "activation_quantization", "norm_quant", "residual_norm"]
    parsed = parse_case(raw)
    parsed = replace(parsed, model=replace(parsed.model, quant=quant_preset("w8a8_int8")))
    ops = target_operators(parsed, phase="prefill", tokens=3)
    norm = next(op for op in ops if op.op_id == "s0.l0.norm1")
    quant = next(op for op in ops if op.family == "quantize" and op.deps == (norm.op_id,))
    assert norm.plan_required and norm.dtype == "bf16"
    assert quant.mte3_bytes > 0 and not quant.plan_required
    assert norm.returns == ()
    assert dict(quant.metadata)["activation_scale_bytes"] > 0


def test_fused_norm_binding_requires_actual_fusion_and_preserves_residual_consumers():
    raw = declared_case()
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "residual_norm"]
    raw["runtime"]["kernels"]["lowerings"]["add_rmsnorm"] = cuda_norm.IMPLEMENTATION
    parsed = parse_case(raw)
    ops = target_operators(parsed, phase="prefill", tokens=3)
    fused = next(op for op in ops if op.family == "add_rmsnorm")
    assert fused.plan_required and fused.returns == ("residual",)
    assert len(fused.deps) == 2
    assert dict(fused.metadata)["dispatch"]["kernel"] == "fused_add_rms_norm"
    assert any(fused.op_id in op.deps and op.recipe == "residual_add" for op in ops)
    row = operator_costs(parsed, (fused,))["operators"][0]
    # First pass reads both inputs and writes the residual; second reads the
    # residual plus gamma and writes normalized output: six BF16 streams.
    assert row["operator"]["hbm_bytes"] == 3 * 32 * 2 * 6


def test_recipe_filter_leaves_distinct_qk_norm_contract_analytic():
    raw = declared_case(norm=True)
    parsed = parse_case(raw)
    layer = replace(parsed.model.layer_plans[0], qk_norm=True)
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)))
    ops = target_operators(parsed, phase="prefill", tokens=3)
    assert next(op for op in ops if op.recipe == "qk_norm").source_boundary == ""
    assert next(op for op in ops if op.op_id == "s0.l0.norm1").plan_required
    all_norms = replace(parsed.runtime, kernels={"lowerings": {"rmsnorm": cuda_norm.IMPLEMENTATION}})
    with pytest.raises(ModelingError, match="normalization_semantic_contract_required:qk_norm"):
        target_operators(replace(parsed, runtime=all_norms), phase="prefill", tokens=3)
