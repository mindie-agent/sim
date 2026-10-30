"""Selected native normalization keeps real outputs and independent declarations."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions
from mindie_sim.structure_subgraphs import subgraph_contract
from mindie_sim.structure_contracts import predicted_contract


ROOT = Path(__file__).resolve().parents[1]


def _case(name, *, quant=False, bias=False, gamma=False):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    passes = ["activation_quantization", *(("norm_quant",) if quant else ()), "residual_norm"]
    raw["runtime"]["lowering_passes"] = passes
    selector = "add_rmsnorm_quant" if quant else "add_rmsnorm"
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[selector] = {
        "implementation": "vllm_ascend_native_norm",
        "options": {"has_bias": bias, "external_gamma_shift": gamma},
    }
    return parse_case(raw)


def _ops(case, tokens=1):
    return target_operators(case, phase="decode", tokens=tokens, context=16)


def _row(capture, kind, rows=1):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    path = Path(mindie_profiler.__file__).resolve().parents[1] / \
        "tests/fixtures/fusion_contracts" / (capture + ".csv")
    return next(row for row in kernel_rows(path)
                if row.kind == kind and row.input_shapes and row.input_shapes[0][0] == rows)


def test_optional_beta_and_rstd_have_real_resource_cost_and_distinct_contracts(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    plain, biased = _case("glm53-a3-w4a8"), _case("glm53-a3-w4a8", bias=True)
    a = next(op for op in _ops(plain) if op.family == "add_rmsnorm")
    b = next(op for op in _ops(biased) if op.op_id == a.op_id)
    rows, width = b.local_shape
    assert b.mte2_bytes - a.mte2_bytes == width * 2
    assert b.vector_ops - a.vector_ops == rows * width
    assert b.mte3_bytes == 2 * rows * width * 2 + rows * 4
    assert set(b.returns) >= {"residual", "rstd"}
    assert b.hbm_bytes == b.mte2_bytes + b.mte3_bytes
    captured = observed_contract(_row("a3_glm53_decode", "AddRmsNormBias"))
    assert predicted_contract(b.to_dict()) == captured
    assert predicted_contract(a.to_dict()) != captured
    assert dict(b.metadata)["norm_parameter_basis"] == "explicit_caller_bias_option"


def test_gemma_shift_is_external_to_native_add_norm_and_has_work():
    case = _case("qwen35-a3-tp4", gamma=True)
    ops = _ops(case)
    norm = next(op for op in ops if op.family == "add_rmsnorm")
    gamma = next(op for op in ops if op.op_id == norm.op_id + ".gamma_shift")
    width = norm.local_shape[-1]
    assert gamma.op_id in norm.deps
    assert gamma.vector_ops == width
    assert gamma.mte2_bytes == gamma.mte3_bytes == width * 2
    assert dict(norm.metadata)["norm_weight_offset"] == 0
    assert dict(norm.metadata)["norm_caller_weight_offset"] == 1
    assert predicted_contract(norm.to_dict())[1] == "AddRmsNormBias"


def test_add_norm_int8_retains_affine_and_residual_but_eliminates_intermediate(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    selected = _case("glm53-a3-w4a8", quant=True, bias=True)
    fused = next(op for op in _ops(selected) if op.family == "add_rmsnorm_quant")
    rows, width = fused.local_shape
    assert predicted_contract(fused.to_dict()) == observed_contract(
        _row("a3_glm53_decode", "AddRmsNormDynamicQuant"))
    assert fused.mte2_bytes == (2 * rows + 2) * width * 2
    assert fused.mte3_bytes == rows * width * 3 + rows * 4
    assert "residual" in fused.returns
    assert "rstd" not in fused.returns


def _kimi_mx_case():
    raw = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    base = parse_case(raw)
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "norm_quant"]
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["rmsnorm_quant"] = \
        "vllm_ascend_native_norm"
    selected = parse_case(raw)
    quant = replace(selected.model.quant, weight_format="fp8_e4m3",
                    expert_weight_format="fp8_e4m3", weight_block=(32, 32),
                    activation_format="fp8_e4m3",
                    activation_group_size=32, activation_scale_format="fp8_e8m0",
                    activation_scale_pack=2, exclusions=())
    selected = replace(selected, model=replace(selected.model, quant=quant))
    return base, selected


def test_kimi_mx_is_an_explicit_precision_scenario_and_keeps_base_bf16(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    base, selected = _kimi_mx_case()
    fused = next(op for op in _ops(selected)
                 if op.family == "rmsnorm_quant" and op.local_shape == (1, 3584))
    assert predicted_contract(fused.to_dict()) == observed_contract(
        _row("a5_kimi_k3", "RmsNormDynamicMxQuant"))
    rows, width = fused.local_shape
    assert fused.mte2_bytes == (rows + 1) * width * 2
    assert fused.mte3_bytes == rows * width + rows * width // 32
    assert base.model.quant.activation_format == "bf16"


def test_kimi_latent_norm_is_after_combine_before_up_projection_with_gamma_read():
    raw = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    ops = _ops(parse_case(raw), tokens=8)
    by_id = {op.op_id: op for op in ops}
    norm = next(op for op in ops if op.recipe == "latent_norm")
    prefix = norm.op_id.removesuffix(".latent_norm")
    assert norm.deps == (prefix + ".unpack",)
    assert by_id[prefix + ".latent_up"].deps == (norm.op_id,)
    assert prefix + ".latent_down" in by_id[prefix + ".pack"].deps
    assert norm.op_id not in by_id[prefix + ".pack"].deps
    assert norm.local_shape == (8, 3584)
    assert norm.mte2_bytes == 9 * 3584 * 2
    assert norm.mte3_bytes == 8 * 3584 * 2


def test_situ_mx_source_profile_removes_only_local_activation_intermediate(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract
    from mindie_sim import graph_costs

    _, case = _kimi_mx_case()
    declaration, members = next(
        (d, members) for d, members in _groups(_ops(case))
        if d["role"] == "situ_quant" and d["tensor_view"]["width"] == 768)
    assert subgraph_contract(declaration) == observed_contract(_row("a5_kimi_k3", "SituMxQuant"))
    kernels = {**case.runtime.kernels, "lowerings": {
        **case.runtime.kernels.get("lowerings", {}),
        "native_fusion": {"implementation": "declared_native_fusion", "options": {
            "roles": ["situ_quant"], "traffic_profile": "situ_mx_quant_arch35_2a9bd1b0"}}}}
    selected = replace(case, runtime=replace(case.runtime, kernels=kernels))
    physical = lower_native_fusions(members, selected.runtime)
    activation = next(op for op in members if op.family == "situ")
    assert len(members) == 2 and len(physical) == 1
    assert physical[0].hbm_bytes == sum(op.hbm_bytes for op in members) - 2 * activation.mte3_bytes
    assert physical[0].vector_ops == sum(op.vector_ops for op in members)
    ids = {op.op_id for op in members}
    external = {dep: 0 for op in members for dep in op.deps if dep not in ids}
    before = graph_costs(case, members, external_inputs=external, isolated=False)
    after = graph_costs(selected, physical, external_inputs=external, isolated=False)
    a, b = before["execution"]["serviced"], after["execution"]["serviced"]
    assert a["vector@r0:s0"] == b["vector@r0:s0"]
    assert a["memory@r0:s0"] - b["memory@r0:s0"] == pytest.approx(2 * activation.mte3_bytes)
