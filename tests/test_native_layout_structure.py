"""Independent precision/view declarations matched to retained native rows."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.structure_contracts import predicted_contract


ROOT = Path(__file__).resolve().parents[1]


def _ops(case, tokens, phase="prefill", *, selected=False):
    raw = json.loads((ROOT / "examples/cases" / f"{case}.json").read_text())
    if selected:
        raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
            "mla_prolog"] = "vllm_ascend_mla_prolog_v3"
    return target_operators(raw, phase=phase, tokens=tokens, context=16 if phase == "decode" else 0)


def _row(capture, kind):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows

    path = Path(mindie_profiler.__file__).resolve().parents[1] / \
        "tests/fixtures/fusion_contracts" / f"{capture}.csv"
    if not path.exists():
        pytest.skip("retained fusion fixture is unavailable")
    return next(row for row in kernel_rows(path) if row.kind == kind)


@pytest.mark.parametrize("tokens,capture,phase", [
    (4096, "a3_glm53_prefill", "prefill"), (1, "a3_glm53_decode", "decode"),
])
def test_indexer_layernorm_fp32_source_boundary_matches_real_capture(tokens, capture, phase, profiler_api):
    from mindie_profiler.body_layout_contracts import observed_layout_contract

    ops = _ops("glm53-a3-w4a8", tokens, phase)
    norm = next(op for op in ops if op.recipe == "index_norm")
    assert predicted_contract(norm.to_dict()) == observed_layout_contract(_row(capture, "LayerNormV3"))
    assert norm.mte2_bytes == (tokens + 2) * 128 * 4
    assert norm.mte3_bytes == tokens * 128 * 4
    assert predicted_contract(replace(norm, dtype="bf16").to_dict()) != \
        predicted_contract(norm.to_dict())


@pytest.mark.parametrize("case,tokens,capture,phase", [
    ("glm53-a3-w4a8", 4096, "a3_glm53_prefill", "prefill"),
    ("glm53-a3-w4a8", 1, "a3_glm53_decode", "decode"),
    ("kimi-k3-a5", 1, "a5_kimi_k3", "decode"),
])
def test_batched_value_projection_preserves_work_without_transpose_copy(
    case, tokens, capture, phase, profiler_api,
):
    from mindie_profiler.body_layout_contracts import observed_layout_contract

    ops = _ops(case, tokens, phase, selected=case.startswith("glm53"))
    projection = next(op for op in ops if op.recipe == "v_expand")
    contract = predicted_contract(projection.to_dict())
    assert contract == observed_layout_contract(_row(capture, "TransposeBatchMatMul"))
    heads, rows, n, k = contract[2]
    assert projection.flops == 2 * heads * rows * n * k
    assert projection.mte2_bytes == (heads * rows * k + heads * k * n) * 2
    assert projection.mte3_bytes == heads * rows * n * 2
    meta = dict(projection.metadata)
    assert meta["native_tensor_view"]["input_permutation"] == (1, 0, 2)
    assert meta["native_implementation"]["selection"] is None
    assert not any(op.recipe.startswith("v_expand_transpose_copy") for op in ops)


def test_unselected_quantized_absorbed_projection_is_not_a_float_native_match():
    op = next(op for op in _ops("glm53-a3-w4a8", 1, "decode") if op.recipe == "v_expand")
    assert dict(op.metadata)["weight_format"] == "int8"
    assert predicted_contract(op.to_dict()) is None


@pytest.mark.parametrize("case,tokens,capture", [
    ("v4f-a3-w8a8-tp8", 4096, "a3_v4f"),
    ("v4pro-a3-w4a8-tp16", 1024, "a3_v4pro"),
])
def test_grouped_output_keeps_native_batch_view_and_declared_precision(
    case, tokens, capture, profiler_api,
):
    from mindie_profiler.body_layout_contracts import observed_layout_contract

    native = observed_layout_contract(_row(capture, "TransposeBatchMatMul"))
    original = next(op for op in _ops(case, tokens) if op.recipe == "grouped_output_a")
    view = dict(original.metadata)["native_tensor_view"]
    assert (view["batch"], view["rows"], view["n"], view["k"]) == native[2]
    assert predicted_contract(original.to_dict()) == native
    # Equal tensor geometry cannot identify a different operand precision.
    raw = json.loads((ROOT / "examples/cases" / f"{case}.json").read_text())
    raw["model"]["quant"]["category_overrides"]["mla.o_a"] = {
        "weight_format": "int8", "activation_format": "int8"}
    quantized = next(op for op in target_operators(raw, phase="prefill", tokens=tokens)
                     if op.recipe == "grouped_output_a")
    assert predicted_contract(quantized.to_dict()) is None
    assert quantized.flops == original.flops
