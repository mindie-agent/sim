"""Independent matrix declarations checked against retained native captures."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure_contracts import predicted_contract, comparison_scope


ROOT = Path(__file__).resolve().parents[1]


def operators(name, tokens, *, phase="prefill"):
    case = parse_case(json.loads((ROOT / "examples/cases" / f"{name}.json").read_text()))
    return target_operators(case, phase=phase, tokens=tokens)


def capture(name, kind):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows

    path = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/fusion_contracts"
    if not path.is_dir():
        pytest.skip("Retained profiler fusion fixtures are unavailable.")
    return [row for row in kernel_rows(path / f"{name}.csv") if row.kind == kind]


@pytest.mark.parametrize("case_name,capture_name,tokens,expected", [
    ("glm53-a3-w4a8", "a3_glm53_prefill", 4096, (21, 78)),
    ("qwen36-a3-tp4", "a3_qwen36", 8192, (48, 64)),
    ("qwen38-a5", "a5_qwen38", 16384, (64, 64)),
])
def test_real_quantized_dense_operators_match_independently_declared_cases(
    case_name, capture_name, tokens, expected, profiler_api,
):
    from mindie_profiler.tensor_contracts import observed_contract

    declared = [predicted_contract(op.to_dict()) for op in operators(case_name, tokens)]
    rows = capture(capture_name, "QuantBatchMatmulV3")
    assert tuple(declared.count(observed_contract(row)) for row in rows) == expected
    assert all(observed_contract(row) is not None for row in rows)


@pytest.mark.parametrize("case_name,capture_name,tokens,count", [
    ("glm53-a3-w4a8", "a3_glm53_prefill", 4096, 75),
    ("qwen35-a3-tp4", "a3_qwen35", 8192, 40),
])
def test_real_grouped_down_preserves_bank_and_unknown_routing_histogram(
    case_name, capture_name, tokens, count, profiler_api,
):
    from mindie_profiler.tensor_contracts import observed_contract

    row = capture(capture_name, "GroupedMatmul")[0]
    actual = observed_contract(row)
    matches = [op for op in operators(case_name, tokens)
               if predicted_contract(op.to_dict()) == actual]
    assert len(matches) == count
    assert actual[0] == "grouped_matrix"
    op = matches[0].to_dict()
    assert "per_expert_row_histogram" in comparison_scope(op)["not_compared"]
    assert op["metadata"]["matrix_weight_bank_groups"] == 256
    assert op["metadata"]["matrix_groups"] == 1
    # A bank and a dense tensor can have identical total M,N,K and precision.
    # Their work assignment is still different and must not share a contract.
    dense = deepcopy(op)
    dense["family"] = "gemm"
    dense["metadata"].pop("matrix_mapping_domain")
    assert predicted_contract(dense)[0] == "matrix"
    assert predicted_contract(dense) != actual
    missing_bank = deepcopy(op)
    missing_bank["metadata"].pop("matrix_weight_bank_groups")
    assert predicted_contract(missing_bank) is None
    fractional = deepcopy(op)
    fractional["metadata"]["expected_rows"] = op["local_shape"][0] - 0.25
    assert predicted_contract(fractional) is None


def test_physical_batch_count_cannot_reuse_flat_dense_contract():
    op = next(op for op in operators("glm53-a3-w4a8", 4096)
              if op.recipe == "index_q")
    ordinary = predicted_contract(op.to_dict())
    assert ordinary[0] == "matrix"
    batched = replace(op, metadata=tuple(
        (key, 2 if key == "matrix_groups" else value) for key, value in op.metadata))
    assert predicted_contract(batched.to_dict()) is None
    call = replace(op, source_boundary="call")
    assert predicted_contract(call.to_dict()) is None


def test_kimi_mxfp4_capture_does_not_override_declared_bf16_compute(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    declared = [predicted_contract(op.to_dict()) for op in operators("kimi-k3-a5", 8192)]
    rows = capture("a5_kimi_k3", "QuantBatchMatmulV4")
    assert all(observed_contract(row) is not None for row in rows)
    assert all(observed_contract(row) not in declared for row in rows)
    assert all(observed_contract(row)[2:4] == ("fp8_e4m3", "mxfp4") for row in rows)
    assert any(item and item[0] == "grouped_matrix" and item[2:4] == ("bf16", "mxfp4")
               for item in declared)


def test_matrix_resource_scope_keeps_w4_correction_separate_from_w8_and_mx():
    glm = operators("glm53-a3-w4a8", 4096)
    w4 = next(op for op in glm if op.recipe == "grouped_down")
    assert w4.vector_ops == 0
    assert comparison_scope(w4.to_dict())["prediction_coverage"] == {
        "kind": "partial_program", "reason": "quantized_epilogue_unmodeled",
        "missing_work": ["packed_int4_output_rescale", "quantization_offset_correction"],
        "basis": "declared_int4_int8_program_has_no_supported_correction_epilogue",
    }
    w8 = next(op for op in glm if op.recipe == "index_q")
    assert dict(w8.metadata)["matrix_epilogue"]["operations_per_output"] > 0
    assert w8.vector_ops > 0
    assert "prediction_coverage" not in comparison_scope(w8.to_dict())
    mx = next(op for op in operators("qwen38-a5", 16384) if op.recipe == "dense_gate_up")
    assert dict(mx.metadata)["activation_scale_bytes"] > 0
    scope = comparison_scope(mx.to_dict())
    assert "prediction_coverage" not in scope
    assert "not separately resolved" in scope["mx_execution_assumption"]


def test_real_matrix_candidate_does_not_select_layer_from_duration(profiler_api):
    from mindie_profiler.structure_mapping import map_structure
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_sim.structure import export_structure

    case = json.loads((ROOT / "examples/cases/glm53-a3-w4a8.json").read_text())
    graph = export_structure(case, phase="prefill", tokens=4096)
    row = capture("a3_glm53_prefill", "QuantBatchMatmulV3")[0]
    event = replace(row.event, rank_id="rank_0", rank_index=0)
    path = Path(row.source)
    stat = path.stat()
    source = SourceRef("rank_0", 0, "csv", str(path), path.name, None,
                       stat.st_size, stat.st_mtime_ns, "retained_capture", "kernel_details")

    def mapping(task):
        obs = Observation("rank_0", 0, Window(task.start_ns, task.end_ns), None, [task],
                          source, QueryStats("rank_0", "matrix_control"))
        return map_structure([obs], graph)

    result = mapping(event)
    mapped = result["event_mappings"][0]
    assert mapped["status"] == "ambiguous"
    assert mapped["candidate_count"] == 21
    assert not mapped["node_ids"]
    assert result["windows"][0]["bound_busy_union_ns"] == 0
    longer = mapping(replace(event, end_ns=event.end_ns + 100000))["event_mappings"][0]
    assert longer["candidate_node_ids"] == mapped["candidate_node_ids"]
