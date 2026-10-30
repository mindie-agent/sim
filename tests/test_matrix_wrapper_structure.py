"""Historical native rows compared to an independently selected source wrapper."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.implementations import Selection
from mindie_sim.prediction import matrix_wrappers
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_contracts import comparison_scope, predicted_contract


ROOT = Path(__file__).resolve().parents[1]


def declared(tokens=1024, phase="prefill", *, graph_mode=None):
    case = parse_case(json.loads((ROOT / "examples/cases/"
        "kimi-k3-a5-current-checkpoint-conditional.json").read_text()))
    if graph_mode is not None:
        case = replace(case, runtime=replace(case.runtime, graph_mode=graph_mode))
    # Restrict tests to the first FFN containing the independent latent pair.
    ops = tuple(op for op in target_operators(case, phase=phase, tokens=tokens,
                                             context=32768 if phase == "decode" else 0)
                if op.layer == 1)
    return case, ops


def selected(*recipes):
    return Selection(matrix_wrappers.IMPLEMENTATION, matrix_wrappers,
                     {"projection_recipes": recipes or ("latent_projection",)})


def captured():
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows

    path = (Path(mindie_profiler.__file__).resolve().parents[1]
            / "tests/fixtures/kimi_grouped_attnres/a5_kimi_k3.csv")
    if not path.is_file():
        pytest.skip("Retained Kimi source-wrapper fixture is unavailable.")
    return tuple(row for row in kernel_rows(path) if row.kind == "GroupedMatmul")


@pytest.mark.parametrize("tokens,phase", [(1024, "prefill"), (1, "decode")])
def test_selected_source_wrapper_matches_real_single_bank_without_changing_work(
    tokens, phase, profiler_api,
):
    from mindie_profiler.tensor_contracts import observed_contract

    case, ops = declared(tokens, phase, graph_mode="eager")
    native = {observed_contract(row) for row in captured()}
    bound = matrix_wrappers.apply_matrix_wrapper(
        ops, selected(), replace(case.runtime, graph_mode="eager"))
    projections = []
    for original, wrapped in zip(ops, bound):
        # Metadata is the entire change, including for dependency/shape/work
        # fields not enumerated by a hand-maintained list in this test.
        assert replace(wrapped, metadata=original.metadata) == original
        if original.recipe != "latent_projection":
            assert wrapped is original
            continue
        projections.append(wrapped)
        assert predicted_contract(original.to_dict())[0] == "matrix"
        assert predicted_contract(original.to_dict()) not in native
        contract = predicted_contract(wrapped.to_dict())
        assert contract in native and contract[0] == "grouped_matrix"
        assert contract[1][0] == 1
        scope = comparison_scope(wrapped.to_dict())
        assert scope["declared_group_list"] == [tokens]
        assert "native_dispatch" in scope["not_compared"]
        assert scope["source_wrapper"]["prediction_boundary"] == "existing_matrix_program"
    assert {op.local_shape[1:] for op in projections} == {(3584, 7168), (7168, 3584)}


def test_default_and_source_programs_retain_original_contracts():
    case, ops = declared()
    assert matrix_wrappers.apply_matrix_wrapper(ops, None, case.runtime) == ops
    op = next(op for op in ops if op.recipe == "latent_projection")
    kernel = replace(op, source_boundary="kernel")
    already_bound = replace(op, metadata=op.metadata + (("implementation_binding", {
        "implementation": "independent_source_program"}),))
    for protected in (kernel, already_bound):
        assert matrix_wrappers.apply_matrix_wrapper(
            (protected,), selected(), replace(case.runtime, graph_mode="eager")) == (protected,)


def test_real_event_is_a_candidate_not_a_layer_binding(profiler_api):
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.structure_mapping import map_structure
    from mindie_profiler.tensor_contracts import observed_contract
    from mindie_sim.structure import export_op_structure

    case, ops = declared()
    projection = next(op for op in ops if op.recipe == "latent_projection")
    wrapped = matrix_wrappers.apply_matrix_wrapper(
        (projection,), selected(), replace(case.runtime, graph_mode="eager"))[0]
    row = next(row for row in captured()
               if observed_contract(row) == predicted_contract(wrapped.to_dict()))
    path = Path(row.source)
    stat = path.stat()
    source = SourceRef("rank_0", 0, "csv", str(path), path.name, None,
                       stat.st_size, stat.st_mtime_ns, "retained_capture", "kernel_details")
    event = replace(row.event, rank_id="rank_0", rank_index=0)
    observation = Observation("rank_0", 0, Window(event.start_ns, event.end_ns),
                              None, [event], source, QueryStats("rank_0", "wrapper_control"))
    result = map_structure([observation], export_op_structure(
        (wrapped,), model=case.model.to_dict(),
        execution={"phase": "prefill", "tokens": 1024}))
    mapping = result["event_mappings"][0]
    assert mapping["status"] == "candidate"
    assert mapping["candidate_node_ids"] == ["rank:0/" + wrapped.op_id]
    assert not mapping["node_ids"]
    assert result["windows"][0]["bound_busy_union_ns"] == 0


def test_source_eager_condition_is_not_inferred_from_missing_graph_metadata():
    case, ops = declared(1, "decode")
    assert any(dict(op.metadata).get("graph_captured") for op in ops)
    # Reproduce an externally supplied graph with no capture evidence. The
    # ordinary target-body frontend now preserves the actual capture decision.
    ops = tuple(replace(op, metadata=tuple((k, v) for k, v in op.metadata
                                          if not k.startswith("graph_"))) for op in ops)
    with pytest.raises(ModelingError, match="requires_explicit_eager_dispatch"):
        matrix_wrappers.apply_matrix_wrapper(ops, selected(), case.runtime)
    projection = next(op for op in ops if op.recipe == "latent_projection")
    captured_op = replace(projection, metadata=projection.metadata + (("graph_captured", True),))
    with pytest.raises(ModelingError, match="requires_eager_dispatch"):
        matrix_wrappers.apply_matrix_wrapper(
            (captured_op,), selected(), replace(case.runtime, graph_mode="eager"))


def test_routed_mxfp4_bank_cannot_become_a_single_dense_wrapper():
    case, ops = declared()
    down = next(op for op in ops if op.recipe == "grouped_down")
    original = predicted_contract(down.to_dict())
    scope = comparison_scope(down.to_dict())
    assert original[1][0] == 28 and original[3] == "mxfp4"
    assert scope["prediction_coverage"]["reason"] == "mxfp4_native_execution_unmodeled"
    with pytest.raises(ModelingError, match="source_caller_unsupported"):
        matrix_wrappers.apply_matrix_wrapper(
            (down,), selected("grouped_down"), replace(case.runtime, graph_mode="eager"))
    assert predicted_contract(down.to_dict()) == original


def test_separate_kda_qkv_projections_cannot_invent_three_source_wrapper_calls():
    case, ops = declared()
    qkv = tuple(op for op in ops if op.recipe in {"linear_q", "linear_k", "linear_v"})
    assert {op.recipe for op in qkv} == {"linear_q", "linear_k", "linear_v"}
    with pytest.raises(ModelingError, match="source_caller_unsupported"):
        matrix_wrappers.apply_matrix_wrapper(
            qkv, selected("linear_q", "linear_k", "linear_v"),
            replace(case.runtime, graph_mode="eager"))
    assert all(predicted_contract(op.to_dict())[0] == "matrix" for op in qkv)


@pytest.mark.parametrize("extra,reason", [
    ({"bias": True, "bias_shape": (3584,)}, "requires_bias_free_projection"),
    ({"activation_scale_shape": ()}, "requires_group32_e8m0_scales"),
    ({"weight_format": "mxfp4"}, "requires_mxfp8_operands"),
    ({"matrix_groups": 2}, "requires_single_dense_matrix"),
    ({"expected_rows": 512}, "requires_all_recorded_rows_active"),
])
def test_ineligible_projection_does_not_gain_a_native_contract(extra, reason):
    case, ops = declared()
    op = next(op for op in ops if op.recipe == "latent_projection")
    changed = replace(op, metadata=tuple({**dict(op.metadata), **extra}.items()))
    with pytest.raises(ModelingError, match=reason):
        matrix_wrappers.apply_matrix_wrapper(
            (changed,), selected(), replace(case.runtime, graph_mode="eager"))
