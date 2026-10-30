"""Independent paged declarations against original mixed-step A3/A5 tensors."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure_contracts import comparison_scope, predicted_contract


ROOT = Path(__file__).resolve().parents[1]


def declared(name, tokens, *, phase="decode", context=8192, sequences=1, paged=True):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    if paged:
        raw["runtime"].setdefault("attention_policy", {})["kv_storage"] = "paged"
    return target_operators(parse_case(raw), phase=phase, tokens=tokens,
                            context=context, sequences=sequences)


def observed(capture, kind, tokens, profiler_api, *, paged=True):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.tensor_contracts import observed_contract

    path = (Path(mindie_profiler.__file__).resolve().parents[1]
            / "tests/fixtures/mixed_step_body" / (capture + ".csv"))
    if not path.is_file():
        pytest.skip(f"retained mixed-step capture is unavailable: {path}")
    return next(observed_contract(row) for row in kernel_rows(path)
                if row.kind == kind and row.input_shapes[0][0] == tokens
                and (kind != "FusedInferAttentionScore"
                     or (observed_contract(row)[0] == "attention_paged_active_qkv") == paged))


@pytest.mark.parametrize("case,capture,tokens,phase", [
    ("qwen35-a3-tp4", "a3_qwen35", 2, "decode"),
    ("qwen36-a3-tp4", "a3_qwen36", 1, "decode"),
    ("qwen36-a3-tp4", "a3_qwen36", 4, "decode"),
    ("qwen38-a5", "a5_qwen38", 1, "decode"),
    ("qwen38-a5", "a5_qwen38", 4, "decode"),
    ("qwen35-a3-tp4", "a3_qwen35", 8192, "prefill"),
    ("qwen36-a3-tp4", "a3_qwen36", 8192, "prefill"),
])
def test_real_paged_fia_matches_each_explicit_query_geometry(case, capture, tokens, phase, profiler_api):
    actual = observed(capture, "FusedInferAttentionScore", tokens, profiler_api)
    attention = [op for op in declared(case, tokens, phase=phase) if op.family == "attention"]
    assert len(attention) > 1
    assert all(predicted_contract(op.to_dict()) == actual for op in attention)
    assert len({op.layer for op in attention}) == len(attention)
    scope = comparison_scope(attention[0].to_dict())
    assert {"phase", "context_length", "verify_or_draft_role"} <= set(scope["not_compared"])
    assert not scope["occurrence_identity"]
    wrong_workload = declared(case, tokens + 1, phase=phase)
    assert all(predicted_contract(op.to_dict()) != actual for op in wrong_workload
               if op.family == "attention")


def test_paged_projection_does_not_invent_storage_or_alias_a_source_call():
    selected = next(op for op in declared("qwen36-a3-tp4", 4) if op.family == "attention")
    expected = predicted_contract(selected.to_dict())
    assert expected[0] == "attention_paged_active_qkv"
    assert predicted_contract(replace(selected, source_boundary="call").to_dict()) is None
    default = declared("qwen36-a3-tp4", 4, paged=False)
    assert all(predicted_contract(op.to_dict()) is None for op in default if op.family == "attention")
    for key, value in [
        ("key_tail_shape", (64, 256)), ("value_tail_shape", (128, 512)),
        ("batch_count", 2), ("output_dtype", "fp16"),
        ("logical_layout", "BSH"), ("kv_storage", "contiguous"),
        ("output_shape", (4, 6, 128)),
    ]:
        op = deepcopy(selected.to_dict())
        op["metadata"]["attention_tensor_view"][key] = value
        assert predicted_contract(op) != expected
    incomplete = deepcopy(selected.to_dict())
    incomplete["metadata"]["attention_tensor_view"].pop("key_tail_shape")
    assert predicted_contract(incomplete) is None


@pytest.mark.parametrize("case,capture,tokens", [
    ("qwen35-a3-tp4", "a3_qwen35", 2),
    ("qwen36-a3-tp4", "a3_qwen36", 4),
    ("qwen38-a5", "a5_qwen38", 4),
])
def test_real_recurrent_geometry_needs_decode_declaration(case, capture, tokens, profiler_api):
    actual = observed(capture, "RecurrentGatedDeltaRule", tokens, profiler_api)
    decode = declared(case, tokens)
    recurrent = [op for op in decode if op.family == "recurrent_vector"]
    assert recurrent
    assert all(predicted_contract(op.to_dict()) == actual for op in recurrent)
    alternate_precision = replace(recurrent[0], dtype="fp16").to_dict()
    assert predicted_contract(alternate_precision) is None
    assert comparison_scope(alternate_precision)["contract_support"] == "unsupported_recurrent_gdn_dtype"
    prefill = declared(case, 8192, phase="prefill", context=0)
    assert all(predicted_contract(op.to_dict()) != actual for op in prefill)


def test_independent_recurrent_state_precision_is_not_inferred_from_storage_width():
    from mindie_sim.prediction.attention import AttentionPolicy
    from mindie_sim.prediction.recurrent import linear
    from mindie_sim.schema.workload import RequestView

    raw = json.loads((ROOT / "examples/cases/qwen36-a3-tp4.json").read_text())
    case = parse_case(raw)
    base = next(plan for plan in case.model.layer_plans if plan.attention == "gdn")
    contracts = {}
    for dtype in ("fp32", "bf16", "fp16"):
        result = linear(
            prefix="state_dtype", plan=replace(base, recurrent_state_dtype=dtype),
            views=(RequestView("r", 4, 8192, False, "decode"),), stage=0, deps="input",
            model=case.model, parallel=case.runtime.parallel, quant=case.model.quant,
            mode="decode", padded_tokens=4, policy=AttentionPolicy(), runtime=case.runtime,
        )
        op = next(op for op in result.ops if op.family == "recurrent_vector").to_dict()
        assert op["metadata"]["state_dtype"] == dtype
        contracts[dtype] = predicted_contract(op)
        if dtype == "fp16":
            assert comparison_scope(op)["contract_support"] == "unsupported_recurrent_gdn_dtype"
        elif dtype == "bf16":
            incomplete = deepcopy(op)
            incomplete["metadata"].pop("state_dtype")
            assert predicted_contract(incomplete) is None
            assert comparison_scope(incomplete)["declared_state_dtype"] is None
    assert contracts["fp32"][-1] == "fp32"
    assert contracts["bf16"][-1] == "bf16"
    assert contracts["fp16"] is None
    assert contracts["fp32"] != contracts["bf16"]


def test_real_paged_and_recurrent_events_form_cohorts_without_layer_or_step_binding(profiler_api):
    import mindie_profiler
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.structure_mapping import map_structure
    from mindie_sim.structure import export_structure

    path = (Path(mindie_profiler.__file__).resolve().parents[1]
            / "tests/fixtures/mixed_step_body/a3_qwen36.csv")
    if not path.is_file():
        pytest.skip(f"retained mixed-step capture is unavailable: {path}")
    rows = [row for row in kernel_rows(path, rank_id="rank_0", rank_index=0)
            if row.kind in {"FusedInferAttentionScore", "RecurrentGatedDeltaRule"}
            and row.input_shapes[0][0] == 4]
    assert len(rows) == 2
    events = [row.event for row in rows]
    stat = path.stat()
    observation = Observation(
        "rank_0", 0, Window(min(row.start_ns for row in rows), max(row.end_ns for row in rows)),
        None, events, SourceRef(
            "rank_0", 0, "csv", str(path), path.name, None, stat.st_size,
            "retained_capture", "kernel_details", mtime_ns=stat.st_mtime_ns),
        QueryStats("rank_0", "mixed_step_contract_regression"),
    )
    raw = json.loads((ROOT / "examples/cases/qwen36-a3-tp4.json").read_text())
    raw["runtime"].setdefault("attention_policy", {})["kv_storage"] = "paged"
    manifest = export_structure(raw, phase="decode", tokens=4, context=8192)
    report = map_structure([observation], manifest)
    assert report["status_counts"] == {"ambiguous": 2}
    window = report["windows"][0]
    assert len(window["cohorts"]) == 2
    assert window["nodes"] == window["layers"] == []
    assert all(group["event_count"] == 1 and group["declared_layer_count"] > 1
               and group["multiplicity"]["steps_in_window"] is None
               for group in window["cohorts"])
    other = export_structure(raw, phase="decode", tokens=1, context=8192)
    assert map_structure([observation], other)["windows"][0]["cohorts"] == []
