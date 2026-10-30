"""Real GLM native interfaces retain missing work across resource fusion."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.cohort_comparison import compare_cohorts
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure
from test_cohort_comparison import _captured_report


ROOT = Path(__file__).resolve().parents[1]
ROLE = "grouped_gate_up_activation_quant"


def _glm_block(*, msd, layers=1):
    case = parse_case(json.loads((ROOT / "examples/cases/glm53-a3-w4a8.json").read_text()))
    block = next(plan for plan in case.model.layer_plans if plan.ffn == "moe" and plan.owns_topk)
    model = replace(case.model, layers=layers, body_layers=layers, layer_plans=tuple(
        replace(block, index=index, kv_owner_layer_id=index, topk_owner_layer_id=index,
                index_k_owner_layer_id=index,
                candidate_source_layer_id=index if block.candidate_source_layer_id >= 0 else -1)
        for index in range(layers)))
    options = {"roles": [ROLE], "traffic_profile": "preserve_workspaces"}
    routing = {}
    machine = case.machine
    if msd:
        options["program_profiles"] = {ROLE: "ascend_w4a8_msd"}
        # This controlled integer histogram and synthetic INT4 rate test the
        # source program boundary; neither is inferred from captured timing.
        routing = {"realizations": [{
            "scope": "body", "layer": index, "phase": "decode", "tokens": 1,
            "expert_rows": [1] * block.top_k + [0] * (block.num_experts - block.top_k),
        } for index in range(layers)]}
        machine = replace(machine, dies=tuple(replace(
            die, compute_peaks=(*die.compute_peaks, ("int4", 1_000_000_000_000.0)))
            for die in machine.dies))
    runtime = replace(case.runtime, pp_layer_partition=(layers,), routing=routing, kernels={
        **case.runtime.kernels, "lowerings": {
            **case.runtime.kernels.get("lowerings", {}),
            "native_fusion": {"implementation": "declared_native_fusion", "options": options},
        },
    })
    return replace(case, model=model, runtime=runtime, machine=machine)


def test_repeated_glm_msd_programs_form_one_real_descriptor_cohort_without_layer_binding(profiler_api):
    case = _glm_block(msd=True, layers=2)
    structure = export_structure(case, phase="decode", tokens=1, scope="body")
    groups = [row for row in structure["subgraphs"] if row["role"] == ROLE]
    assert len(groups) == 2
    assert len({row["physical_signature"] for row in groups}) == 1
    mapping, events = _captured_report(
        case, "a3_glm53_decode.csv", "GroupedMatmulSwigluQuantV2",
        structure=structure, repeat=1)
    assert mapping["status_counts"] == {"ambiguous": 1}
    cohort, = mapping["windows"][0]["cohorts"]
    assert set(cohort["target_ids"]) == {row["subgraph_id"] for row in groups}
    assert cohort["declared_instance_count"] == 2
    assert cohort["declared_layer_count"] == 2
    assert cohort["activity_sum_ns"] == events[0].duration_ns
    assert cohort["occurrence_binding_implied"] is False
    assert cohort["multiplicity"]["steps_in_window"] is None
    row, = compare_cohorts(case, mapping)["windows"][0]["cohorts"]
    assert row["status"] == "compared"
    assert row["prediction"]["per_declared_step_ms"] == 2 * row["prediction"]["per_call_ms"]


@pytest.mark.parametrize("msd", [False, True])
def test_glm_w4a8_single_launch_only_scores_when_source_program_replaces_missing_epilogue(
    profiler_api, msd,
):
    case = _glm_block(msd=msd)
    structure = export_structure(case, phase="decode", tokens=1, scope="body")
    group = next(row for row in structure["subgraphs"] if row["role"] == ROLE)
    assert group["physical_work"]["launches"] == 1
    nodes = {row["node_id"]: row for row in structure["nodes"]}
    gate = next(nodes[key] for key in group["node_ids"] if nodes[key]["family"] == "grouped_gemm")
    assert gate["contract_scope"]["prediction_coverage"]["reason"] == "quantized_epilogue_unmodeled"
    assert gate["event_matchable"] is False
    down = next(row for row in structure["nodes"] if row["recipe"] == "grouped_down")
    assert down["contract_scope"]["prediction_coverage"]["reason"] == "quantized_epilogue_unmodeled"
    mapping, events = _captured_report(
        case, "a3_glm53_decode.csv", "GroupedMatmulSwigluQuantV2",
        structure=structure, repeat=1)
    assert len(mapping["windows"][0]["cohorts"]) == 1
    result = compare_cohorts(case, mapping)
    row = result["windows"][0]["cohorts"][0]
    assert row["target_kind"] == "subgraph"
    assert row["prediction"]["single_device_program"]
    assert row["prediction"]["per_call_ms"] > 0
    assert row["observed"]["activity_sum_ns"] == events[0].duration_ns
    if msd:
        assert row["resource_boundary"]["program_profile"] == "ascend_w4a8_msd"
        assert row["status"] == "compared"
        assert row["comparison"]["per_call_mean"] is not None
    else:
        assert row["status"] == "reference_only"
        assert row["reason"] == "quantized_epilogue_unmodeled"
        assert row["comparison"] is None


@pytest.mark.parametrize("source_down", [False, True])
def test_glm_down_source_program_removes_only_its_implemented_epilogue_gap(
    profiler_api, source_down,
):
    case = _glm_block(msd=True)
    if source_down:
        kernels = {**case.runtime.kernels, "lowerings": {
            **case.runtime.kernels["lowerings"],
            "grouped_gemm": {
                "implementation": "ascend_w4a8_msd_grouped_down",
                "recipes": ["grouped_down", "balanced_grouped_down"],
            },
        }}
        case = replace(case, runtime=replace(case.runtime, kernels=kernels))
    structure = export_structure(case, phase="decode", tokens=1, scope="body")
    recipe = "ascend_w4a8_msd_grouped_down" if source_down else "grouped_down"
    down = next(row for row in structure["nodes"] if row["recipe"] == recipe)
    assert down["physical_work"]["launches"] == 1
    mapping, events = _captured_report(
        case, "a3_glm53_decode.csv", "GroupedMatmul", structure=structure, repeat=1)
    assert len(mapping["windows"][0]["cohorts"]) == 1
    row = compare_cohorts(case, mapping)["windows"][0]["cohorts"][0]
    assert row["target_kind"] == "node"
    assert row["prediction"]["single_device_program"]
    assert row["observed"]["activity_sum_ns"] == events[0].duration_ns
    if source_down:
        assert row["status"] == "compared"
        assert row["comparison"]["per_call_mean"] is not None
        assert "prediction_coverage" not in down["contract_scope"]
        assert down["contract_scope"]["implemented_quantized_epilogue"]["program"] == recipe
        assert down["contract_scope"]["source_program_limitations"]
        assert down["physical_work"]["matrix_flops"] == {"int4": 4 * 8 * 6144 * 256}
    else:
        assert row["status"] == "reference_only"
        assert row["reason"] == "quantized_epilogue_unmodeled"
        assert row["comparison"] is None
