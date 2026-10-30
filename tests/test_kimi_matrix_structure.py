"""Independent Kimi MX declarations checked against retained native interfaces."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure
from mindie_sim.structure_native_contracts import grouped_situ_contract


ROOT = Path(__file__).resolve().parents[1]


def case():
    base = parse_case(json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text()))
    quant = replace(
        base.model.quant, weight_format="mxfp4", expert_weight_format="mxfp4",
        activation_format="fp8_e4m3", activation_group_size=32,
        activation_scale_format="fp8_e8m0", activation_scale_pack=2, exclusions=(),
    )
    return replace(base, model=replace(base.model, quant=quant))


def captured(kind):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows

    path = Path(mindie_profiler.__file__).resolve().parents[1] / \
        "tests/fixtures/fusion_contracts/kimi_matrix_branches.csv"
    if not path.exists():
        pytest.skip("The companion profiler Kimi MX source fixture is unavailable.")
    return [row for row in kernel_rows(path, rank_id="rank_0") if row.kind == kind]


def test_grouped_situ_declares_source_boundary_without_claiming_observed_routing(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    graph = export_structure(case(), phase="prefill", tokens=8192, scope="body")
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    groups = [group for group in graph["subgraphs"] if group["role"] == "grouped_gate_up_situ_quant"]
    assert groups
    expected = (
        "grouped_mlp_parameters_active", "GroupedMatmulSituQuant",
        (28, 3584, 3072), ("fp8_e4m3", "mxfp4", "fp8_e4m3"),
    )
    assert all(observed_contract(row) == expected for row in captured("GroupedMatmulSituQuant"))
    canonical = json.loads(json.dumps(expected))
    assert all(group["contract"] == canonical for group in groups)
    for group in groups:
        assert group["coverage"]["kind"] == "complete_subgraph"
        members = [nodes[node_id]["operator"] for node_id in group["node_ids"]]
        assert {op["family"] for op in members} == {"grouped_gemm", "situ", "quantize"}
        assert len([op for op in members if op["family"] == "grouped_gemm"]) == 1
        assert all(op["recipe"] != "grouped_down" for op in members)
        assert group["contract_scope"]["work_assumption"] == \
            "expected_routing_work_is_declared_not_measured"
        assert group["source"]["source_state"] == "independent_local_delivery_overlay"
        assert group["source"]["scope"] == "independent_source_semantics; not_capture_binary_identity"
        work = [nodes[node_id]["physical_work"] for node_id in group["node_ids"]]
        assert any(item["flops"] > 0 for item in work)
        assert sum(item["vector_ops"] for item in work) > 0
        assert "resource_boundary" not in group
        assert group["contract_scope"]["prediction_coverage"]["kind"] == "partial_program"


def test_grouped_situ_contract_requires_native_precision_and_scale_domain():
    from mindie_sim.prediction.native_fusion import _groups
    from mindie_sim.prediction.operator_analysis import target_operators

    declaration, _ = next(
        (group, members) for group, members in _groups(
            target_operators(case(), phase="prefill", tokens=8))
        if group["role"] == "grouped_gate_up_situ_quant")
    view = declaration["tensor_view"]
    assert grouped_situ_contract(view) is not None
    for key, value in [
        ("weight_dtype", "fp8_e4m3"),
        ("input_dtype", "bf16"),
        ("input_scale_group_size", 64),
        ("output_scale_tail", (48, 1)),
        ("intermediate_dtype", "fp32"),
    ]:
        changed = deepcopy(view)
        changed[key] = value
        assert grouped_situ_contract(changed) is None
    missing = deepcopy(view)
    missing["activation"]["activation_linear_beta"] = None
    assert grouped_situ_contract(missing) is None


def test_dense_situ_and_mxfp4_matrices_share_declared_conversion_without_merging_boundaries(
    profiler_api,
):
    from mindie_profiler.tensor_contracts import observed_contract

    graph = export_structure(case(), phase="prefill", tokens=8192, scope="body")
    contracts = [node["contract"] for node in graph["nodes"]] + [
        group["contract"] for group in graph["subgraphs"]]
    for kind in ("QuantBatchMatmulV4", "SituMxQuant"):
        for row in captured(kind):
            expected = json.loads(json.dumps(observed_contract(row)))
            assert expected is not None
            assert expected in contracts


def test_selected_mxfp4_native_boundaries_produce_reference_costs_without_error_scores(
    profiler_api,
):
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.structure_mapping import map_structure
    from mindie_sim.cohort_comparison import compare_cohorts

    base = case()
    # One independently declared Transformer block exercises both routed and
    # shared FFN programs without replaying the full model for a consumer test.
    block = next(plan for plan in base.model.layer_plans if plan.ffn == "moe")
    base = replace(base, model=replace(
        base.model, layers=1, body_layers=1,
        layer_plans=(replace(block, index=0, kv_owner_layer_id=0),)))
    kernels = {**base.runtime.kernels, "lowerings": {
        **base.runtime.kernels.get("lowerings", {}),
        "native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["grouped_gate_up_situ_quant"],
                        "traffic_profile": "preserve_workspaces"},
        },
    }}
    selected = replace(base, runtime=replace(
        base.runtime, kernels=kernels, pp_layer_partition=(1,)))
    rows = [captured(kind)[0] for kind in ("GroupedMatmulSituQuant", "QuantBatchMatmulV4")]
    events = [row.event for row in rows]
    source = Path(rows[0].source)
    stat = source.stat()
    window = Window(min(event.start_ns for event in events),
                    max(event.end_ns for event in events), step=0)
    observation = Observation(
        events[0].rank_id, 0, window, None, events,
        SourceRef(
            rank_id=events[0].rank_id, rank_index=0, kind="csv", path=str(source),
            basename=source.name, sha256=None, size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns, schema_version="retained_capture",
            schema_variant="op_summary"),
        QueryStats(events[0].rank_id, "retained_capture"),
    )
    structure = export_structure(selected, phase="prefill", tokens=8192, scope="body")
    mapping = map_structure([observation], structure, mapping_limit=0)
    assert sum(row["event_count"] for row in mapping["windows"][0]["cohorts"]) == 2
    result = compare_cohorts(selected, mapping)
    compared = result["windows"][0]["cohorts"]
    assert {row["target_kind"] for row in compared} == {"node", "subgraph"}
    for row in compared:
        assert row["status"] == "reference_only"
        assert row["reason"] == "mxfp4_native_execution_unmodeled"
        assert row["comparison"] is None
        assert row["prediction"]["single_device_program"]
        assert row["prediction"]["per_call_ms"] > 0
        assert row["prediction"]["physical_work_per_instance"]["launches"] == 1
