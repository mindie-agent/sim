"""Source-derived QKV preprocessing boundaries and real capture limitations."""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.attention_common import dcp_partial_merge_ops
from mindie_sim.prediction.attention import gqa
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure import export_op_structure, export_structure
from test_prediction_system import case


ROOT = Path(__file__).resolve().parents[1]


def _declared_gqa(*, gate, qk_norm=True, cp=1, pcp=1):
    parsed = parse_case(case())
    plan = replace(parsed.model.layer_plans[0], qk_norm=qk_norm, q_output_gate=gate)
    parallel = replace(parsed.runtime.parallel, cp=cp, pcp=pcp, cp_plan="auto")
    result = gqa(
        prefix="attention", plan=plan, views=(RequestView("r", 3, 0, True, "prefill"),),
        stage=0, deps="projected_input", model=parsed.model, parallel=parallel,
        quant=parsed.model.quant, mode="prefill", padded_tokens=3,
    )
    return parsed, result.ops


def _addresses(layout):
    return {
        layout["offset"] + sum(index * stride for index, stride in zip(indices, layout["strides"]))
        for indices in product(*(range(size) for size in layout["shape"]))
    }


@pytest.mark.parametrize("gate", [False, True])
def test_qkv_group_covers_real_output_copies_and_stops_at_projection(gate):
    parsed, ops = _declared_gqa(gate=gate)
    by_id = {op.op_id: op for op in ops}
    rope = by_id["attention.rope"]
    declaration = dict(rope.metadata)["structure_subgraphs"][0]
    views = declaration["tensor_view"]["input_views"]
    pitch = (4 * (2 if gate else 1) + 2 * 2) * 8
    addresses = [_addresses(layout) for layout in views.values()]
    # The head-interleaved Q/gate and trailing K/V exactly partition projected
    # rows; neither a shape-only split nor a shared Q/gate alias is valid.
    assert set.union(*addresses) == set(range(3 * pitch))
    assert sum(map(len, addresses)) == 3 * pitch
    if gate:
        assert views["gate"]["offset"] == 8
        assert views["query"]["strides"][1] == views["gate"]["strides"][1] == 16
    copies = [op for op in ops if op.recipe == "qkv_split_materialize"]
    expected_elements = 3 * 2 * 8 + (3 * 4 * 8 if gate else 0)
    assert sum(op.mte2_bytes for op in copies) == expected_elements * 2
    assert sum(op.mte3_bytes for op in copies) == expected_elements * 2
    assert sum(op.vector_ops + op.flops for op in copies) == 0
    assert by_id["attention.qkv.value"].op_id in by_id["attention.cache"].deps
    if gate:
        assert by_id["attention.qkv.gate"].op_id in by_id["attention.output_gate"].deps
        assert "attention.qkv" not in by_id["attention.output_gate"].deps

    manifest = export_op_structure(
        ops, model=parsed.model.to_dict(), execution={"phase": "prefill"})
    group = next(row for row in manifest["subgraphs"] if row["role"] == "qkv_norm_rope")
    assert set(group["node_ids"]) == {
        "rank:0/attention.qk_norm", "rank:0/attention.rope",
        *(f"rank:0/{op.op_id}" for op in copies),
    }
    assert group["external_inputs"] == ["rank:0/attention.qkv"]
    assert "rank:0/attention.qkv.value" in group["external_outputs"]
    assert "rank:0/attention.qkv" not in group["node_ids"]
    assert group["contract"] is None
    assert group["coverage"]["required_output_slots"] == 4


def test_qkv_materialized_value_precedes_pcp_gather_without_extending_fusion():
    _, ops = _declared_gqa(gate=True, pcp=2)
    by_id = {op.op_id: op for op in ops}
    gather = by_id["attention.kv_gather"]
    assert set(gather.deps) == {"attention.rope", "attention.qkv.value"}
    assert by_id["attention.cache"].deps == (gather.op_id,)
    group = dict(by_id["attention.rope"].metadata)["structure_subgraphs"][0]
    assert gather.op_id not in group["op_ids"]
    _, plain = _declared_gqa(gate=False, qk_norm=False)
    assert not any(op.recipe == "qkv_split_materialize" for op in plain)
    assert not any(dict(op.metadata).get("structure_subgraphs") for op in plain)


def test_dcp_attention_update_describes_merge_without_claiming_collective():
    ops, last = dcp_partial_merge_ops(
        prefix="attention", view=RequestView("r", 3, 17, True, "decode"),
        cp=2, local_heads=4, v_dim=8, stage=0, layer=0, mode="decode",
        deps=("partial_attention",), total_pairs=60, local_pairs=32,
        total_rows=20, local_rows=12, interleave=4,
    )
    communication, update = ops
    metadata = dict(update.metadata)
    assert last == update.op_id and update.deps == (communication.op_id,)
    assert communication.link_bytes > 0 and update.link_bytes == 0
    assert not dict(communication.metadata).get("native_implementation")
    assert metadata["native_tensor_view"]["partial_output_shapes"] == ((12, 8),) * 2
    assert metadata["native_tensor_view"]["partial_lse_shapes"] == ((12,),) * 2
    assert metadata["native_tensor_view"]["output_shape"] == (12, 8)
    assert metadata["native_implementation"]["conditions"]["collective_excluded"]
    assert update.vector_ops > 0 and update.hbm_bytes > 0


def test_real_qwen_capture_missing_outputs_does_not_become_a_fusion_contract(profiler_api):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.tensor_contracts import observed_contract, observed_contract_diagnostic

    fixture = (Path(mindie_profiler.__file__).resolve().parents[1] /
               "tests/fixtures/fusion_contracts/a5_qwen38.csv")
    if not fixture.is_file():
        pytest.skip("retained A5 fusion fixture is not available")
    row = next(row for row in kernel_rows(fixture)
               if row.kind == "split_qkv_rmsnorm_mrope_kernel" and row.input_shapes)
    assert len(row.output_shapes) == 1
    assert observed_contract(row) is None
    diagnostic = observed_contract_diagnostic(row)
    assert diagnostic["required_output_slots"] == 4
    assert diagnostic["observed_output_slots"] == 1

    raw = json.loads((ROOT / "examples/cases/qwen38-a5.json").read_text())
    manifest = export_structure(raw, phase="prefill", tokens=16384)
    groups = [group for group in manifest["subgraphs"] if group["role"] == "qkv_norm_rope"]
    assert groups
    nodes = {node["node_id"]: node for node in manifest["nodes"]}
    for group in groups:
        assert group["contract"] is None
        assert {nodes[node_id]["family"] for node_id in group["node_ids"]} == {
            "rmsnorm", "rotary", "tensor_copy",
        }
        assert all(nodes[node_id]["block"] == "attention" for node_id in group["node_ids"])
        assert all(nodes[node_id]["family"] == "qkv_projection"
                   for node_id in group["external_inputs"])
