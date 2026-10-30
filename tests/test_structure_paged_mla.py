"""Source kernel structure uses the same physical work as actual execution."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.paged_mla import build, IMPLEMENTATION
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_op_structure, export_structure
from mindie_sim.structure_contracts import predicted_contract
from mindie_sim.structure_physical import node_physical, subgraph_physical


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples/cases/high-concurrency-20261030/kimi-source-paged-host"


def case(batch, context):
    suffix = "4k" if context == 4096 else "16k"
    return json.loads((CASES / f"a5_kimi_tp8_e16_bf16_c{batch}_p{suffix}_source.json").read_text())


def declared(raw, batch, context):
    parsed = parse_case(raw)
    op = next(op for op in target_graphs(
        parsed, phase="decode", scope="body", tokens=1, sequences=batch,
        context=context, resource_fusion=False)[0] if op.recipe == IMPLEMENTATION)
    return parsed, op, Geometry.from_devices(parsed.machine.dies[:parsed.runtime.parallel.tp])


def expected_contract(batch, page=128):
    query = ((batch, 16, 1, 512), "bf16", "ND")
    rope = ((batch, 16, 1, 64), "bf16", "ND")
    cache = ((1, page, 512), "bf16", "ND")
    rope_cache = ((1, page, 64), "bf16", "ND")
    output = ((16, batch, 1, 512), "bf16", "ND")
    return ("attention_paged_mla_qk_rope", "FusedInferAttentionScore",
            (query, rope), (cache, cache, rope_cache), output, batch)


@pytest.mark.parametrize("batch,context", [(1, 4096), (32, 16384)])
def test_full_ordinary_frontend_export_prepares_source_work_without_timing(batch, context, monkeypatch):
    from mindie_sim.prediction.engine import Engine

    def no_timing(*args, **kwargs):
        raise AssertionError("structure export must not run the timing engine")

    raw = case(batch, context)
    _, op, geometry = declared(raw, batch, context)
    plan = build(op, geometry)
    monkeypatch.setattr(Engine, "advance", no_timing)
    manifest = export_structure(raw, phase="decode", scope="body", tokens=1,
                                sequences=batch, context=context)
    attention = [row for row in manifest["nodes"] if row["recipe"] == IMPLEMENTATION]
    assert len(attention) == 24  # Full 93-layer checkpoint, not a sliced model.
    for row in attention:
        assert row["contract"] == json.loads(json.dumps(expected_contract(batch)))
        work = row["physical_work"]
        assert work["flops"] == sum(plan.work.matrix.values()) > 0
        assert work["matrix_flops"] == plan.work.matrix
        assert work["hbm_bytes"] == plan.work.resources["hbm"] > 0
        assert work["vector_ops"] == plan.work.resources["vector"] > 0
        assert work["mte3_bytes"] == plan.work.resources["mte3"] > 0
        assert work["resource_work"] == {key: value for key, value in plan.work.resources.items()
                                          if key not in {"hbm", "mte3", "vector"}}
        assert work["launches"] == 1
        assert row["physical_work_basis"] == "prepared_source_plan"
        assert row["operator"]["metadata"]["implementation_binding"]["capture_dispatch"] == "not_established"
        assert not row["contract_scope"]["occurrence_identity"]
        assert "captured_binary_identity" in row["contract_scope"]["not_compared"]
    assert len({row["physical_signature"] for row in attention}) == 1


def test_exported_work_matches_actual_engine_and_keeps_directional_resources():
    raw = case(1, 4096)
    parsed, op, geometry = declared(raw, 1, 4096)
    report = export_op_structure([op], model=parsed.model.to_dict(), execution={"phase": "decode"},
                                kernels=parsed.runtime.kernels, resource_geometries={(0, 0): geometry})
    node = report["nodes"][0]
    actual = operator_costs(parsed, [op], input_ready_ms=7)["operators"][0]
    for field in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops", "link_bytes"):
        assert node["physical_work"][field] == actual["operator"][field]
    assert node["physical_work"]["resource_work"] == actual["operator"]["metadata"]["resource_work"]
    assert {"mte1", "fixpipe", "mte2_aic", "mte2_aiv"} <= node["physical_work"]["resource_work"].keys()
    assert not actual["service_before_input"] and actual["live_tasks"] == 0
    assert not any(actual["remaining_buffers"].values())
    # Source-only export still works without hardware, but cannot claim the
    # unprepared kernel declaration is the source planner's resource ledger.
    unprepared = export_op_structure([op], model={}, execution={"phase": "decode"},
                                    kernels=parsed.runtime.kernels)["nodes"][0]
    assert unprepared["contract"] == node["contract"]
    assert unprepared["physical_work_basis"] == "source_declaration_without_resource_geometry"


def test_context_dispatch_and_page_changes_preserve_their_distinct_contract_axes():
    _, op, geometry = declared(case(1, 4096), 1, 4096)
    base = predicted_contract(op.to_dict())
    assert base == expected_contract(1)
    work = build(op, geometry).work
    changed_meta = deepcopy(dict(op.metadata))
    changed_meta["paged_mla_geometry"].update(contexts=(8192,), dispatch=None)
    longer = replace(op, metadata=tuple(changed_meta.items()))
    assert predicted_contract(longer.to_dict()) == base  # No sequence-length values in tensor ABI.
    assert build(longer, geometry).work.resources["hbm"] != work.resources["hbm"]
    assert node_physical(longer)["physical_signature"] != node_physical(op)["physical_signature"]
    changed_meta["paged_mla_geometry"]["page_size"] = 64
    smaller_pages = replace(op, metadata=tuple(changed_meta.items()))
    assert predicted_contract(smaller_pages.to_dict()) == expected_contract(1, 64)
    # Axis order, rotary width and cache sharing are semantic ABI contracts,
    # not dimensions to flatten until they happen to match a capture.
    for field, value in (("output_shape", (1, 16, 1, 512)),
                         ("query_rope_shape", (1, 16, 1, 32)),
                         ("cache_head_shape", (16, 512)), ("cache_dtype", "fp16")):
        bad = deepcopy(dict(op.metadata))
        bad["attention_tensor_view"][field] = value
        assert predicted_contract(replace(op, metadata=tuple(bad.items())).to_dict()) is None


def test_directional_work_survives_group_aggregation():
    raw = case(1, 4096)
    parsed, op, geometry = declared(raw, 1, 4096)
    second = replace(op, op_id="other", deps=(op.op_id,))
    nodes = export_op_structure([op, second], model={}, execution={"phase": "decode"},
        kernels=parsed.runtime.kernels, resource_geometries={(0, 0): geometry})["nodes"]
    group = {"node_ids": [node["node_id"] for node in nodes],
             "external_outputs": [nodes[-1]["node_id"]], "implementation": {}, "coverage": "complete"}
    total = subgraph_physical({node["node_id"]: node for node in nodes}, group)["physical_work"]
    assert total["resource_work"] == {key: 2*value for key, value in nodes[0]["physical_work"]["resource_work"].items()}
    assert total["launches"] == 2


def test_declared_contract_matches_retained_a5_full31_active_tensor_abi(profiler_api):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.tensor_contracts import observed_contract

    path = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/attention_cache_contracts/a5_kimi_mla31.csv"
    if not path.is_file():
        pytest.skip("retained A5 MLA full31 capture fixture unavailable")
    for row, batch in zip(kernel_rows(path), (1, 1, 32)):
        assert observed_contract(row) == expected_contract(batch)
