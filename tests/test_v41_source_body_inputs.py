"""Source declarations must reach actual complete-graph recipe preparation."""
import json
from pathlib import Path

import pytest

from mindie_sim import target_graphs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case


ROOT = Path(__file__).resolve().parents[1]
CASES = sorted((ROOT / "examples/cases/high-concurrency-20261030/v41-source-conditional").glob("*.json"))


@pytest.mark.parametrize("path", CASES, ids=lambda path: path.stem)
def test_complete_source_variant_preserves_geometry_and_prepares_real_programs(path):
    raw = json.loads(path.read_text())
    selected = raw["runtime"]["metadata"]["source_scenario"]
    case = parse_case(raw)
    assert case.model.layers == 40
    assert case.model.quant.for_category("routed_expert").expert_weight_format == "mxfp4"
    graphs = target_graphs(case, **selected["conditions"]["single_active_rank0"])
    assert len(graphs) == 8
    rank0 = graphs[0]
    attention = [op for op in rank0 if op.recipe == "cannbot_mixed_quant_sparse_flash_mla"]
    assert len(attention) == 40
    assert {op.local_shape[0] for op in attention} == {selected["observed_rank0_geometry"]["query_rows"]}
    assert len([op for op in rank0 if dict(op.metadata).get("fusion_role") == "hc_pre"]) == 80
    assert len([op for op in rank0 if dict(op.metadata).get("fusion_role") == "hc_post"]) == 80
    grouped = next(op for op in rank0 if op.family == "grouped_gemm")
    book = RecipeBook.from_mapping(case.runtime.kernels)
    prepared, _ = book.select(grouped).prepare(grouped, Geometry.from_devices(case.machine.dies[:1]))
    execution = dict(prepared.metadata)["program_execution"]
    assert execution["source"]["selector"].startswith("MXA8W4_NZNK/")
    assert prepared.flops > 0
    for rank in range(1, 8):
        assert not any(op.recipe == "cannbot_mixed_quant_sparse_flash_mla" for op in graphs[rank])
        assert len([op for op in graphs[rank] if op.recipe == "expert_combine"]) == 40
