"""Receive-only EP ranks have expert work, but no local HC invocation."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


ROOT = Path(__file__).resolve().parents[1]


def case():
    raw = json.loads((ROOT / "examples/cases/high-concurrency-20261030/v41-dp8/"
                     "a5_v41_fp8_tp1_dp8_ep8_tpot30ms_c64.json").read_text())
    raw["runtime"]["kernels"].setdefault("lowerings", {})["native_fusion"] = {
        "implementation": "declared_native_fusion", "options": {
            "roles": ["hc_pre", "hc_post"], "traffic_profile": {
                "hc_pre": "hc_pre_m_k_split_175ef4d4", "hc_post": "hc_post_d_split_175ef4d4"}}}
    return parse_case(raw)


def test_full40_nonuniform_hc_graph_keeps_receive_only_expert_work():
    parsed = case()
    graphs = target_graphs(parsed, phase="decode", tokens=1, context=2048,
                           sequences=[8] + [0] * 7, scope="body")
    assert len(graphs) == 8
    assert len([op for op in graphs[0] if dict(op.metadata).get("fusion_role") in {"hc_pre", "hc_post"}]) == 160
    for rank, ops in graphs.items():
        groups = [op for op in ops if op.family == "grouped_gemm"]
        assert groups and len({op.layer for op in groups}) == 40
        assert len([op for op in ops if op.recipe == "expert_combine"]) == 40
        if rank:
            assert not any(dict(op.metadata).get("fusion_role") in {"hc_pre", "hc_post"} for op in ops)
            assert not any(op.family == "attention" for op in ops)

    # Execute the complete first-layer cohort as a bounded physical control;
    # the full 40-layer graph above retains the real catalog geometry.
    layer = parsed.model.layer_plans[:1]
    short = replace(parsed, model=replace(parsed.model, layers=1, body_layers=1, layer_plans=layer),
                    runtime=replace(parsed.runtime, pp_layer_partition=(1,)))
    short_graphs = target_graphs(short, phase="decode", tokens=1, context=2048,
                                 sequences=[8] + [0] * 7, scope="body")
    result = graph_costs(short, short_graphs, input_ready_ms=7,
                         external_inputs={"analysis_input": 7}, isolated=False)
    assert all(row["ready_ms"] >= 7 for row in result["timeline"])
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert not any(abs(v) > 1e-6 for v in result["execution"]["live_allocated"].values())
    assert all(any(row["replica"] == rank and row["family"] == "grouped_gemm"
                   for row in result["timeline"]) for rank in range(8))


def test_active_plain_residual_still_rejects_unmatched_hc_selection():
    parsed = case()
    plain = tuple(replace(layer, hyper_connections=False) for layer in parsed.model.layer_plans)
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=plain))
    with pytest.raises(ModelingError, match="native_fusion_selected_roles_unmatched:hc_post,hc_pre"):
        target_graphs(parsed, phase="decode", tokens=1, sequences=[1] + [0] * 7, scope="body")
