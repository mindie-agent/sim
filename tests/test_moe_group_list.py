"""Count-list conversion stays a separate native call before fused INT8 GMM."""
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs


def case():
    path = Path(__file__).resolve().parents[1] / "examples/cases/longctx-decode-20261030/qwen35-35b-a3-tp4.json"
    raw = json.loads(path.read_text())
    raw["model"]["layers"] = 2
    lower = raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})
    lower["group_list_prefix"] = "vllm_ascend_group_list_prefix"
    lower["native_fusion"] = {"implementation": "declared_native_fusion", "options": {
        "roles": ["grouped_gate_up_activation_quant"]}}
    return raw


@pytest.mark.parametrize("batch", [1, 4])
def test_zero_count_experts_are_scanned_once_and_fusion_waits_for_prefix(batch):
    raw = case()
    ops = target_graphs(raw, phase="decode", tokens=2, sequences=batch,
                        context=4096, scope="body")[0]
    prefixes = [op for op in ops if op.recipe == "expert_group_list_cumsum"]
    assert len(prefixes) == 2
    for scan in prefixes:
        # Only 16 or 64 assignments: the ABI still contains all 256 experts.
        assert scan.local_shape == (256,)
        assert scan.hbm_bytes == 2 * 256 * 8 + 8
        assert dict(scan.metadata)["logical_integer_adds"] == 255
        fused = next(op for op in ops if op.layer == scan.layer
                     and dict(op.metadata).get("fusion_role") == "grouped_gate_up_activation_quant")
        assert scan.op_id in fused.deps
        assert scan.op_id not in dict(fused.metadata)["fusion_semantic_ids"]
        down = next(op for op in ops if op.layer == scan.layer
                    and op.recipe == "balanced_grouped_down")
        assert "group_list_producer" not in dict(down.metadata)
        result = operator_costs(raw, [scan], input_ready_ms=7)["operators"][0]
        assert sum(result["service_before_input"].values()) == 0
        assert result["serviced"]["memory@r0:s0"] == 4104
        assert not any("vector" in key for key in result["serviced"])
        assert result["live_tasks"] == 0
    report = graph_costs(raw, {0: ops}, external_inputs={"analysis_input": 0}, isolated=False)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_declared_aicpu_response_is_inclusive_once_per_conversion():
    raw = case()
    raw["runtime"]["kernels"]["service_regimes"] = [{
        "id": "test-aicpu-scan-service", "match": {"recipe": "expert_group_list_cumsum"},
        "response": {"fixed_us": 40},
        "provenance": {"kind": "test-service-control-not-measurement"}}]
    ops = target_graphs(raw, phase="decode", tokens=2, sequences=1,
                        context=4096, scope="body")[0]
    scan = next(op for op in ops if op.recipe == "expert_group_list_cumsum")
    report = operator_costs(raw, [scan])["operators"][0]
    assert report["duration_ms"] == pytest.approx(.040)
    assert report["serviced"]["memory@r0:s0"] == 4104


def test_unselected_caller_keeps_original_graph():
    raw = case()
    del raw["runtime"]["kernels"]["lowerings"]["group_list_prefix"]
    ops = target_graphs(raw, phase="decode", tokens=2, sequences=1,
                        context=4096, scope="body")[0]
    assert not any(op.recipe == "expert_group_list_cumsum" for op in ops)
