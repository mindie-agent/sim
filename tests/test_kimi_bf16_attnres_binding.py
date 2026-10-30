"""Full current BF16 caller, including padding and live prefix bank edges."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_operators


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples/cases/high-concurrency-20261030/kimi-variants"


@pytest.mark.parametrize("batch", (1, 32))
def test_complete_bf16_caller_replaces_split_residual_and_retains_prefix(batch):
    raw = json.loads((CASES / f"a5_kimi_tp8_e16_bf16_c{batch}_p4k.json").read_text())
    ops = target_operators(raw, phase="decode", tokens=1, sequences=batch,
                           context=4096, scope="body")
    by_id = {op.op_id: op for op in ops}
    points = [op for op in ops if op.recipe == "attn_res_fwd"]
    assert len(points) == 187
    local = (batch + 7) // 8
    for index, point in enumerate(points[:-1]):
        layer, ffn = divmod(index, 2)
        view = dict(point.metadata)["attn_res_view"]
        assert point.local_shape == (local, 7168)
        assert view["valid_blocks"] == (layer + 11) // 12 + int(ffn and layer % 12 == 0)
        assert view["block_write_idx"] == (layer // 12 if not ffn and layer % 12 == 0 else -1)
        assert view["bank_capacity"] == 8
        assert view["block_token_stride"] == 8 * 7168
        assert not view["optimize_prefill"]
        if not ffn:
            assert by_id[f"s0.l{layer}.attention.sp_gather"].deps == (point.op_id,)
    final = points[-1]
    expected_banks = tuple(f"s0.l{layer}.attention.attn_res_fwd" for layer in range(0, 93, 12))
    assert dict(final.metadata)["attn_res_bank_ids"] == expected_banks
    assert set(expected_banks) <= set(final.deps)
    assert by_id["sp_output_gather"].deps == (final.op_id,)
    assert not any(op.op_id.endswith((".norm1", ".norm2")) for op in ops)
    assert not any(op.recipe in {"residual_add", "attention_residual_scores"} for op in ops)

    # Run the complete prefix DAG with all cut attention/FFN values arriving
    # late. This is a residual-only control, never a claimed full-body span.
    ids = {op.op_id for op in points}
    cut_inputs = {dep: 13 for op in points for dep in op.deps if dep not in ids}
    report = graph_costs(raw, points, external_inputs=cut_inputs,
                         input_ready_ms=7, isolated=False, trace_limit=10000)
    assert all(row["ready_ms"] >= 13 for row in report["timeline"])
    assert all(row["start_ms"] >= 13 for row in report["events"] if row["demand"])
    execution = report["execution"]
    assert execution["live_tasks"] == execution["live_programs"] == 0
    assert not any(execution["live_allocated"].values())
    isolated = operator_costs(raw, points, input_ready_ms=7)
    assert isolated["operators"][-1]["duration_ms"] > isolated["operators"][0]["duration_ms"]
    for row in isolated["operators"]:
        view = row["operator"]["metadata"]["attn_res_view"]
        assert view["used_cores"] == local
        assert not row["service_before_input"]
    assert {row["operator"]["metadata"]["attn_res_view"]["resident"]
            for row in isolated["operators"]} == {True, False}

    legacy = deepcopy(raw)
    del legacy["runtime"]["kernels"]["lowerings"]["attention_residual"]
    unbound = target_operators(legacy, phase="decode", tokens=1, sequences=batch,
                               context=4096, scope="body")
    assert not any(op.recipe == "attn_res_fwd" for op in unbound)
    assert any(op.recipe == "attention_residual_scores" for op in unbound)
