"""Explicit target-plus-draft prefill at the public pure-device boundary."""
import json
from pathlib import Path

from mindie_sim import graph_costs, target_graphs, target_operators
from mindie_sim.__main__ import parser, run
from test_draft_and_media import draft_case


def test_target_and_one_mtp_layer_preserve_dependencies_without_serving_commit(tmp_path):
    case = draft_case()
    target = target_graphs(case, phase="prefill", tokens=4)
    prepared = target_graphs(case, phase="prefill", tokens=4, draft_prepare=True)
    assert not any(op.op_id.startswith("draft_prefill.") for op in target[0])
    assert len([op for op in prepared[0] if op.recipe == "qkv_projection"]) == 3
    assert len([op for op in target[0] if op.recipe == "qkv_projection"]) == 2
    assert not any(op.family in {"sampling", "token_commit"} for op in prepared[0])
    result = graph_costs(case, prepared)
    baseline = graph_costs(case, target)
    assert result["duration_ms"] > baseline["duration_ms"]
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert any(op.op_id.startswith("draft_prefill.") for op in target_operators(
        case, phase="prefill", tokens=4, scope="model", draft_prepare=True))
    source = tmp_path / "declared-mtp.json"
    source.write_text(json.dumps(case.to_dict()))
    cli = run(parser().parse_args(["graph", str(source), "--phase", "prefill",
                                   "--tokens", "4", "--draft-prepare"]))
    assert cli["duration_ms"] == result["duration_ms"]
    assert cli["draft_prepare"] is True


def test_flash_mtp1_declaration_emits_43_target_and_one_draft_attention():
    source = Path(__file__).parents[1] / (
        "tests/fixtures/verification/v4flash-mtp-case.json")
    raw = json.loads(source.read_text())
    ops = target_graphs(raw, phase="prefill", tokens=4, draft_prepare=True)[0]
    attention = [op for op in ops if op.family == "attention"]
    assert len(attention) == 44
    assert sum(op.op_id.startswith("draft_prefill.") for op in attention) == 1
    assert not any(dep in {"token_commit", "sampling"} for op in ops for dep in op.deps)
