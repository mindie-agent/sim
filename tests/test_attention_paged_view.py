"""Independently declared paged storage preserves geometry without pricing capacity."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case


ROOT = Path(__file__).resolve().parents[1]


def case(name, **policy):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    raw["runtime"]["attention_policy"] = policy
    return parse_case(raw)


@pytest.mark.parametrize("name,q,heads,kv_heads", [
    ("qwen35-a3-tp4", 2, 4, 1),
    ("qwen36-a3-tp4", 1, 6, 1),
    ("qwen36-a3-tp4", 4, 6, 1),
    ("qwen38-a5", 1, 24, 4),
    ("qwen38-a5", 4, 24, 4),
    ("qwen35-a3-tp4", 8192, 4, 1),
])
def test_paged_views_keep_local_kv_geometry_and_do_not_change_work(name, q, heads, kv_heads):
    phase = "prefill" if q == 8192 else "decode"
    kwargs = dict(phase=phase, tokens=q, context=16384, sequences=2)
    original = target_operators(case(name), **kwargs)
    selected = target_operators(case(name, kv_storage="paged"), **kwargs)
    assert tuple(replace(op, metadata=()) for op in selected) == tuple(
        replace(op, metadata=()) for op in original)
    attention = [op for op in selected if op.family == "attention"]
    assert attention
    for op in attention:
        view = dict(op.metadata)["attention_tensor_view"]
        assert view["query_shape"] == view["output_shape"] == (2 * q, heads, 256)
        assert view["key_tail_shape"] == view["value_tail_shape"] == (128, kv_heads * 256)
        assert view["batch_count"] == 2
        assert view["kv_storage"] == "paged"


def test_page_tail_and_actual_batch_follow_declared_storage_and_fusion():
    base = case("qwen38-a5", kv_storage="paged")
    separate = replace(base, runtime=replace(
        base.runtime, kv_block_size=64,
        attention_policy={"kv_storage": "paged", "fused_batch": False}))
    ops = target_operators(separate, phase="decode", tokens=2, context=2048, sequences=3)
    attention = [op for op in ops if op.family == "attention"]
    for op in attention:
        view = dict(op.metadata)["attention_tensor_view"]
        assert view["query_shape"] == (2, 24, 256)
        assert view["key_tail_shape"] == (64, 1024)
        assert view["batch_count"] == 1
    assert len(attention) == 3 * sum(
        layer.attention == "gqa" for layer in base.model.layer_plans)


def test_unmodeled_context_partition_does_not_claim_an_unsharded_paged_abi():
    base = case("qwen38-a5", kv_storage="paged")
    sharded = replace(base, runtime=replace(
        base.runtime, parallel=replace(base.runtime.parallel, cp=2)))
    ops = target_operators(sharded, phase="decode", tokens=1, context=2048)
    attention = [op for op in ops if op.family == "attention"]
    assert attention
    assert all("attention_tensor_view" not in dict(op.metadata) for op in attention)
