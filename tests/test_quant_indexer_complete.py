"""Complete QLI work and source boundaries on the ordinary execution engine.

[Role] Verify QuantLightningIndexer V2 work accounting and kernel boundary lowerings.
[Invariants] No double counting of replaced score/topk ops; real producers preserved.
[Timeouts & Locks] Tier 0 (<5s), no external locks.
[Side-Effect Policy] Pure read-only in-memory unit tests.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import graph_costs, operator_costs, target_operators
from mindie_sim.prediction.family_quant_indexer import (
    RECIPE, _check_domain, declare, generic_work,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_structure


def raw_case(selected=True):
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/high-concurrency-20261030/new17/v4flash-a3-tp8-c4.json").read_text())
    raw["model"]["layers"] = 3
    raw["model"]["quant"].update(index_dtype="int8", index_scale_dtype="fp16")
    if selected:
        raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["quant_lightning_indexer"] = {
            "implementation": "cann_quant_lightning_indexer_v2_complete"}
    return raw


def test_one_boundary_replaces_score_and_topk_but_preserves_real_producers():
    raw = raw_case()
    ops = target_operators(raw, phase="decode", tokens=2, sequences=3, context=4096)
    selected = [op for op in ops if op.recipe == "quant_lightning_indexer_v2"]
    assert len(selected) == 1
    op = selected[0]
    pairs = 3 * (4097 // 4 + 4098 // 4)
    assert dict(op.matrix_flops) == {"int8": 2 * pairs * 64 * 128,
                                    "fp16": 2 * pairs * 64 * 16}
    assert op.local_shape == (6, 1, 512) and op.source_boundary == "kernel"
    assert not {"index_scores", "index_topk"} & {item.recipe for item in ops}
    assert len(op.deps) == 3
    by_id = {item.op_id: item for item in ops}
    # The native indexer consumes the BF16 head-weight projection directly
    # (no ``index_weights_fp16`` launch), and dynamic quantization plus cache
    # scatter is one ``indexer_quant_cache`` epilog (no standalone store).
    assert {by_id[dep].recipe for dep in op.deps} == {
        "index_query_quantize", "index_weights", "indexer_quant_cache"}
    assert any(item.recipe == "index_query_hadamard" for item in ops)
    assert any(item.recipe == "index_query_rotary" for item in ops)
    ledger = dict(op.metadata)["quant_indexer_work"]
    assert ledger["reduced_score_workspace_read_write_bytes"] == 8 * pairs
    assert ledger["selection_output_bytes"] == 6 * 512 * 4
    assert ledger["fixpipe_bytes"] == pairs * (64 * 2 + 4)
    assert ledger["aic_read_bytes"] + ledger["aiv_read_bytes"] == op.mte2_bytes
    assert op.mte3_bytes == ledger["weight_workspace_read_write_bytes"] // 2 + ledger["selection_output_bytes"]
    assert op.hbm_bytes == op.mte2_bytes + op.mte3_bytes + 4 * pairs


def test_split_read_pipes_and_native_contract_are_enforced():
    raw = raw_case()
    ops = target_operators(raw, phase="decode", tokens=2, context=131072)
    qli = next(op for op in ops if op.recipe == "quant_lightning_indexer_v2")
    ext = {dep: 0 for dep in qli.deps}
    base_ms = graph_costs(raw, (qli,), external_inputs=ext)["complete_ms"]
    slow_aic = deepcopy(raw)
    slow_aic.setdefault("machine", {}).setdefault("device", {}).setdefault("rates", {})["mte2_aic"] = 1e6
    assert graph_costs(slow_aic, (qli,), external_inputs=ext)["complete_ms"] > base_ms * 10
    slow_aiv = deepcopy(raw)
    slow_aiv.setdefault("machine", {}).setdefault("device", {}).setdefault("rates", {})["mte2_aiv"] = 1e6
    assert graph_costs(slow_aiv, (qli,), external_inputs=ext)["complete_ms"] > base_ms * 2
    slow_mte1 = deepcopy(raw)
    slow_mte1.setdefault("machine", {}).setdefault("device", {}).setdefault("rates", {})["mte1"] = 1e12
    mte1_res = graph_costs(slow_mte1, (qli,), external_inputs=ext, isolated=False)
    pairs = dict(qli.metadata)["quant_indexer_work"]["query_key_pairs"]
    assert mte1_res["duration_ms"] >= pairs * 64 * 2 / 1e12 * 1000
    assert any(v > 0 for k, v in mte1_res["execution"]["serviced"].items() if k.startswith("mte1"))
    manifest = export_structure(raw, phase="decode", tokens=2, context=4096)
    node = next(n for n in manifest["nodes"] if n["recipe"] == "quant_lightning_indexer_v2")
    assert node["contract"] is not None
    assert node["contract"][0] == "quant_lightning_indexer_active"
    assert node["contract"][1] == "VllmAscendQuantLightningIndexerV2"


def test_explicit_selection_preserves_ordinary_semantic_path_and_late_inputs():
    ordinary = target_operators(raw_case(False), phase="decode", tokens=2, context=4096)
    assert {"index_scores", "index_topk"} <= {op.recipe for op in ordinary}
    raw = raw_case()
    ops = target_operators(raw, phase="decode", tokens=2, context=4096)
    qli = next(op for op in ops if op.recipe == "quant_lightning_indexer_v2")
    result = graph_costs(raw, (qli,), input_ready_ms=7,
                        external_inputs={dep: 7 for dep in qli.deps}, isolated=False)
    assert result["complete_ms"] > 7
    assert all(row["ready_ms"] >= 7 for row in result["timeline"])
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


@pytest.mark.parametrize("field,value", [("index_dtype", "bf16"), ("index_scale_dtype", "fp32")])
def test_explicit_wrong_representation_is_a_visible_domain_error(field, value):
    raw = raw_case()
    raw["model"]["quant"][field] = value
    with pytest.raises(ModelingError, match="quant_lightning_indexer_v2_complete_domain_mismatch"):
        target_operators(raw, phase="decode", tokens=2, context=4096)


def _domain_plan(**overrides):
    values = dict(candidate_filter=False, index_kpool=1, index_heads=64,
                  index_dim=128, index_topk=512, compress_ratio=4)
    values.update(overrides)
    return SimpleNamespace(**values)


def _domain_quant(**overrides):
    values = dict(index_dtype="int8", index_scale_dtype="fp16")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_quant_domain_uses_the_named_arch22_constraints():
    parallel = SimpleNamespace(cp=1, pcp=1)
    for plan in (_domain_plan(compress_ratio=2), _domain_plan(compress_ratio=8),
                 _domain_plan(compress_ratio=128), _domain_plan(index_topk=1024),
                 _domain_plan(index_topk=2048)):
        _check_domain(plan, _domain_quant(), parallel)
    for block_size in (16, 128, 1024):
        _check_domain(_domain_plan(), _domain_quant(), parallel,
                      block_size=block_size)
    for plan, quant, *block in (
            (_domain_plan(index_dim=64), _domain_quant()),
            (_domain_plan(index_heads=32), _domain_quant()),
            (_domain_plan(index_dim=72), _domain_quant()),
            (_domain_plan(index_topk=0), _domain_quant()),
            (_domain_plan(index_topk=4096), _domain_quant()),
            (_domain_plan(compress_ratio=0), _domain_quant()),
            (_domain_plan(compress_ratio=3), _domain_quant()),
            (_domain_plan(compress_ratio=256), _domain_quant()),
            (_domain_plan(index_kpool=4), _domain_quant()),
            (_domain_plan(candidate_filter=True), _domain_quant()),
            (_domain_plan(), _domain_quant(index_dtype="bf16")),
            (_domain_plan(), _domain_quant(index_scale_dtype="fp32")),
            (_domain_plan(), _domain_quant(), 120),
            (_domain_plan(), _domain_quant(), 0),
            (_domain_plan(), _domain_quant(), 2048)):
        with pytest.raises(ModelingError,
                           match="quant_lightning_indexer_v2_complete_domain_mismatch"):
            _check_domain(plan, quant, parallel,
                          block_size=block[0] if block else None)


def test_generic_quant_prediction_keeps_unsampled_legal_geometry():
    # D=64/ratio=2/topk=1024 is not the arch22 native binary's domain, but the
    # analytic formula still predicts it and must not be rejected wholesale.
    work = generic_work(spans=((5, 1000),), heads=32, dim=64, topk=1024, ratio=2)
    pairs = sum((1000 + i + 1) // 2 for i in range(5))
    assert work["query_key_pairs"] == pairs
    assert work["qk_int8_flops"] == 2 * pairs * 32 * 64
    assert work["selection_output_bytes"] == 5 * 1024 * 4
    with pytest.raises(ModelingError,
                       match="quant_lightning_indexer_v2_complete_domain_mismatch"):
        _check_domain(_domain_plan(index_dim=64, compress_ratio=2,
                                   index_topk=1024),
                      _domain_quant(), SimpleNamespace(cp=1, pcp=1))


class _FakeBuilder:
    def __init__(self):
        self.ops = []

    def work(self, name, deps, shape, *, flops=0, vector=0, read=0, write=0,
             family="attention", dtype=None, metadata=()):
        op = OpSpec(
            op_id="L0." + name, deps=tuple(deps), recipe=name, family=family,
            phase="decode", stage=0, layer=0, requests=(),
            semantic_shape=shape, local_shape=shape, padded_shape=shape,
            launch_shape=shape, flops=flops, vector_ops=vector,
            hbm_bytes=read + write, mte2_bytes=read, mte3_bytes=write,
            dtype=dtype or "bf16", metadata=tuple(metadata))
        self.ops.append(op)
        return op


def _declare(plan, tokens, context, **kwargs):
    builder = _FakeBuilder()
    declare(builder, deps=("query", "weights", "cache"),
            views=(SimpleNamespace(q_len=tokens, context=context,
                                   query_ranges=()),),
            plan=plan, quant=_domain_quant(),
            parallel=SimpleNamespace(cp=1, pcp=1), **kwargs)
    return builder.ops[-1]


def test_new_native_geometry_predicts_from_the_same_ledger():
    tokens, context, ratio, topk, heads, width = 5, 1000, 2, 2048, 64, 128
    plan = SimpleNamespace(index_kpool=1, candidate_filter=False,
                           compress_ratio=ratio, index_heads=heads,
                           index_dim=width, index_topk=topk)
    op = _declare(plan, tokens, context, block_size=128)
    pairs = sum((context + i + 1) // ratio for i in range(tokens))
    assert op.recipe == RECIPE and op.local_shape == (tokens, 1, topk)
    assert dict(op.matrix_flops) == {"int8": 2 * pairs * heads * width,
                                     "fp16": 2 * pairs * heads * 16}
    ledger = dict(op.metadata)["quant_indexer_work"]
    assert ledger["query_key_pairs"] == pairs
    assert ledger["selection_output_bytes"] == tokens * topk * 4
    assert ledger["key_block_size"] == 128
    assert ledger["key_block_size_source"] == "runtime"
    assert dict(op.metadata)["native_tensor_view"]["compression_ratio"] == ratio
    assert dict(op.metadata)["native_tensor_view"]["key_block_size"] == 128
    assert op.hbm_bytes == op.mte2_bytes + op.mte3_bytes + 4 * pairs


def test_quant_top1_counts_the_maximum_reduction():
    tokens, context, ratio, topk = 5, 100, 4, 1
    plan = SimpleNamespace(index_kpool=1, candidate_filter=False,
                           compress_ratio=ratio, index_heads=64,
                           index_dim=128, index_topk=topk)
    op = _declare(plan, tokens, context, block_size=128)
    pairs = sum((context + i + 1) // ratio for i in range(tokens))
    visible_rows = sum(1 for i in range(tokens) if (context + i + 1) // ratio > 0)
    ledger = dict(op.metadata)["quant_indexer_work"]
    assert ledger["query_key_pairs"] == pairs
    assert ledger["visible_query_rows"] == visible_rows
    assert ledger["topk_comparisons"] == pairs - visible_rows > 0
    generic = generic_work(spans=((tokens, context),), heads=32, dim=128,
                           topk=1, ratio=ratio)
    assert generic["topk_comparisons"] == pairs - visible_rows


def test_short_prefix_sentinel_initialization_applies_vector_invocation():
    raw = raw_case()
    for tokens in (1, 2, 3):
        op = next(o for o in target_operators(raw, phase="prefill", tokens=tokens, context=0)
                  if o.recipe == "quant_lightning_indexer_v2")
        ledger = dict(op.metadata)["quant_indexer_work"]
        assert ledger["query_key_pairs"] == 0
        assert ledger["query_bytes"] == 0
        assert ledger["aic_read_bytes"] == 0
        assert ledger["sentinel_fill_elements"] == tokens * 512
        assert op.mte2_bytes == 0
        assert op.vector_ops == tokens * 512
        res = graph_costs(raw, (op,), external_inputs={d: 0 for d in op.deps}, isolated=False)
        assert res["complete_ms"] * 1000 == pytest.approx(2.5)


def test_weight_preparation_scales_with_query_owners_not_key_blocks():
    raw = raw_case()
    short_op = next(o for o in target_operators(raw, phase="decode", tokens=1, sequences=1, context=4096)
                    if o.recipe == "quant_lightning_indexer_v2")
    long_op = next(o for o in target_operators(raw, phase="decode", tokens=1, sequences=1, context=1048575)
                   if o.recipe == "quant_lightning_indexer_v2")
    short_ledger = dict(short_op.metadata)["quant_indexer_work"]
    long_ledger = dict(long_op.metadata)["quant_indexer_work"]
    assert long_ledger["query_key_pairs"] > short_ledger["query_key_pairs"] * 100
    assert short_ledger["weight_workspace_read_write_bytes"] == long_ledger["weight_workspace_read_write_bytes"]
    assert long_ledger["weight_workspace_read_write_bytes"] == 1 * 2 * 64 * 16 * 2
