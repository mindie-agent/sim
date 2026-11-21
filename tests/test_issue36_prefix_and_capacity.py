"""Speculative cache geometry and attention operator work.

[Role]
  Check per-layer draft context attribution, MLA QK/PV width splits and
  speculative query width in cache capacity.

[Invariants]
  * Draft bytes appear only when a proposal width or a live DSpark spec exists.
  * Each draft context op names its producing layer; the shared context norm
    stays outside the per-layer loop.
  * MLA QK work is proportional to its QK width, not one half of the matrix.

[Timeouts & Locks]
  Pure in-memory arithmetic on two-to-three-layer fixtures; no I/O, waits,
  subprocesses or locks.

[Side-Effect Policy]
  Builds immutable cases and private ledgers in memory.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from _support.builders import dense_case, dense_layer, dense_model, linear_model
from mindie_sim.catalog.cache_ownership import draft_cache_components
from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.prediction.batch_prefix_attention import BatchPrefixAttention
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.draft_context import prepare
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.operators import OpSpec


BLOCK = 128


def dspark_draft() -> DraftSpec:
    return DraftSpec(
        kind="dspark", layers=2, hidden_size=32, intermediate_size=64,
        num_heads=4, num_kv_heads=2, head_dim=16, q_width=64, kv_width=32,
        attention_kind="gqa", block_size=5, markov_rank=16, vocab_size=128)


def context_draft() -> DraftSpec:
    return replace(
        dspark_draft(), attention_kind="mla", q_lora_rank=32,
        kv_lora_rank=64, qk_rope_head_dim=8)


def test_draft_context_attributes_every_layer_op():
    draft = context_draft()
    model = replace(dense_model(layers=1),
                    auxiliary=AuxiliarySpec(dspark=draft))
    projection = OpSpec("proj", "gemm", "draft", 0, -1, (),
                        (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1))
    ops = []
    prepare(ops, projection=projection, model=model, draft=draft, rows=4,
            query_rows=3, prefix="d:", stage=0, requests=())
    attributed = {op.op_id: op.layer for op in ops}
    # One fused V3 prolog plus one paged scatter per draft layer; the five
    # fragmented eager producers are gone.
    for layer in range(draft.layers):
        for suffix in ("mla_prolog_v3", "kv_scatter"):
            assert attributed[f"d:context_kv{layer}.{suffix}"] == layer
    assert not any("qkv_a" in op_id or "kv_copy" in op_id for op_id in attributed)
    assert attributed["d:context_norm"] == -1


def _stage_work(shape):
    op = OpSpec("attn", "attention", "decode", 0, 0, (), shape, shape, shape,
                shape, metadata=(("prefix_read_reuse_applied", True),))
    context = SimpleNamespace(op=op, work={"cube": 100.0}, setup_ms=0.0,
                              minimum_ms=0.0, resource_names={})
    return {stage.name: dict(stage.work)
            for stage in BatchPrefixAttention().lower(context).stages}


def test_mla_matrix_work_uses_qk_and_pv_head_widths():
    mla = _stage_work((1, 128, 4, 3, 1))
    assert mla["qk"]["cube"] == pytest.approx(75.0)
    assert mla["pv"]["cube"] == pytest.approx(25.0)
    gqa = _stage_work((1, 128, 4, 16))
    assert gqa["qk"]["cube"] == pytest.approx(50.0)
    assert gqa["pv"]["cube"] == pytest.approx(50.0)


def test_capacity_query_width_and_windowed_mtp_draft_geometry():
    kda = replace(
        dense_case(layers=1), model=linear_model("kda", layers=1),
        runtime=replace(dense_case(layers=1).runtime,
                        dspark=DsparkSpec(6, (1.0,) * 6), metadata=()))
    memory = Memory(kda)
    assert memory.query_tokens == 6
    assert memory.cache_bytes(128, query_tokens=6) > memory.cache_bytes(128)

    mtp = DraftSpec(
        kind="mtp", layers=1, hidden_size=32, intermediate_size=64,
        num_heads=4, num_kv_heads=2, head_dim=16, q_width=64, kv_width=32,
        attention_kind="gqa", target_layer_ids=(0,))
    base = dense_case(layers=1)
    plans = (replace(dense_layer(0), window=64),)
    target = replace(dense_model(layers=1), layer_plans=plans)
    mtp = replace(mtp, layer_plans=native_draft_layer_plans(target, mtp))
    model = replace(target, auxiliary=AuxiliarySpec(mtp=mtp))
    drafted = replace(
        base, model=model, runtime=replace(
            base.runtime,
            metadata=(("draft_mode", "mtp"), ("draft_proposal_tokens", 2))))
    plain = replace(
        base, model=replace(model, auxiliary=None),
        runtime=replace(base.runtime, metadata=()))
    expected = draft_cache_components(
        model, drafted.runtime.parallel, draft_kind="mtp",
        sequence_tokens=512, capacity_padded=True, runtime=drafted.runtime,
        query_tokens=Memory(drafted).query_tokens)["draft_kv"]
    delta = (Memory(drafted).cache_bytes(512)[-1]
             - Memory(plain).cache_bytes(512)[-1])
    assert delta == pytest.approx(expected)
    assert delta == BLOCK * (2 * 2 * 16) * 2
