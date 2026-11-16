"""Issue #36 paired-page, shared-prefix and speculative-width invariants.

[Role]
  Exercise the capacity fixes directly: DSpark page geometry, DCP index-block
  slicing, per-layer draft context attribution, MLA QK/PV width splits, physical
  paired-page deduplication, speculative query width and shared-prefix windows.

[Invariants]
  * Draft bytes appear only when a proposal width or a live DSpark spec exists.
  * Each draft context op names its producing layer; the shared context norm
    stays outside the per-layer loop.
  * MLA QK work is proportional to its QK width, not one half of the matrix.
  * A complete shared prefix page is charged once per (key, stage); remaining
    history and every state page stay per request.
  * Every matching shared-page owner entry is released, not just the first.

[Timeouts & Locks]
  Pure in-memory arithmetic on two-to-three-layer fixtures; no I/O, waits,
  subprocesses or locks.

[Side-Effect Policy]
  Builds immutable cases and private ledgers only; never mutates global state.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from issue34_helpers import (
    dense_case, dense_layer, dense_model, linear_layer, linear_model, view,
)
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.cache_ownership import (
    batch_cache_footprint, draft_cache_components,
)
from mindie_sim.catalog.paired_sfa import page_groups
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.prediction.batch_prefix_attention import BatchPrefixAttention
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.capacity_features import batch_cache_bytes
from mindie_sim.prediction.draft_context import prepare
from mindie_sim.prediction.efficiency import capacity_window
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.model import LayerPlan, RequestSpec, WorkloadSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec


BLOCK = 128
LAYER_STAGE = {0: 0, 1: 0}


def mla_layer(index: int) -> LayerPlan:
    return LayerPlan(
        index=index, attention="mla", ffn="dense", intermediate_size=128,
        num_heads=4, head_dim=16, q_lora_rank=64, kv_lora_rank=64,
        qk_nope_head_dim=64, qk_rope_head_dim=8, v_head_dim=64,
        kv_cache_elements=72, kv_owner_layer_id=index, owns_long_kv=True)


def dspark_draft() -> DraftSpec:
    return DraftSpec(
        kind="dspark", layers=2, hidden_size=32, intermediate_size=64,
        num_heads=4, num_kv_heads=2, head_dim=16, q_width=64, kv_width=32,
        attention_kind="gqa", block_size=5, markov_rank=16, vocab_size=128)


def context_draft() -> DraftSpec:
    return replace(
        dspark_draft(), attention_kind="mla", q_lora_rank=32,
        kv_lora_rank=64, qk_rope_head_dim=8)


def paired_case(**runtime) -> object:
    base = dense_case(layers=2, tp=1)
    model = replace(
        dense_model(layers=2), attention="mla",
        layer_plans=(mla_layer(0), linear_layer(1, "kda")),
        auxiliary=AuxiliarySpec(dspark=dspark_draft()),
        metadata=(("cache_allocator", "paired_sfa"),))
    options = {"prefix_cache_mode": "shared", "metadata": (), **runtime}
    return replace(base, model=model, runtime=replace(base.runtime, **options))


def test_paired_page_groups_use_dspark_width_and_gate_draft_bytes():
    direct = page_groups(
        paired_case(dspark=DsparkSpec(4, (1.0,) * 4)), LAYER_STAGE, 1)[0]
    declared = page_groups(paired_case(
        metadata=(("draft_proposal_tokens", 3), ("draft_mode", "dspark"))),
        LAYER_STAGE, 4)[0]
    empty = page_groups(paired_case(
        metadata=(("draft_proposal_tokens", 0),)), LAYER_STAGE, 1)[0]
    main_page = BLOCK * (64 + 8) * 2
    draft_bytes = BLOCK * (2 * 2 * 16) * 2 * 2
    assert empty["bytes_per_group_page"] == main_page
    assert direct["bytes_per_group_page"] == main_page + draft_bytes
    assert declared["bytes_per_group_page"] == direct["bytes_per_group_page"]
    assert direct["recycling_pages"] == 2 + 4 - 1
    assert empty["recycling_pages"] == 2


def test_dcp_index_sharding_slices_compressed_interleave_blocks():
    layer = LayerPlan(
        index=0, attention="mla", ffn="dense", intermediate_size=128,
        num_heads=4, head_dim=16, q_lora_rank=64, kv_lora_rank=64,
        qk_nope_head_dim=64, qk_rope_head_dim=8, v_head_dim=64,
        kv_cache_elements=72, kv_owner_layer_id=0, owns_long_kv=False,
        compress_ratio=32, owns_index_k=True, index_dim=16, index_heads=4,
        index_topk=8, indexer=True, indexer_type="full")
    model = replace(
        dense_model(layers=1), attention="mla", layer_plans=(layer,),
        metadata=(("attention_execution", {"dcp_indexer_sharded": True}),))
    parallel = ParallelSpec(tp=1, dp=1, cp=2)
    components = layer_cache_components(
        model, layer, parallel, 100, capacity_padded=True, interleave=BLOCK)
    # 100 tokens / 32 compression = 3 rows -> one 4-row block; the two CP ranks
    # own one whole block each, not a 128-row interleave slice.
    assert components["index_k"] == 4 * (16 + 2)


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
    for layer in range(draft.layers):
        for suffix in ("qkv_a", "kv_copy", "norm", "rope", "store"):
            assert attributed[f"d:context_kv{layer}.{suffix}"] == layer
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


def test_paired_batch_footprint_deduplicates_complete_shared_pages():
    case = paired_case(dspark=DsparkSpec(4, (1.0,) * 4))
    requests = (view("a", 1, 383, prefix=256), view("b", 1, 383, prefix=256))
    shared = batch_cache_footprint(
        case, requests=requests, replica=0, phase="decode", padded=True)
    private = batch_cache_footprint(
        replace(case, runtime=replace(
            case.runtime, prefix_cache_mode="private")),
        requests=requests, replica=0, phase="decode", padded=True)
    group = page_groups(case, LAYER_STAGE, 1)[0]
    page = group["bytes_per_group_page"]
    tokens_per_page = group["tokens_per_page"]
    recycling = group["recycling_pages"]
    history = (384 + tokens_per_page - 1) // tokens_per_page
    prefix_pages = 256 // tokens_per_page
    assert shared.shared_once_total == prefix_pages * page
    assert shared.private_total == 2 * (history - prefix_pages + recycling) * page
    assert shared.state_total == 2 * recycling * page
    assert shared.total == shared.shared_once_total + shared.private_total
    assert private.shared_once_total == 0
    assert private.total == 2 * (history + recycling) * page


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
    model = replace(dense_model(layers=1), layer_plans=plans,
                    auxiliary=AuxiliarySpec(mtp=mtp))
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


def test_release_shared_pages_releases_every_matching_owner_entry():
    memory = Memory(dense_case(layers=1))
    assert memory.retain_shared_pages(
        0, "owner", (1024.0,), physical_key="p1") is True
    assert memory.retain_shared_pages(
        0, "owner", (2048.0,), physical_key="p2") is True
    assert len(memory.shared_pages) == 2
    assert memory.release_shared_pages(0, "owner") is True
    assert memory.shared_pages == {}
    assert len(memory.release_events) == 2
    assert memory.release_shared_pages(0, "owner") is False


def test_shared_prefix_capacity_window_charges_pool_pages_once():
    base = dense_case(layers=2)
    case = replace(
        base, runtime=replace(base.runtime, prefix_cache_mode="shared"),
        workload=WorkloadSpec((
            RequestSpec("r", 0.0, 128, 2, prefix_tokens=128),)))
    window = capacity_window(
        case, phase="decode", context=128, tokens_per_seq=1, sequences=2,
        output_budget=8)
    active = window["window"]["active_tokens_per_sequence"]
    footprint = batch_cache_bytes(
        case, sequences=2, context=128, tokens_per_seq=active - 128,
        prefix_tokens=128, phase="decode", padded=True)
    assert window["active_kv_bytes"] == pytest.approx(
        sum(footprint.total_bytes))
    assert window["peak_bytes"] == pytest.approx(window["active_kv_bytes"])
    private = capacity_window(
        replace(case, runtime=replace(
            case.runtime, prefix_cache_mode="private")),
        phase="decode", context=128, tokens_per_seq=1, sequences=2,
        output_budget=8)
    assert private["active_kv_bytes"] == pytest.approx(
        2 * sum(Memory(case).cache_bytes(active)))
