"""Physical prefix read union, causal query rows and applied traffic deltas."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.prediction.prefix_reads import (
    CacheAccess, apply_prefix_read_plan, plan_prefix_reads,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def access(request="a", *, context=999, q_len=1, prefix=128, key="k",
           row=128, scale=16, layer=0):
    return CacheAccess(request, context, q_len=q_len, prefix_tokens=prefix,
                       prefix_key=key, row_bytes=row, scale_bytes=scale,
                       layer=layer)


def test_dense_batch_union_matches_the_independent_oracle():
    accesses = (access("a"), access("b"))
    plan = plan_prefix_reads(accesses, mode="batch_shared", kernel_reuse="tile")
    rows = 128 + 2 * (1000 - 128)
    assert plan.logical_query_bytes == pytest.approx(2 * (1000 * 128 + 16))
    scale = 2 * 16 * rows / (2 * 1000)
    assert plan.unique_hbm_bytes == pytest.approx(rows * 128 + scale)
    # MTE2 remains each access's per-tile refill service; only the external
    # HBM fill deduplicates the shared prefix.
    assert plan.l2_to_core_bytes == pytest.approx(2 * (1000 * 128 + 16))
    assert plan.query_read_bytes == pytest.approx(2 * 128)
    assert plan.read_reuse_applied is True
    assert plan.unique_hbm_bytes < plan.logical_query_bytes


def test_per_sequence_preserves_old_traffic_even_with_pooled_capacity():
    accesses = (access("a"), access("b"))
    plan = plan_prefix_reads(accesses, mode="per_sequence", kernel_reuse="tile")
    assert plan.unique_hbm_bytes == pytest.approx(plan.logical_query_bytes)
    assert plan.read_reuse_applied is False
    assert "not_batch_shared" in plan.reason


def test_different_keys_and_layers_are_never_pooled():
    plain = plan_prefix_reads((access("a"), access("b")),
                              mode="batch_shared", kernel_reuse="tile")
    keys = plan_prefix_reads((access("a"), access("b", key="other")),
                             mode="batch_shared", kernel_reuse="tile")
    layers = plan_prefix_reads((access("a"), access("b", layer=1)),
                               mode="batch_shared", kernel_reuse="tile")
    assert keys.unique_hbm_bytes > plain.unique_hbm_bytes
    assert layers.unique_hbm_bytes > plain.unique_hbm_bytes


def test_q_gt_one_keeps_causal_query_rows_and_scales():
    value = plan_prefix_reads((access("a", context=9, q_len=3, prefix=0, scale=4),),
                              mode="auto", kernel_reuse="tile")
    assert value.logical_query_bytes == pytest.approx((10 + 11 + 12) * 128 + 3 * 4)
    assert value.query_read_bytes == pytest.approx(3 * 128)
    assert value.read_reuse_applied is False


def test_shared_recipe_requires_a_declared_supported_kernel():
    with pytest.raises(ModelingError, match="shared_prefix_kernel_reuse_unsupported"):
        plan_prefix_reads((access("a"), access("b")), mode="batch_shared",
                          kernel_reuse="native_per_sequence")
    auto = plan_prefix_reads((access("a"), access("b")), mode="auto",
                             kernel_reuse="native_per_sequence")
    assert auto.read_reuse_applied is False
    assert auto.unique_hbm_bytes == auto.logical_query_bytes


def test_apply_plan_changes_only_the_declared_key_read_delta():
    op = OpSpec("index", "attention_index", "decode", 0, 0, ("a", "b"),
                (1000,), (1000,), (1024,), (1000,),
                hbm_bytes=1000.0 * 128 + 32, mte2_bytes=1000.0 * 128 + 32,
                metadata=(("component", "index_k"),))
    plan = plan_prefix_reads((access("a"), access("b")),
                             mode="batch_shared", kernel_reuse="tile")
    applied = apply_prefix_read_plan(op, plan)
    metadata = dict(applied.metadata)
    assert metadata["prefix_read_reuse_applied"] is True
    assert applied.hbm_bytes == pytest.approx(
        op.hbm_bytes - plan.logical_query_bytes + plan.unique_hbm_bytes)
    assert applied.mte2_bytes == pytest.approx(
        op.mte2_bytes - plan.logical_query_bytes + plan.l2_to_core_bytes)
