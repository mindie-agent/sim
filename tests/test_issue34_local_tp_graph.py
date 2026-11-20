"""Node-local dense projection rewrite: exact bytes, shapes and dependencies."""
from __future__ import annotations

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.prediction.local_projection_tp import lower_local_projection_tp
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from _support.builders import hardware, quant, view


def software(width=4, *, dp=4):
    return Software(parallel=ParallelSpec(
        dp=dp, ep=dp, o_proj_tp=width, linear_proj_tp=width, lm_head_tp=width))


def make_op(category, *, m=8, n=64, k=128):
    return gemm_op(op_id="matrix", deps=("x",), stage=0, layer=0,
                   requests=("r",), m=m, n=n, k=k, quant=quant(),
                   category=category, phase="decode")


def run(op, *, width=4, views=None):
    return lower_local_projection_tp(
        (op,), model=None, runtime=software(width), machine=hardware(4),
        replica=0, cohort_views=views or {}, scope="target")


def test_column_parallel_exchange_matches_the_independent_oracle():
    op = make_op("kda.q")
    gather, gemm, restore = run(op)
    assert gather.family == "allgather"
    assert gather.link_bytes == pytest.approx((4 - 1) * 8 * 128 * 2)
    assert gemm.local_shape == (32, 16, 128)
    assert gemm.flops == pytest.approx(2 * 8 * 64 * 128)
    assert restore.family == "alltoall"
    assert restore.link_bytes == pytest.approx((4 - 1) * 8 * 16 * 2)
    assert restore.op_id == op.op_id
    assert gemm.deps == (gather.op_id,)
    assert restore.deps == (gemm.op_id,)
    metadata = dict(restore.metadata)
    assert metadata["group_axis"] == "local_tp"
    assert metadata["group_ranks"] == (0, 1, 2, 3)
    assert metadata["projection_role"] == "linear_input"


def test_row_parallel_exchange_and_reduce_scatter_match_the_oracle():
    op = make_op("mla.o")
    exchange, gemm, restore = run(op)
    assert exchange.family == "alltoall"
    assert exchange.link_bytes == pytest.approx((4 - 1) * 8 * 32 * 2)
    assert gemm.local_shape == (32, 64, 32)
    assert gemm.flops == pytest.approx(2 * 8 * 64 * 128)
    assert restore.family == "reduce_scatter"
    assert restore.link_bytes == pytest.approx((4 - 1) * 8 * 64 * 2)
    assert restore.op_id == op.op_id and restore.local_shape == (8, 64)
    assert dict(restore.metadata)["projection_role"] == "attention_output"


def test_unequal_and_idle_rows_use_exact_destination_counts():
    op = make_op("kda.q")
    views = {0: (view("a", 8),), 1: (view("b", 4),), 2: (), 3: (view("d", 2),)}
    gather, gemm, restore = run(op, views=views)
    assert gather.link_bytes == pytest.approx(3 * 8 * 128 * 2)
    assert gemm.local_shape == (14, 16, 128)
    assert restore.link_bytes == pytest.approx(3 * 8 * 16 * 2)


def test_steps_are_idempotent_and_source_placement_fails():
    op = make_op("kda.q")
    once = run(op)
    twice = lower_local_projection_tp(
        once, model=None, runtime=software(), machine=hardware(4),
        replica=0, cohort_views={}, scope="target")
    assert twice == once
    source = gemm_op(
        op_id="src", deps=(), stage=0, layer=0, requests=(), m=8, n=64, k=128,
        quant=quant(), category="kda.q", phase="decode",
        metadata=(("implementation", "unknown_source"),))
    with pytest.raises(ModelingError, match="local_tp_source_placement_unsupported"):
        run(source)


def test_width_one_is_byte_identical():
    op = make_op("kda.q")
    assert run(op, width=1) == (op,)
