"""Directed unfused MoE exchange receipts.

[Role] Pin conservation for explicit source->destination cells: payload FP8,
E8M0 scale and BF16 return are distinct ledgers; sparse/ragged placement is
never reconstructed from row/column marginals.
"""
import pytest

from mindie_sim.prediction.piecewise.moe_unfused_exchange import (
    DirectedCell, price_three_exchange, uniform_three_exchange,
)
from mindie_sim.schema.common import ModelingError

RANKS = 16
NODE_RANKS = 8
MTU = 4096
PAYLOAD_ROW = 10240
SCALE_ROW = 160
RETURN_ROW = 20480


def _uniform(rows=12288):
    return uniform_three_exchange(
        ranks=RANKS, rows=rows, payload_row_bytes=PAYLOAD_ROW,
        scale_row_bytes=SCALE_ROW, return_row_bytes=RETURN_ROW,
        node_ranks=NODE_RANKS, mtu=MTU)


def test_uniform_group_all_is_self_excluded_and_not_an_eightfold_wire():
    receipt = _uniform()
    payload = receipt.payload
    assert payload.total_bytes == 12288 * PAYLOAD_ROW
    assert payload.remote_bytes == (12288 // RANKS) * PAYLOAD_ROW * (RANKS - 1)
    assert payload.self_bytes == (12288 // RANKS) * PAYLOAD_ROW
    assert payload.aggregate_wire_bytes == payload.remote_bytes
    assert payload.aggregate_wire_bytes <= payload.total_bytes
    assert payload.aggregate_wire_bytes != 943_718_400
    assert payload.send_max == payload.remote_bytes // RANKS
    assert payload.receive_max == payload.remote_bytes // RANKS
    assert receipt.aggregate_wire_bytes == (
        payload.aggregate_wire_bytes + receipt.scale.aggregate_wire_bytes
        + receipt.return_.aggregate_wire_bytes)


def test_three_exchanges_keep_distinct_dtypes_and_totals():
    receipt = _uniform()
    assert (receipt.payload.dtype, receipt.scale.dtype, receipt.return_.dtype) == (
        "fp8_e4m3", "e8m0_scale", "bf16")
    assert receipt.payload.total_bytes == 12288 * PAYLOAD_ROW
    assert receipt.scale.total_bytes == 12288 * SCALE_ROW
    assert receipt.return_.total_bytes == 12288 * RETURN_ROW
    assert receipt.return_.remote_bytes == 2 * receipt.payload.remote_bytes


def test_ragged_two_source_cells_use_actual_source_payloads():
    payload = [
        DirectedCell(0, 1, 6144 * 5120),
        DirectedCell(1, 0, 6 * 5120),
    ]
    scale = [DirectedCell(0, 1, 6144 * 160), DirectedCell(1, 0, 6 * 160)]
    ret = [DirectedCell(0, 1, 6144 * 10240), DirectedCell(1, 0, 6 * 10240)]
    receipt = price_three_exchange(
        payload, scale, ret, ranks=RANKS, node_ranks=NODE_RANKS, mtu=MTU,
        payload_total=(6144 + 6) * 5120,
        scale_total=(6144 + 6) * 160,
        return_total=(6144 + 6) * 10240,
    )
    assert receipt.payload.remote_bytes == 6144 * 5120 + 6 * 5120
    assert receipt.payload.self_bytes == 0
    assert receipt.payload.send_max == 6144 * 5120
    assert receipt.payload.receive_max == 6144 * 5120
    assert receipt.payload.aggregate_wire_bytes >= receipt.payload.remote_bytes
    assert receipt.scale.send_max == 6144 * 160
    assert receipt.return_.send_max == 6144 * 10240


def test_marginals_alone_do_not_create_a_unique_matrix():
    a = [DirectedCell(0, 1, 4096), DirectedCell(1, 2, 4096),
         DirectedCell(2, 0, 4096)]
    b = [DirectedCell(0, 2, 4096), DirectedCell(1, 0, 4096),
         DirectedCell(2, 1, 4096)]
    # Both have one 4096-byte send and receive per rank, but different cells.
    ra = price_three_exchange(a, (), (), ranks=3, node_ranks=3, mtu=MTU,
                              payload_total=3 * 4096, scale_total=0,
                              return_total=0)
    rb = price_three_exchange(b, (), (), ranks=3, node_ranks=3, mtu=MTU,
                              payload_total=3 * 4096, scale_total=0,
                              return_total=0)
    assert ra.payload.send_max == rb.payload.send_max == 4096
    assert ra.payload.receive_max == rb.payload.receive_max == 4096
    assert ra.payload.aggregate_wire_bytes == rb.payload.aggregate_wire_bytes


def test_empty_rank_and_non_ep_contribute_no_wire():
    empty = price_three_exchange([], [], [], ranks=4, node_ranks=2, mtu=MTU,
                                 payload_total=8192, scale_total=0,
                                 return_total=0)
    assert empty.payload.remote_bytes == 0
    assert empty.payload.self_bytes == 8192
    assert empty.payload.aggregate_wire_bytes == 0
    non_ep = price_three_exchange([], [], [], ranks=1, node_ranks=1, mtu=MTU,
                                  payload_total=8192, scale_total=0,
                                  return_total=0)
    assert non_ep.aggregate_wire_bytes == 0


def test_uniform_requires_divisible_rows_and_self_cells_are_rejected():
    with pytest.raises(ModelingError):
        uniform_three_exchange(ranks=16, rows=12289, payload_row_bytes=10240,
                               scale_row_bytes=160, return_row_bytes=20480,
                               node_ranks=8, mtu=4096)
    with pytest.raises(ModelingError):
        price_three_exchange([DirectedCell(0, 0, 4096)], [], [],
                             ranks=2, node_ranks=1, mtu=4096,
                             payload_total=4096, scale_total=0, return_total=0)


def test_capacity_uses_capture_config_not_a_generic_512_default():
    from mindie_sim.prediction.piecewise.moe_unfused_exchange import (
        mc2_tokens_capacity)
    assert mc2_tokens_capacity(max_num_reqs=64, uniform_decode_query_len=1,
                               capture_sizes=(1, 2, 4, 8, 16, 32, 64),
                               tp=1, enable_fused_mc2=True) == 64
    assert mc2_tokens_capacity(max_num_reqs=64, uniform_decode_query_len=1,
                               capture_sizes=(), tp=1,
                               enable_fused_mc2=True) == 64
    assert mc2_tokens_capacity(max_num_reqs=64, uniform_decode_query_len=1,
                               capture_sizes=(1, 2, 4, 8, 16, 32, 64),
                               tp=2, enable_fused_mc2=True) == 64


def test_two_requests_can_colocate_or_use_distinct_sources():
    from mindie_sim.prediction.piecewise.moe_unfused_exchange import (
        active_source_geometry)
    colocated = active_source_geometry(((0, 1024), (0, 1024)))
    distinct = active_source_geometry(((0, 1024), (1, 1024)))
    assert colocated["group_total_tokens"] == distinct["group_total_tokens"] == 2048
    assert colocated["active_sources"] == 1
    assert colocated["max_tokens_per_rank"] == 2048
    assert distinct["active_sources"] == 2
    assert distinct["max_tokens_per_rank"] == 1024
