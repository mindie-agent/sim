"""FIA physical leading rows and BF16 IO at the normal A5 attention boundary.

[Role]
  The Qwen C1/C8 decode A0 launch carries a real graph leading row count from
  the static dense-SP lanes, while the quantized matmul activation remains a
  separate compute detail.  C1 has one valid request padded to two graph rows;
  C8 has two real rows.  Both Q and output are BF16, 12,288 B here.
[Invariants]
  * `fia_query_tokens` stays the logical q-token sum; `fia_physical_rows` is the
    graph leading row count; valid tokens stay separate.
  * Q/output HBM bytes use physical rows and the declared BF16 attention IO
    boundary, while KV/QK/PV work remains per valid request (no invented cache).
  * A2 and dk128 negative scopes do not take the physical-row correction.
  * A pure fresh prefill uses its real leading width, not a decode dummy.
[Timeouts & Locks] Pure CPU compilation/evaluation of one A0 operator.
[Side-Effect Policy] In-memory only; no files, device or locks.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry,
)
from test_piecewise_operators_attention_mla import (
    _HBM_BOUND, _page_attention_cost,
)

A5 = replace(_HBM_BOUND, arch_family="A5")
A2 = replace(_HBM_BOUND, arch_family="A2")


def _lanes(requests, *, graph_bucket, local_batch, phase="decode"):
    return WorkloadGeometry(
        requests, graph_bucket=graph_bucket, local_batch=local_batch,
        phase=phase, dense_sequence_parallel=True, tp_width=2,
        graph_dispatch="full", capture_sizes=(2, 4, 8))


def _cost(supply, head_dim, requests, *, page=128):
    compiled, _ = _page_attention_cost(
        12, 2, head_dim, requests, page=page, page_attention=1,
        supply=supply, split_target=4096)
    geometry = _lanes(
        tuple(RequestGeometry(q, ctx, phase="decode")
              for q, ctx in requests),
        graph_bucket=len(requests), local_batch=len(requests))
    return compiled.evaluate(geometry)


def test_c1_c8_physical_rows_and_bf16_q_output_bytes():
    c1 = _cost(A5, 256, ((1, 4160),))
    assert c1.get("fia_query_tokens") == 1
    assert c1.get("fia_physical_rows") == 2
    assert c1.get("fia_valid_query_tokens") == 1
    assert c1.get("fia_io_dtype_bytes") == 2
    assert c1.get("q_bytes") == 2 * 12 * 256 * 2 == 12_288
    assert c1.get("output_bytes") == 2 * 12 * 256 * 2 == 12_288

    c8 = _cost(A5, 256, ((1, 4160), (1, 4160)))
    assert c8.get("fia_query_tokens") == 2
    assert c8.get("fia_physical_rows") == 2
    assert c8.get("fia_valid_query_tokens") == 2
    assert c8.get("q_bytes") == 12_288
    assert c8.get("output_bytes") == 12_288


def test_negative_a2_and_dk128_keep_prior_io_scope():
    a2 = _cost(A2, 256, ((1, 4160),))
    assert a2.get("fia_physical_rows") == 0
    assert a2.get("fia_query_tokens") == 1
    # The helper's declared activation bytes are 2; negative A2 keeps the old
    # valid-query-only width, not the A5 physical graph rows.
    assert a2.get("q_bytes") == 12 * 256 * 2
    assert a2.get("output_bytes") == 12 * 256 * 2

    dk128 = _cost(A5, 128, ((1, 4160),))
    assert dk128.get("fia_physical_rows") == 0
    assert dk128.get("q_bytes") == 12 * 128 * 2
    assert dk128.get("output_bytes") == 12 * 128 * 2


def test_fresh_prefill_uses_real_leading_width():
    compiled, _ = _page_attention_cost(
        12, 2, 256, ((4096, 0),), page=128, page_attention=1,
        supply=A5, split_target=4096)
    geometry = WorkloadGeometry(
        (RequestGeometry(4096, 0, phase="prefill"),), phase="prefill")
    cost = compiled.evaluate(geometry)
    assert cost.get("fia_physical_rows") == 4096
    assert cost.get("fia_query_tokens") == 4096
    assert cost.get("q_bytes") == 4096 * 12 * 256 * 2
    assert cost.get("output_bytes") == 4096 * 12 * 256 * 2
