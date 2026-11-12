"""EP/DP routed source partitions for lifted prefill F2.

[Role]
  Pin the actual EP/DP expert-transport source scope: a prefill F2 subgraph is
  evaluated over the EP cohort, but only the active DP source ranks own token
  rows.  Decode remains one local source per rank.

[Invariants]
  * `source_partitions` is `min(ranks, cohort_batch)` for prefill and `ranks`
    for decode.
  * `source_routed_rows` and `source_payload_bytes` use that partition count;
    `routed_rows` and `peer_bytes` remain the lifted group totals.
  * The result is derived from declared queued rows and parallel axes, not
    from an observed duration.
"""
from fractions import Fraction

from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply, WorkloadGeometry,
)
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature

SUPPLY = PipeSupply((
    ("cube", 1000.0), ("vector", 1000.0), ("hbm", 1000.0),
    ("mte1", 1000.0), ("mte2", 1000.0), ("mte3", 1000.0),
    ("fixpipe", 1000.0), ("link", 1000.0),
    ("link_hccs", 2000.0), ("link_roce", 500.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=1, moe_row_tile=16,
                              split_target=10 ** 9, max_splits=8)


def _dispatch(*, ranks=16, node_ranks=8, mtu=4096, row_bytes=10240, top_k=6,
              placement="equal"):
    template = OperatorTemplate(name="dispatch", family="transport",
                                role="dispatch")
    signature = StaticSignature(
        role="dispatch", matrix=(("K", 1), ("N", 1)),
        quant=(("row_bytes", row_bytes),), parallel=(("members", ranks),),
        tiles=(("cores", 1),),
        extras=(("ranks_per_node", node_ranks), ("mtu_bytes", mtu),
                ("chunks", 4), ("top_k", top_k),
                ("source_placement", placement)),
    )
    return BoundOperator(template, signature)


def _evaluate(geometry, placement="equal"):
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        _dispatch(placement=placement))
    return compiled.evaluate(geometry)


def test_prefill_lifted_f2_uses_active_dp_sources_not_all_ep_ranks():
    geometry = WorkloadGeometry.homogeneous(
        1, 1024, 1024, phase="prefill", cohort_batch=2, local_batch=1).expanded_to(2)
    cost = _evaluate(geometry)
    derived = dict(cost.derived)
    assert derived["routed_rows"] == 2048 * 6
    assert derived["source_partitions"] == 2
    assert derived["source_routed_rows"] == 1024 * 6
    assert derived["source_payload_bytes"] == 1024 * 6 * 10240
    assert derived["peer_bytes"] == 2048 * 6 * 10240
    assert derived["aggregate_wire_bytes"] == 2 * derived["send_bytes"]
    assert derived["aggregate_wire_bytes"] <= derived["peer_bytes"]
    assert derived["aggregate_wire_bytes"] != 943_718_400


def test_decode_f2_keeps_one_rank_source_partition():
    geometry = WorkloadGeometry.homogeneous(
        1, 1024, 1024, phase="decode", cohort_batch=2, local_batch=1)
    cost = _evaluate(geometry)
    derived = dict(cost.derived)
    assert derived["routed_rows"] == 1024 * 6
    assert derived["source_partitions"] == 16
    assert derived["source_routed_rows"] == Fraction(1024 * 6, 16)
    assert derived["source_payload_bytes"] == Fraction(1024 * 6 * 10240, 16)


def test_ragged_placement_is_rejected_and_unknown_keeps_rank_denominator():
    geometry = WorkloadGeometry.homogeneous(
        1, 1024, 1024, phase="prefill", cohort_batch=2, local_batch=1).expanded_to(2)
    try:
        _evaluate(geometry, "ragged")
    except Exception as error:
        assert "source_placement_ragged" in str(error)
    else:
        raise AssertionError("ragged placement was uniformly averaged")
    unknown = dict(_evaluate(geometry, "unknown").derived)
    assert unknown["source_partitions"] == 16
