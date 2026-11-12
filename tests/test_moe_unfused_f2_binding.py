"""Ordinary F2 binding for three unfused AllToAllV exchanges.

[Role]
  The piecewise transport compiler and the F2 builder consume one declared
  route.  A byte matrix is priced as declared.  A row route expands once into
  FP8 payload, separate E8M0 scale and reversed BF16 return.  Paired source
  and received marginals qualify a remote-row floor and still leave the
  matrix unknown.

[Invariants]
  * Ragged placement reads a square ``traffic_matrix`` before it can demand
    ``source_rows``.  Those cells are not multiplied by a row width.
  * ``routing_unit=rows`` is the only row route.  Payload and scale keep the
    direction; return reverses it.  Diagonal rows stay local.
  * Equal placement with a typed width uses the per-source framed split.
    Unequal ``source_rows`` are rejected rather than averaged.
  * Paired marginals use max(0, source-received) and max(0, received-source).
    Group remote rows are the larger sum.  Unknown ranks stay unknown.
    A lower bound of zero is not an exact empty forecast.
  * A declared source workload with no received rows still emits three legs
    and does not fall through to the theoretical equal split.
  * MC2, EP1, dense FFN, and a route-less graph do not grow a third
    alltoallv.  Chunks partition one leg and do not discount it.

[Timeouts & Locks]
  Pure CPU compilation.  No device, network, or filesystem writes.

[Side-Effect Policy]
  Read-only.  Assertions use independent arithmetic from the declared cells.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math

import pytest

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.piecewise.compiler import (
    CompiledSubgraph, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
)
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, instantiate_layer,
)
from mindie_sim.prediction.subgraphs.state import StateView
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature,
)

SUPPLY = PipeSupply((
    ("cube", 1_000_000.0), ("vector", 1_000_000.0), ("hbm", 1_000_000.0),
    ("mte1", 1_000_000.0), ("mte2", 1_000_000.0), ("mte3", 1_000_000.0),
    ("fixpipe", 1_000_000.0), ("link", 1_000_000.0),
    ("link_hccs", 2_000_000.0), ("link_roce", 500_000.0),
), invocations=(
    ("link_latency_ms", 2.0), ("link_latency_intra_ms", 2.0),
    ("link_latency_node_ms", 2.0), ("link_latency_inter_ms", 5.0),
    ("vector", 0.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=1, moe_row_tile=16,
                              split_target=10 ** 9, max_splits=8)
NODE_LAT = Fraction(2)
WIDTHS = {"payload_row_bytes": 5120, "scale_row_bytes": 160,
          "return_row_bytes": 10240}
# Local diagonal rows are real routed rows and are not peer cells.
ROUTING = ((5, 2, 1), (0, 0, 4), (3, 0, 7))


def _frame(value, mtu):
    if value <= 0:
        return 0
    if mtu <= 1:
        return int(value)
    return math.ceil(int(value) / int(mtu)) * int(mtu)


def _matrix_volumes(matrix, node_ranks, mtu):
    ranks = len(matrix)
    cells = [[_frame(matrix[i][j], mtu) if i != j else 0
              for j in range(ranks)] for i in range(ranks)]
    sends = [sum(cells[i][j] for j in range(ranks) if i != j)
             for i in range(ranks)]
    recvs = [sum(cells[i][j] for i in range(ranks) if i != j)
             for j in range(ranks)]

    def level(off_node):
        same = lambda i, j: (i // node_ranks == j // node_ranks) != off_node
        out = [sum(cells[i][j] for j in range(ranks) if i != j and same(i, j))
               for i in range(ranks)]
        inc = [sum(cells[i][j] for i in range(ranks) if i != j and same(i, j))
               for j in range(ranks)]
        return max(out + inc)

    return {
        "send_bytes": max(sends),
        "recv_hotspot": max(recvs),
        "aggregate_wire_bytes": sum(sends),
        "local_hbm_bytes": max(send + recv for send, recv in zip(sends, recvs)),
        "intra_bytes": level(False),
        "inter_bytes": level(True),
    }


def _hotspot(cells, latency):
    per_source = {}
    for source, _destination in cells:
        per_source[source] = per_source.get(source, 0) + latency
    return max(per_source.values(), default=0)


def _total_pipe(cost, pipe):
    return sum((dict(stage).get(pipe, Fraction(0)) for stage in cost.stage_work),
               Fraction(0))


def _compile(operator, supply=SUPPLY):
    return OperatorPiecewiseCompiler(supply, POLICY).compile_operator(operator)


def _evaluate(operator, geometry, supply=SUPPLY):
    return _compile(operator, supply).evaluate(geometry)


def _transport(*, ranks=4, placement="unknown", name="dispatch", **extras):
    extra = (("source_placement", placement),
             ("ranks_per_node", extras.pop("ranks_per_node", 8)),
             ("chunks", extras.pop("chunks", 1)),
             ("top_k", extras.pop("top_k", 1)))
    for key, value in extras.items():
        if value is not None:
            extra += ((key, value),)
    signature = StaticSignature(
        role="dispatch", matrix=(("K", 1), ("N", 1)),
        quant=(("row_bytes", extras.get("row_bytes", 8)),),
        parallel=(("members", ranks),), tiles=(("cores", 1),), extras=extra)
    return BoundOperator(OperatorTemplate(name=name, family="transport",
                                          role="dispatch"), signature)


def _geometry(tokens, *, phase="decode", cohort=0):
    return WorkloadGeometry.homogeneous(
        1, tokens, max(tokens, 1), phase=phase, cohort_batch=cohort)


def _model(*, experts, top_k):
    plan = dict(index=0, attention="gqa", ffn="moe", num_heads=4,
                num_kv_heads=2, head_dim=16, intermediate_size=128,
                num_experts=experts, top_k=top_k, moe_intermediate_size=32)
    return ModelSpec.from_mapping(dict(
        name="moe-binding", hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=128,
        num_experts=experts, top_k=top_k, moe_intermediate_size=32,
        layers=1, layer_plans=[plan]))


def _instances(model, **overrides):
    features = FeatureConfig.from_model(model, **overrides)
    return instantiate_layer(model, model.layer_plans[0], features)


def _names(instance):
    return [operator.name for operator in instance.operators]


def test_ragged_traffic_matrix_is_priced_before_the_source_row_guard():
    matrix = ((0, 4096, 8192), (0, 0, 4096), (12288, 0, 0))
    cost = _evaluate(_transport(
        ranks=3, placement="ragged", traffic_matrix=matrix, mtu_bytes=4096,
        ranks_per_node=3), _geometry(4))
    expected = _matrix_volumes(matrix, 3, 4096)
    for key, value in expected.items():
        assert cost.get(key) == value
    assert cost.get("directed_forecast") == 1
    assert cost.get("aggregate_wire_bytes") == 4096 + 8192 + 4096 + 12288

    scaled = ((0, 10), (4, 0))
    per_token = _evaluate(_transport(
        ranks=2, placement="ragged", traffic_matrix=scaled, mtu_bytes=1,
        matrix_per_token=1, ranks_per_node=2), _geometry(4))
    assert per_token.get("send_bytes") == 40
    assert per_token.get("aggregate_wire_bytes") == 56

    untouched = _evaluate(_transport(
        ranks=2, placement="ragged", traffic_matrix=scaled, mtu_bytes=1,
        row_bytes=7, top_k=1, exchange_leg="payload",
        payload_row_bytes=5120, scale_row_bytes=160, return_row_bytes=10240,
        ranks_per_node=2), _geometry(4))
    assert untouched.get("send_bytes") == 10
    assert untouched.get("peer_bytes") == 4 * 7
    assert untouched.get("transport_row_bytes") == 7

    preserved = _evaluate(_transport(
        ranks=2, placement="unknown", traffic_matrix=scaled, mtu_bytes=1,
        ranks_per_node=2), _geometry(1))
    assert "directed_forecast" not in dict(preserved.derived)

    with pytest.raises(ModelingError, match="collective_traffic_matrix_cell_invalid"):
        _evaluate(_transport(
            ranks=2, placement="ragged", traffic_matrix=((0, -5), (1, 0)),
            mtu_bytes=1), _geometry(1))
    with pytest.raises(ModelingError, match="source_placement_ragged"):
        _evaluate(_transport(ranks=2, placement="ragged"), _geometry(1))


def test_routing_expands_once_and_return_reverses():
    widths = dict(payload_row_bytes=10, scale_row_bytes=2, return_row_bytes=20)
    routing = ROUTING
    payload = _matrix_volumes(
        ((0, 20, 10), (0, 0, 40), (30, 0, 0)), 3, 1)
    scale = _matrix_volumes(
        ((0, 4, 2), (0, 0, 8), (6, 0, 0)), 3, 1)
    returned = _matrix_volumes(
        ((0, 0, 60), (40, 0, 0), (20, 80, 0)), 3, 1)
    common = dict(ranks=3, placement="ragged", routing_matrix=routing,
                  routing_unit="rows", ranks_per_node=3, mtu_bytes=1,
                  row_bytes=999, **widths)
    payload_cost = _evaluate(_transport(exchange_leg="payload", **common),
                             _geometry(9))
    scale_cost = _evaluate(_transport(
        exchange_leg="scale", name="dispatch_scale", **common), _geometry(3))
    return_cost = _evaluate(_transport(
        exchange_leg="return", name="combine", **common), _geometry(1))
    for cost, expected, width in (
            (payload_cost, payload, 10), (scale_cost, scale, 2),
            (return_cost, returned, 20)):
        for key, value in expected.items():
            assert cost.get(key) == value
        assert cost.get("transport_row_bytes") == width
        assert cost.get("directed_forecast") == 1
        assert cost.get("routed_rows") == 22
    assert payload_cost.get("peer_bytes") == 100
    assert return_cost.get("peer_bytes") == 200
    assert return_cost.get("send_bytes") == 100
    assert return_cost.get("recv_hotspot") == 80
    assert payload_cost.get("send_bytes") == 40
    assert scale_cost.get("send_bytes") == 8

    cells = _compile(_transport(exchange_leg="payload", **common)).program.binder(
        _geometry(1), StateView())["transport_active_cells"]
    assert cells == ((0, 1), (0, 2), (1, 2), (2, 0))
    reversed_cells = _compile(_transport(
        exchange_leg="return", name="combine", **common)).program.binder(
            _geometry(1), StateView())["transport_active_cells"]
    assert reversed_cells == ((0, 2), (1, 0), (2, 0), (2, 1))
    separate = (_hotspot(cells, NODE_LAT) + _hotspot(cells, NODE_LAT)
                + _hotspot(reversed_cells, NODE_LAT))
    merged = _hotspot(tuple(cells) + tuple(cells) + tuple(reversed_cells),
                      NODE_LAT)
    assert payload_cost.get("service_dependency_ms") == _hotspot(cells, NODE_LAT)
    assert return_cost.get("service_dependency_ms") == _hotspot(
        reversed_cells, NODE_LAT)
    assert separate == 3 * NODE_LAT * 2
    assert separate > merged


def _equal_aggregate(tokens, width, partitions, ranks=16, mtu=4096):
    local = Fraction(tokens * 6 * width, partitions)
    per_peer = local / ranks
    framed = Fraction(math.ceil(per_peer / mtu) * mtu)
    send = (ranks - 1) * framed
    return partitions * send


def test_equal_typed_width_keeps_the_framed_group_wire():
    widths = dict(payload_row_bytes=10240, scale_row_bytes=160,
                  return_row_bytes=20480)
    prefill = WorkloadGeometry.homogeneous(
        1, 1024, 1024, phase="prefill", cohort_batch=2).expanded_to(2)
    payload = _evaluate(_transport(
        ranks=16, placement="equal", exchange_leg="payload", row_bytes=1,
        top_k=6, mtu_bytes=4096, **widths), prefill)
    assert payload.get("routed_rows") == 2048 * 6
    assert payload.get("source_partitions") == 2
    assert payload.get("aggregate_wire_bytes") == _equal_aggregate(
        2048, 10240, 2) == 117_964_800
    assert payload.get("transport_row_bytes") == 10240
    assert payload.get("directed_forecast") == 1
    scale = _evaluate(_transport(
        ranks=16, placement="equal", exchange_leg="scale", name="dispatch_scale",
        row_bytes=1, top_k=6, mtu_bytes=4096, **widths), prefill)
    assert scale.get("aggregate_wire_bytes") == _equal_aggregate(
        2048, 160, 2) == 1_843_200
    decode = _evaluate(_transport(
        ranks=16, placement="equal", exchange_leg="payload", row_bytes=1,
        top_k=6, mtu_bytes=4096, **widths), _geometry(1024, phase="decode"))
    assert decode.get("source_partitions") == 16
    assert decode.get("aggregate_wire_bytes") == _equal_aggregate(
        1024, 10240, 16)


def _pad(known, ranks):
    return tuple(known) + (None,) * (ranks - len(known))


def test_partial_rows_record_the_row_floor_without_a_mixed_wire():
    ranks = 16
    source = _pad((6144, 6), ranks)
    received = _pad((1648, 705), ranks)
    egress = 6144 - 1648
    ingress = 705 - 6
    operator = _transport(
        ranks=ranks, placement="ragged", source_rows=source,
        received_rows=received, chunks=4, mtu_bytes=4096, **WIDTHS)
    cost = _evaluate(operator, _geometry(7, phase="decode"))
    assert cost.get("source_routed_rows") == 6144 + 6
    assert cost.get("source_partitions") == 2
    assert cost.get("source_payload_bytes") == 6144 * 5120 + 6 * 5120
    assert cost.get("known_source_payload_max") == 6144 * 5120
    assert cost.get("local_scale_bytes") == 6144 * 160 + 6 * 160
    assert cost.get("known_source_scale_max") == 6144 * 160
    assert cost.get("local_return_bytes") == 6144 * 10240 + 6 * 10240
    assert cost.get("known_source_return_max") == 6144 * 10240
    assert cost.get("known_received_rows") == 1648 + 705
    assert cost.get("known_received_payload_bytes") == (1648 + 705) * 5120
    assert cost.get("known_received_payload_max") == 1648 * 5120
    # No leg: one number would mix FP8, E8M0, and BF16.  The row floor is
    # still recorded, and it is the larger deficit rather than their sum.
    assert cost.get("outbound_rows_lower") == egress == 4496
    assert cost.get("inbound_rows_lower") == ingress == 699
    assert cost.get("directed_rows_lower") == max(egress, ingress) == 4496
    assert cost.get("directed_rows_lower") != egress + ingress
    assert cost.get("diagonal_upper_rows") == min(6144, 1648) + min(6, 705)
    assert cost.get("aggregate_wire_bytes") == 0
    assert cost.get("send_bytes") == 0
    assert cost.get("intra_bytes") == 0
    assert cost.get("inter_bytes") == 0
    assert cost.get("directed_forecast") == 0
    assert cost.get("directed_lower_bound") == 1
    assert cost.get("forecast_complete") == 0
    assert cost.get("peer_workload_declared") == 1
    assert cost.get("diagonal_unknown") == 1
    assert cost.get("peer_bytes") == (
        cost.get("source_payload_bytes") + cost.get("local_scale_bytes")
        + cost.get("local_return_bytes"))
    assert cost.get("local_hbm_bytes") == cost.get("peer_bytes")
    assert cost.get("service_dependency_ms") == 0
    assert "transport_active_cells" not in _compile(operator).program.binder(
        _geometry(7), StateView())
    assert len(cost.stage_times) == 2
    assert _total_pipe(cost, "hbm") == cost.get("local_hbm_bytes")
    assert _total_pipe(cost, "link_hccs") == 0

    one = _evaluate(_transport(
        ranks=ranks, placement="ragged", source_rows=_pad((6144,), ranks),
        mtu_bytes=4096, **WIDTHS), _geometry(4))
    assert one.get("source_partitions") == 1
    assert one.get("source_payload_bytes") == 6144 * 5120
    assert one.get("aggregate_wire_bytes") == 0
    assert one.get("directed_lower_bound") == 0
    assert one.get("forecast_complete") == 0
    assert one.get("peer_workload_declared") == 0
    assert one.get("directed_rows_lower") == 0


def test_f2_routing_is_three_sequential_exchanges():
    model = _model(experts=3, top_k=2)
    metadata = dict(source_placement="ragged", routing_matrix=ROUTING,
                    routing_unit="rows", mtu_bytes=1,
                    payload_row_bytes=10, scale_row_bytes=2, return_row_bytes=20)
    built = _instances(model, ep=3, mc2_chunks=4, metadata=metadata)
    dispatch_graph = built.find("F2", "transport")
    combine_graph = built.find("F2", "transport_combine")
    assert _names(dispatch_graph) == [
        "quantize", "dispatch", "dispatch_scale", "notify"]
    assert _names(combine_graph) == ["combine"]
    compiled = CompiledSubgraph(dispatch_graph, SUPPLY, POLICY)
    combine = CompiledSubgraph(combine_graph, SUPPLY, POLICY)
    geometry = _geometry(8, phase="decode")
    costs = {cost.name: cost for cost in compiled.evaluate(geometry).operators}
    returned = combine.evaluate(geometry).operators[0]
    assert costs["dispatch"].get("send_bytes") == 40
    assert costs["dispatch_scale"].get("send_bytes") == 8
    assert returned.get("send_bytes") == 100
    assert returned.get("recv_hotspot") == 80
    assert costs["dispatch"].get("routed_rows") == 22
    for cost, pipe_bytes in (
            (costs["dispatch"], costs["dispatch"].get("intra_bytes")),
            (costs["dispatch_scale"], costs["dispatch_scale"].get("intra_bytes")),
            (returned, returned.get("intra_bytes"))):
        assert cost.get("chunks") == 4
        assert cost.get("overlap") == 0
        assert len(cost.stage_times) == 2
        assert _total_pipe(cost, "link_hccs") == pipe_bytes
        assert _total_pipe(cost, "hbm") == cost.get("local_hbm_bytes")
        assert dict(cost.stage_work[0])["link_hccs"] * 4 == pipe_bytes
    deps = [costs["dispatch"].get("service_dependency_ms"),
            costs["dispatch_scale"].get("service_dependency_ms"),
            returned.get("service_dependency_ms")]
    assert deps == [NODE_LAT * 2, NODE_LAT * 2, NODE_LAT * 2]
    assert sum(deps) > max(deps)
    again = compiled.operator("dispatch").evaluate(_geometry(1, phase="decode"))
    assert again.get("send_bytes") == costs["dispatch"].get("send_bytes")


def test_f2_equal_legs_reproduce_the_two_source_wire():
    model = _model(experts=16, top_k=6)
    metadata = dict(source_placement="equal", payload_row_bytes=10240,
                    scale_row_bytes=160, return_row_bytes=20480)
    built = _instances(model, ep=16, metadata=metadata)
    forward = CompiledSubgraph(built.find("F2", "transport"), SUPPLY, POLICY)
    backward = CompiledSubgraph(
        built.find("F2", "transport_combine"), SUPPLY, POLICY)
    geometry = WorkloadGeometry.homogeneous(
        1, 1024, 1024, phase="prefill", cohort_batch=2)
    costs = {item.name: item for item in forward.evaluate(geometry).operators}
    combine = backward.evaluate(geometry).operators[0]
    assert _names(forward.instance) == [
        "quantize", "dispatch", "dispatch_scale", "notify"]
    assert costs["dispatch"].get("aggregate_wire_bytes") == 117_964_800
    assert costs["dispatch_scale"].get("aggregate_wire_bytes") == 1_843_200
    assert combine.get("aggregate_wire_bytes") == _equal_aggregate(
        2048, 20480, 2)
    assert costs["dispatch"].get("source_partitions") == 2
    assert costs["dispatch"].get("transport_row_bytes") == 10240
    assert int(model.hidden_size) * 2 == 128


def test_f2_partial_rows_are_three_typed_legs():
    model = _model(experts=16, top_k=6)
    metadata = dict(source_placement="ragged", source_rows=_pad((6144, 6), 16),
                    received_rows=_pad((1648, 705), 16), mtu_bytes=4096, **WIDTHS)
    built = _instances(model, ep=16, metadata=metadata)
    forward = built.find("F2", "transport")
    assert _names(forward) == ["quantize", "dispatch", "dispatch_scale", "notify"]
    assert forward.operator("dispatch").theta.find("exchange_leg") == "payload"
    assert forward.operator("dispatch_scale").theta.find("exchange_leg") == "scale"
    assert built.find("F2", "transport_combine").operator("combine").theta.find(
        "exchange_leg") == "return"
    costs = {cost.name: cost for cost in CompiledSubgraph(
        forward, SUPPLY, POLICY).evaluate(_geometry(7)).operators}
    combine = CompiledSubgraph(
        built.find("F2", "transport_combine"), SUPPLY, POLICY).evaluate(
            _geometry(7)).operators[0]
    egress, ingress = 6144 - 1648, 705 - 6
    directed = max(egress, ingress)
    payload, scale = costs["dispatch"], costs["dispatch_scale"]
    assert directed == 4496 and ingress == 699
    assert payload.get("aggregate_wire_bytes") == directed * 5120 == 23_019_520
    assert payload.get("send_bytes") == egress * 5120
    assert payload.get("recv_hotspot") == ingress * 5120 == 3_578_880
    assert scale.get("aggregate_wire_bytes") == directed * 160 == 719_360
    assert scale.get("send_bytes") == egress * 160
    assert scale.get("recv_hotspot") == ingress * 160 == 111_840
    assert combine.get("aggregate_wire_bytes") == directed * 10240 == 46_039_040
    assert combine.get("send_bytes") == ingress * 10240 == 7_157_760
    assert combine.get("recv_hotspot") == egress * 10240
    for cost, width in ((payload, 5120), (scale, 160), (combine, 10240)):
        assert cost.get("directed_forecast") == 0
        assert cost.get("directed_lower_bound") == 1
        assert cost.get("forecast_complete") == 0
        assert cost.get("diagonal_unknown") == 1
        assert cost.get("intra_bytes") == 0
        assert cost.get("inter_bytes") == 0
        assert cost.get("service_dependency_ms") == 0
        assert cost.get("source_routed_rows") == 6150
        assert cost.get("peer_bytes") == (6144 + 6) * width
        assert cost.get("local_hbm_bytes") == (6144 + 1648) * width
        assert cost.get("aggregate_wire_bytes") != cost.get("peer_bytes")
    assert payload.get("peer_bytes") + scale.get("peer_bytes") + combine.get(
        "peer_bytes") == (6144 + 6) * (5120 + 160 + 10240)
    assert payload.get("known_received_payload_max") == 1648 * 5120
    assert "transport_active_cells" not in _compile(
        forward.operator("dispatch")).program.binder(_geometry(7), StateView())


def test_return_rows_are_unpacked_tokens_not_routed_rows():
    model = _model(experts=16, top_k=6)
    metadata = dict(
        source_placement="ragged", source_rows=_pad((6144, 6), 16),
        return_rows=_pad((1024, 1), 16), mtu_bytes=4096, **WIDTHS)
    built = _instances(model, ep=16, metadata=metadata)
    forward = built.find("F2", "transport")
    combine = CompiledSubgraph(
        built.find("F2", "transport_combine"), SUPPLY, POLICY).evaluate(
            _geometry(7)).operators[0]
    costs = {cost.name: cost for cost in CompiledSubgraph(
        forward, SUPPLY, POLICY).evaluate(_geometry(7)).operators}
    assert costs["dispatch"].get("routed_rows") == 6150
    assert combine.get("routed_rows") == 1025
    assert combine.get("source_partitions") == 2
    assert combine.get("peer_bytes") == 1025 * 10240
    assert combine.get("local_return_bytes") == 1025 * 10240
    assert combine.get("known_source_return_max") == 1024 * 10240
    assert combine.get("aggregate_wire_bytes") == 0
    # The payload and scale legs keep the routed rows; only the BF16 return
    # is unpacked to one row per token.
    assert costs["dispatch"].get("peer_bytes") == 6150 * 5120
    assert costs["dispatch_scale"].get("peer_bytes") == 6150 * 160
    with pytest.raises(ModelingError, match="moe_return_rows_requires_ragged"):
        _evaluate(_transport(
            ranks=2, placement="equal", exchange_leg="return",
            name="combine", return_rows=(1, 2), **WIDTHS), _geometry(1))
    with pytest.raises(ModelingError, match="moe_return_rows_rank_count_mismatch"):
        _evaluate(_transport(
            ranks=3, placement="ragged", exchange_leg="return",
            name="combine", source_rows=(1, 2, None), return_rows=(1,),
            **WIDTHS), _geometry(1))


def test_marginal_counterexamples_keep_exact_unknown_and_overlap_apart():
    width = {"payload_row_bytes": 10, "scale_row_bytes": 1, "return_row_bytes": 20}
    # Two egress ranks and one ingress rank.  13 and 6 may be the same edges.
    overlap_source, overlap_received = (10, 4, 1), (1, 0, 7)
    overlap = _evaluate(_transport(
        ranks=3, placement="ragged", exchange_leg="payload",
        source_rows=overlap_source, received_rows=overlap_received,
        mtu_bytes=4096, **width), _geometry(3))
    assert overlap.get("outbound_rows_lower") == 9 + 4
    assert overlap.get("inbound_rows_lower") == 6
    assert overlap.get("directed_rows_lower") == 13
    assert overlap.get("directed_rows_lower") != 13 + 6
    assert overlap.get("aggregate_wire_bytes") == 13 * 10
    assert overlap.get("send_bytes") == 9 * 10
    assert overlap.get("recv_hotspot") == 6 * 10
    assert overlap.get("aggregate_wire_bytes") != 4096
    assert overlap.get("local_hbm_bytes") == (10 + 1) * 10
    assert overlap.get("diagonal_upper_rows") == 1 + 0 + 1
    assert overlap.get("forecast_complete") == 0
    assert overlap.get("directed_forecast") == 0
    returned = _evaluate(_transport(
        ranks=3, placement="ragged", exchange_leg="return", name="combine",
        source_rows=overlap_source, received_rows=overlap_received,
        mtu_bytes=4096, **width), _geometry(3))
    assert returned.get("send_bytes") == 6 * 20
    assert returned.get("recv_hotspot") == 9 * 20
    assert returned.get("aggregate_wire_bytes") == 13 * 20

    balanced = _evaluate(_transport(
        ranks=2, placement="ragged", exchange_leg="payload",
        source_rows=(4, 4), received_rows=(4, 4), mtu_bytes=1, **width),
        _geometry(1))
    assert balanced.get("directed_rows_lower") == 0
    assert balanced.get("aggregate_wire_bytes") == 0
    assert balanced.get("directed_lower_bound") == 1
    assert balanced.get("directed_forecast") == 0
    assert balanced.get("forecast_complete") == 0
    assert balanced.get("diagonal_upper_rows") == 8
    assert balanced.get("local_hbm_bytes") == (4 + 4) * 10

    # A known zero is a rank.  It participates in the deficit and is not dropped.
    zero = _evaluate(_transport(
        ranks=2, placement="ragged", exchange_leg="scale", name="dispatch_scale",
        source_rows=(5, 0), received_rows=(0, 4), mtu_bytes=1, **width),
        _geometry(1))
    assert zero.get("source_partitions") == 2
    assert zero.get("outbound_rows_lower") == 5
    assert zero.get("inbound_rows_lower") == 4
    assert zero.get("directed_rows_lower") == 5
    assert zero.get("aggregate_wire_bytes") == 5 * 1

    # The unknown second rank is not filled with the first rank's marginals.
    lone = _evaluate(_transport(
        ranks=2, placement="ragged", exchange_leg="payload",
        source_rows=(6144, None), received_rows=(1648, None),
        mtu_bytes=4096, **WIDTHS), _geometry(1))
    assert lone.get("source_partitions") == 1
    assert lone.get("directed_rows_lower") == 6144 - 1648
    assert lone.get("inbound_rows_lower") == 0
    assert lone.get("aggregate_wire_bytes") == (6144 - 1648) * 5120
    assert lone.get("local_hbm_bytes") == (6144 + 1648) * 5120
    assert lone.get("known_received_rows") == 1648

    source_only = _instances(_model(experts=16, top_k=6), ep=16, metadata=dict(
        source_placement="ragged", source_rows=_pad((6144, 6), 16), **WIDTHS))
    forward = source_only.find("F2", "transport")
    assert _names(forward) == ["quantize", "dispatch", "dispatch_scale", "notify"]
    payload = CompiledSubgraph(forward, SUPPLY, POLICY).evaluate(
        _geometry(7)).operators
    dispatch = next(cost for cost in payload if cost.name == "dispatch")
    assert dispatch.get("source_routed_rows") == 6150
    assert dispatch.get("source_partitions") == 2
    assert dispatch.get("aggregate_wire_bytes") == 0
    assert dispatch.get("send_bytes") == 0
    assert dispatch.get("recv_hotspot") == 0
    assert dispatch.get("directed_lower_bound") == 0
    assert dispatch.get("peer_workload_declared") == 0
    assert dispatch.get("forecast_complete") == 0
    assert dispatch.get("directed_forecast") == 0
    assert dispatch.get("peer_bytes") == (6144 + 6) * 5120
    assert dispatch.get("local_hbm_bytes") == 6144 * 5120
    assert dispatch.get("routed_rows") != 7 * 6


def test_invalid_routes_are_rejected():
    with pytest.raises(ModelingError, match="moe_routing_unit_invalid"):
        _evaluate(_transport(
            ranks=2, routing_matrix=((0, 1), (1, 0)), routing_unit="bytes",
            exchange_leg="payload", **WIDTHS), _geometry(1))
    with pytest.raises(ModelingError, match="moe_routing_cell_invalid"):
        _evaluate(_transport(
            ranks=2, routing_matrix=((0, -1), (0, 0)), routing_unit="rows",
            exchange_leg="payload", **WIDTHS), _geometry(1))
    with pytest.raises(ModelingError, match="collective_traffic_matrix_rank_count_mismatch"):
        _evaluate(_transport(
            ranks=3, routing_matrix=((0, 1), (0, 0)), routing_unit="rows",
            exchange_leg="payload", **WIDTHS), _geometry(1))
    with pytest.raises(ModelingError, match="moe_equal_placement_conflicts_with_source_rows"):
        _evaluate(_transport(
            ranks=2, placement="equal", source_rows=(3, 9),
            exchange_leg="payload", **WIDTHS), _geometry(1))
    with pytest.raises(ModelingError, match="moe_exchange_leg_requires_route"):
        _evaluate(_transport(
            ranks=2, placement="unknown", exchange_leg="scale", **WIDTHS),
            _geometry(1))
    with pytest.raises(ModelingError, match="moe_source_rows_rank_count_mismatch"):
        _evaluate(_transport(
            ranks=4, placement="ragged", source_rows=(1, 2), **WIDTHS),
            _geometry(1))
    model = _model(experts=3, top_k=2)
    with pytest.raises(ModelingError, match="moe_exchange_widths_required"):
        _instances(model, ep=3, metadata={
            "routing_matrix": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
            "routing_unit": "rows"})
    with pytest.raises(ModelingError, match="moe_routing_unit_invalid"):
        _instances(model, ep=3, metadata={
            "routing_matrix": ROUTING, "routing_unit": "bytes", **WIDTHS})
    conflict = _instances(model, ep=3, metadata={
        "source_placement": "equal", "source_rows": (1, 2, 3), **dict(
            payload_row_bytes=10, scale_row_bytes=2, return_row_bytes=20)})
    with pytest.raises(ModelingError, match="moe_equal_placement_conflicts_with_source_rows"):
        _evaluate(conflict.find("F2", "transport").operator("dispatch"),
                  _geometry(1))


def _mx_quant(*, activation_scale_format="fp8_e8m0"):
    return QuantSpec(
        "declared-mxfp8", weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="fp8_e4m3", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        activation_group_size=32, activation_scale_format=activation_scale_format,
        activation_scale_pack=2)


def test_route_less_graph_stays_one_bf16_dispatch():
    plain = _instances(_model(experts=2, top_k=2), ep=2)
    forward = plain.find("F2", "transport")
    assert _names(forward) == ["quantize", "dispatch", "notify"]
    assert forward.operator("dispatch").theta.find("exchange_leg") is None
    dispatch = CompiledSubgraph(forward, SUPPLY, POLICY).evaluate(
        _geometry(4)).operators
    cost = next(item for item in dispatch if item.name == "dispatch")
    assert cost.get("ranks") == 2
    assert cost.get("peer_bytes") == 4 * 2 * 64 * 2
    assert cost.get("transport_row_bytes") == 128
    assert "directed_lower_bound" not in dict(cost.derived)
    assert "forecast_complete" not in dict(cost.derived)

    proven = replace(_model(experts=2, top_k=2), quant=_mx_quant())
    untouched = _instances(proven, ep=2)
    same = untouched.find("F2", "transport")
    assert _names(same) == ["quantize", "dispatch", "notify"]
    same_cost = next(item for item in CompiledSubgraph(
        same, SUPPLY, POLICY).evaluate(_geometry(4)).operators
        if item.name == "dispatch")
    assert same_cost.get("peer_bytes") == 4 * 2 * 64 * 2
    assert same_cost.get("transport_row_bytes") == 128


def test_proven_native_widths_fill_only_a_declared_route():
    hidden = 64
    groups = (hidden + 32 - 1) // 32
    pack = 2
    scale_row = ((groups + pack - 1) // pack) * pack
    payload_row = hidden
    return_row = hidden * 2
    assert scale_row == 2 and payload_row == 64 and return_row == 128
    model = replace(_model(experts=3, top_k=2), quant=_mx_quant())
    built = _instances(model, ep=3, metadata={
        "routing_matrix": ROUTING, "routing_unit": "rows", "mtu_bytes": 1})
    forward = built.find("F2", "transport")
    assert _names(forward) == ["quantize", "dispatch", "dispatch_scale", "notify"]
    costs = {cost.name: cost for cost in CompiledSubgraph(
        forward, SUPPLY, POLICY).evaluate(_geometry(1)).operators}
    combine = CompiledSubgraph(
        built.find("F2", "transport_combine"), SUPPLY, POLICY).evaluate(
            _geometry(1)).operators[0]
    assert costs["dispatch"].get("transport_row_bytes") == payload_row
    assert costs["dispatch_scale"].get("transport_row_bytes") == scale_row
    assert combine.get("transport_row_bytes") == return_row
    assert costs["dispatch"].get("send_bytes") == 4 * payload_row
    assert costs["dispatch"].get("directed_forecast") == 1
    assert "forecast_complete" not in dict(costs["dispatch"].derived)

    with pytest.raises(ModelingError, match="moe_exchange_widths_required:scale_row_bytes"):
        _instances(model, ep=3, metadata={
            "routing_matrix": ROUTING, "routing_unit": "rows",
            "payload_row_bytes": payload_row})
    with pytest.raises(ModelingError, match="moe_exchange_widths_required"):
        _instances(replace(_model(experts=3, top_k=2), quant=_mx_quant(
            activation_scale_format="fp32")), ep=3, metadata={
                "source_placement": "ragged", "source_rows": (1, 2, 3)})


def _fp4_quant(*, activation_scale_format="fp8_e8m0"):
    return QuantSpec(
        "declared-mxfp4", weight_format="mxfp4", activation_format="mxfp4",
        expert_weight_format="mxfp4", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        activation_group_size=32, activation_scale_format=activation_scale_format,
        activation_scale_pack=2)


def test_fp4_route_keeps_payload_and_grouped_scale_as_separate_legs():
    from mindie_sim.prediction.moe_transport import default_dispatch_spec
    model = replace(_model(experts=3, top_k=2), quant=_fp4_quant())
    spec = default_dispatch_spec(model, width=model.hidden_size)
    # MXFP4 is 0.5 B E2M1 plus one E8M0 byte per 32 elements; the aggregate
    # 17/32 B row must not be published as one payload without its scale leg.
    assert spec["payload_bytes_per_row"] == 32
    assert spec["scale_bytes_per_row"] == 2
    assert spec["dispatch_row_bytes"] == 34
    assert spec["combine_row_bytes"] == 128
    built = _instances(model, ep=3, metadata={
        "source_placement": "ragged", "source_rows": (2, None, 4), "mtu_bytes": 1})
    forward = built.find("F2", "transport")
    assert _names(forward) == ["quantize", "dispatch", "dispatch_scale", "notify"]
    costs = {cost.name: cost for cost in CompiledSubgraph(
        forward, SUPPLY, POLICY).evaluate(_geometry(1)).operators}
    combine = CompiledSubgraph(
        built.find("F2", "transport_combine"), SUPPLY, POLICY).evaluate(
            _geometry(1)).operators[0]
    payload, scale = costs["dispatch"], costs["dispatch_scale"]
    assert payload.get("transport_row_bytes") == 32
    assert scale.get("transport_row_bytes") == 2
    assert combine.get("transport_row_bytes") == 128
    assert payload.get("routed_rows") == 6
    assert payload.get("source_partitions") == 2
    assert payload.get("known_source_payload_max") == 128
    assert scale.get("local_scale_bytes") == 12
    assert combine.get("local_return_bytes") == 768
    # The known hot/cold rows are a local floor, not an invented peer matrix.
    for cost in (payload, scale, combine):
        assert cost.get("aggregate_wire_bytes") == 0
        assert cost.get("directed_lower_bound") == 0
        assert cost.get("forecast_complete") == 0
    assert payload.get("peer_bytes") == 6 * 32
    with pytest.raises(ModelingError, match="moe_exchange_widths_required"):
        _instances(replace(model, quant=_fp4_quant(
            activation_scale_format="fp32")), ep=3, metadata={
                "source_placement": "ragged", "source_rows": (2, None, 4)})


def test_ep1_mc2_and_dense_do_not_grow_a_third_exchange():
    routed = dict(routing_matrix=ROUTING, routing_unit="rows", **dict(
        payload_row_bytes=10, scale_row_bytes=2, return_row_bytes=20))
    single = _instances(_model(experts=1, top_k=1), ep=1, metadata=routed)
    assert _names(single.find("F2", "transport")) == [
        "quantize", "dispatch", "notify"]
    assert single.find("F2", "transport").operator("dispatch").theta.find(
        "exchange_leg") is None
    mc2 = _instances(_model(experts=2, top_k=1), ep=2, moe_recipe="mc2",
                     metadata=routed)
    assert "dispatch_scale" not in _names(mc2.find("F2", "transport"))
    assert mc2.find("F2", "transport").operator("dispatch").theta.find(
        "overlap") == 1
    dense_plan = dict(index=0, attention="gqa", ffn="dense", num_heads=4,
                      num_kv_heads=2, head_dim=16, intermediate_size=128)
    dense = ModelSpec.from_mapping(dict(
        name="dense", hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=128, layers=1,
        layer_plans=[dense_plan]))
    built = _instances(dense, ep=2, metadata=routed)
    assert all(instance.kind != "F2" for instance in built.subgraphs)
