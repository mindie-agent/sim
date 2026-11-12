"""Routed MoE transport: per-source framed wire and group ledger.

[Role]
  Establish the ordinary routed dispatch/combine transport byte formulas from
  the declared geometry alone: one source rank's genuine routed rows, the
  self-share excluded from egress, per-remote MTU framing, directed per-source
  bottlenecks, the group wire ledger, and the per-rank HBM ledger.

[Invariants]
  * ``routed_rows``/``peer_bytes`` are the declared group routed rows/payload;
    ``source_routed_rows``/``source_payload_bytes`` are the genuine source-rank
    share (group / ranks).  The two ledgers are never substituted for each
    other.
  * A source splits its own payload over all ``ranks`` group members and sends
    only the other ``ranks - 1`` shares.  Each remote share is framed at the
    declared MTU before it becomes a directed cell.
  * ``send_bytes``/``recv_hotspot``/``intra_bytes``/``inter_bytes`` are
    per-rank bottlenecks; ``aggregate_wire_bytes`` is the group wire total.
    The busiest declared group index selects each node level; no average split
    replaces a real per-rank placement.
  * Chunking partitions the same declared work and does not discount it.
  * Elapsed time is declared stage work over the declared pipe rate, never a
    fitted multiplier or an assertion's expected value.

[Timeouts & Locks]
  Pure CPU expression compilation and evaluation; no I/O, locks or device.

[Side-Effect Policy]
  Read-only declarations and in-memory records.  One test-only monkeypatch
  replaces the routed helper with the rejected old/own-share formulas; the
  production module is not modified.
"""
from __future__ import annotations

from fractions import Fraction
import math

import pytest

from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply, WorkloadGeometry,
)
from mindie_sim.prediction.piecewise import families_moe_collective
from mindie_sim.prediction.piecewise import families_moe_transport
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature,
)

SUPPLY = PipeSupply((
    ("cube", 1000.0), ("vector", 1000.0), ("hbm", 1000.0),
    ("mte1", 1000.0), ("mte2", 1000.0), ("mte3", 1000.0),
    ("fixpipe", 1000.0), ("link", 1000.0),
    ("link_hccs", 2000.0), ("link_roce", 500.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=1, moe_row_tile=16,
                              split_target=10 ** 9, max_splits=8)


def _make_op(*, ranks, node_ranks, mtu, row_bytes, top_k, chunks=1,
             matrix=None, per_token=0, transport_algorithm=None):
    extra = (("ranks_per_node", node_ranks), ("mtu_bytes", mtu),
             ("chunks", chunks), ("matrix_per_token", per_token),
             ("top_k", top_k))
    if transport_algorithm is not None:
        extra += (("transport_algorithm", transport_algorithm),)
    if matrix is not None:
        extra += (("traffic_matrix", matrix),)
    template = OperatorTemplate(name="dispatch", family="transport",
                                role="dispatch")
    signature = StaticSignature(
        role="dispatch", matrix=(("K", 1), ("N", 1)),
        quant=(("row_bytes", row_bytes),), parallel=(("members", ranks),),
        tiles=(("cores", 1),), extras=extra,
    )
    return BoundOperator(template, signature)


def _cost(*, tokens, supply=SUPPLY, **kwargs):
    compiled = OperatorPiecewiseCompiler(supply, POLICY).compile_operator(
        _make_op(**kwargs))
    geometry = WorkloadGeometry.homogeneous(1, tokens, tokens + 1)
    return compiled.evaluate(geometry)


def _expected(*, tokens, top_k, row_bytes, ranks, node_ranks, mtu):
    """Independent physical ledger; never reads the production helper."""
    rows = tokens * top_k
    group_payload = Fraction(rows * row_bytes)
    if ranks <= 1:
        zero = Fraction(0)
        return {
            "routed_rows": rows, "peer_bytes": group_payload,
            "source_routed_rows": Fraction(rows),
            "source_payload_bytes": group_payload,
            "per_peer_useful_bytes": zero, "send_bytes": zero,
            "recv_hotspot": zero, "aggregate_wire_bytes": zero,
            "intra_bytes": zero, "inter_bytes": zero, "local_hbm_bytes": zero,
        }
    source_payload = group_payload / ranks
    per_peer_useful = source_payload / ranks
    per_peer_wire = Fraction(math.ceil(per_peer_useful / mtu) * mtu)
    send = (ranks - 1) * per_peer_wire
    width = max(1, min(ranks, node_ranks))
    intra_peers = max(
        sum(1 for peer in range(ranks)
            if peer != source and source // width == peer // width)
        for source in range(ranks))
    inter_peers = max(
        sum(1 for peer in range(ranks)
            if peer != source and source // width != peer // width)
        for source in range(ranks))
    return {
        "routed_rows": rows, "peer_bytes": group_payload,
        "source_routed_rows": Fraction(rows, ranks),
        "source_payload_bytes": source_payload,
        "per_peer_useful_bytes": per_peer_useful,
        "send_bytes": send, "recv_hotspot": send,
        "aggregate_wire_bytes": send * ranks,
        "intra_bytes": intra_peers * per_peer_wire,
        "inter_bytes": inter_peers * per_peer_wire,
        "local_hbm_bytes": 2 * send,
    }


def _assert_ledger(cost, expected):
    for key, value in expected.items():
        assert Fraction(cost.get(key)) == value, (key, cost.get(key), value)


# The original A3 fixture geometry plus explicitly changed ranks/placements.
GEOMETRIES = (
    # q6/B4, TP16/EP16, one 16-die node: original counterexample.
    dict(tokens=24, top_k=8, row_bytes=8192, ranks=16, node_ranks=16, mtu=4096),
    # q1/B4, same group: one routed row per source rank.
    dict(tokens=4, top_k=8, row_bytes=8192, ranks=16, node_ranks=16, mtu=4096),
    # EP8 on one 8-die node: whole-payload split still frames 3 remotes.
    dict(tokens=24, top_k=8, row_bytes=8192, ranks=8, node_ranks=8, mtu=4096),
    # EP8 across two 4-die hosts: directed node split is not averaged.
    dict(tokens=24, top_k=8, row_bytes=8192, ranks=8, node_ranks=4, mtu=4096),
    # Odd 3-rank group on a 2-die host: the singleton node is the max egress.
    dict(tokens=6, top_k=2, row_bytes=1024, ranks=3, node_ranks=2, mtu=1024),
    # Odd 5-rank group across three 2-die hosts.
    dict(tokens=9, top_k=3, row_bytes=512, ranks=5, node_ranks=2, mtu=512),
    # Declared one-rank transport has no peer wire and no local peer copy.
    dict(tokens=5, top_k=2, row_bytes=256, ranks=1, node_ranks=1, mtu=4096),
)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_routed_transport_uses_source_rows_and_frames_each_remote(geometry):
    cost = _cost(**geometry)
    expected = _expected(**geometry)
    _assert_ledger(cost, expected)

    # The rejected old policy treated the whole group payload as one source's
    # payload and split it over ep-1 peers; with more than two ranks it must
    # differ from the source-row ledger, so these checks still discriminate.
    ranks = geometry["ranks"]
    if ranks > 2:
        group_payload = expected["peer_bytes"]
        old_peer = math.ceil(group_payload / (ranks - 1) / geometry["mtu"]) * geometry["mtu"]
        old_send = (ranks - 1) * old_peer
        assert expected["send_bytes"] != old_send
    # The own share is excluded: a source never sends all ranks' shares.
    if ranks > 1:
        assert expected["send_bytes"] < expected["source_payload_bytes"] * ranks


def test_routed_transport_chunks_conserve_declared_work():
    geometry = dict(tokens=24, top_k=8, row_bytes=8192, ranks=8,
                    node_ranks=4, mtu=4096)
    single = _cost(**geometry, chunks=1)
    chunked = _cost(**geometry, chunks=4)
    for key in ("routed_rows", "peer_bytes", "source_payload_bytes",
                "per_peer_useful_bytes", "send_bytes", "recv_hotspot",
                "aggregate_wire_bytes", "intra_bytes", "inter_bytes",
                "local_hbm_bytes"):
        assert Fraction(chunked.get(key)) == Fraction(single.get(key)), key
    for pipe in ("link_hccs", "link_roce", "hbm"):
        total = sum(Fraction(dict(stage).get(pipe, 0))
                    for stage in chunked.stage_work)
        assert total == Fraction(dict(single.stage_work[0]).get(pipe, 0))
    assert len(single.stage_times) == 1
    assert len(chunked.stage_times) == 2
    assert (Fraction(dict(chunked.stage_work[0]).get("link_hccs", 0))
            == Fraction(chunked.get("intra_bytes")) / 4)


def test_routed_transport_unequal_matrix_keeps_per_rank_bottleneck():
    # Asymmetric directed histogram: no rank's row/column is the group average.
    matrix = ((0, 100, 400, 700),
              (100, 0, 100, 100),
              (100, 100, 0, 100),
              (100, 100, 100, 0))
    mtu, ranks, node_ranks = 256, 4, 2
    cost = _cost(tokens=1, top_k=1, row_bytes=1, ranks=ranks,
                 node_ranks=node_ranks, mtu=mtu, matrix=matrix)
    cells = tuple(tuple(Fraction(math.ceil(value / mtu) * mtu) for value in row)
                  for row in matrix)
    # The self cell is not a peer wire cell.
    sends = [sum(cells[i][j] for j in range(ranks) if i != j)
             for i in range(ranks)]
    recvs = [sum(cells[i][j] for i in range(ranks) if i != j)
             for j in range(ranks)]

    def level(off_node):
        out = [sum(cells[i][j] for j in range(ranks)
                   if i != j and (i // node_ranks == j // node_ranks) != off_node)
               for i in range(ranks)]
        inc = [sum(cells[i][j] for i in range(ranks)
                   if i != j and (i // node_ranks == j // node_ranks) != off_node)
               for j in range(ranks)]
        return max(out + inc)

    assert Fraction(cost.get("send_bytes")) == max(sends)
    assert Fraction(cost.get("recv_hotspot")) == max(recvs)
    assert Fraction(cost.get("aggregate_wire_bytes")) == sum(sends)
    assert Fraction(cost.get("intra_bytes")) == level(False)
    assert Fraction(cost.get("inter_bytes")) == level(True)
    # Pair each rank's own egress and ingress before taking the hotspot; the
    # old max(rows)+max(cols) added two unrelated ranks.
    assert Fraction(cost.get("local_hbm_bytes")) == max(
        out + inc for out, inc in zip(sends, recvs))
    # Conservation of the explicit off-diagonal cells, not an average.
    assert max(sends) != sum(sends) / ranks
    assert max(recvs) != sum(recvs) / ranks


@pytest.mark.parametrize("name,matrix,expected", (
    # diagonal-only self copy: no peer wire, no implicit HBM stage
    ("all-local", ((8192, 0), (0, 0)), (0, 0, 0, 0)),
    # one directed edge: source0 egress and dest1 ingress are the same 8192
    ("one-link", ((0, 8192), (0, 0)), (8192, 8192, 8192, 8192)),
    # mixed cells with a diagonal and a reverse edge: self excluded, HBM pairs
    ("both", ((512, 4096), (8192, 6144)), (8192, 8192, 12288, 12288)),
    # a one-rank transport has no peer wire and no self-copy stage
    ("one-rank", ((8192,),), (0, 0, 0, 0)),
))
def test_transport_matrix_excludes_self_and_pairs_rank_hbm(name, matrix, expected):
    cost = _cost(tokens=1, top_k=1, row_bytes=8192, ranks=len(matrix),
                 node_ranks=len(matrix), mtu=4096, matrix=matrix)
    actual = tuple(Fraction(cost.get(key)) for key in
                   ("send_bytes", "recv_hotspot", "aggregate_wire_bytes",
                    "local_hbm_bytes"))
    assert actual == tuple(Fraction(value) for value in expected), name
    stage = dict(cost.stage_work[0])
    if name == "one-link":
        # The real physical stage is 8.192 ms at the 1000-byte/ms supply,
        # not the old 16.384 ms that added two unrelated rank maxima.
        assert float(cost.duration_ms) == pytest.approx(8.192)
        assert Fraction(stage["hbm"]) == 8192
        assert Fraction(stage["link_hccs"]) == 8192
    if name in {"all-local", "one-rank"}:
        assert float(cost.duration_ms) == 0.0
        assert sum(Fraction(value) for value in stage.values()) == 0


def test_transport_matrix_perturbation_keeps_edge_owners():
    base = ((0, 8192, 0), (0, 0, 0), (0, 0, 0))
    perturbed = ((0, 8192, 0), (0, 0, 0), (0, 4096, 0))
    base_cost = _cost(tokens=1, top_k=1, row_bytes=8192, ranks=3,
                      node_ranks=3, mtu=4096, matrix=base)
    changed = _cost(tokens=1, top_k=1, row_bytes=8192, ranks=3,
                    node_ranks=3, mtu=4096, matrix=perturbed)
    assert (Fraction(base_cost.get("send_bytes")),
            Fraction(base_cost.get("recv_hotspot")),
            Fraction(base_cost.get("aggregate_wire_bytes")),
            Fraction(base_cost.get("local_hbm_bytes"))) == (
                8192, 8192, 8192, 8192)
    # The new 2->1 cell belongs to rank2 egress and rank1 ingress; their
    # paired HBM is 12,288.  It must not become max(send)+max(recv)=20,480.
    assert (Fraction(changed.get("send_bytes")),
            Fraction(changed.get("recv_hotspot")),
            Fraction(changed.get("aggregate_wire_bytes")),
            Fraction(changed.get("local_hbm_bytes"))) == (
                8192, 12288, 12288, 12288)
    assert Fraction(changed.get("aggregate_wire_bytes")) == 8192 + 4096


def test_routed_transport_elapsed_is_declared_work_over_rate():
    geometry = dict(tokens=2, top_k=2, row_bytes=256, ranks=4,
                    node_ranks=4, mtu=64, chunks=1)
    cost = _cost(**geometry)
    expected = _expected(**{key: value for key, value in geometry.items()
                            if key != "chunks"})
    # One-node group: only link_hccs and hbm carry declared work.
    stage = dict(cost.stage_work[0])
    assert Fraction(stage["link_hccs"]) == expected["intra_bytes"]
    assert Fraction(stage["hbm"]) == expected["local_hbm_bytes"]
    assert Fraction(stage["vector"]) == expected["local_hbm_bytes"] / 2
    independent = max(
        expected["intra_bytes"] / Fraction(2000),
        expected["local_hbm_bytes"] / Fraction(1000),
        (expected["local_hbm_bytes"] / 2) / Fraction(1000),
    )
    assert float(cost.duration_ms) == pytest.approx(float(independent))

    # A declaration-level counterfactual: doubling the genuine source rows
    # doubles the per-source wire and the elapsed body time.  It cannot be a
    # fitted constant because the bytes are recomputed from the geometry.
    doubled = _cost(**{**geometry, "tokens": geometry["tokens"] * 2})
    assert Fraction(doubled.get("send_bytes")) == 2 * expected["send_bytes"]
    assert Fraction(doubled.get("aggregate_wire_bytes")) == 2 * expected[
        "aggregate_wire_bytes"]
    assert float(doubled.duration_ms) == pytest.approx(2 * float(independent))


def test_routed_transport_rejects_x16_and_source_total_mutants(monkeypatch):
    geometry = dict(tokens=24, top_k=8, row_bytes=8192, ranks=16,
                    node_ranks=16, mtu=4096)
    expected = _expected(**geometry)
    base = _cost(**geometry)
    assert Fraction(base.get("send_bytes")) == expected["send_bytes"]

    # x16 is a real declaration perturbation, not a time multiplier: the
    # physical ledger scales by 16 and a checker bound to the base geometry
    # must reject it.
    x16_geometry = {**geometry, "tokens": geometry["tokens"] * 16}
    x16_expected = _expected(**x16_geometry)
    x16 = _cost(**x16_geometry)
    assert Fraction(x16.get("send_bytes")) == x16_expected["send_bytes"]
    assert Fraction(x16.get("send_bytes")) != expected["send_bytes"]
    assert Fraction(x16.get("aggregate_wire_bytes")) == x16_expected[
        "aggregate_wire_bytes"]
    assert Fraction(x16.get("aggregate_wire_bytes")) != expected[
        "aggregate_wire_bytes"]
    # Framing is not linear: at x16 the per-peer share is already MTU-aligned,
    # so the wire is smaller than a naive 16x of the padded base.  The ledger
    # must be recomputed from geometry, not multiplied by an assertion result.
    assert Fraction(x16.get("send_bytes")) < 16 * expected["send_bytes"]

    # Test-only mutant A: the old product formula (whole group payload as one
    # source, split over ep-1 peers).  The independent checker must reject it.
    def old_product(local_payload, ranks, node_ranks, mtu):
        group = families_moe_collective.Mul(
            (local_payload, families_moe_collective.Const(ranks)))
        intra, inter, send, hotspot, aggregate = (
            families_moe_collective._uniform_alltoallv(
                group, ranks, node_ranks, mtu))
        return intra, inter, send, hotspot, aggregate, families_moe_collective.Add(
            (send, hotspot))

    monkeypatch.setattr(families_moe_transport, "_routed_alltoallv",
                        old_product)
    mutant = _cost(**geometry)
    assert Fraction(mutant.get("send_bytes")) != expected["send_bytes"]
    assert Fraction(mutant.get("aggregate_wire_bytes")) != expected[
        "aggregate_wire_bytes"]
    monkeypatch.undo()

    # Test-only mutant B: charge the source's own share as egress (split over
    # ranks-1 instead of ranks).  Also rejected by the per-source ledger.
    def own_share(local_payload, ranks, node_ranks, mtu):
        per_peer = families_moe_collective._round_expr(
            families_moe_collective.Div(
                local_payload, families_moe_collective.Const(ranks - 1)), mtu)
        send = families_moe_collective.Mul(
            (families_moe_collective.Const(ranks - 1), per_peer))
        return (send, send, send, send,
                families_moe_collective.Mul(
                    (send, families_moe_collective.Const(ranks))),
                families_moe_collective.Add((send, send)))

    # Use a geometry whose own-share split crosses an MTU boundary; the
    # 16-rank original happens to frame both splits to the same 8192 bytes.
    own_geometry = dict(tokens=4, top_k=2, row_bytes=2048, ranks=4,
                        node_ranks=4, mtu=1024)
    own_expected = _expected(**own_geometry)
    monkeypatch.setattr(families_moe_transport, "_routed_alltoallv",
                        own_share)
    own = _cost(**own_geometry)
    assert Fraction(own.get("send_bytes")) != own_expected["send_bytes"]
    assert Fraction(own.get("send_bytes")) > own_expected["send_bytes"]
    monkeypatch.undo()


def test_routed_transport_reports_source_rank_placement():
    # Two 8-die hosts: every source has 7 same-host and 8 off-host peers.
    cost = _cost(tokens=24, top_k=8, row_bytes=8192, ranks=16,
                 node_ranks=8, mtu=4096)
    expected = _expected(tokens=24, top_k=8, row_bytes=8192, ranks=16,
                         node_ranks=8, mtu=4096)
    assert Fraction(cost.get("intra_bytes")) == 7 * expected["send_bytes"] / 15
    assert Fraction(cost.get("inter_bytes")) == 8 * expected["send_bytes"] / 15
    assert (Fraction(cost.get("intra_bytes")) + Fraction(cost.get("inter_bytes"))
            == expected["send_bytes"])
    assert int(cost.get("ranks")) == 16
    assert int(cost.get("nodes")) == 2
    assert int(cost.get("ranks_per_node")) == 8


def _matrix_expected(matrix, mtu, node_ranks):
    """Independent per-edge ledger for an explicit transport matrix."""
    ranks = len(matrix)
    if ranks <= 1:
        return {key: Fraction(0) for key in
                ("send_bytes", "recv_hotspot", "aggregate_wire_bytes",
                 "local_hbm_bytes", "intra_bytes", "inter_bytes")}
    cells = tuple(tuple(Fraction(math.ceil(value / mtu) * mtu) for value in row)
                  for row in matrix)
    sends = [sum(cells[i][j] for j in range(ranks) if i != j)
             for i in range(ranks)]
    recvs = [sum(cells[i][j] for i in range(ranks) if i != j)
             for j in range(ranks)]

    def level(off_node):
        out = [sum(cells[i][j] for j in range(ranks)
                   if i != j and (i // node_ranks == j // node_ranks) != off_node)
               for i in range(ranks)]
        inc = [sum(cells[i][j] for i in range(ranks)
                   if i != j and (i // node_ranks == j // node_ranks) != off_node)
               for j in range(ranks)]
        return max(out + inc)

    return {
        "send_bytes": max(sends),
        "recv_hotspot": max(recvs),
        "aggregate_wire_bytes": sum(sends),
        "local_hbm_bytes": max(out + inc for out, inc in zip(sends, recvs)),
        "intra_bytes": level(False),
        "inter_bytes": level(True),
    }


@pytest.mark.parametrize("name,matrix,mtu,node_ranks", (
    # per-rank offset: only source0 egresses to dest1
    ("offset-forward", ((0, 8192, 0), (0, 0, 0), (0, 0, 0)), 4096, 3),
    # the same edge in the reverse owner direction
    ("offset-reverse", ((0, 0, 0), (8192, 0, 0), (0, 0, 0)), 4096, 3),
    # half directions: each source owns one next-rank edge
    ("half-ring", ((0, 4096, 0), (0, 0, 4096), (0, 0, 0)), 4096, 3),
    # odd/non-power-of-two MTU and two different edge amounts
    ("odd-mtu", ((0, 1000, 0), (0, 0, 1001), (0, 0, 0)), 3, 3),
    # six ranks on three 2-die hosts, with a hot cross-host edge and skew
    ("multi-host-congestion",
     ((0, 4096, 8192, 0, 0, 0),
      (0, 0, 4096, 0, 0, 0),
      (0, 0, 0, 4096, 0, 0),
      (0, 0, 0, 0, 4096, 0),
      (0, 0, 0, 0, 0, 4096),
      (4096, 0, 0, 0, 0, 0)), 4096, 2),
))
def test_transport_matrix_offsets_half_odd_mtu_and_multihost(
        name, matrix, mtu, node_ranks):
    ranks = len(matrix)
    cost = _cost(tokens=1, top_k=1, row_bytes=8192, ranks=ranks,
                 node_ranks=min(node_ranks, ranks), mtu=mtu, matrix=matrix)
    expected = _matrix_expected(matrix, mtu, min(node_ranks, ranks))
    for key, value in expected.items():
        assert Fraction(cost.get(key)) == value, (name, key, cost.get(key), value)
    # A zero diagonal is not charged to any wire or HBM ledger.
    assert all(matrix[i][i] == 0 for i in range(ranks))


# The D2 geometry: 16 members, top_k=8, row_bytes=4096, MTU=4096.
# One source owns T*8/16 = T/2 rows, so each remote gets T*128 useful bytes.
# T*128 is exactly 64 frames at T=2048 and crosses to 65 frames at T=2049.
_DISPATCH_SUPPLY = PipeSupply((
    ("link", 392_000_000.0), ("hbm", 1_600_000_000.0),
    ("vector", 6_758_400_000.0),
))


def _d2_dispatch_cost(tokens):
    return _cost(tokens=tokens, top_k=8, row_bytes=4096, ranks=16,
                 node_ranks=8, mtu=4096, supply=_DISPATCH_SUPPLY)


def test_d2_dispatch_frame_boundary_is_source_peer_share_not_group_total():
    low = _d2_dispatch_cost(2047)
    mid = _d2_dispatch_cost(2048)
    high = _d2_dispatch_cost(2049)
    # Correct boundary is 2048 -> 2049, not 2047 -> 2048.
    assert Fraction(mid.get("send_bytes")) == Fraction(3_932_160)
    assert Fraction(mid.get("send_bytes")) - Fraction(low.get("send_bytes")) == 0
    assert Fraction(high.get("send_bytes")) - Fraction(mid.get("send_bytes")) == 61_440
    assert float(mid.duration_ms) - float(low.duration_ms) == pytest.approx(0.0)
    assert float(high.duration_ms) - float(mid.duration_ms) == pytest.approx(
        61_440 / 392_000_000.0)


def test_d2_old_2048_assertion_passes_the_group_total_mutant(monkeypatch):
    """The old +61,440 at 2047->2048 assertion is satisfied by the defect."""
    def old_product(local_payload, ranks, node_ranks, mtu):
        group = families_moe_collective.Mul(
            (local_payload, families_moe_collective.Const(ranks)))
        intra, inter, send, hotspot, aggregate = (
            families_moe_collective._uniform_alltoallv(
                group, ranks, node_ranks, mtu))
        return (intra, inter, send, hotspot, aggregate,
                families_moe_collective.Add((send, hotspot)))

    monkeypatch.setattr(families_moe_transport, "_routed_alltoallv",
                        old_product)
    low = _d2_dispatch_cost(2047)
    mid = _d2_dispatch_cost(2048)
    # This is exactly what the existing D2 test asserts; the old group-total
    # defect passes it, so the assertion cannot distinguish the two models.
    assert Fraction(mid.get("send_bytes")) - Fraction(low.get("send_bytes")) == 61_440
    monkeypatch.undo()
    corrected_low = _d2_dispatch_cost(2047)
    corrected_mid = _d2_dispatch_cost(2048)
    assert Fraction(corrected_mid.get("send_bytes")) - Fraction(
        corrected_low.get("send_bytes")) == 0


def _tiered_supply(package_width, package_ms=0.0008, node_ms=0.0025,
                   inter_ms=0.0018):
    return PipeSupply((
        ("link", 196_000_000.0), ("link_hccs", 196_000_000.0),
        ("link_roce", 196_000_000.0), ("hbm", 1_600_000_000.0),
        ("vector", 6_758_400_000.0),
    ), invocations=(
        ("link_latency_ms", package_ms), ("link_latency_intra_ms", package_ms),
        ("link_latency_package_ms", package_ms), ("link_latency_node_ms", node_ms),
        ("link_latency_inter_ms", inter_ms), ("ranks_per_package", package_width),
        ("vector", 0.0),
    ))


def _dep_ms(cost):
    return Fraction(dict(cost.derived).get("service_dependency_ms", 0))


def test_transport_uniform_unpaired_last_die_uses_its_real_routes():
    # Package pairs are (0,1),(2,3),...; rank 2 in a 3-rank group and rank 4
    # in a 5-rank group have no same-package partner, so only node routes.
    for ranks, expected_us in ((3, 5.0), (5, 10.0), (16, 35.8)):
        cost = _cost(tokens=1, top_k=1, row_bytes=8192.0, ranks=ranks,
                     node_ranks=16, mtu=4096,
                     supply=_tiered_supply(2))
        assert _dep_ms(cost) == Fraction(str(expected_us)) / 1000, ranks


def test_transport_sparse_matrix_uses_only_active_source_routes():
    ranks = 16

    def matrix(*cells):
        rows = [[0] * ranks for _ in range(ranks)]
        for source, destination in cells:
            rows[source][destination] = 6144
        return tuple(tuple(row) for row in rows)

    supply = _tiered_supply(2)
    for name, cells, expected_us in (
        ("package-only", ((0, 1),), 0.8),
        ("node-only", ((0, 2),), 2.5),
        ("both-one-source", ((0, 1), (0, 2)), 3.3),
        ("reverse-only", ((2, 0),), 2.5),
        ("diagonal-only", (), 0.0),
    ):
        cost = _cost(tokens=1, top_k=1, row_bytes=8192.0, ranks=ranks,
                     node_ranks=16, mtu=4096, supply=supply,
                     matrix=matrix(*cells))
        assert _dep_ms(cost) == Fraction(str(expected_us)) / 1000, name


def test_transport_package4_node4_sparse_active_class():
    # Four-die packages on four-die nodes: 0->1 package, 0->4 cross-node,
    # 4->6 package.  The critical source is 0: 0.8 + 1.8 = 2.6 us.
    ranks = 8
    rows = [[0] * ranks for _ in range(ranks)]
    for source, destination in ((0, 1), (0, 4), (4, 6)):
        rows[source][destination] = 6144
    cost = _cost(tokens=1, top_k=1, row_bytes=8192.0, ranks=ranks,
                 node_ranks=4, mtu=4096, supply=_tiered_supply(4),
                 matrix=tuple(tuple(row) for row in rows))
    assert _dep_ms(cost) == Fraction(26, 10_000)


def test_transport_zero_useful_charges_no_propagation():
    cost = _cost(tokens=1, top_k=1, row_bytes=0.0, ranks=16, node_ranks=16,
                 mtu=4096, supply=_tiered_supply(2))
    assert _dep_ms(cost) == 0


def test_transport_declared_algorithm_modes_are_distinct():
    ranks = 16
    rows = [[0] * ranks for _ in range(ranks)]
    rows[0][1] = 6144
    matrix = tuple(tuple(row) for row in rows)
    supply = _tiered_supply(2)
    default = _cost(tokens=1, top_k=1, row_bytes=8192.0, ranks=ranks,
                    node_ranks=16, mtu=4096, supply=supply, matrix=matrix)
    pairwise = _cost(tokens=1, top_k=1, row_bytes=8192.0, ranks=ranks,
                     node_ranks=16, mtu=4096, supply=supply, matrix=matrix,
                     transport_algorithm="pairwise_rounds")
    upper = _cost(tokens=1, top_k=1, row_bytes=8192.0, ranks=ranks,
                  node_ranks=16, mtu=4096, supply=supply, matrix=matrix,
                  transport_algorithm="max_path_rounds")
    assert _dep_ms(default) == Fraction(8, 10_000)
    assert _dep_ms(pairwise) == Fraction(8, 10_000)
    assert _dep_ms(upper) == Fraction(12, 1000)
