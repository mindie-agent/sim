"""Physical invariants for matrix, MoE, norm/activation and collective families.

[Role]
  Exercise the piecewise family compilers directly with declared static
  geometry and assert independent arithmetic identities.

[Invariants]
  Fractal rounding, capacity-screened tiles, fill/steady pipeline makespan,
  granule amplification, ragged-expert tile sums, zero-expert weight reads,
  UB row alignment and directed AllToAllV hotspots are checked independently.

[Timeouts & Locks]
  Pure bounded arithmetic on immutable records.  No I/O, waits or locks.

[Side-Effect Policy]
  Test-local construction only; no device, file or global mutation.
"""
from __future__ import annotations

from fractions import Fraction
import math

import pytest

from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply, RequestGeometry,
    WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.families_moe_collective import (
    DEFAULT_L0A_BYTES, DEFAULT_L0B_BYTES, DEFAULT_L0C_BYTES, DEFAULT_L1_BYTES,
)
from mindie_sim.prediction.subgraphs.state import MoeRoutingState, StateView
from mindie_sim.schema.common import ModelingError
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


def make_op(name, family, *, role=None, matrix=(), experts=(), quant=(),
            parallel=(), tiles=(), extras=()):
    template = OperatorTemplate(name=name, family=family, role=role or name)
    signature = StaticSignature(
        role=name, matrix=matrix, experts=experts, quant=quant,
        parallel=parallel, tiles=tiles, extras=extras)
    return BoundOperator(template, signature)


def compile_op(operator, *, policy=POLICY, supply=SUPPLY):
    return OperatorPiecewiseCompiler(supply, policy).compile_operator(operator)


def cost_for(operator, tokens, *, state=None, policy=POLICY, supply=SUPPLY):
    compiled = compile_op(operator, policy=policy, supply=supply)
    geometry = WorkloadGeometry.homogeneous(1, tokens, tokens + 1)
    return compiled, compiled.evaluate(geometry, state)


def gemm_theta(k, n, *, mt=16, nt=16, cores=1, act_bytes=2.0,
               weight_format="bf16", act_format="bf16", extras=()):
    weight_width = {"bf16": 2.0, "int8": 1.0, "int4": 0.5,
                    "mxfp8": 33.0 / 32.0}.get(weight_format, 2.0)
    return (
        (("K", k), ("N", n)),
        (("act_bytes", act_bytes), ("act_format", act_format),
         ("weight_format", weight_format),
         ("weight_total", n * k * weight_width),
         ("weight_payload", n * k * weight_width)),
        (("mt", mt), ("nt", nt), ("cores", cores)),
        tuple(extras),
    )


def gemm_op(name="MatMulV2", k=16, n=16, **kwargs):
    matrix, quant, tiles, extras = gemm_theta(k, n, **kwargs)
    return make_op(name, "gemm", role="projection", matrix=matrix,
                   quant=quant, tiles=tiles, extras=extras)


def stage_pipe(cost, stage, pipe):
    return dict(cost.stage_work[stage]).get(pipe, Fraction(0))


def total_pipe(cost, pipe):
    return sum((dict(stage).get(pipe, Fraction(0)) for stage in cost.stage_work),
               Fraction(0))


# 1. Fractal alignment generalises over power-of-two and odd extents.


def test_fractal_alignment_and_capacity_screened_tiling():
    operator = gemm_op(k=250, n=300, mt=256, nt=256)
    compiled, cost = cost_for(operator, 100)
    assert cost.get("fractal_m") == 112
    assert cost.get("fractal_n") == 304
    assert cost.get("fractal_k") == 256
    assert cost.get("tile_m") % 16 == 0
    assert cost.get("tile_n") % 16 == 0
    assert cost.get("tile_k") % 16 == 0
    assert cost.get("tile_capacity_fits") == 1
    assert cost.get("l1_tile_bytes") <= DEFAULT_L1_BYTES
    assert cost.get("l0a_tile_bytes") <= DEFAULT_L0A_BYTES
    assert cost.get("l0b_tile_bytes") <= DEFAULT_L0B_BYTES
    assert cost.get("l0c_tile_bytes") <= DEFAULT_L0C_BYTES
    assert cost.get("tile_padded_m") >= cost.get("fractal_m")
    assert cost.get("padded_flops") == (
        2 * cost.get("tile_padded_m") * 304 * 256)
    assert cost.get("padded_flops") >= cost.get("flops")

    for tokens in (1, 2, 15, 16, 17, 31, 32, 33, 100):
        _, probe = cost_for(gemm_op(k=33, n=17), tokens)
        tile = int(probe.get("tile_m"))
        assert probe.get("fractal_m") == math.ceil(tokens / 16) * 16
        assert probe.get("fractal_n") == 32
        assert probe.get("fractal_k") == 48
        assert int(probe.get("tile_padded_m")) == math.ceil(tokens / tile) * tile
        assert probe.get("padded_flops") == (
            2 * probe.get("tile_padded_m") * 32 * 48)

    int8 = gemm_op(k=33, n=17, act_bytes=1.0, weight_format="int8",
                   act_format="int8")
    _, int8_cost = cost_for(int8, 17)
    assert int8_cost.get("k_align") == 32
    assert int8_cost.get("fractal_k") == 64

    tight = gemm_op(k=256, n=256, mt=256, nt=256,
                    extras=(("l1_bytes", 64 * 1024),
                            ("l0c_bytes", 16 * 1024)))
    _, tight_cost = cost_for(tight, 100)
    assert tight_cost.get("tile_capacity_fits") == 1
    assert tight_cost.get("l1_tile_bytes") <= 64 * 1024
    assert tight_cost.get("l0c_tile_bytes") <= 16 * 1024


# 2. A single tile is serial; multiple tiles use fill + steady intervals.


def test_single_tile_serial_and_multi_tile_fill_steady_pipeline():
    operator = gemm_op(k=16, n=16)
    _, single = cost_for(operator, 16)
    _, double = cost_for(operator, 32)

    assert single.get("j_tiles") == 1 and single.get("pipeline_q") == 1
    load = single.get("tile_load_bytes")
    compute = single.get("tile_compute_flops")
    assert single.duration_ms == pytest.approx(float(
        load / 1000 + compute / 1000 + Fraction(512, 1000)))
    assert single.stage_times[3] == 0

    assert double.get("j_tiles") == 2 and double.get("pipeline_q") == 2
    fill = load / 1000 + compute / 1000
    steady = max(load, compute, Fraction(0)) / 1000
    merge = 32 * 16 * 2 / 1000
    assert double.duration_ms == pytest.approx(float(fill + steady + merge))
    assert double.stage_times[3] == pytest.approx(float(steady))

    serial = gemm_op(k=16, n=16, extras=(("pipeline_slots", 1),))
    _, serial_cost = cost_for(serial, 32)
    assert serial_cost.duration_ms == pytest.approx(float(fill + fill + merge))
    assert serial_cost.duration_ms > double.duration_ms
    assert serial_cost.stage_times[3] == pytest.approx(float(load / 1000))
    assert serial_cost.stage_times[4] == pytest.approx(float(compute / 1000))


# 3. MTE2 granule rounding amplifies a sub-granule tile load.


def test_mte2_transaction_granule_rounding_and_efficiency():
    _, exact = cost_for(gemm_op(k=16, n=16, extras=(("mte2_granule", 512),)), 16)
    _, coarse = cost_for(gemm_op(k=16, n=16, extras=(("mte2_granule", 4096),)), 16)
    assert exact.get("mte2_useful_bytes") == exact.get("tile_load_bytes") == 1024
    assert exact.get("mte2_padded_bytes") == 1024
    assert exact.get("mte2_efficiency") == 1
    assert coarse.get("mte2_useful_bytes") == 1024
    assert coarse.get("mte2_padded_bytes") == 4096
    assert coarse.get("mte2_efficiency") == Fraction(1, 4)
    assert total_pipe(coarse, "mte2") > total_pipe(exact, "mte2")


# 4. Split-K is bounded by physical K panels and keeps padded cube work.


def test_split_k_trigger_merge_stage_and_panel_cap():
    operator = gemm_op(k=512, n=64, nt=16)
    fast = ImplementationPolicy(mt=16, nt=16, cores=1, split_target=50_000,
                                max_splits=4)
    slow = ImplementationPolicy(mt=16, nt=16, cores=1, split_target=10 ** 12,
                                max_splits=4)
    _, split = cost_for(operator, 16, policy=fast)
    _, whole = cost_for(operator, 16, policy=slow)
    assert split.get("k_panels") == 4
    assert dict(split.regime)["SPLITS"] == 4
    assert split.get("k_split_panels") == 1
    assert dict(whole.regime)["SPLITS"] == 1
    assert split.get("padded_flops") == whole.get("padded_flops")
    assert total_pipe(split, "cube") == total_pipe(whole, "cube")
    # Each split tile stores one partial (16x16x4); merge reads only the
    # (S-1) extra partial rows.  Neither side is multiplied by S twice.
    assert split.get("partial_write_bytes") == 16 * 1024
    assert split.get("partial_read_bytes") == 3 * 16 * 64 * 4
    assert split.get("partial_bytes") == 16 * 1024 + 3 * 16 * 64 * 4
    assert total_pipe(split, "mte3") == split.get("partial_write_bytes") + 16 * 64 * 2
    # The split partial is published only after every K panel of the split
    # accumulates, and the merge is the final stage.
    assert stage_pipe(split, -2, "mte3") == split.get("partial_write_bytes")
    assert stage_pipe(split, -1, "mte2") == split.get("partial_read_bytes")
    assert split.get("mte2_useful_bytes") == whole.get("mte2_useful_bytes")

    capped = ImplementationPolicy(mt=16, nt=16, cores=1, split_target=1,
                                  max_splits=8)
    _, capped_cost = cost_for(gemm_op(k=32, n=16), 4, policy=capped)
    assert dict(capped_cost.regime)["SPLITS"] <= capped_cost.get("k_panels")


# 5. MoE ragged tiles, skew waves and zero-token expert weight reads.


def grouped_op(*, e=8, local=8, top_k=2, k=64, n=128, row_tile=16,
               cores=2):
    return make_op(
        "expert_gate_up", "grouped_gemm", role="gate_up",
        matrix=(("K", k), ("N", n)),
        experts=(("E", e), ("local", local), ("k", top_k)),
        quant=(("act_bytes", 2.0), ("act_format", "bf16"),
               ("weight_format", "bf16"), ("weight_total", n * k * 2.0)),
        tiles=(("cores", cores), ("nt", 16), ("moe_rows", row_tile)),
        extras=(("row_tile", row_tile),))


def test_moe_skew_waves_and_zero_expert_weight_reads():
    operator = grouped_op()
    balanced = StateView(routing=((0, MoeRoutingState((2,) * 8, 2, 8)),))
    skewed = StateView(routing=((0, MoeRoutingState((0,) * 7 + (16,), 2, 8)),))
    _, left = cost_for(operator, 8, state=balanced)
    _, right = cost_for(operator, 8, state=skewed)
    assert left.get("moe_rows") == right.get("moe_rows") == 16
    assert left.get("active_experts") == 8
    assert right.get("active_experts") == 1
    assert right.get("zero_experts") == 7
    assert left.get("weight_bytes") == 8 * 128 * 64 * 2
    assert right.get("weight_bytes") == 128 * 64 * 2
    assert left.get("padded_rows") == 8 * 16
    assert right.get("padded_rows") == 16
    assert left.get("waves") > right.get("waves")

    row_tile = int(left.get("row_tile"))
    n_tiles = int(left.get("n_tile"))
    k_panels = int(left.get("k_panels"))
    for counts, cost in (((2,) * 8, left), ((0,) * 7 + (16,), right)):
        row_tiles = sum(math.ceil(value / row_tile) for value in counts if value)
        assert cost.get("row_tiles") == row_tiles
        assert cost.get("tiles") == row_tiles * n_tiles * k_panels

    ragged = StateView(routing=((0, MoeRoutingState((1, 0, 3, 0, 5, 0, 7, 0), 2, 8)),))
    _, ragged_cost = cost_for(operator, 8, state=ragged)
    row_tiles = sum(math.ceil(value / row_tile) for value in
                    (1, 0, 3, 0, 5, 0, 7, 0) if value)
    assert ragged_cost.get("row_tiles") == row_tiles
    assert ragged_cost.get("tiles") == row_tiles * n_tiles * k_panels
    assert ragged_cost.get("padded_rows") == 4 * 16


def test_moe_init_routing_histogram_sort_permute():
    operator = make_op(
        "MoEInitRouting", "router", role="routing",
        matrix=(("elements", 8),), experts=(("E", 8), ("k", 2)),
        tiles=(("mt", 16), ("cores", 1),))
    _, cost = cost_for(operator, 4)
    expected = 8 * (1 - (1 - 2 / 8) ** 4)
    assert abs(float(cost.get("expected_active_experts")) - expected) < 1e-12
    assert cost.get("histogram_bins") == 8
    assert cost.get("sort_ops") == 4 * 2 * 1
    assert cost.get("permute_indices") == 4 * 2
    assert cost.get("routing_tokens") == 4


def test_moe_uniform_expected_active_experts_is_fractional_moment():
    _, cost = cost_for(grouped_op(e=8, top_k=2), 4)
    expected = 8 * (1 - (1 - 2 / 8) ** 4)
    assert abs(float(cost.get("expected_active_experts")) - expected) < 1e-12
    _, saturated = cost_for(grouped_op(e=8, top_k=8), 4)
    assert saturated.get("expected_active_experts") == 8


# 6. Directed AllToAllV matrix: send volume, receive hotspot, node split.


def traffic_op(name="alltoallv", family="transport", role="dispatch",
               matrix=None, ranks=4, node_ranks=2, mtu=1, chunks=1,
               row_bytes=8.0, top_k=1):
    extra = (("ranks_per_node", node_ranks), ("mtu_bytes", mtu),
             ("chunks", chunks), ("matrix_per_token", 0),
             ("top_k", top_k))
    if matrix is not None:
        extra += (("traffic_matrix", matrix),)
    return make_op(
        name, family, role=role, matrix=(("K", 1), ("N", 1)),
        quant=(("row_bytes", row_bytes),),
        tiles=(("cores", 1),), parallel=(("members", ranks),), extras=extra)


def test_alltoallv_directed_matrix_receive_hotspot():
    matrix = ((0, 100, 200, 300), (400, 0, 500, 600),
              (700, 800, 0, 900), (1000, 1100, 1200, 0))
    _, cost = cost_for(traffic_op(matrix=matrix), 2)
    rows = [sum(row) for row in matrix]
    columns = [sum(matrix[row][column] for row in range(4))
               for column in range(4)]
    def node_send(same_node):
        return [sum(matrix[i][j] for j in range(4)
                    if i != j and (i // 2 == j // 2) == same_node)
                for i in range(4)]

    def node_recv(same_node):
        return [sum(matrix[i][j] for i in range(4)
                    if i != j and (i // 2 == j // 2) == same_node)
                for j in range(4)]

    intra = max(max(node_send(True)), max(node_recv(True)))
    inter = max(max(node_send(False)), max(node_recv(False)))
    assert cost.get("send_bytes") == max(rows)
    assert cost.get("recv_hotspot") == max(columns)
    assert cost.get("aggregate_wire_bytes") == sum(rows)
    # Link pipes carry the single-rank send/receive bottleneck, not the
    # all-rank aggregate; the aggregate remains a derived ledger only.
    assert cost.get("intra_bytes") == intra
    assert cost.get("inter_bytes") == inter
    assert cost.get("local_hbm_bytes") == max(rows) + max(columns)
    assert stage_pipe(cost, 0, "link_hccs") == intra
    assert stage_pipe(cost, 0, "link_roce") == inter

    local = ((10, 100), (200, 20))
    _, local_cost = cost_for(traffic_op(matrix=local, ranks=2,
                                        node_ranks=2), 1)
    assert local_cost.get("send_bytes") == 220
    assert local_cost.get("recv_hotspot") == 210
    assert local_cost.get("aggregate_wire_bytes") == 300
    assert local_cost.get("local_hbm_bytes") == 430


def test_alltoallv_mtu_frame_padding_changes_hotspot():
    matrix = ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0))
    _, exact = cost_for(traffic_op(matrix=matrix, mtu=1), 1)
    _, framed = cost_for(traffic_op(matrix=matrix, mtu=64), 1)
    assert exact.get("recv_hotspot") == 3
    assert exact.get("aggregate_wire_bytes") == 12
    assert framed.get("recv_hotspot") == 3 * 64
    assert framed.get("aggregate_wire_bytes") == 12 * 64


# 7. Collective topology volumes and chunk pipeline stages.


def collective_op(name, ranks=8, node_ranks=4, wire=10.0, chunks=1,
                  mtu=1, topology=None):
    extra = (("ranks_per_node", node_ranks), ("chunks", chunks),
             ("mtu_bytes", mtu))
    if topology is not None:
        extra += (("topology", topology),)
    return make_op(
        name, "collective", role="tp",
        quant=(("wire_bytes", wire),), parallel=(("members", ranks),),
        tiles=(("cores", 1),), extras=extra)


def test_collective_topology_volumes_without_model_discount():
    _, hierarchical = cost_for(collective_op("allreduce"), 3)
    vector = 30
    assert hierarchical.get("intra_bytes") == Fraction(45)
    assert hierarchical.get("inter_bytes") == Fraction(15, 2)
    assert hierarchical.get("local_hbm_bytes") == 2 * vector
    assert stage_pipe(hierarchical, 0, "link_hccs") == Fraction(45)
    assert stage_pipe(hierarchical, 0, "link_roce") == Fraction(15, 2)

    _, flat = cost_for(collective_op("allreduce", node_ranks=8), 3)
    assert flat.get("intra_bytes") == Fraction(105, 2)
    assert flat.get("inter_bytes") == 0
    assert flat.get("local_hbm_bytes") == 2 * vector
    assert (hierarchical.get("intra_bytes") + hierarchical.get("inter_bytes")
            == flat.get("intra_bytes") + flat.get("inter_bytes"))

    _, scatter = cost_for(collective_op("reduce_scatter"), 3)
    assert scatter.get("intra_bytes") == Fraction(45, 2)
    assert scatter.get("inter_bytes") == Fraction(15, 4)
    _, gather = cost_for(collective_op("all_gather"), 3)
    assert gather.get("intra_bytes") == Fraction(90)
    assert gather.get("inter_bytes") == Fraction(120)
    assert (gather.get("intra_bytes") + gather.get("inter_bytes")
            == 7 * vector)

    _, uniform = cost_for(collective_op("alltoallv"), 3)  # 30 bytes: 5/peer
    assert uniform.get("intra_bytes") == 3 * 5
    assert uniform.get("inter_bytes") == 4 * 5

    _, hccs = cost_for(collective_op("allreduce", topology="hccs"), 3)
    assert hccs.get("inter_bytes") == 0
    assert hccs.get("intra_bytes") == Fraction(105, 2)
    _, roce = cost_for(collective_op("allreduce", node_ranks=8,
                                     topology="roce"), 3)
    assert roce.get("intra_bytes") == 0
    assert roce.get("inter_bytes") == Fraction(105, 2)


def test_collective_chunk_pipeline_partitions_work_without_discount():
    _, single = cost_for(collective_op("allreduce", chunks=1), 3)
    _, chunked = cost_for(collective_op("allreduce", chunks=4), 3)
    assert len(single.stage_times) == 1
    assert len(chunked.stage_times) == 2
    for pipe in ("link_hccs", "link_roce", "hbm"):
        assert total_pipe(single, pipe) == total_pipe(chunked, pipe)
    assert chunked.get("chunks") == 4
    assert stage_pipe(chunked, 0, "link_hccs") == total_pipe(chunked, "link_hccs") / 4


def test_mc2_transport_uses_rows_and_directed_hotspot():
    operator = traffic_op(name="combine", family="transport", role="combine",
                          ranks=4, node_ranks=2, mtu=1, chunks=4,
                          row_bytes=1000.0, top_k=2)
    _, cost = cost_for(operator, 3)
    message = 3 * 2 * 1000
    assert cost.get("routed_rows") == 6
    assert cost.get("peer_bytes") == message
    assert cost.get("send_bytes") == message
    assert cost.get("recv_hotspot") == message
    assert cost.get("intra_bytes") == Fraction(message, 3)
    assert cost.get("inter_bytes") == Fraction(2 * message, 3)
    assert len(cost.stage_times) == 2
    assert cost.get("local_hbm_bytes") == 2 * message
    for pipe in ("link_hccs", "link_roce", "hbm"):
        assert total_pipe(cost, pipe) == stage_pipe(cost, 0, pipe) * 4


# 8. UB row alignment, reductions and quantised residual normalisation.


def elementwise_op(name, elements=100, read=200.0, write=200.0, ops=5,
                   extras=()):
    return make_op(
        name, "elementwise", role=name,
        matrix=(("elements", elements),),
        quant=(("act_bytes", 2.0),),
        tiles=(("mt", 16), ("cores", 1),),
        extras=(("ops_per_element", ops), ("read_bytes", read),
                ("write_bytes", write), *extras))


def test_rmsnorm_ub_row_alignment_and_vector_reduction():
    _, cost = cost_for(elementwise_op("RmsNorm", elements=100), 3)
    assert cost.get("ub_row_bytes") == 224
    assert cost.get("ub_write_row_bytes") == 224
    assert cost.get("ub_padded_bytes") == 3 * 224
    assert cost.get("vector_reduction_ops") == 3 * 2
    assert stage_pipe(cost, 0, "mte2") == 672
    assert stage_pipe(cost, 0, "mte3") == 672
    _, aligned = cost_for(elementwise_op("RmsNorm", elements=96,
                                          read=192.0, write=192.0), 3)
    assert aligned.get("ub_row_bytes") == 192
    assert aligned.get("ub_padded_bytes") == 3 * 192


def test_add_rmsnorm_quant_keeps_residual_read_and_quantised_write():
    operator = elementwise_op(
        "AddRmsNormQuant", elements=100, read=400.0, write=200.0,
        extras=(("quantized", 1),))
    _, cost = cost_for(operator, 3)
    assert cost.get("quantized_output") == 1
    assert cost.get("ub_row_bytes") == 416
    assert cost.get("ub_write_row_bytes") == 128
    assert stage_pipe(cost, 0, "mte2") == 3 * 416
    assert stage_pipe(cost, 0, "mte3") == 3 * 128
    assert cost.get("vector_reduction_ops") == 6


def test_swiglu_geglu_gating_and_dequant_swiglu_quant():
    _, gate = cost_for(elementwise_op("SwiGLU", elements=128, read=512.0,
                                      write=256.0, ops=4), 3)
    assert gate.get("gated_inputs") == 3 * 128 * 2
    assert gate.get("ub_row_bytes") == 512
    assert gate.get("ub_write_row_bytes") == 256
    _, quant = cost_for(elementwise_op(
        "DequantSwigluQuant", elements=128, read=512.0, write=256.0, ops=4,
        extras=(("quantized", 1),)), 3)
    assert quant.get("quantized_output") == 1
    assert quant.get("ub_write_row_bytes") == 128
    _, gelu = cost_for(elementwise_op("GeGLU", elements=96, read=384.0,
                                      write=192.0, ops=4), 2)
    assert gelu.get("gated_inputs") == 2 * 96 * 2


def test_gatherv2_index_read_is_separate_from_payload():
    operator = make_op(
        "GatherV2", "copy", role="gather",
        matrix=(("elements", 64),), quant=(("act_bytes", 2.0),),
        tiles=(("mt", 16), ("cores", 1),), extras=(("index_bytes", 4.0),))
    _, cost = cost_for(operator, 5)
    assert cost.get("gather_indices") == 5
    assert cost.get("copy_bytes") == 5 * 64 * 2
    assert stage_pipe(cost, 0, "mte2") == 5 * (64 * 2 + 4)
    assert stage_pipe(cost, 0, "mte3") == 5 * 64 * 2


def test_add_rmsnorm_qknorm_mhc_and_attention_residual():
    for name in ("AddRmsNorm", "QKNorm"):
        _, cost = cost_for(elementwise_op(name, elements=64, read=128.0,
                                          write=128.0), 2)
        assert cost.get("vector_reduction_ops") == 2 * 1
        assert cost.get("ub_row_bytes") == 128

    _, mhc = cost_for(elementwise_op("mHC", elements=64, read=128.0,
                                     write=128.0, ops=6), 2)
    assert mhc.get("ub_row_bytes") == 128
    assert mhc.get("vector_ops") == 2 * 64 * 6

    _, residual = cost_for(elementwise_op("AttentionResidual", elements=64,
                                          read=256.0, write=128.0, ops=2), 2)
    assert residual.get("ub_row_bytes") == 256
    assert stage_pipe(residual, 0, "mte2") == 2 * 256
    assert stage_pipe(residual, 0, "mte3") == 2 * 128


# 9. Quantised GEMM modes and transpose batch traffic.


def test_quantized_gemm_modes_and_transpose_batch_matmul():
    _, bf16 = cost_for(gemm_op("MatMulV2", k=64, n=64), 8)
    int8 = gemm_op("QuantBatchMatmulV3", k=64, n=64, act_bytes=1.0,
                   weight_format="int8", act_format="int8")
    compiled_int8, int8_cost = cost_for(int8, 8)
    assert dict(compiled_int8.program.annotations)["mode"] == "quant_batch_matmul_v3"
    assert int8_cost.get("quantized") == 1
    assert int8_cost.get("weight_total") < bf16.get("weight_total")
    assert int8_cost.pipe_work("vector") > bf16.pipe_work("vector")

    w4 = gemm_op("QuantBatchMatmulV3", k=64, n=64, act_bytes=1.0,
                 weight_format="int4", act_format="int8")
    _, w4_cost = cost_for(w4, 8)
    assert w4_cost.get("weight_total") < int8_cost.get("weight_total")

    mx = gemm_op("QuantBatchMatmulV3", k=64, n=64, act_bytes=1.0,
                 weight_format="mxfp8", act_format="fp8_e4m3")
    compiled_mx, _ = cost_for(mx, 8)
    assert dict(compiled_mx.program.annotations)["mode"] == "quant_batch_matmul_v3_mxfp8"

    weight_quant = gemm_op("WeightQuantBatchMatmul", k=64, n=64)
    compiled_wq, _ = cost_for(weight_quant, 8)
    assert dict(compiled_wq.program.annotations)["mode"] == "weight_quant_batch_matmul"

    transpose = gemm_op("TransposeBatchMatMul", k=16, n=32,
                        extras=(("batch", 4),))
    compiled_t, transpose_cost = cost_for(transpose, 8)
    assert dict(compiled_t.program.annotations)["mode"] == "transpose_batch_matmul"
    assert transpose_cost.get("transpose_bytes") == 4 * 32 * 16 * 2 * 2
    assert transpose_cost.get("batch") == 4


# 10. Non-aligned expert dimensions and zero-router work remain exact.


def test_grouped_gemm_non_divisible_dimensions_pad_each_expert():
    operator = grouped_op(e=6, local=6, top_k=3, k=70, n=90, row_tile=16,
                          cores=3)
    counts = (4, 0, 7, 0, 9, 4)
    state = StateView(routing=((0, MoeRoutingState(counts, 3, 8)),))
    _, cost = cost_for(operator, 8, state=state)
    assert sum(counts) == 3 * 8
    row_tile = int(cost.get("row_tile"))
    n_tiles = int(cost.get("n_tile"))
    k_panels = int(cost.get("k_panels"))
    expected = sum(math.ceil(value / row_tile) * n_tiles * k_panels
                   for value in counts if value)
    assert cost.get("tiles") == expected
    assert cost.get("padded_rows") == sum(
        math.ceil(value / row_tile) * row_tile for value in counts if value)
    assert cost.get("flops") == 2 * sum(counts) * 90 * 70
    assert cost.get("padded_flops") == 2 * cost.get("padded_rows") * 96 * 80


# 11. Split-K partial ownership, partial-wave loads and internal K panels.


def test_gemm_split_partial_and_tail_core_physics():
    # A partial wave loads only the real tiles, while Cube stays on the
    # per-core wave lattice.
    _, tail = cost_for(gemm_op(k=16, n=16, cores=4), 16)
    assert tail.get("j_tiles") == 1 and tail.get("waves") == 1
    assert total_pipe(tail, "mte2") == tail.get("mte2_padded_bytes") == 1024
    assert total_pipe(tail, "cube") == 4 * 2 * 16 * 16 * 16
    assert total_pipe(tail, "mte3") == 16 * 16 * 2

    # One wave with four internal K panels pipelines the panels instead of
    # serialising them inside one full-K tile.
    _, panel = cost_for(gemm_op(k=512, n=16, cores=4), 16)
    assert panel.get("k_panels") == 4 and panel.get("k_split_panels") == 4
    assert total_pipe(panel, "mte2") == 4 * (16 * 128 * 2 + 128 * 16 * 2)
    assert total_pipe(panel, "cube") == 4 * 2 * 16 * 16 * 512
    assert panel.stage_work[3] and dict(panel.stage_work[3])["cube"] == (
        4 * 3 * 2 * 16 * 16 * 128)


def test_gemm_batch_multiplicity_metadata_and_capacity():
    _, batched = cost_for(gemm_op(k=16, n=16,
                                  extras=(("batch", 4),)), 8)
    assert batched.get("flops") == 2 * 4 * 8 * 16 * 16
    assert batched.get("j_tiles") == 4
    assert batched.get("output_bytes") == 4 * 8 * 16 * 2
    assert batched.get("mte2_useful_bytes") == 4 * (16 * 16 * 2 + 16 * 16 * 2)

    quantized = make_op(
        "QuantBatchMatmulV3", "gemm", role="projection",
        matrix=(("K", 32), ("N", 16)),
        quant=(("act_bytes", 1.0), ("act_format", "int8"),
               ("weight_format", "int8"), ("weight_total", 640),
               ("weight_payload", 512), ("weight_scale", 64),
               ("weight_zero", 64)),
        tiles=(("mt", 16), ("nt", 16), ("cores", 1)))
    _, metadata = cost_for(quantized, 16)
    assert metadata.get("tile_load_bytes") == 512 + 512 + 128
    assert total_pipe(metadata, "mte2") == 1536

    tight = gemm_op(k=16, n=16, extras=(("l0c_bytes", 1),))
    with pytest.raises(ModelingError, match="tile_plan_capacity_exceeded"):
        compile_op(tight)


# 12. Grouped GEMM K panels accumulate one output tile and zero work is valid.


def test_grouped_gemm_k_panels_single_output_and_zero_work():
    operator = grouped_op(e=1, local=1, top_k=1, k=512, n=16, row_tile=16,
                          cores=1)
    state = StateView(routing=((0, MoeRoutingState((16,), 1, 16)),))
    _, cost = cost_for(operator, 16, state=state)
    assert cost.get("k_panels") == 4
    assert cost.get("tiles") == 1 and cost.get("waves") == 1
    assert cost.get("output_bytes") == 16 * 16 * 2
    assert total_pipe(cost, "mte3") == 16 * 16 * 2
    assert total_pipe(cost, "cube") == 2 * 16 * 16 * 512

    empty = grouped_op(e=2, local=2, top_k=2, k=16, n=16, row_tile=16,
                       cores=2)
    zero = StateView(routing=((0, MoeRoutingState((0, 0), 2, 0)),))
    _, no_work = cost_for(empty, 4, state=zero)
    assert no_work.get("active_experts") == 0
    assert no_work.get("tiles") == 0 and no_work.get("waves") == 0
    assert no_work.duration_ms == 0
    for pipe in ("cube", "vector", "hbm", "mte1", "mte2", "mte3",
                 "fixpipe"):
        assert total_pipe(no_work, pipe) == 0
    assert dict(no_work.materialization)["expert_gate_up_output"] == 0


def test_grouped_gemm_quant_metadata_enters_tile_load():
    operator = make_op(
        "expert_gate_up", "grouped_gemm", role="gate_up",
        matrix=(("K", 32), ("N", 16)),
        experts=(("E", 1), ("local", 1), ("k", 1)),
        quant=(("act_bytes", 1.0), ("act_format", "int8"),
               ("weight_format", "int8"), ("weight_total", 640),
               ("weight_payload", 512), ("weight_scale", 64),
               ("weight_zero", 64)),
        tiles=(("cores", 1), ("nt", 16), ("moe_rows", 16)))
    state = StateView(routing=((0, MoeRoutingState((16,), 1, 16)),))
    _, cost = cost_for(operator, 16, state=state)
    assert cost.get("tile_load_bytes") == 512 + 512 + 128
    assert total_pipe(cost, "mte2") == 512 + 512 + 128
    assert cost.get("weight_bytes") == 640


# 13. Default MoE activation rows conserve the grouped-GEMM top-k fallback.


def test_elementwise_moe_default_rows_follow_top_k():
    operator = make_op(
        "expert_activation", "elementwise", role="activation",
        matrix=(("elements", 32),),
        quant=(("act_bytes", 2.0),),
        tiles=(("mt", 16), ("cores", 1)),
        extras=(("ops_per_element", 4), ("read_bytes", 128.0),
                ("write_bytes", 64.0), ("k", 2), ("local", 1)))
    compiled = OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        operator, kind="F3")
    cost = compiled.evaluate(WorkloadGeometry.homogeneous(1, 4, 5))
    assert cost.get("elements") == 8 * 32
    assert cost.pipe_work("mte2") == 8 * 128

    grouped = grouped_op(e=4, local=4, top_k=2, k=16, n=16, row_tile=16,
                         cores=1)
    _, grouped_cost = cost_for(grouped, 4)
    assert grouped_cost.get("moe_rows") == 8
    sharded = grouped_op(e=4, local=1, top_k=2, k=16, n=16, row_tile=16,
                         cores=1)
    _, sharded_cost = cost_for(sharded, 4)
    assert sharded_cost.get("moe_rows") == 2


# 14. Uniform traffic matrices and the uniform AllToAllV fallback agree.


def test_uniform_traffic_matrix_matches_uniform_alltoallv_fallback():
    matrix = ((0, 128, 128, 128), (128, 0, 128, 128),
              (128, 128, 0, 128), (128, 128, 128, 0))
    keys = ("intra_bytes", "inter_bytes", "send_bytes", "recv_hotspot",
            "local_hbm_bytes", "aggregate_wire_bytes")
    for mtu in (1, 4096):
        fallback = cost_for(collective_op("alltoallv", ranks=4, node_ranks=2,
                                          wire=384.0, mtu=mtu), 1)[1]
        explicit = cost_for(make_op(
            "alltoallv", "collective", role="tp",
            quant=(("wire_bytes", 384.0),), parallel=(("members", 4),),
            tiles=(("cores", 1),),
            extras=(("ranks_per_node", 2), ("chunks", 1),
                    ("mtu_bytes", mtu), ("traffic_matrix", matrix))), 1)[1]
        for key in keys:
            assert explicit.get(key) == fallback.get(key)
        assert explicit.duration_ms == fallback.duration_ms
        transport = dict(name="combine", family="transport", role="combine",
                         ranks=4, node_ranks=2, mtu=mtu, chunks=1,
                         row_bytes=128.0, top_k=3)
        fallback_t = cost_for(traffic_op(**transport), 1)[1]
        explicit_t = cost_for(traffic_op(**transport, matrix=matrix), 1)[1]
        for key in keys:
            assert explicit_t.get(key) == fallback_t.get(key)
        assert explicit_t.duration_ms == fallback_t.duration_ms

# 14b. Selected-row GEMM and EP-localized default routing.

def test_gemm_selected_rows_bind_one_row_per_request():
    operator = gemm_op("lm_head", k=16, n=16,
                       extras=(("selected_rows", 1),))
    compiled = compile_op(operator)
    prefill = WorkloadGeometry(
        tuple(RequestGeometry(8, 0, phase="prefill") for _ in range(3)),
        phase="prefill")
    cost = compiled.evaluate(prefill)
    assert cost.get("flops") == 2 * 3 * 16 * 16
    assert cost.get("output_bytes") == 3 * 16 * 2


def test_grouped_gemm_defaults_and_global_uniform_counts_localize_to_ep():
    operator = grouped_op(e=4, local=2, top_k=2, k=16, n=16, row_tile=16,
                          cores=1)
    _, implicit = cost_for(operator, 32)
    global_uniform = StateView(
        routing=((0, MoeRoutingState((16, 16, 16, 16), 2, 32)),))
    _, explicit = cost_for(operator, 32, state=global_uniform)
    assert implicit.get("moe_rows") == explicit.get("moe_rows") == 32
    assert implicit.duration_ms == explicit.duration_ms


# 14c. Explicit communication membership beats an unrelated world width.

def test_collective_axis_members_select_the_transport_group():
    op = make_op(
        "dispatch", "transport", role="dispatch",
        quant=(("row_bytes", 64.0),),
        parallel=(("ep", 2), ("tp", 4), ("etp", 2), ("world", 8)),
        tiles=(("cores", 1),),
        extras=(("axis", "ep"), ("ep", 2), ("etp", 2), ("top_k", 1),
                ("mtu_bytes", 1), ("ranks_per_node", 2)))
    _, cost = cost_for(op, 1)
    assert cost.get("ranks") == 2


def test_transport_applies_top_k_to_one_latent_row_only():
    common = dict(
        name="combine", family="transport", role="combine",
        parallel=(("ep", 2), ("tp", 4), ("etp", 2), ("world", 4)),
        tiles=(("cores", 1),),
        extras=(("latent", 64), ("top_k", 2), ("mtu_bytes", 1),
                ("ranks_per_node", 2)))
    expanded = make_op(quant=(("row_bytes", 256.0), ("wire_bytes", 2.0)),
                       **common)
    _, cost = cost_for(expanded, 4)
    assert cost.get("ranks") == 2
    assert cost.get("peer_bytes") == 4 * 2 * 64 * 2 == 1024
    ordinary = make_op(quant=(("row_bytes", 128.0), ("wire_bytes", 2.0)),
                       **common)
    _, plain = cost_for(ordinary, 4)
    assert plain.get("peer_bytes") == 1024


# 15. Hierarchical reduce-scatter always returns M/R per rank.


def test_hierarchical_reduce_scatter_output_is_full_rank_shard():
    _, cost = cost_for(collective_op("reduce_scatter", ranks=8,
                                     node_ranks=4, wire=10.0), 3)
    assert cost.get("message_bytes") == 30
    assert cost.get("output_bytes") == Fraction(30, 8)
    assert cost.get("local_hbm_bytes") == 30 + Fraction(30, 8)
