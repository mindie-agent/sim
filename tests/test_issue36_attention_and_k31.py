"""Issue #36 attention geometry and K3.1 DCP/QSFA regression coverage.

[Role] Exercise local-TP packing, noncausal truncation, per-rank CP service,
  owned sharded index stores and QSFA output packing through real builders.
[Invariants] Local placement pads to the weight-scale granule; dense noncausal
  TopK keeps the smaller positive cap; a causal window tile reads the visible
  row union; DCP stores count rank-owned positions; QSFA publishes FP32
  statistics and packs four BF16 LSE lanes before its output AllToAll.
[Timeouts & Locks] Bounded CPU-only graphs; no locks, subprocesses or devices.
[Side-Effect Policy] In-memory records only; no filesystem or host mutation.
"""
from dataclasses import replace
from types import SimpleNamespace

from mindie_sim import target_graphs
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes
from mindie_sim.prediction.attention import AttentionPolicy, gqa
from mindie_sim.prediction.generic_latent_attention import (
    owned_store_rows, sparse_over_parts,
)
from mindie_sim.prediction.local_projection_tp import (
    _local_dimension, _packed_local_width,
)
from mindie_sim.prediction.latent_dcp import PACK_LSE_OPS
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView


def _gqa(window, views, cp=2, interleave=4):
    from test_prediction_system import case
    parsed = parse_case(case(output=1))
    plans = tuple(replace(plan, window=window) for plan in parsed.model.layer_plans)
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=plans))
    parallel = replace(parsed.runtime.parallel, cp=cp,
                       cp_plan="dcp" if cp > 1 else "none")
    result = gqa(
        prefix="probe", plan=parsed.model.layer_plans[0], views=views, stage=0,
        deps=("input",), model=parsed.model, parallel=parallel,
        quant=parsed.model.quant, mode="decode", padded_tokens=sum(v.q_len for v in views),
        policy=AttentionPolicy(), cp_interleave_size=interleave,
        runtime=parsed.runtime, cache_ids={}, stage_of=lambda layer: 0)
    return parsed, parallel, result


def _kimi_graph():
    from test_kimi_k31_alignment import case as kimi_case
    cached = getattr(_kimi_graph, "graph", None)
    if cached is None:
        cached = target_graphs(
            kimi_case(cp=8), phase="decode", tokens=1, context=100,
            scope="body", resource_fusion=False)[0]
        _kimi_graph.graph = cached
    return cached


def test_local_projection_pads_to_the_weight_scale_block():
    quant = QuantSpec(name="block128", weight_format="fp8_e4m3",
                      activation_format="fp8_e4m3", weight_block=(128, 128))
    assert _local_dimension(512, 8, quant) == 128
    packed = {"packed_projection_slices": ({"shape": (2, 512)},)}
    assert _packed_local_width(packed, 8, 512, quant) == 128
    parallel = ParallelSpec(linear_proj_tp=8)
    term = WeightTerm("kda_q", 512, 256, "kda.q", "out_tp", quant=quant,
                      placement_role="linear_input")
    assert term_bytes(term, quant, parallel) == quant.matrix_bytes(128, 256, "kda.q")


def test_dense_noncausal_topk_keeps_the_smaller_window_cap():
    plan = SimpleNamespace(
        index_topk=16, compress_ratio=1, window=4, index_kpool=1,
        index_kpool_always_select_tail=False, noncausal_attention=True,
        cache_kind="dense")
    traffic = sparse_over_parts(
        ((2, 100),), plan, 128, dense=True, kv_len=102, topk=4)
    assert traffic.gather_rows == 8
    assert traffic.total_cache_rows == 8


def test_causal_window_tile_reads_the_visible_union_not_only_the_window():
    parsed, _, result = _gqa(
        64, (RequestView("r", 32, 64, False, "prefill"),), cp=1)
    attention = next(op for op in result.ops if op.family == "attention")
    meta = dict(attention.metadata)
    assert meta["kv_sweeps"] == 95
    query_bytes = 32 * 4 * 8 * dtype_bytes(parsed.model.activation_dtype)
    kv_row = 2 * 2 * 8 * dtype_bytes(parsed.model.activation_dtype)
    assert attention.mte2_bytes - query_bytes == 95 * kv_row


def test_batched_dcp_keeps_per_rank_pairs_and_sweeps_separate():
    views = (RequestView("a", 1, 0, False, "decode"),
             RequestView("b", 3, 11, False, "decode"))
    _, _, result = _gqa(8, views, cp=2, interleave=4)
    attention = next(op for op in result.ops if op.family == "attention")
    meta = dict(attention.metadata)
    assert meta["cp_per_rank_pairs"] == (13.0, 12.0)
    assert meta["cp_per_rank_sweeps"] == (5, 6)
    assert meta["kv_sweeps"] == 6
    assert meta["cp_bottleneck_pairs_rank"] == 0
    assert meta["cp_bottleneck_sweeps_rank"] == 1


def test_pcp_dcp_prefill_stores_all_rank_owned_rows_without_raw_division():
    parallel = ParallelSpec(tp=8, cp=4, pcp=2, cp_plan="pcp_dcp")
    views = (SimpleNamespace(context=0, phase="prefill", q_len=8),
             SimpleNamespace(context=100, phase="decode", q_len=1))
    assert owned_store_rows([8, 1], views, parallel, 1536) == 9


def test_kimi_index_store_and_qsfa_publish_owned_rows_and_statistics():
    graph = _kimi_graph()
    store = next(op for op in graph if op.recipe == "index_store")
    assert store.local_shape[0] == 1
    qsfa = next(op for op in graph if op.recipe == "paired_fp8_qsfa_analytic")
    meta = dict(qsfa.metadata)
    latent_b = 512 * 2
    statistics_b = 96 * 2 * 4
    assert meta["gathered_q_heads"] == 96
    assert meta["qsfa_output_bytes"] == 96 * (latent_b + 8)
    assert meta["qsfa_statistics_bytes"] == statistics_b
    assert qsfa.mte3_bytes == 96 * (latent_b + 8)
    packs = [op for op in graph if op.recipe == "dcp_output_pack"]
    assert len(packs) == 24
    pack = packs[0]
    assert pack.mte2_bytes == 96 * (latent_b + 8)
    assert pack.mte3_bytes == 96 * (512 + 4) * 2
    assert pack.vector_ops == 96 * PACK_LSE_OPS
    alltoall = next(op for op in graph if op.recipe == "dcp_output_alltoall")
    assert alltoall.link_bytes == 12 * (512 + 4) * 2 * (8 - 1)
