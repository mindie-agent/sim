"""First-principles DSA/SFA sparse Top-K DCP straggler rank-skew tests.

[Role]
  Verify the canonical closed-form sparse DCP rank-skew model
  (``split_sparse_dcp_ranks`` / ``split_sparse_dcp_batch``) and its wiring
  across MLA views, packed-C8 paired-SFA geometry, sparse GQA attention,
  MLA op metadata, and sharded-indexer prefill alignment.

[Invariants]
  * ``cp == 1`` is exact identity (``imbalance == 1.0``).
  * ``K >= L`` matches dense ``owned_positions`` identically, assigning zero
    work to ranks that physically own zero KV tokens when ``L < cp * B``.
  * Block-interleave coherence amplifies skew at realistic ``B`` (768/1536)
    versus near-uniform ``B == 1``, while never exceeding physical capacity.
  * Batch pooling reduces semantic variance across independent requests while
    conserving ``sum_r K_r == sum_i K_i`` and ``0 <= K_r <= sum_i C_{i, r}``
    under both default and explicit override modes.
  * End-to-end sparse DCP lowering on MLA, packed-C8 QSFA, and sparse GQA
    exposes ``cp_sparse_imbalance > 1.0`` and increases straggler attention
    latency relative to the uniform ``1 / cp`` baseline.

[Timeouts & Locks]
  Tier-0 pure in-memory arithmetic, graph lowering, and engine replay; no I/O.

[Side-Effect Policy]
  Read-only catalog/case composition and operator graph analysis.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from mindie_sim.catalog.cache_format import CacheFormat
from mindie_sim.catalog.context_parallel import (
    SparseDcpSequence,
    dcp_sparse_imbalance_override,
    owned_positions,
    round_sparse_dcp_matrix,
    sparse_dcp_capacities,
    split_causal_pairs,
    split_sparse_dcp_batch,
    split_sparse_dcp_batch_per_sequence,
    split_sparse_dcp_ranks,
)
from mindie_sim.compose import compose_case
from mindie_sim.prediction.efficiency import _lower_and_run_graph
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.operators.mla.generic_latent_attention_views import (
    sharded_index_visible_rows,
)
from mindie_sim.prediction.sharding.context_parallel import packed_sfa_geometry
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView

pytestmark = pytest.mark.tier0


def _attention_elapsed_ms(result: dict, recipe: str | None = None) -> float:
    entries = [
        item for item in result["timeline"]
        if item["family"] == "attention" and (recipe is None or item["recipe"] == recipe)
    ]
    assert entries, f"missing attention timeline entries (recipe={recipe})"
    return float(entries[0]["elapsed_ms"])


def test_cp1_identity_and_degenerate_boundaries():
    split_cp1 = split_sparse_dcp_ranks(kv_len=131072, selected_rows=2048, cp=1, interleave=1536)
    assert split_cp1.cp == 1
    assert split_cp1.per_rank == (2048.0,)
    assert split_cp1.max_rank == 2048.0
    assert split_cp1.imbalance == pytest.approx(1.0)

    zero_k = split_sparse_dcp_ranks(kv_len=8192, selected_rows=0, cp=8, interleave=768)
    assert zero_k.per_rank == (0.0,) * 8
    assert zero_k.imbalance == pytest.approx(1.0)

    zero_l = split_sparse_dcp_ranks(kv_len=0, selected_rows=2048, cp=8, interleave=768)
    assert zero_l.per_rank == (0.0,) * 8
    assert zero_l.imbalance == pytest.approx(1.0)

    empty_batch = split_sparse_dcp_batch((), cp=8, interleave=768)
    assert empty_batch.per_rank == (0.0,) * 8
    assert empty_batch.imbalance == pytest.approx(1.0)

    for bad in (0.5, 0.0, -2.0, float("nan"), float("inf"), True, "bad"):
        rt = SimpleNamespace(metadata=(("dcp_sparse_imbalance", bad),))
        with pytest.raises(ModelingError, match="dcp_sparse_imbalance_invalid"):
            dcp_sparse_imbalance_override(rt)


def test_topk_ge_kv_len_exact_owned_positions_continuity():
    # L = 8192 < dcp * B = 16 * 768 = 12288: ranks 11..15 physically own 0 KV tokens.
    kv_len, cp, interleave = 8192, 16, 768
    expected_caps = tuple(
        float(owned_positions(0, kv_len - 1, cp, interleave, rank))
        for rank in range(cp)
    )
    assert expected_caps[:10] == (768.0,) * 10
    assert expected_caps[10] == 512.0
    assert expected_caps[11:] == (0.0,) * 5

    # Boundary 2: K >= L must match owned_positions identically.
    full_sel = split_sparse_dcp_ranks(kv_len=kv_len, selected_rows=16384, cp=cp, interleave=interleave)
    assert full_sel.per_rank == expected_caps
    assert full_sel.total == pytest.approx(float(kv_len))

    # Compressed KV continuity (compress_ratio=8, L_eff = ceil(8192 / 8) = 1024).
    ratio = 8
    eff_l = math.ceil(kv_len / ratio)
    eff_b = max(1, interleave // ratio)
    comp_caps = tuple(
        float(owned_positions(0, eff_l - 1, cp, eff_b, rank))
        for rank in range(cp)
    )
    comp_full = split_sparse_dcp_ranks(
        kv_len=kv_len, selected_rows=2048, cp=cp, interleave=interleave, compress_ratio=ratio,
    )
    assert comp_full.per_rank == comp_caps

    # Sparse K < L when L < cp * B: empty ranks 11..15 MUST remain strictly 0.0.
    sparse_8k = split_sparse_dcp_ranks(kv_len=kv_len, selected_rows=2048, cp=cp, interleave=interleave)
    assert sum(sparse_8k.per_rank) == pytest.approx(2048.0)
    assert sparse_8k.per_rank[11:] == (0.0,) * 5
    for r in range(cp):
        assert 0.0 <= sparse_8k.per_rank[r] <= expected_caps[r] + 1e-9
    assert sparse_8k.imbalance > 3.5


def test_interleave_block_coherence_amplifies_straggler_skew():
    kv_len, topk, cp = 131072, 2048, 8
    split_b1 = split_sparse_dcp_ranks(kv_len, topk, cp, interleave=1)
    split_b128 = split_sparse_dcp_ranks(kv_len, topk, cp, interleave=128)
    split_b768 = split_sparse_dcp_ranks(kv_len, topk, cp, interleave=768)
    split_b1536 = split_sparse_dcp_ranks(kv_len, topk, cp, interleave=1536)

    # B=1 spreads sink and tail uniformly with g_eff=1 -> near-uniform (~1.06x).
    assert 1.0 < split_b1.imbalance < 1.10
    # Realistic large interleave blocks concentrate sink/tail + semantic coherence.
    assert split_b128.imbalance > split_b1.imbalance
    assert split_b768.imbalance > split_b128.imbalance
    assert split_b1536.imbalance > 2.5

    for split, interleave in (
        (split_b1, 1), (split_b128, 128), (split_b768, 768), (split_b1536, 1536),
    ):
        assert sum(split.per_rank) == pytest.approx(float(topk))
        for rank in range(cp):
            cap = float(owned_positions(0, kv_len - 1, cp, interleave, rank))
            assert 0.0 <= split.per_rank[rank] <= cap + 1e-9

    # Explicit always_select_tail within a single full tail block concentrates that rank.
    aligned_kv_len = 86 * 1536  # Last block [130560, 132095] owns all 1536 tokens on one rank.
    split_aligned_default = split_sparse_dcp_ranks(aligned_kv_len, topk, cp, interleave=1536)
    split_aligned_tail = split_sparse_dcp_ranks(
        aligned_kv_len, topk, cp, interleave=1536, always_select_tail=1024,
    )
    assert split_aligned_tail.imbalance > split_aligned_default.imbalance
    assert sum(split_aligned_tail.per_rank) == pytest.approx(float(topk))


def test_batch_variance_pooling_conservation_and_override_bounds():
    cp, interleave, topk = 8, 1536, 2048
    # 1) Same-context batch pools semantic variance by 1/sqrt(N_seq).
    single = split_sparse_dcp_batch(
        (SparseDcpSequence(131072, topk),), cp, interleave,
    )
    same_16 = split_sparse_dcp_batch(
        tuple(SparseDcpSequence(131072, topk) for _ in range(16)), cp, interleave,
    )
    assert 1.0 < same_16.imbalance < single.imbalance
    assert sum(same_16.per_rank) == pytest.approx(16.0 * topk)

    # 2) Diverse contexts stagger tail owners across ranks -> stronger pooling.
    diverse_16 = split_sparse_dcp_batch(
        tuple(SparseDcpSequence(120000 + i * interleave, topk) for i in range(16)),
        cp, interleave,
    )
    assert 1.0 < diverse_16.imbalance < same_16.imbalance

    # 3) Heterogeneous batch with MTP q_weight=4 and short seq (K == L): exact row+col conservation.
    hetero = (
        SparseDcpSequence(kv_len=131072, selected_rows=2048, q_weight=4.0),
        SparseDcpSequence(kv_len=2000, selected_rows=2000, q_weight=1.0),
        SparseDcpSequence(kv_len=16384, selected_rows=2048, q_weight=2.0),
    )
    for override in (None, 1.0, 1.75, 50.0):
        rt = (
            SimpleNamespace(metadata=(("sfa_dcp_imbalance", override),))
            if override is not None else None
        )
        batch_split, per_seq = split_sparse_dcp_batch_per_sequence(
            hetero, cp, interleave, runtime=rt,
        )
        expected_total = sum(s.q_weight * min(s.kv_len, s.selected_rows) for s in hetero)
        assert batch_split.total == pytest.approx(expected_total)
        assert sum(batch_split.per_rank) == pytest.approx(expected_total)
        seq_caps = [sparse_dcp_capacities(s, cp, interleave) for s in hetero]
        for i, seq in enumerate(hetero):
            seq_tot = seq.q_weight * min(seq.kv_len, seq.selected_rows)
            assert sum(per_seq[i]) == pytest.approx(seq_tot)
            for r in range(cp):
                assert -1e-9 <= per_seq[i][r] <= seq_caps[i][r] + 1e-9
        for r in range(cp):
            col_sum = sum(per_seq[i][r] for i in range(len(hetero)))
            assert batch_split.per_rank[r] == pytest.approx(col_sum)
            assert batch_split.per_rank[r] <= sum(c[r] for c in seq_caps) + 1e-9

        # Integer sweep rounding preserves exact total and zero-capacity bounds.
        seq_int_totals = [int(round(s.q_weight * min(s.kv_len, s.selected_rows))) for s in hetero]
        mat = round_sparse_dcp_matrix(per_seq, seq_int_totals, seq_caps)
        assert sum(sum(row) for row in mat) == sum(seq_int_totals)
        for i in range(len(hetero)):
            for r in range(cp):
                assert 0 <= mat[i][r] <= seq_caps[i][r]

    # Explicit override alpha=1.75 on homogeneous batch hits 1.75x exactly.
    hom_rt = SimpleNamespace(metadata=(("dcp_sparse_imbalance", 1.75),))
    hom_override = split_sparse_dcp_batch(
        tuple(SparseDcpSequence(131072, topk) for _ in range(4)),
        cp, interleave, runtime=hom_rt,
    )
    assert hom_override.imbalance == pytest.approx(1.75, rel=1e-6)


def test_sharded_indexer_visible_rows_aligns_prefill_and_decode():
    pf_views = (RequestView("p0", 512, 128, False, "prefill"),)
    parallel = ParallelSpec(tp=4, cp=4, cp_plan="dcp")
    expected_pf = split_causal_pairs(512, 128, 4, 128).max_rank
    sharded_pf = sharded_index_visible_rows(pf_views, parallel, 128, 1, sharded_dcp=4)
    unsharded_pf = sharded_index_visible_rows(pf_views, parallel, 128, 1)
    assert sharded_pf == pytest.approx(expected_pf)
    assert sharded_pf < unsharded_pf * 0.5

    # End-to-end prefill graph with sharded indexer shards indexer FLOPs across DCP.
    c_rep = compose_case(
        "deepseek-v3.2", hardware="a5", dp=1, tp=4, ep=1, dcp=4, num_devices=4,
        layers=1, va_recommended=False,
    )
    c_shd = compose_case(
        "deepseek-v3.2", hardware="a5", dp=1, tp=4, ep=1, dcp=4, num_devices=4,
        layers=1, va_recommended=False, enable_sfa_dcp_sharded_indexer=True,
    )
    g_rep = target_graphs(parse_case(c_rep), phase="prefill", tokens=512, context=128, sequences=1, scope="body")[0]
    g_shd = target_graphs(parse_case(c_shd), phase="prefill", tokens=512, context=128, sequences=1, scope="body")[0]
    idx_rep = next(op for op in g_rep if op.family == "indexer")
    idx_shd = next(op for op in g_shd if op.family == "indexer")
    assert idx_shd.flops == pytest.approx(idx_rep.flops * (expected_pf / unsharded_pf), rel=1e-6)
    assert idx_shd.flops < idx_rep.flops * 0.5


def test_end_to_end_kimi_and_qwen_dcp_sparse_imbalance_and_latency():
    # 1) kimi-k3.1 on A5 (dcp=8, B=1536, FP8 QSFA) at 128K context.
    kimi_a5_s = compose_case(
        "kimi-k3.1", hardware="a5", tp=8, dp=1, ep=8, dcp=8, num_devices=8,
        layers=4, max_model_len=140000,
    )
    kimi_a5_u = compose_case(
        "kimi-k3.1", hardware="a5", tp=8, dp=1, ep=8, dcp=8, num_devices=8,
        layers=4, max_model_len=140000,
    )
    kimi_a5_u["runtime"]["metadata"]["dcp_sparse_imbalance"] = 1.0
    g_a5, _, r_a5_s = _lower_and_run_graph(
        parse_case(kimi_a5_s), phase="decode", tokens=1, context=131072,
        sequences=64, scope="body", include_submission=False, fast_span_only=False,
    )
    _, _, r_a5_u = _lower_and_run_graph(
        parse_case(kimi_a5_u), phase="decode", tokens=1, context=131072,
        sequences=64, scope="body", include_submission=False, fast_span_only=False,
    )
    attn_a5 = next(op for op in g_a5[0] if op.family == "attention")
    meta_a5 = dict(attn_a5.metadata)
    assert meta_a5["cp_sparse_imbalance"] > 2.0
    assert sum(meta_a5["cp_per_rank_pairs"]) == pytest.approx(64.0 * 2048.0)
    assert _attention_elapsed_ms(r_a5_s) > _attention_elapsed_ms(r_a5_u) * 2.0

    # 2) kimi-k3.1 on A3 (dcp=16, B=768, packed INT8 C8 QSFA) at 8K and 128K.
    kimi_a3_s = compose_case(
        "kimi-k3.1", hardware="a3", tp=16, dp=1, ep=16, dcp=16, num_devices=16,
        layers=4, max_model_len=140000,
    )
    kimi_a3_u = compose_case(
        "kimi-k3.1", hardware="a3", tp=16, dp=1, ep=16, dcp=16, num_devices=16,
        layers=4, max_model_len=140000,
    )
    kimi_a3_u["runtime"]["metadata"]["dcp_sparse_imbalance"] = 1.0
    # 8K context: L < dcp * B -> ranks 11..15 have 0 pairs & 0 sweeps.
    g_a3_8k = target_graphs(
        parse_case(kimi_a3_s), phase="decode", tokens=1, context=8192,
        sequences=1, scope="body",
    )[0]
    attn_a3_8k = next(op for op in g_a3_8k if op.family == "attention")
    meta_a3_8k = dict(attn_a3_8k.metadata)
    assert meta_a3_8k["cp_per_rank_pairs"][11:] == (0.0,) * 5
    assert meta_a3_8k["cp_per_rank_sweeps"][11:] == (0,) * 5
    assert meta_a3_8k["indexed_attention_geometry"]["topk"] > 2048 // 16
    assert meta_a3_8k["cp_sparse_imbalance"] == pytest.approx(
        meta_a3_8k["indexed_attention_geometry"]["cp_sparse_imbalance"],
    )

    # 128K context B=8: strong packed-C8 straggler latency jump vs uniform 1/dcp.
    g_a3_128k, _, r_a3_s = _lower_and_run_graph(
        parse_case(kimi_a3_s), phase="decode", tokens=1, context=131072,
        sequences=8, scope="body", include_submission=False, fast_span_only=False,
    )
    _, _, r_a3_u = _lower_and_run_graph(
        parse_case(kimi_a3_u), phase="decode", tokens=1, context=131072,
        sequences=8, scope="body", include_submission=False, fast_span_only=False,
    )
    attn_a3_128k = next(op for op in g_a3_128k[0] if op.family == "attention")
    meta_a3_128k = dict(attn_a3_128k.metadata)
    assert meta_a3_128k["cp_sparse_imbalance"] > 3.0
    assert meta_a3_128k["indexed_attention_geometry"]["topk"] > 2048 // 16
    assert _attention_elapsed_ms(r_a3_s) > _attention_elapsed_ms(r_a3_u) * 2.0

    # Direct packed_sfa_geometry override check (sfa_dcp_imbalance=2.5).
    c8_plan = SimpleNamespace(
        kv_lora_rank=512, qk_rope_head_dim=64, cache_row_elements=576,
        index_topk=2048, indexer=True,
    )
    c8_fmt = CacheFormat("int8", 128, "fp32", 64, "bf16")
    c8_views = tuple(RequestView(f"r{i}", 1, 131072, False, "decode") for i in range(4))
    c8_rt = SimpleNamespace(
        kv_block_size=768, prefix_cache_mode="private",
        metadata=(("sfa_dcp_imbalance", 2.5),),
    )
    c8_geom = packed_sfa_geometry(
        {"attention_tensor_view": {"cache_dtype": "int8"}}, 96, c8_plan, c8_views, 16,
        cache_dtype="int8", cache_fmt=c8_fmt, runtime=c8_rt,
    )
    assert c8_geom is not None
    assert c8_geom["cp_sparse_imbalance"] == pytest.approx(2.5, rel=1e-6)
    assert c8_geom["topk"] == math.ceil(2.5 * (2048 / 16))

    # 3) qwen3.8-flash-next on A5 (sparse GQA with dcp=4) at 32K context.
    qwen_s = compose_case(
        "qwen3.8-flash-next", hardware="a5", tp=8, dp=1, ep=8, dcp=4, num_devices=8,
        layers=4, max_model_len=140000, va_recommended=False,
    )
    qwen_u = compose_case(
        "qwen3.8-flash-next", hardware="a5", tp=8, dp=1, ep=8, dcp=4, num_devices=8,
        layers=4, max_model_len=140000, va_recommended=False,
    )
    qwen_u["runtime"]["metadata"]["dcp_sparse_imbalance"] = 1.0
    g_qwen, _, r_qwen_s = _lower_and_run_graph(
        parse_case(qwen_s), phase="decode", tokens=1, context=32768,
        sequences=64, scope="body", include_submission=False, fast_span_only=False,
    )
    _, _, r_qwen_u = _lower_and_run_graph(
        parse_case(qwen_u), phase="decode", tokens=1, context=32768,
        sequences=64, scope="body", include_submission=False, fast_span_only=False,
    )
    attn_qwen = next(op for op in g_qwen[0] if op.family == "attention")
    meta_qwen = dict(attn_qwen.metadata)
    assert meta_qwen["cp_sparse_imbalance"] > 1.0
    assert sum(meta_qwen["cp_per_rank_pairs"]) == pytest.approx(64.0 * 2048.0)
    assert _attention_elapsed_ms(r_qwen_s) > _attention_elapsed_ms(r_qwen_u)
