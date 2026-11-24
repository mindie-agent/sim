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


def test_g1_sp_padding_p_eq_w_sharded_sort_m1_m1b_m5_and_m10():
    """Verify G1 SP laws, P==W switch, sharded tiling/A5 zero-Cast, M1/M1b/M5/M10, and AI_CPU."""
    from types import SimpleNamespace
    from mindie_sim.prediction.memory_traffic import (
        AttentionAccessSpec,
        AttentionSchedule,
        plan_attention_traffic,
    )
    from mindie_sim.prediction.operators.hyper_connections import family_attnres
    from mindie_sim.prediction.operators.mla.generic_latent_attention_views import (
        _owned_completed_rows_for_rank,
        unique_index_key_rows,
    )
    from mindie_sim.prediction.operators.mla.mla_epilogs import (
        SMLA_METADATA_SIZE_A5,
        SMLA_METADATA_SIZE_A23,
        validate_m10_mla_legality,
    )
    from mindie_sim.prediction.operators.mla_prolog import resolve_mla_preprocess_mode
    from mindie_sim.prediction.sharding import context_parallel as cp_mod
    from mindie_sim.prediction.sharding import sequence_parallel as sp_mod
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.parallel import ParallelSpec
    from mindie_sim.schema.physical_mechanisms import MemoryHierarchySpec

    # G1: per-occurrence SP padding law + draft scope + mc2_mask + MMRS + directional bytes.
    rt = SimpleNamespace(
        parallel=ParallelSpec(tp=4, ep=4, dp=1, pp=1),
        kernels={"sequence_parallel": {
            **sp_mod.DECLARATION, "padding": "right_to_tp", "mc2_active_mask_active": True,
        }},
    )
    draft_model = SimpleNamespace(layer_plans=(
        SimpleNamespace(scope="draft", ffn="moe", residual_block_size=0, hyper_connections=False, hc_mult=0),
    ))
    plan = sp_mod.select(rt, draft_model, (SimpleNamespace(q_len=3, phase="decode", padding=False),), tokens=3)
    assert plan.reduce_scatter_pad_rows(4) == 0
    assert plan.reduce_scatter_pad_rows(3) == 1
    assert plan.directional_remote_link_bytes("allgather", 4, 7168, 2) == 4 * 7168 * 2 * 3 // 4
    pad_op = sp_mod.emit_reduce_scatter_pad(
        plan=plan, op_id="l0.shared.sp_scatter.pad", deps=("x",), stage=0, layer=0,
        requests=("r0",), phase="decode", width=7168, dtype="bf16", source_rows=3,
    )
    assert pad_op is not None and dict(pad_op.metadata)["aclnn_symbol"] == "aclnnConstantPadNd"
    assert sp_mod.emit_reduce_scatter_pad(
        plan=plan, op_id="l0.dense.pad", deps=("x",), stage=0, layer=0,
        requests=("r0",), phase="decode", width=7168, dtype="bf16", source_rows=4,
    ) is None
    mask_pad = sp_mod.emit_mc2_mask_pad(
        plan=plan, op_id="l0.mc2_mask.pad", deps=("m",), stage=0, layer=0,
        requests=("r0",), phase="decode",
    )
    assert mask_pad is not None and dict(mask_pad.metadata)["mc2_mask_cast_launches"] == 3
    assert sp_mod.mmrs_eligible(tp=4, dtype="bf16", has_bias=False, k_dim=4096, enable_mmrs_fusion=True)
    assert not sp_mod.mmrs_eligible(tp=4, dtype="bf16", has_bias=True, k_dim=4096, enable_mmrs_fusion=True)

    # G1: sp_scope="off" returns None; bucket=16/tp=8/valid=1 pad matches scatter shape (16, 7168).
    rt_off = SimpleNamespace(
        parallel=ParallelSpec(tp=8, ep=8, dp=1, pp=1),
        kernels={"sequence_parallel": {**sp_mod.DECLARATION, "padding": "right_to_tp", "sp_scope": "off"}},
    )
    assert sp_mod.select(rt_off, draft_model, (SimpleNamespace(q_len=1, phase="decode", padding=False),), tokens=1) is None

    rt_b16 = SimpleNamespace(
        parallel=ParallelSpec(tp=8, ep=8, dp=1, pp=1),
        kernels={"sequence_parallel": {
            **sp_mod.DECLARATION, "padding": "right_to_tp", "capture_bucket_rows": 16,
        }},
    )
    plan_b16 = sp_mod.select(rt_b16, draft_model, (SimpleNamespace(q_len=1, phase="decode", padding=False),), tokens=1)
    assert plan_b16 is not None and plan_b16.tokens_global == 16
    b16_pad = sp_mod.emit_reduce_scatter_pad(
        plan=plan_b16, op_id="l0.shared.sp_scatter.pad", deps=("x",), stage=0, layer=0,
        requests=("r0",), phase="decode", width=7168, dtype="bf16", source_rows=1,
        target_rows=plan_b16.tokens_global,
    )
    assert b16_pad is not None and b16_pad.local_shape == (16, 7168)

    # P==W>1 (no spurious GatherV3), mixed-prefill bypass, sharded tiling + 75% occupancy, A3 vs A5 Cast diff.
    pw_plan = cp_mod.build_dcp_layer_execution_plan(
        layer_id=0, rank_id=0, request_phase="decode", pcp_factor=2, tp_gather=1, dp_width=1,
        sharded_indexer=False, arch="A3", num_prod_rows=4, topk=2048,
    )
    assert pw_plan.runs_query_allgather is False
    assert pw_plan.output_exchange_collective == "pcp_allgather_only"
    assert "hcom_alltoall_" not in pw_plan.native_calls and "GatherV3" not in pw_plan.native_calls
    pw_ops: list = []
    cp_mod.emit_dcp_output_exchange_combine(
        pw_ops, prefix="l0", stage=0, layer=0, requests=("r0",), mode="decode",
        deps=("attn",), rows=4, local_heads=16, v_dim=512, act_dtype="bf16",
        pcp_factor=2, tp_gather=1, dp_width=1,
    )
    assert not any("pcp_hidden_restore" in op.op_id for op in pw_ops)

    bypass = cp_mod.build_dcp_layer_execution_plan(
        layer_id=0, rank_id=0, request_phase="mixed", pcp_factor=1, tp_gather=4, dp_width=1,
        sharded_indexer=False, arch="A5", is_global_mixed_prefill=True,
    )
    assert bypass.runs_query_allgather is False and bypass.output_exchange_collective == "none"

    assert cp_mod.sharded_indexer_tile_rows(4, 2048) == 4096
    assert cp_mod.sharded_candidate_allgather_tiles(0, 4, 2048) == 0
    assert cp_mod.sharded_candidate_allgather_tiles(4097, 4, 2048) == 2
    assert cp_mod.sharded_fused_sort_eligible(world_size=6, topk=2048) is False
    assert cp_mod.sharded_fused_sort_eligible(world_size=7, topk=2048) is True
    assert cp_mod.sharded_fused_sort_eligible(world_size=7, topk=2048, rows=10, vector_cores=48) is False
    assert cp_mod.sharded_fused_sort_eligible(world_size=7, topk=2048, rows=36, vector_cores=48) is True
    p_a3 = cp_mod.build_dcp_layer_execution_plan(
        layer_id=0, rank_id=0, request_phase="decode", pcp_factor=1, tp_gather=4, dp_width=1,
        sharded_indexer=True, arch="A3", num_prod_rows=4097, topk=2048,
    )
    p_a5 = cp_mod.build_dcp_layer_execution_plan(
        layer_id=0, rank_id=0, request_phase="decode", pcp_factor=1, tp_gather=4, dp_width=1,
        sharded_indexer=True, arch="A5", num_prod_rows=4, topk=2048,
    )
    # 1 query AllGather + 2 tiled candidate AllGathers for 4097 rows at W=4, K=2048.
    assert p_a3.native_calls.count("hcom_allGather_") == 3
    assert "Cast" in p_a3.native_calls and "Cast" not in p_a5.native_calls
    p_k128 = cp_mod.build_dcp_layer_execution_plan(
        layer_id=0, rank_id=0, request_phase="decode", pcp_factor=1, tp_gather=4, dp_width=1,
        sharded_indexer=True, arch="A5", num_prod_rows=4, topk=128,
    )
    assert p_k128.native_calls.index("_pack_candidates") < p_k128.native_calls.index("Sort")
    assert "_unpack_candidates" in p_k128.native_calls and "_select_local_candidates" in p_k128.native_calls

    # M1 5-branch MLA preprocess selector (BF16 required per mla_v1.py:1919; strided adapter requires A5 draft).
    assert resolve_mla_preprocess_mode(arch="A5", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, weight_quant="bf16") == "prolog_v3"
    assert resolve_mla_preprocess_mode(arch="A5", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, act_dtype="fp16", weight_quant="fp16") == "native_decomposed"
    assert resolve_mla_preprocess_mode(arch="A3", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, weight_quant="bf16") == "prolog_v3"
    assert resolve_mla_preprocess_mode(arch="A3", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, weight_quant="mxfp8") == "native_decomposed"
    assert resolve_mla_preprocess_mode(arch="A3", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, enable_mlapo=True) == "mlapo"
    assert resolve_mla_preprocess_mode(arch="A5", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, is_strided_kv_cache=True, is_draft_model=True, input_size=14336) == "strided_kv_adapter"
    assert resolve_mla_preprocess_mode(arch="A5", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, is_strided_kv_cache=True, is_draft_model=True, input_size=7168) == "native_decomposed"
    assert resolve_mla_preprocess_mode(arch="A5", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, is_strided_kv_cache=False, is_draft_model=True) == "native_decomposed"
    assert resolve_mla_preprocess_mode(arch="A5", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, has_learned_rope_tail=True) == "prolog_v3_learned_rope_tail"

    # M1b zero-row sentinel & M5 store-on-completion (I % R != 0) & prefix_block_size vs CP interleave.
    hier = MemoryHierarchySpec(l2_capacity_bytes=64 * 1024 * 1024, l2_effective_fraction=0.8, cache_line_bytes=128, write_allocate=True)
    zero_tp = plan_attention_traffic(
        AttentionAccessSpec(q_lengths=(1,), context_lengths=(4096,), unique_rows=0, refill_rows=0,
                            unique_rows_known=True, refill_rows_known=True, row_bytes=1152.0),
        hier, AttentionSchedule(),
    )
    assert zero_tp.hbm_read_bytes == 0.0 and zero_tp.logical_read_bytes == 0.0
    assert _owned_completed_rows_for_rank(12, 3, 2, 4, 0) == 3
    assert _owned_completed_rows_for_rank(12, 3, 2, 4, 1) == 1
    v_m5_a = SimpleNamespace(q_len=1, context=127, kv_len=128, prefix_tokens=64, prefix_key="k0", query_ranges=())
    v_m5_b = SimpleNamespace(q_len=1, context=127, kv_len=128, prefix_tokens=64, prefix_key="k0", query_ranges=())
    u_m5, r_m5 = unique_index_key_rows((v_m5_a, v_m5_b), cp=2, interleave=128, compression=4, prefix_block_size=64)
    assert (u_m5, r_m5) == (48, 64)

    # M10 legality guards + AttnResFwd on A3 & A5 + SMLA metadata size constants + unmodeled structure fallback.
    from mindie_sim.structure_implementations import operator_birth_certificate
    for fam in ("elementwise", "attention", "unregistered_fam"):
        unknown_cert = operator_birth_certificate({"op_id": "x", "family": fam, "recipe": "no_such_recipe", "local_shape": (1, 1), "dtype": "bf16", "metadata": {}})
        assert unknown_cert.backend_stack == "unmodeled" and unknown_cert.native_kinds == () and unknown_cert.readiness == "unmodeled"

    assert SMLA_METADATA_SIZE_A23 == 1024 and SMLA_METADATA_SIZE_A5 == 2048
    assert family_attnres._attnres_arch_supported("A3") and family_attnres._attnres_arch_supported("A5")
    with pytest.raises(ModelingError, match="m10_kpool_requires_dcp_eq_1"):
        validate_m10_mla_legality(
            plan=SimpleNamespace(index_kpool=4, mla_variant="latent_mla", cache_kind="dense", indexer=True),
            eff_dcp=2, quant_kv_dtype="bf16", arch="A5", cp_interleave_size=128,
        )
    with pytest.raises(ModelingError, match="m10_sparse_flash_mla_a23_forbids_int8_kv"):
        validate_m10_mla_legality(
            plan=SimpleNamespace(index_kpool=1, mla_variant="latent_mqa", cache_kind="window_compressed", indexer=True),
            eff_dcp=1, quant_kv_dtype="int8", arch="A3", cp_interleave_size=128,
        )

    # Noncausal SWA trailing-window + current-block law & CP2 exact conservation (665 = 25 + 640).
    from mindie_sim.prediction.memory_traffic import _causal_pairs as mt_causal_pairs
    from mindie_sim.prediction.operators.mla.generic_latent_attention_views import (
        _noncausal_window_rank_pairs,
        window_tile_rows,
    )
    swa_plan = SimpleNamespace(window=128, noncausal_attention=True)
    assert window_tile_rows(swa_plan, context=256, query=5, start=0, query_tile=128, kv_len=261) == 133
    assert window_tile_rows(swa_plan, context=256, query=6, start=0, query_tile=128, kv_len=262) == 134
    assert mt_causal_pairs(5, 256, 128, True, 0) == 665.0
    assert mt_causal_pairs(6, 256, 128, True, 0) == 804.0
    cp2_pairs = _noncausal_window_rank_pairs(5, 256, 128, 2, 128)
    assert cp2_pairs == [25.0, 640.0] and sum(cp2_pairs) == 665.0

    # P1-7: PCP lane_views preserves global_q_len=5 -> 266+399=665 == 25+640.
    from mindie_sim.prediction.execution.rank_ledger import lane_views
    from mindie_sim.prediction.operators.mla.generic_latent_attention_views import attention_view_record, window_cache_read_by_rank
    from mindie_sim.schema.workload import RequestView
    from mindie_sim.catalog.types import QuantSpec
    lv0, lv1 = lane_views((RequestView("r0", 5, 251, False, "prefill"),), 2, 0)[0], lane_views((RequestView("r0", 5, 251, False, "prefill"),), 2, 1)[0]
    p_nc = SimpleNamespace(cache_kind="dense", window=128, noncausal_attention=True, index_topk=0, index_kpool=1, index_kpool_always_select_tail=False, compress_ratio=1, qk_rope_head_dim=64)
    m_nc = SimpleNamespace(activation_dtype="bf16", compute_dtype="bf16", quant=QuantSpec("bf16"), kv_cache_formats=(), metadata=())
    rr0 = attention_view_record(lv0, plan=p_nc, model=m_nc, quant=QuantSpec("bf16"), query_tile=128, eff_cp=2, block=128, interleave=128, tp=1, dsa=False, topk_present=False, heads=16, latent_dim=512)
    rr1 = attention_view_record(lv1, plan=p_nc, model=m_nc, quant=QuantSpec("bf16"), query_tile=128, eff_cp=2, block=128, interleave=128, tp=1, dsa=False, topk_present=False, heads=16, latent_dim=512)
    assert rr0["pairs"] == 266.0 and rr1["pairs"] == 399.0 and [rr0["v_pairs"][i] + rr1["v_pairs"][i] for i in range(2)] == [25.0, 640.0]

    # P1-9: packed SFA interleave-boundary horizon conservation (256==256, 384==384, 513==513).
    from mindie_sim.prediction.operators.paged_mla.indexed_quant_attention import _row_partition, tile_stream
    sfa_p = SimpleNamespace(index_topk=2048, kv_lora_rank=512, qk_rope_head_dim=64, cache_row_elements=576, index_kpool=1, index_kpool_always_select_tail=False)
    for vs, exp_k in (((RequestView("r0", 2, 127, False, "decode"),), 256), ((RequestView("r0", 3, 127, False, "decode"),), 384)):
        g_sfa = cp_mod.packed_sfa_geometry({}, 96, sfa_p, vs, 2, cache_dtype="int8", interleave=128)
        _, sts = tile_stream(g_sfa["sequences"], g_sfa["topk"], 2, 128, 1, False, per_sequence_topk=g_sfa["per_sequence_topk"])
        assert sum(k for parts in sts for _, ts in parts for k, _, _ in ts) == exp_k == sum(map(sum, _row_partition(g_sfa, g_sfa["sequences"], g_sfa["topk"])[1]))

    # P1-5 & P1-8 M1b: per-tile fused sort (2049 rows -> tail unfused), valid=0, K=128, & production [1,0] M1b (128B / 0B + per-layer 64B).
    from mindie_sim.prediction.lowering import Builder
    from mindie_sim.prediction.operators.mla.mla import _plan_rank_attention_traffic
    from mindie_sim.prediction.operators.mla.mla_epilogs import _emit_sharded_candidate_merge
    m_stub = SimpleNamespace(activation_dtype="bf16", compute_dtype="bf16", quant=QuantSpec("bf16"), kv_cache_formats=(), metadata=(("layer_cache_representations", {0: {"compressed_kv": {"dtype": "int8"}}}),))
    p_stub = SimpleNamespace(index=0, scope="target", cache_kind="window_compressed", window=128,
                             cache_row_elements=64, index_topk=0, index_kpool=1, index_kpool_always_select_tail=False, noncausal_attention=False, compress_ratio=4, qk_rope_head_dim=0)
    mach48 = SimpleNamespace(dies=(SimpleNamespace(aiv_count=48, aic_count=24),))
    b_t = Builder("s0", p_stub, 0, (SimpleNamespace(request_id="r0", q_len=2049, context=2049, phase="prefill"),), m_stub, "prefill")
    _emit_sharded_candidate_merge(b_t, "topk", valid=2049, index_topk=2048, eff_dcp=8, runtime=SimpleNamespace(metadata=(("device_arch", "A3"),)), machine=mach48)
    packs_t = [o for o in b_t.ops if o.recipe.startswith("candidate_pack")]
    assert len(packs_t) == 2 and dict(packs_t[0].metadata)["sharded_fused_sort"] is True and dict(packs_t[1].metadata)["sharded_fused_sort"] is False
    b_zero = Builder("s0", p_stub, 0, (), m_stub, "prefill")
    assert _emit_sharded_candidate_merge(b_zero, "topk", valid=0, index_topk=2048, eff_dcp=8, runtime=SimpleNamespace(metadata=(("device_arch", "A3"),)), machine=mach48) == "topk" and not b_zero.ops
    b_k128 = Builder("s0", p_stub, 0, (SimpleNamespace(request_id="r0", q_len=64, context=64, phase="prefill"),), m_stub, "prefill")
    _emit_sharded_candidate_merge(b_k128, "topk", valid=64, index_topk=128, eff_dcp=8, runtime=SimpleNamespace(metadata=(("device_arch", "A3"),)), machine=mach48)
    assert dict(b_k128.ops[0].metadata)["native_kinds"] == ("_pack_candidates",)
    assert dict(b_k128.ops[-1].metadata)["native_kinds"] == (
        "_unpack_candidates", "Sort", "Cast", "GatherElementsV2", "Sort", "Cast", "GatherElementsV2", "_select_local_candidates",
    )
    b_k128_a5 = Builder("s0", p_stub, 0, (SimpleNamespace(request_id="r0", q_len=64, context=64, phase="prefill"),), m_stub, "prefill")
    _emit_sharded_candidate_merge(b_k128_a5, "topk", valid=64, index_topk=128, eff_dcp=8, runtime=SimpleNamespace(metadata=(("device_arch", "A5"),)), machine=mach48)
    assert dict(b_k128_a5.ops[0].metadata)["native_kinds"] == ("_pack_candidates",)
    assert dict(b_k128_a5.ops[-1].metadata)["native_kinds"] == (
        "_unpack_candidates", "Sort", "GatherElements", "Sort", "GatherElements", "_select_local_candidates",
    )
    assert "Cast" not in dict(b_k128_a5.ops[-1].metadata)["native_kinds"]

    m1b_view = RequestView("r0", 1, 0, False, "decode")
    m1b_rec = attention_view_record(m1b_view, plan=p_stub, model=m_stub, quant=QuantSpec("bf16"), query_tile=128, eff_cp=2, block=128, interleave=128, tp=1, dsa=False, topk_present=False, heads=16, latent_dim=64)
    assert m1b_rec["v_sweeps"] == [1, 0] and m1b_rec["v_rows"] == [1, 0]
    _, r0_bytes = _plan_rank_attention_traffic(plan=p_stub, model=m_stub, view=m1b_view, parts=((1, 0),), rec=m1b_rec, eff_cp=2, worst_pairs_r=0, worst_sweeps_r=0, kv_row_b=128.0, query_tile=128, hierarchy=hier, mode="decode", topk=None, latent=True, interleave=128)
    _, r1_bytes = _plan_rank_attention_traffic(plan=p_stub, model=m_stub, view=m1b_view, parts=((1, 0),), rec=m1b_rec, eff_cp=2, worst_pairs_r=1, worst_sweeps_r=1, kv_row_b=128.0, query_tile=128, hierarchy=hier, mode="decode", topk=None, latent=True, interleave=128)
    assert r0_bytes == 128.0 and r1_bytes == 0.0
    by_r_pl = window_cache_read_by_rank(p_stub, RequestView("r0", 1, 7, False, "decode"), ((1, 7),), [10, 0], [10, 0], 2, 128, 64, m_stub, 128)
    assert by_r_pl[0][4] == 128.0 and by_r_pl[0][5] == 64.0
