"""Generalized parallelism, composable Local-TP, tutorial profiles, and parallel sweeps.

[Role]
  Verify algebraic 3-axis DCP factorization, all 7 composable Local-TP axes in
  model scope (exact weight/GEMM/collective parity, packed whole-KV-head
  sharding, exact vocab sharding, lane-offset groups, and active/idle FFN
  symmetry), matched-global-workload Local-TP vs DCP physics crossover,
  transparent non-blocking ``kv_cache_replicated_tp`` metadata, per-SKU
  tutorial profiles and KeyPool backend dtype contracts, hardware/threshold-
  gated stack mechanisms, and deterministic streaming ``ProcessPoolExecutor``
  sweep parity with strict exception propagation.

[Invariants]
  * No artificial model-name allowlists gate structural parallelism or stack
    optimization mechanisms; hardware-specific paths stay gated on true arch
    and runtime eligibility.
  * Parallel sweep execution (``jobs=2``) matches serial execution (``jobs=1``)
    in candidate order, metrics, static rejection accounting, and error types.
  * ``tp > 1, dcp == 1`` on 1-KV-head latent attention is never banned and
    always surfaces ``kv_cache_replicated_tp=True`` with exact factor ``tp``.

[Timeouts & Locks]
  Tier-0 pure in-memory composition, graph lowering, and bounded sweep checks.

[Side-Effect Policy]
  Read-only in-memory evaluation; zero filesystem writes.
"""
from __future__ import annotations

import math
import pickle
from unittest import mock

import pytest

from mindie_sim.catalog.context_parallel import kv_cache_replication_info
from mindie_sim.catalog.weights import stage_weight_bytes
from mindie_sim.compose import compose_case, va_recommended_config
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import _lower_and_run_graph
from mindie_sim.prediction.sharding.local_tp import local_tp_groups
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec, factor_dcp_axes
from mindie_sim.sweep import sweep_topologies
from mindie_sim.sweep.resize import enumerate_parallel_candidates

pytestmark = pytest.mark.tier0

HW_DEFAULT = "a5"
HW_A2 = "Ascend910B4"
AUDIT_LAYERS = 4
QUANT_AR_BELOW_TOKENS = 64
QUANT_AR_ABOVE_TOKENS = 4096


def test_composable_local_tp_all_axes_model_scope_and_idle_ffn_symmetry():
    with pytest.raises(ModelingError, match="local_tp_conflict"):
        ParallelSpec(dp=4, tp=1, ep=4, dcp_linear_tp=4, o_proj_tp=2)

    # 1) Full model-scope Local-TP composition across TP=2 + DCP=2 + LM head.
    raw_base = compose_case(
        "deepseek-v3", hardware=HW_DEFAULT, dp=4, tp=2, ep=8, dcp=2,
        num_devices=8, layers=AUDIT_LAYERS, va_recommended=False,
    )
    raw_combo = compose_case(
        "deepseek-v3", hardware=HW_DEFAULT, dp=4, tp=2, ep=8, dcp=2,
        q_proj_tp=4, o_proj_tp=4, mlp_tp=4, shared_expert_tp=4, lm_head_tp=4,
        num_devices=8, layers=AUDIT_LAYERS, va_recommended=False,
    )
    cs_base, cs_combo = parse_case(raw_base), parse_case(raw_combo)
    groups = local_tp_groups(cs_combo.machine, cs_combo.runtime.parallel, width=4, stage=0, stride=2)
    assert set(groups) == {(0, 2, 4, 6), (1, 3, 5, 7)}
    assert max(stage_weight_bytes(cs_combo.model, cs_combo.runtime.parallel, 8)) < max(
        stage_weight_bytes(cs_base.model, cs_base.runtime.parallel, 8)
    )

    # Model-scope LM head exact vocab shard: 129,280 / 4 = 32,320 rows, exact resident == GEMM weight bytes!
    g_model, _, rep_m = _lower_and_run_graph(
        cs_combo, phase="decode", tokens=1, context=1024, sequences=2,
        scope="model", include_submission=False, fast_span_only=True,
    )
    assert float(rep_m["duration_ms"]) > 0.0
    lm_head = next(op for op in g_model[0] if op.recipe == "vocab_logits")
    assert lm_head.local_shape[1] == 32_320
    assert lm_head.mte2_bytes == pytest.approx(463_339_520.0 + 4 * 7168 * 2, rel=1e-6)
    from mindie_sim.schema.common import dtype_bytes

    def _assert_local_tp_wire_floor(ops_list: list, label: str) -> None:
        for op in ops_list:
            meta = dict(op.metadata)
            width = int(meta.get("local_tp_width") or 0)
            numel = math.prod(int(d) for d in op.local_shape) if op.local_shape else 0
            assert width >= 2 and numel > 0, f"{label}:{op.op_id}:invalid_shape_or_width"
            elem_b = float(dtype_bytes(op.dtype))
            floor_b = 0.95 * float(numel) * elem_b * float(width - 1) / float(width)
            ceil_b = float(numel) * elem_b * float(width)
            assert math.isfinite(op.link_bytes) and floor_b - 1.0 <= op.link_bytes <= ceil_b + 1.0, (
                f"{label}:{op.op_id}:link_bytes={op.link_bytes}:bounds=({floor_b},{ceil_b})"
            )

    for r, ops in g_model.items():
        ltp_ops = [op for op in ops if dict(op.metadata).get("group_axis") == "local_tp"]
        assert len(ltp_ops) == 42, f"rank {r} missing model-scope local_tp collectives"
        _assert_local_tp_wire_floor(ltp_ops, f"combo:r{r}")
        ltp_ids = {op.op_id for op in ltp_ops}
        assert {"logits:local_tp:gather", "logits", "s0.l0.q_a:local_tp:gather", "s0.l0.output", "s0.l0.down", "s0.l3.shared.down"} <= ltp_ids

    # Remaining 2 Local-TP axes (dcp_linear_tp & linear_proj_tp): exact analytical wire floors.
    for model_id, axis_kw in (
        ("deepseek-v3", dict(dp=4, tp=1, ep=1, dcp_linear_tp=4, num_devices=4, layers=1)),
        ("glm-5.3-flash", dict(dp=4, tp=1, ep=1, linear_proj_tp=4, num_devices=4, layers=1)),
    ):
        raw_ax = compose_case(model_id, hardware=HW_DEFAULT, va_recommended=False, **axis_kw)
        g_ax, _, _ = _lower_and_run_graph(
            parse_case(raw_ax), phase="decode", tokens=1, context=512, sequences=1,
            scope="model", include_submission=False, fast_span_only=True,
        )
        ax_ops = [op for op in g_ax[0] if dict(op.metadata).get("group_axis") == "local_tp"]
        assert ax_ops
        _assert_local_tp_wire_floor(ax_ops, model_id)

    # 2) Packed GQA whole-KV-head Local-TP: kv_heads=4, q_proj_tp=8 -> N=768, slices cover all 8 gathered rows!
    c_pack = compose_case(
        "qwen3-30b-a3b", hardware=HW_DEFAULT, dp=8, tp=1, ep=8, q_proj_tp=8,
        num_devices=8, layers=1, va_recommended=False,
    )
    cs_pack = parse_case(c_pack)
    g_pack, _, _ = _lower_and_run_graph(
        cs_pack, phase="decode", tokens=1, context=512, sequences=1,
        scope="body", include_submission=False, fast_span_only=True,
    )
    qkv = next(op for op in g_pack[0] if op.recipe == "qkv_projection")
    assert qkv.local_shape == (8, 768, 2048)
    assert qkv.mte2_bytes == pytest.approx(3_145_728.0 + 8 * 2048 * 2, rel=1e-4)
    slices = [(s["name"], tuple(s["shape"])) for s in dict(qkv.metadata)["packed_projection_slices"]]
    assert slices == [("q", (8, 512)), ("k", (8, 128)), ("v", (8, 128))]

    # 3) Active/idle [1, 0] and ragged [1, 3] FFN Local-TP padded-row parity on dense mlp_tp & MoE shared_expert_tp.
    for model_id, kw in (
        ("qwen3-32b", dict(dp=2, tp=1, ep=1, mlp_tp=2, num_devices=2, layers=1)),
        ("deepseek-v3", dict(dp=2, tp=1, ep=2, shared_expert_tp=2, num_devices=2, layers=4)),
    ):
        c_ffn = compose_case(model_id, hardware=HW_DEFAULT, va_recommended=False, **kw)
        g_ffn, _, _ = _lower_and_run_graph(
            parse_case(c_ffn), phase="decode", tokens=1, context=512, sequences=[1, 0],
            scope="body", include_submission=False, fast_span_only=True,
        )
        r0_local = [op.family for op in g_ffn[0] if dict(op.metadata).get("group_axis") == "local_tp"]
        r1_local = [op.family for op in g_ffn[1] if dict(op.metadata).get("group_axis") == "local_tp"]
        assert r0_local and r0_local == r1_local
        g_rag, _, _ = _lower_and_run_graph(
            parse_case(c_ffn), phase="decode", tokens=1, context=512, sequences=[1, 3],
            scope="body", include_submission=False, fast_span_only=True,
        )
        for r in (0, 1):
            gu = next(op for op in g_rag[r] if op.op_id.endswith("gate_up:local_tp:gemm") and op.layer == kw["layers"] - 1)
            dn_rs = next(op for op in g_rag[r] if op.family == "reduce_scatter" and op.layer == kw["layers"] - 1)
            assert gu.local_shape[0] == 6 and dn_rs.local_shape[0] == 3


def test_matched_global_workload_local_tp_vs_dcp_physics_and_kv_crossover():
    # Identical 8-die hardware and IDENTICAL global workloads (B=8 @ ctx=2048 and B=64 @ ctx=65536):
    # DP8 * local_seqs == global_seqs vs DP1/TP8/DCP8 * global_seqs.
    raw_ltp = compose_case(
        "glm-5.3-flash", hardware=HW_DEFAULT, dp=8, tp=1, ep=8, dcp=1,
        linear_proj_tp=8, o_proj_tp=8, lm_head_tp=8,
        num_devices=8, layers=AUDIT_LAYERS, va_recommended=False,
    )
    raw_dcp = compose_case(
        "glm-5.3-flash", hardware=HW_DEFAULT, dp=1, tp=8, ep=8, dcp=8,
        num_devices=8, layers=AUDIT_LAYERS, va_recommended=False,
    )
    cs_ltp, cs_dcp = parse_case(raw_ltp), parse_case(raw_dcp)

    g_ltp, _, rep_ltp = _lower_and_run_graph(
        cs_ltp, phase="decode", tokens=1, context=2048, sequences=1,
        scope="body", include_submission=False, fast_span_only=True,
    )
    g_dcp, _, rep_dcp = _lower_and_run_graph(
        cs_dcp, phase="decode", tokens=1, context=2048, sequences=8,
        scope="body", include_submission=False, fast_span_only=True,
    )
    _, _, rep_ltp_64k = _lower_and_run_graph(
        cs_ltp, phase="decode", tokens=1, context=65536, sequences=8,
        scope="body", include_submission=False, fast_span_only=True,
    )
    _, _, rep_dcp_64k = _lower_and_run_graph(
        cs_dcp, phase="decode", tokens=1, context=65536, sequences=64,
        scope="body", include_submission=False, fast_span_only=True,
    )
    t_ltp = float(rep_ltp["duration_ms"])
    t_dcp = float(rep_dcp["duration_ms"])
    t_ltp_64k = float(rep_ltp_64k["duration_ms"])
    t_dcp_64k = float(rep_dcp_64k["duration_ms"])
    assert all(math.isfinite(v) and v > 0.0 for v in (t_ltp, t_dcp, t_ltp_64k, t_dcp_64k))
    # Workload scaling monotonicity & cross-layout physical parity (no unphysical 100x+ distortion).
    assert t_ltp_64k > t_ltp and t_dcp_64k > t_dcp
    assert 0.2 <= (t_dcp / t_ltp) <= 5.0 and 0.2 <= (t_dcp_64k / t_ltp_64k) <= 5.0
    # Local-TP emits local_tp exchanges and zero DCP exchanges; TP+DCP emits DCP Q-gather + A2A.
    assert any(dict(op.metadata).get("group_axis") == "local_tp" for op in g_ltp[0])
    assert not any(op.recipe == "dcp_output_alltoall" for op in g_ltp[0])
    assert any(op.recipe == "dcp_output_alltoall" for op in g_dcp[0])

    # Honest single-sequence vs matched-global private-batch cache residency at ctx=65536:
    # 1) Single-sequence per-die footprint: DCP=8 shards KV across 8 dies (< 25% of DP8+Local-TP).
    # 2) Matched B=64 private-sequence batch per die (8 local seqs on DP8 vs 64 shared seqs on DCP8):
    #    hybrid KDA+KeyPool retains unsharded recurrent/indexer state per sequence, so DP8 < DCP8!
    kv_seq_ltp = max(Memory(cs_ltp).cache_bytes(65536, padded=False))
    kv_seq_dcp = max(Memory(cs_dcp).cache_bytes(65536, padded=False))
    assert 0.0 < kv_seq_dcp < kv_seq_ltp * 0.25
    assert (kv_seq_ltp * 8) < (kv_seq_dcp * 64)


def test_canonical_parallelism_and_kv_replication_transparency():
    raw_dp = compose_case("deepseek-v3.2", hardware=HW_DEFAULT, dp=8, tp=1, ep=8, dcp=1, num_devices=8, layers=AUDIT_LAYERS)
    raw_tp_rep = compose_case("deepseek-v3.2", hardware=HW_DEFAULT, dp=1, tp=8, ep=8, dcp=1, num_devices=8, layers=AUDIT_LAYERS)
    raw_tp_dcp = compose_case("deepseek-v3.2", hardware=HW_DEFAULT, dp=1, tp=8, ep=8, dcp=8, num_devices=8, layers=AUDIT_LAYERS)
    assert raw_dp["runtime"]["metadata"]["kv_cache_replicated_tp"] is False
    assert raw_dp["runtime"]["metadata"]["kv_cache_replication_factor"] == 1
    assert raw_tp_rep["runtime"]["metadata"]["kv_cache_replicated_tp"] is True
    assert raw_tp_rep["runtime"]["metadata"]["kv_cache_replication_factor"] == 8
    assert raw_tp_rep["runtime"]["metadata"]["canonical_kv_sharding"] == "tp_kv_replicated"
    assert raw_tp_dcp["runtime"]["metadata"]["kv_cache_replicated_tp"] is False
    assert raw_tp_dcp["runtime"]["metadata"]["kv_cache_replication_factor"] == 1
    info = kv_cache_replication_info(parse_case(raw_tp_rep).model, parse_case(raw_tp_rep).runtime.parallel)
    assert info["kv_cache_replicated_tp"] is True and info["kv_cache_replication_factor"] == 8


def test_per_sku_tutorial_profiles_kpool_dtypes_and_gated_pr_mechanisms():
    # 1) Per-SKU tutorial flags match recommended_topology without contradiction and apply under va_recommended=True.
    prof_a3 = va_recommended_config("glm-5.3-flash", "a3")
    prof_a5 = va_recommended_config("glm-5.3-flash", "a5")
    assert prof_a3["tutorial_flags"]["kv_cache_dtype"] == "int8"
    assert "indexer_kv_dtype" not in prof_a3["tutorial_flags"]
    assert prof_a5["tutorial_flags"]["kv_cache_dtype"] == "fp8"
    assert prof_a5["tutorial_flags"]["indexer_kv_dtype"] == "fp8"

    c_a3 = compose_case("glm-5.3-flash", hardware="a3", dp=1, tp=16, ep=16, num_devices=16, layers=AUDIT_LAYERS, va_recommended=True)
    c_a5 = compose_case("glm-5.3-flash", hardware="a5", dp=1, tp=8, ep=8, num_devices=8, layers=AUDIT_LAYERS, va_recommended=True)
    c_a5_ovr = compose_case("glm-5.3-flash", hardware="a5", dp=1, tp=8, ep=8, num_devices=8, layers=AUDIT_LAYERS, va_recommended=True, kv_cache_dtype="bf16", indexer_kv_dtype="auto")
    assert c_a3["model"]["quant"]["kv_dtype"] == "int8" and c_a3["model"]["quant"]["index_dtype"] == "bf16"
    assert c_a5["model"]["quant"]["kv_dtype"] == "fp8_e4m3" and c_a5["model"]["quant"]["index_dtype"] == "fp8_e4m3"
    assert c_a5_ovr["model"]["quant"]["kv_dtype"] == "bf16" and c_a5_ovr["model"]["quant"]["index_dtype"] == "bf16"
    assert max(Memory(parse_case(c_a5)).cache_bytes(1024, padded=False)) < max(Memory(parse_case(c_a5_ovr)).cache_bytes(1024, padded=False))

    # 2) KeyPool backend indexer_kv_dtype accepts auto/bf16/fp8 and rejects int8.
    with pytest.raises(ModelingError, match="kpool_indexer_kv_dtype_unsupported:int8"):
        compose_case("glm-5.3-flash", hardware=HW_DEFAULT, dp=1, tp=1, ep=1, num_devices=1, layers=AUDIT_LAYERS, va_recommended=False, indexer_kv_dtype="int8")

    # 3) quantized_allreduce: NEVER active on A5; on A2 BF16 TP=4 compresses ONLY above 16M-element threshold!
    raw_a5_pr = compose_case("deepseek-v3.2", hardware=HW_DEFAULT, dp=1, tp=4, pcp=2, dcp=4, ep=8, num_devices=8, layers=AUDIT_LAYERS, feature_track="unmerged_pr", _stack_phase="decode")
    a5_active = {e["id"] for e in raw_a5_pr["runtime"]["metadata"]["feature_track_report"]["active_mechanisms"]}
    assert {"universal_dcp_query_gather", "sfa_dcp_sharded_indexer", "pcp_w8a8_quantized_allgather"} <= a5_active
    assert "quantized_allreduce" not in a5_active

    c_a2_q = compose_case("qwen3-32b", hardware=HW_A2, dp=1, tp=4, ep=1, num_devices=4, layers=1, quant="bf16", va_recommended=False, optimizations=("quantized_allreduce",))
    g_lo, _, rep_lo = _lower_and_run_graph(parse_case(c_a2_q), phase="prefill", tokens=QUANT_AR_BELOW_TOKENS, context=0, sequences=1, scope="body", include_submission=False, fast_span_only=True)
    g_hi, _, rep_hi = _lower_and_run_graph(parse_case(c_a2_q), phase="prefill", tokens=QUANT_AR_ABOVE_TOKENS, context=0, sequences=1, scope="body", include_submission=False, fast_span_only=True)
    assert float(rep_lo["duration_ms"]) > 0.0 and float(rep_hi["duration_ms"]) > 0.0
    ar_lo = next(op for op in g_lo[0] if op.op_id == "s0.l0.tp_attn")
    assert ar_lo.family == "allreduce" and not dict(ar_lo.metadata).get("quantized_allreduce")
    int8_hi = next(op for op in g_hi[0] if op.op_id == "s0.l0.tp_attn.quant_ar_int8_reduce")
    assert int8_hi.dtype == "int8" and dict(int8_hi.metadata).get("quantized_allreduce") is True
    assert int8_hi.link_bytes == pytest.approx(31_457_280.0, rel=1e-6)
    assert any(op.recipe == "quant_allreduce_quantize" for op in g_hi[0])
    assert any(op.recipe == "quant_allreduce_dequantize" for op in g_hi[0])

    # 4) pcp_w8a8_quantized_allgather emits real physical quant -> int8 gather -> dequant on hidden gathers!
    c_pcp_q = compose_case("glm-5.3-flash", hardware=HW_DEFAULT, dp=1, tp=1, pcp=2, dcp=1, ep=1, num_devices=2, layers=1, quant="w8a8", va_recommended=False, optimizations=("pcp_w8a8_quantized_allgather",))
    g_pq, _, rep_pq = _body_graphs_local(c_pcp_q, phase="prefill", tokens=128, context=0, sequences=1)
    assert float(rep_pq["duration_ms"]) > 0.0
    assert any(op.recipe == "kda_pcp_gather_quant" for op in g_pq[0])
    assert any(op.recipe == "kda_pcp_gather" and op.dtype == "int8" for op in g_pq[0])
    assert any(op.recipe == "kda_pcp_gather_dequant" for op in g_pq[0])


def _body_graphs_local(raw_case: dict, *, phase: str, tokens: int, context: int, sequences: int):
    return _lower_and_run_graph(
        parse_case(raw_case), phase=phase, tokens=tokens, context=context, sequences=sequences,
        scope="body", include_submission=False, fast_span_only=True,
    )


def test_dcp_factorization_sweep_streaming_errors_and_parallel_parity():
    # Non-power-of-two gcd DCP factorization: DP=3, TP=4, PCP=1, DCP=6 -> (pcp=1, tp=2, dp=3).
    assert factor_dcp_axes(6, pcp=1, tp=4, dp=3) == (1, 2, 3)

    # Enumeration honors V4.1 replicated-cache DCP refusal and GQA layer divisibility.
    v41_cands = enumerate_parallel_candidates("deepseek-v4.1-flash", num_devices=8, allow_dp=True, allow_ep=True, allow_dcp=True)
    assert v41_cands and all(c["dcp"] == 1 for c in v41_cands)

    base_cands = enumerate_parallel_candidates("deepseek-v3", num_devices=8, allow_dp=True, allow_ep=True)
    assert base_cands and all(set(c.keys()) == {"dp", "tp", "ep", "pp", "pcp"} for c in base_cands)
    dcp_cands = enumerate_parallel_candidates("deepseek-v3", num_devices=8, allow_dp=True, allow_ep=True, allow_dcp=True)
    assert any(c["tp"] == 1 and c["dp"] == 8 and c["dcp"] == 1 for c in dcp_cands)
    assert any(c["tp"] == 4 and c["dp"] == 2 and c["dcp"] == 4 for c in dcp_cands)

    # Streaming sweep_topologies (jobs=1 vs jobs=2): identical candidates, metrics, and rejection counters.
    raw_case = compose_case("deepseek-v3", hardware=HW_DEFAULT, dp=4, tp=1, ep=1, num_devices=4, layers=2, va_recommended=False)
    explicit = [
        {"dp": 4, "tp": 1, "ep": 1, "pp": 1, "pcp": 1, "dcp": 1},
        {"dp": 4, "tp": 1, "ep": 1, "pp": 1, "pcp": 1, "dcp": 1, "o_proj_tp": 4},
        {"dp": 1, "tp": 4, "ep": 1, "pp": 1, "pcp": 1, "dcp": 1},
        {"dp": 1, "tp": 4, "ep": 1, "pp": 1, "pcp": 1, "dcp": 4},
        {"dp": 1, "tp": 4, "ep": 1, "pp": 1, "pcp": 1, "dcp": 3},
    ]
    rep_s = sweep_topologies(raw_case, candidates=explicit, num_devices=4, sla_ms=50.0, context=1024, phase="decode", scope="body", output_budget=64, max_sequences_per_dp=8, jobs=1)
    rep_p = sweep_topologies(raw_case, candidates=explicit, num_devices=4, sla_ms=50.0, context=1024, phase="decode", scope="body", output_budget=64, max_sequences_per_dp=8, jobs=2)
    assert rep_s["cache_stats"]["static_topology_rejections"] == 1
    assert rep_p["cache_stats"]["static_topology_rejections"] == 1
    assert [r["candidate_id"] for r in rep_s["candidates"]] == [r["candidate_id"] for r in rep_p["candidates"]]
    for rs, rp in zip(rep_s["candidates"], rep_p["candidates"], strict=True):
        assert rs["feasible"] == rp["feasible"]
        assert rs["step_ms_at_b1"] == pytest.approx(rp["step_ms_at_b1"], rel=1e-9)

    # UnmodeledCapabilityError MUST survive pickling (ProcessPoolExecutor) and propagate untouched.
    err = UnmodeledCapabilityError(
        capability="test_cap", entry="sweep", code="injected_gap", message="injected_gap",
    )
    unpickled = pickle.loads(pickle.dumps(err))
    assert isinstance(unpickled, UnmodeledCapabilityError) and unpickled.gap["code"] == "injected_gap"
    with mock.patch("mindie_sim.sweep.validate_candidate_statically", side_effect=err):
        with pytest.raises(UnmodeledCapabilityError, match="injected_gap"):
            sweep_topologies(raw_case, candidates=explicit[:2], num_devices=4, sla_ms=50.0, context=1024, phase="decode", scope="body", output_budget=64, max_sequences_per_dp=8, jobs=1)
    # Invalid candidate entries fail up-front before dispatch with identical behavior in jobs=1 & jobs=2.
    bad_cands = [explicit[0], {"dp": 4, "tp": "bad", "ep": 1, "pp": 1, "pcp": 1}]
    for jobs in (1, 2):
        with pytest.raises(ValueError):
            sweep_topologies(raw_case, candidates=bad_cands, num_devices=4, sla_ms=50.0, context=1024, phase="decode", scope="body", output_budget=64, max_sequences_per_dp=8, jobs=jobs)
