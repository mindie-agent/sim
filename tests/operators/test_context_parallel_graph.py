"""Context-parallel (DCP/PCP/PCP+DCP) graph rewrite and conservation invariants.

[Role]
  Verify 1:1 context-parallel graph lowering across MLA, sparse MLA/SFA,
  window-compressed latent MQA, GQA, hybrid sliding/full GQA, KDA, and GDN:
  per-rank/per-layer cross-rank Attention FLOPs, collective wire traffic, and
  bounded KV HBM conservation; ragged/idle cross-DP DCP cohorts and subgroup
  readiness; executable multi-stream MTP overlap; fast 4B vs generic 8B LSE
  exchange; opt-in Q replication weight+GEMM parity; SFA index remap, prefill
  KV gather, sharded indexer, and KeyPool pre-store ordering; V4/V4.1 PCP
  hidden/ring gathers and replicated-cache DCP refusal; DualChunkSwap sliding
  windows; KDA/GDN PCP wire/DAG contracts; and PCP o_proj weight sharding.

[Invariants]
  * Useful attention FLOPs summed across participating ranks match CP=1 per
    layer within discrete tile rounding; every rank emits positive FLOPs and
    positive DCP collective bytes, and KV HBM obeys both lower and upper bounds.
  * Every emitted operator's dependencies strictly precede it in graph order
    and resolve either to an earlier op or a declared graph input root.
  * Opt-in sharding flags alter resident bytes, collectives, and streams
    deterministically without silent fallback.

[Timeouts & Locks]
  Tier-0 pure in-memory graph lowering and engine replay; no device or I/O.

[Side-Effect Policy]
  Read-only case composition and operator graph construction.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mindie_sim.catalog.context_parallel import split_pcp_pairs
from mindie_sim.compose import compose_case
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.efficiency import _lower_and_run_graph
from mindie_sim.prediction.graph_analysis import _bind_collective_readiness
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy, gqa
from mindie_sim.prediction.sharding.context_parallel import (
    CP_CHUNKED_PREFILL_STREAM,
    CP_DECODE_COMM_STREAM,
    CP_WEIGHT_STREAM,
    FAST_LSE_WIRE_BYTES,
    GENERIC_LSE_WIRE_BYTES,
    is_fast_sfa_dcp_exchange,
    should_split_mtp_dcp,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView

pytestmark = pytest.mark.tier0

HW_DEFAULT = "a5"
AUDIT_LAYERS = 4
DECODE_CONTEXT = 1024
FLOPS_REL_TOL = 0.02
TWO_RANKS = 2
FOUR_RANKS = 4
BODY_INPUT_ROOTS = frozenset({"analysis_input"})
DCP_WIRE_RECIPES = frozenset({
    "dcp_query_allgather",
    "dcp_dp_query_allgather",
    "dcp_output_alltoall",
    "dcp_pcp_allgather",
    "dcp_kv_allgather",
    "dcp_prefill_kv_gather",
    "candidate_gather",
})
DCP_COLLECTIVE_AXES = frozenset({
    "cp",
    "dcp",
    "dcp_tp",
    "dcp_pcp",
    "dcp_dp",
    "dcp_a2a",
    "tp",
    "pcp",
})
DTYPE_BYTES_FP8 = 1
DTYPE_BYTES_BF16 = 2
DTYPE_BYTES_FP32 = 4


def _assert_strict_dag(ops, *, external_roots: frozenset[str] = BODY_INPUT_ROOTS) -> None:
    seen: set[str] = set()
    for op in ops:
        assert op.op_id not in seen, f"DAG violation: duplicate op_id {op.op_id}"
        for dep in op.deps:
            if dep in external_roots:
                continue
            assert dep in seen, f"DAG violation: {op.op_id} depends on undefined/later {dep}"
        seen.add(op.op_id)


def _expected_min_dcp_wire(op, dcp_degree: int) -> float:
    shape = tuple(int(d) for d in (op.local_shape or ()))
    numel = math.prod(shape) if shape else 0
    if numel <= 0 or dcp_degree <= 1:
        return 0.0
    if op.recipe == "candidate_gather":
        return float(numel)
    ds = str(op.dtype or "").lower()
    eb = DTYPE_BYTES_FP8 if "8" in ds else (DTYPE_BYTES_FP32 if "32" in ds else DTYPE_BYTES_BF16)
    return float(numel * eb) * float(dcp_degree - 1) / float(dcp_degree)


def _body_graphs(
    raw_case: dict,
    *,
    phase: str,
    tokens: int,
    context: int,
    sequences: int | list[int] = 2,
    fast_span_only: bool = True,
):
    case = parse_case(raw_case)
    graphs, engine, rep = _lower_and_run_graph(
        case, phase=phase, tokens=tokens, context=context, sequences=sequences,
        scope="body", include_submission=False, fast_span_only=fast_span_only,
    )
    return case, graphs, engine, rep


def _layer_flops_by_rank(graphs) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for rank, ops in graphs.items():
        per_layer: dict[str, float] = {}
        for op in ops:
            if op.family != "attention":
                continue
            lk = op.op_id.rsplit(".", 1)[0] if "." in op.op_id else op.op_id
            per_layer[lk] = per_layer.get(lk, 0.0) + float(op.flops)
        out[int(rank)] = per_layer
    return out


def _assert_kv_bounds(kv_base: tuple[float, ...], kv_dcp: tuple[float, ...], dcp: int, exact: float | None) -> None:
    assert kv_base and kv_dcp and min(kv_base) > 0.0 and min(kv_dcp) > 0.0
    hi_base, hi_dcp, lo_dcp = max(kv_base), max(kv_dcp), min(kv_dcp)
    if exact is not None:
        assert hi_dcp == pytest.approx(hi_base * exact, rel=1e-6)
        assert lo_dcp == pytest.approx(hi_base * exact, rel=1e-6)
    else:
        assert lo_dcp >= hi_base / float(dcp) - 1.0
        assert hi_dcp < hi_base


@pytest.mark.parametrize(
    "model_id,par_base,par_dcp,exact_full_ratio",
    [
        ("deepseek-v3", dict(dp=1, tp=4, ep=4, dcp=1), dict(dp=1, tp=4, ep=4, dcp=4), 0.25),
        ("deepseek-v3.2", dict(dp=1, tp=4, ep=4, dcp=1), dict(dp=1, tp=4, ep=4, dcp=4), None),
        ("kimi-k3", dict(dp=2, tp=1, ep=2, dcp=1), dict(dp=2, tp=1, ep=2, dcp=2), None),
        ("qwen3-32b", dict(dp=2, tp=1, ep=1, dcp=1), dict(dp=2, tp=1, ep=1, dcp=2), 0.5),
        ("qwen3.5-35b-a3b", dict(dp=2, tp=1, ep=2, dcp=1), dict(dp=2, tp=1, ep=2, dcp=2), None),
    ],
)
def test_dcp_per_rank_per_layer_conservation_and_negative_controls(
    model_id: str, par_base: dict, par_dcp: dict, exact_full_ratio: float | None,
):
    c_base = compose_case(
        model_id, hardware=HW_DEFAULT, layers=AUDIT_LAYERS,
        num_devices=par_base["dp"] * par_base["tp"], va_recommended=False, **par_base,
    )
    c_dcp = compose_case(
        model_id, hardware=HW_DEFAULT, layers=AUDIT_LAYERS,
        num_devices=par_dcp["dp"] * par_dcp["tp"], va_recommended=False, **par_dcp,
    )
    case_base, g_base, _, _ = _body_graphs(c_base, phase="decode", tokens=1, context=DECODE_CONTEXT)
    case_dcp, g_dcp, _, _ = _body_graphs(c_dcp, phase="decode", tokens=1, context=DECODE_CONTEXT)
    for ops in g_dcp.values():
        _assert_strict_dag(ops)

    scale_b = (par_base["dp"] / len(g_base)) * par_base["tp"]
    scale_d = (par_dcp["dp"] / len(g_dcp)) * par_dcp["tp"]
    fl_b, fl_d = _layer_flops_by_rank(g_base), _layer_flops_by_rank(g_dcp)
    layers = sorted({lk for m in fl_b.values() for lk in m})
    dcp_layers = sorted({lk for m in fl_d.values() for lk in m})
    assert layers and dcp_layers == layers
    layer_pairs: dict[str, tuple[float, ...]] = {}
    for ops in g_dcp.values():
        for op in ops:
            if op.family != "attention":
                continue
            lk = op.op_id.rsplit(".", 1)[0] if "." in op.op_id else op.op_id
            pairs = dict(op.metadata).get("cp_per_rank_pairs")
            if pairs and lk not in layer_pairs:
                layer_pairs[lk] = tuple(float(v) for v in pairs)
    sorted_ranks = sorted(fl_d)
    for lk in layers:
        tot_b = sum(fl_b[r].get(lk, 0.0) for r in fl_b) * scale_b
        assert math.isfinite(tot_b) and tot_b > 0.0
        pairs = layer_pairs.get(lk)
        pair_sum = sum(pairs) if pairs else 0.0
        for pos, r in enumerate(sorted_ranks):
            r_fl = fl_d[r].get(lk, 0.0)
            assert math.isfinite(r_fl) and r_fl > 0.0
            share = (
                1.0
                if len(sorted_ranks) == 1
                else (pairs[pos] / pair_sum if (pairs and pair_sum > 0.0 and len(pairs) >= len(sorted_ranks)) else 1.0 / len(sorted_ranks))
            )
            exp_r = (tot_b / scale_d) * share
            assert abs(r_fl - exp_r) / exp_r <= FLOPS_REL_TOL
        tot_d = sum(fl_d[r].get(lk, 0.0) for r in sorted_ranks) * scale_d
        assert abs(tot_d - tot_b) / tot_b <= FLOPS_REL_TOL

    for r, ops in g_dcp.items():
        dcp_ops = [op for op in ops if op.recipe in DCP_WIRE_RECIPES]
        assert len(dcp_ops) >= len(layers), f"rank {r} missing per-layer DCP collectives"
        for op in dcp_ops:
            assert dict(op.metadata).get("group_axis") in DCP_COLLECTIVE_AXES
            min_wire = _expected_min_dcp_wire(op, par_dcp["dcp"])
            assert math.isfinite(op.link_bytes) and op.link_bytes >= min_wire - 1.0 and op.link_bytes > 0.0

    kv_base = tuple(Memory(case_base).cache_bytes(DECODE_CONTEXT, padded=False))
    kv_dcp = tuple(Memory(case_dcp).cache_bytes(DECODE_CONTEXT, padded=False))
    _assert_kv_bounds(kv_base, kv_dcp, par_dcp["dcp"], exact_full_ratio)

    # Negative controls: undefined dep, duplicate op_id, and zeroed KV bytes MUST fail.
    bad_op = replace(g_dcp[0][0], deps=("never-produced",))
    with pytest.raises(AssertionError, match="undefined/later"):
        _assert_strict_dag([bad_op])
    dup_op = replace(g_dcp[0][0], deps=(g_dcp[0][0].op_id,))
    with pytest.raises(AssertionError, match="duplicate op_id"):
        _assert_strict_dag([g_dcp[0][0], dup_op])
    with pytest.raises(AssertionError):
        _assert_kv_bounds(kv_base, tuple(0.0 for _ in kv_dcp), par_dcp["dcp"], exact_full_ratio)


def test_cross_dp_dcp_ragged_idle_hybrid_groups_and_subgroup_readiness():
    # 1) Ragged [1, 3] and idle [1, 0] on DP2/TP1/DCP2 GQA: exact cohort union & peer traffic.
    c_gqa = compose_case(
        "qwen3-32b", hardware=HW_DEFAULT, dp=2, tp=1, ep=1, dcp=2,
        num_devices=2, layers=1, va_recommended=False,
    )
    _, g_rag, _, _ = _body_graphs(c_gqa, phase="decode", tokens=1, context=511, sequences=[1, 3])
    assert sorted(g_rag.keys()) == [0, 1]
    for r in (0, 1):
        _assert_strict_dag(g_rag[r])
        attn = next(op for op in g_rag[r] if op.family == "attention")
        assert attn.flops == pytest.approx(33_554_432.0, rel=1e-6)
    a2a0 = next(op for op in g_rag[0] if op.recipe == "dcp_output_alltoall")
    a2a1 = next(op for op in g_rag[1] if op.recipe == "dcp_output_alltoall")
    assert a2a0.family == "alltoallv" and a2a0.link_bytes == 50_688
    assert a2a1.family == "alltoallv" and a2a1.link_bytes == 16_896

    _, g_idle, _, _ = _body_graphs(c_gqa, phase="decode", tokens=1, context=511, sequences=[1, 0])
    for r in (0, 1):
        _assert_strict_dag(g_idle[r])
        assert any(op.recipe == "dcp_dp_query_allgather" and op.link_bytes == 16_384 for op in g_idle[r])
        assert next(op for op in g_idle[r] if op.family == "attention").flops == pytest.approx(8_388_608.0, rel=1e-6)
    assert any(op.recipe == "dcp_lse_merge" for op in g_idle[0])
    assert not any(op.recipe == "dcp_lse_merge" for op in g_idle[1])

    # Partial interleave block ownership: ctx=127 puts all 512 pairs on rank 0 and 0 on rank 1.
    cs_gqa, g_partial, _, _ = _body_graphs(c_gqa, phase="decode", tokens=1, context=127, sequences=[1, 3])
    assert next(op for op in g_partial[0] if op.family == "attention").flops == pytest.approx(16_777_216.0, rel=1e-6)
    assert next(op for op in g_partial[1] if op.family == "attention").flops == pytest.approx(0.0, abs=1e-6)
    alloc_128 = Memory(cs_gqa).rank_cache_allocation(128)
    assert [int(row["valid"]["total_bytes_per_stage_rank"][0]) for row in alloc_128] == [524_288, 0]

    # 2) Hybrid GQA DP2/TP2/DCP4 resolves 4-rank (kv=1) vs 2-rank same-TP-lane (kv=2).
    raw_hyb = compose_case(
        "qwen3-32b", hardware=HW_DEFAULT, dp=2, tp=2, ep=1, dcp=1,
        num_devices=4, layers=2, va_recommended=False,
    )
    raw_hyb["model"]["layer_plans"][0]["num_kv_heads"] = 1
    raw_hyb["model"]["layer_plans"][1]["num_kv_heads"] = 2
    raw_hyb["runtime"]["parallel"]["cp"] = 4
    raw_hyb["runtime"]["parallel"]["dcp"] = 4
    cs_hyb = parse_case(raw_hyb)
    topo = Topology(cs_hyb.machine, cs_hyb.runtime.parallel)
    assert topo.groups("dcp_a2a", stage=0, replica=0, metadata={"dcp_pcp_factor": 1, "dcp_tp_gather": 2, "dcp_dp_width": 2}) == ((0, 1, 2, 3),)
    assert topo.groups("dcp_a2a", stage=0, replica=0, metadata={"dcp_pcp_factor": 1, "dcp_tp_gather": 1, "dcp_dp_width": 2}) == ((0, 2), (1, 3))

    # 3) DP4/EP4/DCP2 readiness joins per 2-rank DCP subgroup (0,1)->2 and (2,3)->100, never parent EP=4.
    from types import SimpleNamespace

    c_dp4 = compose_case(
        "deepseek-v3", hardware=HW_DEFAULT, dp=4, tp=1, ep=4, dcp=2,
        num_devices=4, layers=AUDIT_LAYERS, va_recommended=False,
    )
    cs_dp4, g_dp4, _, _ = _body_graphs(c_dp4, phase="decode", tokens=1, context=512, sequences=[1, 1, 1, 1])
    sample_ag = next(op for op in g_dp4[0] if op.recipe == "dcp_dp_query_allgather")
    group_for = {r: (0, 1, 2, 3) for r in range(4)}
    records = [
        dict(replica=r, op=sample_ag, collective_algorithm=False,
             ready=SimpleNamespace(time=v), predecessors=(f"r{r}",))
        for r, v in enumerate((1.0, 2.0, 99.0, 100.0))
    ]
    _bind_collective_readiness(cs_dp4, records, group_for, SimpleNamespace(native_group_launches={}))
    assert [rec["collective_ready_ms"] for rec in records] == [2.0, 2.0, 100.0, 100.0]


def test_pcp_plus_dcp_exchange_and_executable_stream_mtp_overlap():
    # 1) Joint PCP=2 + DCP=4 two-stage exchange and generic 8B LSE wire bytes.
    c_base = compose_case(
        "deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=2, pcp=2, dcp=1, ep=4,
        num_devices=4, layers=AUDIT_LAYERS, va_recommended=False,
    )
    c_joint = compose_case(
        "deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=2, pcp=2, dcp=4, ep=4,
        num_devices=4, layers=AUDIT_LAYERS, va_recommended=False,
    )
    _, g_base, _, _ = _body_graphs(c_base, phase="decode", tokens=1, context=DECODE_CONTEXT)
    _, g_joint, _, _ = _body_graphs(c_joint, phase="decode", tokens=1, context=DECODE_CONTEXT)
    ops_joint = g_joint[0]
    _assert_strict_dag(ops_joint)
    recipes = {op.recipe for op in ops_joint}
    assert {"dcp_query_allgather", "dcp_output_alltoall", "dcp_pcp_allgather", "dcp_lse_merge"} <= recipes
    a2a_op = next(op for op in ops_joint if op.recipe == "dcp_output_alltoall")
    assert dict(a2a_op.metadata)["lse_wire_bytes"] == GENERIC_LSE_WIRE_BYTES
    fl_base = sum(op.flops for op in g_base[0] if op.family == "attention")
    fl_joint = sum(op.flops for op in ops_joint if op.family == "attention")
    assert abs(fl_joint * 2.0 - fl_base) / fl_base <= FLOPS_REL_TOL

    # 2) MTP decode dual-stream split: executable op.stream drives true timeline overlap.
    c_mtp = compose_case(
        "deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=4, ep=1, dcp=4,
        num_devices=4, layers=1, va_recommended=False,
    )
    _, g_mtp, _, rep_mtp = _body_graphs(
        c_mtp, phase="decode", tokens=4, context=DECODE_CONTEXT, sequences=2, fast_span_only=False,
    )
    ops_mtp = g_mtp[0]
    _assert_strict_dag(ops_mtp)
    cur_op = next(op for op in ops_mtp if op.recipe == "dcp_mtp_current_attention")
    assert cur_op.stream == CP_DECODE_COMM_STREAM
    assert dict(cur_op.metadata)["sparse_mode"] == 3
    spans = {ev["op_id"]: (float(ev["ready_ms"]), float(ev["complete_ms"])) for ev in rep_mtp["timeline"]}
    a2a_id = next(op.op_id for op in ops_mtp if op.recipe == "dcp_output_alltoall")
    cur_s, cur_e = spans[cur_op.op_id]
    a2a_s, a2a_e = spans[a2a_id]
    assert cur_s < a2a_e and a2a_s < cur_e, f"MTP current ({cur_s},{cur_e}) serialized vs A2A ({a2a_s},{a2a_e})"

    # Sliding-window layers never split history/current with sparse_mode=0.
    raw_win = compose_case("qwen3-32b", hardware=HW_DEFAULT, dp=1, tp=4, ep=1, layers=1, va_recommended=False)
    win_plan = replace(parse_case(raw_win).model.layer_plans[0], window=128)
    assert should_split_mtp_dcp(win_plan, (RequestView("r0", 4, 256, False, "decode"),), layer_dcp=2, sparse=False) is False


def test_lse_wire_modes_and_dcp_q_replicate_resident_and_gemm_parity():
    assert is_fast_sfa_dcp_exchange(pcp_factor=1, dp_width=1, tp_gather=4, local_heads=32, v_dim=512, act_dtype="bf16", defer_combine=False) is True
    assert is_fast_sfa_dcp_exchange(pcp_factor=2, dp_width=1, tp_gather=2, local_heads=32, v_dim=512, act_dtype="bf16", defer_combine=False) is False
    assert is_fast_sfa_dcp_exchange(pcp_factor=1, dp_width=1, tp_gather=4, local_heads=32, v_dim=512, act_dtype="fp16", defer_combine=False) is False
    assert GENERIC_LSE_WIRE_BYTES == 2 * FAST_LSE_WIRE_BYTES

    c_fast = compose_case("deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=4, ep=1, dcp=4, num_devices=4, layers=1, va_recommended=False)
    c_rep = compose_case("deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=4, ep=1, dcp=4, num_devices=4, layers=1, va_recommended=False, dcp_q_replicate=True)
    cs_fast, g_fast_d, _, _ = _body_graphs(c_fast, phase="decode", tokens=1, context=DECODE_CONTEXT)
    cs_rep, g_rep_d, _, _ = _body_graphs(c_rep, phase="decode", tokens=1, context=DECODE_CONTEXT)
    _, g_rep_p, _, _ = _body_graphs(c_rep, phase="prefill", tokens=64, context=0, sequences=1)

    fast_ex = next(op for op in g_fast_d[0] if op.recipe == "dcp_output_alltoall")
    assert dict(fast_ex.metadata)["lse_wire_bytes"] == FAST_LSE_WIRE_BYTES
    assert any(op.recipe == "dcp_query_allgather" for op in g_fast_d[0])
    assert not any(op.recipe == "dcp_query_allgather" for op in g_rep_d[0])
    # Replicating q_b (+28,318,464 B) and W_UK_T half of kv_b (+6,292,992 B) = exact +34,611,456 B.
    w_fast = max(Memory(cs_fast).weights)
    w_rep = max(Memory(cs_rep).weights)
    assert (w_rep - w_fast) == pytest.approx(34_611_456.0, abs=1.0)
    # Both prefill and decode project all 128 * 192 = 24,576 replicated Q rows.
    for ops in (g_rep_d[0], g_rep_p[0]):
        q_b = next(op for op in ops if op.recipe == "q_b")
        assert q_b.local_shape[1] == 24_576


def test_sfa_prefill_kv_gather_sharded_indexer_and_keypool_pre_store_dag():
    # 1) Dense chunked MLA prefill gathers compact KV across DCP ranks.
    c_dense = compose_case("deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=2, ep=1, dcp=2, num_devices=2, layers=1, va_recommended=False)
    _, g_dense_chk, _, _ = _body_graphs(c_dense, phase="prefill", tokens=128, context=512, sequences=1)
    _assert_strict_dag(g_dense_chk[0])
    assert next(op for op in g_dense_chk[0] if op.recipe == "dcp_prefill_kv_gather").link_bytes == 294_912

    # 2) Replicated vs sharded SFA indexer in decode & prefill.
    c_base = compose_case("deepseek-v3.2", hardware=HW_DEFAULT, dp=1, tp=4, ep=4, dcp=1, num_devices=4, layers=AUDIT_LAYERS, va_recommended=False)
    c_rep = compose_case("deepseek-v3.2", hardware=HW_DEFAULT, dp=1, tp=4, ep=4, dcp=4, num_devices=4, layers=AUDIT_LAYERS, va_recommended=False)
    c_shd = compose_case("deepseek-v3.2", hardware=HW_DEFAULT, dp=1, tp=4, ep=4, dcp=4, num_devices=4, layers=AUDIT_LAYERS, va_recommended=False, enable_sfa_dcp_sharded_indexer=True)
    cs_base, _, _, _ = _body_graphs(c_base, phase="decode", tokens=1, context=DECODE_CONTEXT)
    cs_rep, g_rep, _, _ = _body_graphs(c_rep, phase="decode", tokens=1, context=DECODE_CONTEXT)
    cs_shd, g_shd, _, _ = _body_graphs(c_shd, phase="decode", tokens=1, context=DECODE_CONTEXT)
    _assert_strict_dag(g_rep[0])
    _assert_strict_dag(g_shd[0])
    assert any(op.recipe == "sparse_index_remap" for op in g_rep[0])
    assert any(op.recipe == "candidate_gather" for op in g_shd[0])
    kv_base = max(Memory(cs_base).cache_bytes(DECODE_CONTEXT, padded=False))
    kv_rep = max(Memory(cs_rep).cache_bytes(DECODE_CONTEXT, padded=False))
    kv_shd = max(Memory(cs_shd).cache_bytes(DECODE_CONTEXT, padded=False))
    assert kv_shd == pytest.approx(kv_base / 4.0, rel=1e-6) and kv_shd < kv_rep < kv_base

    # Sharded SFA prefill keeps KV local (no sfa_dcp_prefill_kv_gather) and runs Q gather + LSE exchange.
    _, g_rep_pf, _, _ = _body_graphs(c_rep, phase="prefill", tokens=512, context=128, sequences=1)
    _, g_shd_pf, _, _ = _body_graphs(c_shd, phase="prefill", tokens=512, context=128, sequences=1)
    assert any(op.recipe == "sfa_dcp_prefill_kv_gather" and op.link_bytes == 552_960 for op in g_rep_pf[0])
    assert not any(op.recipe == "sfa_dcp_prefill_kv_gather" for op in g_shd_pf[0])
    assert any(op.recipe == "dcp_output_alltoall" for op in g_shd_pf[0])

    # 3) KeyPool PCP prefill gathers full hidden states BEFORE cann_key_pool (zero post-store index gather);
    #    KeyPool models strictly reject DCP > 1.
    with pytest.raises(ModelingError, match="kpool_indexer_requires_dcp_1"):
        compose_case("glm-5.3-flash", hardware=HW_DEFAULT, dp=1, tp=1, pcp=TWO_RANKS, dcp=TWO_RANKS, ep=TWO_RANKS, num_devices=TWO_RANKS, layers=AUDIT_LAYERS)
    c_kp = compose_case("glm-5.3-flash", hardware=HW_DEFAULT, dp=1, tp=1, pcp=TWO_RANKS, dcp=1, ep=TWO_RANKS, num_devices=TWO_RANKS, layers=AUDIT_LAYERS)
    _, g_kp, _, _ = _body_graphs(c_kp, phase="prefill", tokens=1024, context=0, sequences=1)
    ops_kp = g_kp[0]
    _assert_strict_dag(ops_kp)
    idx_gather = next(i for i, op in enumerate(ops_kp) if op.recipe == "pcp_keypool_hidden_gather")
    idx_pool = next(i for i, op in enumerate(ops_kp) if op.recipe == "cann_key_pool")
    assert idx_gather < idx_pool
    assert ops_kp[idx_gather].link_bytes == 4_194_304
    assert not any(op.recipe == "pcp_index_gather" for op in ops_kp)


def test_window_compressed_pcp_ring_refuses_dcp_and_gqa_chunked_window():
    # 1) Replicated-cache latent MQA: PCP=4, DCP=1 passes with pre-store C2 ring gather; DCP>1 raises!
    with pytest.raises(ModelingError, match="v41_replicated_cache_requires_dcp_1"):
        compose_case("deepseek-v4.1-flash", hardware=HW_DEFAULT, dp=2, tp=1, pcp=4, dcp=4, ep=8, num_devices=8, layers=AUDIT_LAYERS)
    c_c2_eager = compose_case(
        "deepseek-v4.1-flash", hardware=HW_DEFAULT, dp=1, tp=1, pcp=TWO_RANKS, dcp=1,
        ep=TWO_RANKS, num_devices=TWO_RANKS, layers=AUDIT_LAYERS, graph_capture_mode="none",
    )
    _, g_c2_eager, _, _ = _body_graphs(c_c2_eager, phase="prefill", tokens=1024, context=0, sequences=1)
    ops_c2_eager = g_c2_eager[0]
    _assert_strict_dag(ops_c2_eager)
    i_ring = next(i for i, op in enumerate(ops_c2_eager) if op.recipe == "pcp_c2_ring_tail_gather")
    i_pool = next(i for i, op in enumerate(ops_c2_eager) if op.recipe == "compress_pool")
    i_write = next(i for i, op in enumerate(ops_c2_eager) if op.recipe == "pcp_c2_ring_write")
    i_store = next(i for i, op in enumerate(ops_c2_eager) if op.recipe == "compressed_store")
    assert i_ring < i_pool < i_write < i_store and ops_c2_eager[i_ring].link_bytes == 135_168
    assert ops_c2_eager[i_pool].op_id in ops_c2_eager[i_write].deps
    c_vals = next(op for op in ops_c2_eager if op.recipe == "compress_values")
    assert c_vals.dtype == "fp32"
    c_c2_full = compose_case(
        "deepseek-v4.1-flash", hardware=HW_DEFAULT, dp=1, tp=1, pcp=TWO_RANKS, dcp=1,
        ep=TWO_RANKS, num_devices=TWO_RANKS, layers=AUDIT_LAYERS, graph_capture_mode="full",
    )
    # Prefill runs eager (GraphPolicy scope == "none") even when full capture is configured.
    _, g_c2_full, _, _ = _body_graphs(c_c2_full, phase="prefill", tokens=1024, context=0, sequences=1)
    _assert_strict_dag(g_c2_full[0])
    ring_full = next(op for op in g_c2_full[0] if op.recipe == "pcp_c2_ring_tail_gather")
    assert ring_full.link_bytes == 135_168

    # 2) Window-compressed compress_ratio != 2 gathers hidden states ONCE before cache stores.
    c_v4 = compose_case("deepseek-v4-flash-0731", hardware=HW_DEFAULT, dp=1, tp=1, pcp=TWO_RANKS, dcp=1, ep=TWO_RANKS, num_devices=TWO_RANKS, layers=2, va_recommended=False)
    _, g_v4, _, _ = _body_graphs(c_v4, phase="prefill", tokens=128, context=0, sequences=1)
    _assert_strict_dag(g_v4[0])
    assert any(op.recipe == "pcp_hidden_gather" and op.link_bytes == 524_288 for op in g_v4[0])
    assert not any(op.recipe == "pcp_kv_gather" for op in g_v4[0])

    # 3) Sliding-window DualChunkSwap conserves exact pairs; GQA chunked PCP+DCP emits query gather + q=1 current.
    split = split_pcp_pairs(2048, 0, FOUR_RANKS, window=128)
    assert abs(sum(split.per_rank) - sum(min(pos + 1, 128) for pos in range(2048))) <= 1e-6
    c_gqa = compose_case("qwen3-32b", hardware=HW_DEFAULT, dp=1, tp=2, pcp=2, dcp=2, ep=1, num_devices=4, layers=1, va_recommended=False)
    cs_gqa = parse_case(c_gqa)
    res_chk = gqa(
        prefix="chk", plan=cs_gqa.model.layer_plans[0],
        views=(RequestView("r0", 16, 256, False, "prefill"),), stage=0, deps=("analysis_input",),
        model=cs_gqa.model, parallel=cs_gqa.runtime.parallel, quant=cs_gqa.model.quant,
        mode="prefill", padded_tokens=16, policy=AttentionPolicy(), cp_interleave_size=128,
        runtime=cs_gqa.runtime, cache_ids={}, stage_of=lambda _: 0, machine=cs_gqa.machine,
    )
    assert any(op.recipe == "pcp_query_allgather" and op.stream == CP_CHUNKED_PREFILL_STREAM for op in res_chk.ops)
    # Disjoint sliding window history pairs: q=16, ctx=1024, window=128 -> exact 2048 total pairs (0 excess).
    win_plan = replace(cs_gqa.model.layer_plans[0], window=128)
    res_win = gqa(
        prefix="win", plan=win_plan,
        views=(RequestView("r0", 16, 1024, False, "prefill"),), stage=0, deps=("analysis_input",),
        model=cs_gqa.model, parallel=replace(cs_gqa.runtime.parallel, pcp=1, cp=2, dcp=2, cp_plan="dcp"),
        quant=cs_gqa.model.quant, mode="prefill", padded_tokens=16, policy=AttentionPolicy(),
        cp_interleave_size=128, runtime=cs_gqa.runtime, cache_ids={}, stage_of=lambda _: 0,
        machine=cs_gqa.machine,
    )
    attn_flops = sum(op.flops for op in res_win.ops if op.family == "attention")
    head_flops = (win_plan.num_heads // 2) * (2.0 * (win_plan.head_dim + win_plan.head_dim))
    assert attn_flops / head_flops == pytest.approx(2048.0, rel=1e-6)


def test_kda_gdn_pcp_contracts_and_pcp_o_proj_weight_sharding():
    # 1) Head-sharded KDA PCP vs batch-chunk-norm KDA PCP (prefill ReduceScatter vs gather+scatter).
    c_kda = compose_case("kimi-k3.1", hardware=HW_DEFAULT, dp=1, tp=2, pcp=2, dcp=2, ep=4, sp=True, num_devices=4, layers=AUDIT_LAYERS)
    _, g_kda_pf, _, _ = _body_graphs(c_kda, phase="prefill", tokens=1024, context=0, sequences=1)
    _, g_kda_dec, _, _ = _body_graphs(c_kda, phase="decode", tokens=1, context=DECODE_CONTEXT, sequences=1)
    _assert_strict_dag(g_kda_pf[0])
    assert {"kda_pcp_gather", "kda_pcp_scatter"} <= {op.recipe for op in g_kda_pf[0]}
    assert {"kda_pcp_gather", "kda_pcp_scatter", "kda_pcp_restore"} <= {op.recipe for op in g_kda_dec[0]}
    assert not any(op.recipe == "pcp_recurrent_state_step" for op in g_kda_pf[0])

    c_bcn = compose_case("glm-5.3-flash", hardware=HW_DEFAULT, dp=1, tp=1, pcp=2, dcp=1, ep=1, num_devices=2, layers=1, va_recommended=False)
    _, g_bcn_pf, _, _ = _body_graphs(c_bcn, phase="prefill", tokens=128, context=0, sequences=1)
    _, g_bcn_dec, _, _ = _body_graphs(c_bcn, phase="decode", tokens=1, context=512, sequences=1)
    assert any(op.recipe == "kda_pcp_reduce_scatter" and op.link_bytes == 524_288 for op in g_bcn_pf[0])
    assert not any(op.recipe == "kda_pcp_gather" for op in g_bcn_dec[0])

    # 2) GDN PCP: *(pcp-1) wire bytes, explicit rerun dep+value port, and rank-0 skip vs rank-2 multi-seq DAG.
    c_gdn2 = compose_case("qwen3.5-35b-a3b", hardware=HW_DEFAULT, dp=1, tp=2, pcp=2, dcp=1, ep=4, num_devices=4, layers=1, va_recommended=True)
    c_gdn4 = compose_case("qwen3.5-35b-a3b", hardware=HW_DEFAULT, dp=1, tp=2, pcp=4, dcp=1, ep=8, num_devices=8, layers=1, va_recommended=True)
    cs_gdn2, g_gdn2, _, _ = _body_graphs(c_gdn2, phase="prefill", tokens=1024, context=0, sequences=1)
    _, g_gdn4, _, _ = _body_graphs(c_gdn4, phase="prefill", tokens=1024, context=0, sequences=1)
    _assert_strict_dag(g_gdn2[0])
    _assert_strict_dag(g_gdn4[0])
    sg2 = next(op for op in g_gdn2[0] if op.recipe == "pcp_gdn_state_gather")
    sg4 = next(op for op in g_gdn4[0] if op.recipe == "pcp_gdn_state_gather")
    assert sg2.link_bytes == 1_048_576 and sg4.link_bytes == 3_145_728
    rerun_op = next(op for op in g_gdn2[0] if op.recipe == "pcp_gdn_fwd_h_rerun")
    out_ops = [op for op in g_gdn2[0] if op.op_id.endswith(("gdn_output0", "gdn_output1"))]
    assert len(out_ops) == TWO_RANKS
    for out_op in out_ops:
        assert rerun_op.op_id in out_op.deps
        producers = {p.get("producer") for p in dict(out_op.metadata).get("state_value_inputs", ())}
        assert rerun_op.op_id in producers
    # Explicit physical ranks: pcp_rank=0 skips rerun; pcp_rank=1 runs multi-seq cleanly with zero duplicate IDs.
    ops_r0 = target_operators(cs_gdn2, phase="prefill", tokens=16, context=0, sequences=2, scope="body", physical_rank=0)
    ops_r2 = target_operators(cs_gdn2, phase="prefill", tokens=16, context=0, sequences=2, scope="body", physical_rank=2)
    _assert_strict_dag(ops_r0, external_roots=BODY_INPUT_ROOTS | {"sp_shard"})
    _assert_strict_dag(ops_r2, external_roots=BODY_INPUT_ROOTS | {"sp_shard"})
    assert not any(op.recipe == "pcp_gdn_fwd_h_rerun" for op in ops_r0)
    r2_prefix = next(op for op in ops_r2 if op.recipe == "pcp_gdn_prefix_scan")
    r2_rerun = next(op for op in ops_r2 if op.recipe == "pcp_gdn_fwd_h_rerun")
    r2_scatter = next(op for op in ops_r2 if op.op_id.endswith("scatter_recurrent_state"))
    r2_outs = [op for op in ops_r2 if op.op_id.endswith(("gdn_output0", "gdn_output1"))]
    assert r2_prefix.op_id in r2_scatter.deps and r2_rerun.op_id not in r2_scatter.deps
    assert r2_prefix.op_id in {p.get("producer") for p in dict(r2_scatter.metadata).get("state_value_inputs", ())}
    assert r2_outs and all(r2_rerun.op_id in op.deps for op in r2_outs)

    # 3) enable_pcp_o_proj_weight_sharding: standard MLA, grouped o_proj, & token-sharded decode.
    c_no = compose_case("deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=1, pcp=4, dcp=4, ep=1, num_devices=4, layers=1, va_recommended=False)
    c_ws = compose_case("deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=1, pcp=4, dcp=4, ep=1, num_devices=4, layers=1, va_recommended=False, enable_pcp_o_proj_weight_sharding=True)
    cs_no, g_no_pf, _, _ = _body_graphs(c_no, phase="prefill", tokens=1024, context=0, sequences=1)
    cs_ws, g_ws_pf, _, _ = _body_graphs(c_ws, phase="prefill", tokens=1024, context=0, sequences=1)
    _, g_ws_dec, _, _ = _body_graphs(c_ws, phase="decode", tokens=1, context=DECODE_CONTEXT, sequences=1)
    assert max(Memory(cs_ws).weights) < max(Memory(cs_no).weights)
    ws_ag = next(op for op in g_ws_pf[0] if op.recipe == "pcp_o_proj_weight_allgather")
    assert ws_ag.stream == CP_WEIGHT_STREAM
    assert any(op.recipe == "pcp_o_proj_allreduce" for op in g_ws_dec[0])
    # Grouped o_proj weight sharding (o_groups > 0): prefill weight AG on CP_WEIGHT_STREAM.
    c_grp = compose_case("deepseek-v4.1-flash", hardware=HW_DEFAULT, dp=1, tp=1, pcp=2, dcp=1, ep=2, num_devices=2, layers=1, va_recommended=False, enable_pcp_o_proj_weight_sharding=True)
    _, g_grp_pf, _, _ = _body_graphs(c_grp, phase="prefill", tokens=128, context=0, sequences=1)
    grp_ag = next(op for op in g_grp_pf[0] if op.recipe == "pcp_o_proj_weight_allgather")
    assert grp_ag.stream == CP_WEIGHT_STREAM and grp_ag.link_bytes == 37_785_600
    # Token-sharded decode o_proj (shard_decode_requests): gathers full output width & projects q*pcp rows.
    assert not any(op.recipe == "pcp_o_proj_weight_allgather" for op in g_no_pf[0])
    c_tok = compose_case("deepseek-v3", hardware=HW_DEFAULT, dp=1, tp=1, pcp=2, dcp=1, ep=1, num_devices=2, layers=1, va_recommended=False, enable_pcp_o_proj_weight_sharding=True)
    c_tok["runtime"]["parallel"]["pcp_lane_policy"] = "shard_decode_requests"
    _, g_tok, _, _ = _body_graphs(c_tok, phase="decode", tokens=1, context=DECODE_CONTEXT, sequences=4)
    out_ag = next(op for op in g_tok[0] if op.recipe == "pcp_o_proj_token_allgather")
    out_gemm = next(op for op in g_tok[0] if op.recipe == "output")
    out_rs = next(op for op in g_tok[0] if op.recipe == "pcp_o_proj_reduce_scatter")
    assert out_ag.link_bytes == 131_072 and out_gemm.local_shape[0] == 8 and out_rs.link_bytes == 57_344
