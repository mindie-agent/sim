"""Deterministic 500-case property suite locking the 5 Universal Spine Conservation Laws."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import random

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.catalog.context_parallel import legal_vllm_ascend_dcp_values, validate_vllm_ascend_dcp_codomain
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case, resolve_case_contracts
from mindie_sim.export.agent_brief import build_agent_brief
from mindie_sim.export.bundle import build_export_bundle
from mindie_sim.export.ooxml import Workbook
from mindie_sim.export.trace_exporter import iter_msprof_trace_events
from mindie_sim.export.xlsx_memory import populate_memory_sheet
from mindie_sim.export.xlsx_pending import populate_pending_kernel_details_sheet
from mindie_sim.export.xlsx_summary import populate_summary_sheet
from mindie_sim.optimization.ladder import solve_target_step_receipt
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.efficiency.sla import (
    _cohort_die_ranks, _cohort_hw_signature, _distinct_eval_cohorts,
)
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import paged_mla, sparse_indexer
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.matrix.refill_steady import compact_periodic_pipeline
from mindie_sim.prediction.operators.mx_quant_matrix.mxfp4_geometry import (
    Config as Mxfp4Config, activation_l2_fraction, weight_l2_fraction,
)
from mindie_sim.prediction.operators.sparse_indexer.index_pool import selection as pool_selection
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, RecipeContext, Stage
from mindie_sim.prediction.program.coalesce import coalesce_execution_program
from mindie_sim.prediction.results import OpOccurrence, StepRequest
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.runtime.exposed_observer import (
    ObservedInterval, ServiceGrantObserver, attribute_stream0_intervals,
)
from mindie_sim.prediction.runtime.l2_residency import (
    AdmissionMemoryView, LineSegmentLruCache, RawDieL2Bytes, UsableL2Bytes,
    resolve_engine_admission_view, resolve_usable_l2_bytes,
)
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, AddressRunPlan, MemoryAccessPattern, ReuseKey, VisibilityContract,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from mindie_sim.schema.model import LayerPlan, ModelSpec, RequestSpec, WorkloadSpec
from mindie_sim.schema.parallel import GlobalRank, ParallelSpec, ReplicaLocalRank
from mindie_sim.schema.workload import RequestView
from mindie_sim.stacks.vllm_ascend.launch_codec import VllmAscendLaunchCodec
from mindie_sim.sweep.pd import _build_pd_stage_candidates, _solve_pareto_pd_ratios_closed_form

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
SPINE_STRICT_TYPED_MODULES = (
    "mindie_sim/schema/parallel.py", "mindie_sim/schema/hardware.py", "mindie_sim/schema/datapath.py",
    "mindie_sim/schema/workload.py", "mindie_sim/schema/scenario.py", "mindie_sim/schema/stack.py",
    "mindie_sim/schema/bench_matrix.py", "mindie_sim/prediction/runtime/memory_physics.py",
    "mindie_sim/prediction/runtime/l2_residency.py", "mindie_sim/prediction/runtime/exposed_observer.py",
    "mindie_sim/prediction/operators/contracts.py",
)

@lru_cache(maxsize=8)
def _base_sku_hardware(tier: str) -> Hardware:
    return Hardware.from_mapping({"backend": "ascend", "sku": REPRESENTATIVE_SKUS[tier], "num_devices": 1})

@lru_cache(maxsize=8)
def _base_sku_geometry(tier: str) -> Geometry:
    return Geometry.from_devices(_base_sku_hardware(tier).dies)

@lru_cache(maxsize=4)
def _base_export_bundle(tier: str):
    raw_1p = compose_case("qwen3-32b", REPRESENTATIVE_SKUS[tier], layers=1, tp=1, dp=1, ep=1, prompt_tokens=32, output_tokens=4)
    return build_export_bundle(raw_1p, request=StepRequest("decode", tokens=1, context=32, sequences=2), apply_target_envelope=False)

def _synthetic_moe_model(
    *, layers: int, num_experts: int, attention: str = "gqa", num_heads: int = 16, num_kv_heads: int = 4, head_dim: int = 128,
) -> ModelSpec:
    if attention == "mla":
        plans = tuple(
            LayerPlan(
                index=idx, attention="mla", mla_variant="latent_mla", ffn="moe", num_heads=num_heads, num_kv_heads=1,
                head_dim=192, q_lora_rank=512, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
                v_head_dim=128, num_experts=num_experts, top_k=2, moe_intermediate_size=512, native_layer_id=idx,
            ) for idx in range(layers)
        )
        return ModelSpec(
            name="synthetic-moe-mla", layers=layers, hidden_size=1024, num_heads=num_heads, head_dim=192,
            intermediate_size=2048, vocab_size=4096, attention="mla", ffn="moe", num_kv_heads=1,
            num_experts=num_experts, top_k=2, moe_intermediate_size=512, layer_plans=plans,
        )
    plans = tuple(
        LayerPlan(
            index=idx, attention="gqa", ffn="moe", num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            num_experts=num_experts, top_k=2, moe_intermediate_size=512, native_layer_id=idx,
        ) for idx in range(layers)
    )
    return ModelSpec(
        name="synthetic-moe-gqa", layers=layers, hidden_size=1024, num_heads=num_heads, head_dim=head_dim,
        intermediate_size=2048, vocab_size=4096, attention="gqa", ffn="moe", num_kv_heads=num_kv_heads,
        num_experts=num_experts, top_k=2, moe_intermediate_size=512, layer_plans=plans,
        quant=QuantSpec(name="bf16", index_dtype="bf16"),
    )


@pytest.mark.parametrize("seed", range(100))
def test_law1_cross_rank_capacity_and_coordinate_partition(seed: int) -> None:
    rng = random.Random(10_000 + seed)
    tp = rng.choice((1, 2, 4))
    pcp = rng.choice((1, 2)) if tp <= 2 else 1
    dp = rng.choice((1, 2))
    external_dp = rng.choice((1, 2))
    pp = rng.choice((1, 2))
    ep_domain = dp * tp * pcp
    ep = rng.choice(tuple(e for e in (1, 2, 4, 8) if ep_domain % e == 0))
    etp = 1 if ep > 1 else tp
    offset = rng.choice((0, 4, 8, 16))
    num_experts = rng.choice((5, 7, 12, 17))
    interleave = rng.choice((1, 16, 64))
    attn = rng.choice(("gqa", "mla"))
    model = _synthetic_moe_model(
        layers=pp * 2, num_experts=num_experts, attention=attn, num_kv_heads=1,
    )
    dcp = rng.choice(legal_vllm_ascend_dcp_values(model, tp=tp, pcp=pcp))
    parallel = ParallelSpec(
        tp=tp, dp=dp, external_dp=external_dp, pp=pp, ep=ep, pcp=pcp, cp=dcp, expert_tp=etp,
    )
    parallel.validate(model)

    n_dies = parallel.world_size
    base_hw = _base_sku_hardware("a5")
    base_die = base_hw.dies[0]
    dies = tuple(
        replace(
            base_die, rank=offset + idx,
            memory_bytes=int(base_die.memory_bytes * (1.0 + 0.1 * (idx % 3))),
            node=idx // max(1, tp * pcp),
        )
        for idx in range(n_dies)
    )
    all_ranks = tuple(d.rank for d in dies)
    links = (Link(name="mesh", capacity_bytes_per_ms=2e8, members=all_ranks, scope="cross_node"),) if n_dies >= 2 else ()
    hw = replace(base_hw, dies=dies, links=links)
    placement = hw.validate_placement(parallel)

    # 1. RankCoord bijection & exact disjoint EP group partition.
    width = tp * pcp
    for rep in range(dp * external_dp):
        for stage in range(pp):
            for local in range(width):
                coord = placement.coord_for(rep, stage, ReplicaLocalRank(local))
                assert placement.coord_for_physical_rank(coord.global_rank) == coord
                assert int(coord.global_rank) >= offset
    if ep > 1:
        ep_groups = {
            placement.ep_group_ranks(rep, 0, lane=ReplicaLocalRank(local))
            for rep in range(dp) for local in range(width)
        }
        assert all(len(group) == ep for group in ep_groups)
        flat = [rk for group in ep_groups for rk in group]
        assert len(flat) == len(set(flat)) == ep_domain

    # 2. Cross-rank weight conservation, per-rank KV bijection & physical-offset invariance.
    runtime = Software(parallel=parallel, kv_block_size=max(16, interleave), attention_policy={"cp_kv_cache_interleave_size": interleave})
    workload = WorkloadSpec(requests=(RequestSpec(request_id="r0", prompt_tokens=64, output_tokens=8, arrival_ms=0.0),))
    case = Case(model=model, machine=hw, runtime=runtime, workload=workload)
    dies_zero = tuple(replace(d, rank=idx) for idx, d in enumerate(dies))
    links_zero = (Link(name="mesh", capacity_bytes_per_ms=2e8, members=tuple(range(n_dies)), scope="cross_node"),) if n_dies >= 2 else ()
    case_zero = replace(case, machine=replace(base_hw, dies=dies_zero, links=links_zero))
    mem, mem_zero = Memory(case), Memory(case_zero)
    assert mem.rank_weights == mem_zero.rank_weights
    rcpt = mem.resident_memory_receipt(phase="decode", sequence_tokens=64, batch_per_dp=1, query_tokens=1)
    rcpt0 = mem_zero.resident_memory_receipt(phase="decode", sequence_tokens=64, batch_per_dp=1, query_tokens=1)
    assert [d.checkpoint_weight_bytes for d in rcpt.dies] == [d.checkpoint_weight_bytes for d in rcpt0.dies]
    assert [d.kv_cache_allocated_bytes for d in rcpt.dies] == [d.kv_cache_allocated_bytes for d in rcpt0.dies]
    assert sum(d.checkpoint_weight_bytes for d in rcpt.dies) == pytest.approx(
        sum(mem.rank_weights[r][s][l] for r in range(dp) for s in range(pp) for l in range(width)), rel=1e-9,
    )
    for stage in range(pp):
        plans = mem.rank_capacity_plans(0, stage, tokens=64, include_cache=True)
        assert len(plans) == width
        for local, plan in enumerate(plans):
            w_alloc = sum(a.bytes for a in plan.allocations if a.category == "weights")
            assert w_alloc == pytest.approx(mem.rank_weights[0][stage][local], rel=1e-9)

    # 3. Physical rank offset invariance & equivalence-class completeness on _distinct_eval_cohorts & _cohort_die_ranks.
    cohorts = _distinct_eval_cohorts(case)
    assert cohorts == _distinct_eval_cohorts(case_zero)
    covered = [r for group in cohorts for r in group]
    assert cohorts and len(covered) == len(set(covered)) and set(covered) <= set(range(dp * external_dp))
    rep_sigs = {_cohort_hw_signature(case, group) for group in cohorts}
    assert len(rep_sigs) == len(cohorts)
    group_size = len(cohorts[0])
    assert all(len(group) == group_size for group in cohorts)
    assert all(
        _cohort_hw_signature(case, tuple(range(base, base + group_size))) in rep_sigs
        for base in range(0, dp * external_dp, group_size)
    )
    for group in cohorts:
        ranks = _cohort_die_ranks(case, group)
        assert len(ranks) == len(group) * pp * width and all(int(rk) >= offset for rk in ranks)

    # 4. Heterogeneous shrink induced_links_for preserves surviving >= 2-member links.
    if n_dies >= 2:
        keep_n = max(2, n_dies - 1) if n_dies >= 3 else 2
        induced = placement.induced_links_for(frozenset(GlobalRank(r) for r in all_ranks[:keep_n]))
        assert len(induced) == 1 and set(induced[0].members) == set(all_ranks[:keep_n])
        assert placement.induced_links_for(frozenset((GlobalRank(all_ranks[0]),))) == ()

    # 5. Multi-hop transit-link closure in Topology.resources_for_ranks.
    chain_dies = tuple(replace(base_die, rank=offset + i, node=i) for i in range(3))
    chain_links = (
        Link(name="hop01", capacity_bytes_per_ms=1e8, members=(offset, offset + 1), scope="cross_node"),
        Link(name="hop12", capacity_bytes_per_ms=1e8, members=(offset + 1, offset + 2), scope="cross_node"),
    )
    topo = Topology(replace(base_hw, dies=chain_dies, links=chain_links), ParallelSpec(tp=3))
    res_names = {r.name for r in topo.resources_for_ranks((offset, offset + 2))}
    assert any("hop01" in n for n in res_names) and any("hop12" in n for n in res_names)


@pytest.mark.parametrize("seed", range(100))
def test_law2_launch_codec_bijection_and_sweep_completeness(seed: int) -> None:
    rng = random.Random(20_000 + seed)
    codec = VllmAscendLaunchCodec()
    tp = rng.choice((1, 2, 4, 8))
    dp = rng.choice((1, 2, 4))
    pp = rng.choice((1, 2, 4))
    pcp = rng.choice((1, 2)) if tp <= 4 else 1
    attention = rng.choice(("gqa", "mla"))
    model = _synthetic_moe_model(
        layers=max(4, pp * 2), num_experts=8, attention=attention, num_heads=16, num_kv_heads=8,
    )
    legal_dcps = legal_vllm_ascend_dcp_values(model, tp=tp, pcp=pcp)
    dcp = rng.choice(legal_dcps)
    ep_full = dp * tp * pcp
    ep = rng.choice((1, ep_full))
    max_len = rng.choice((4096, 8192, 16384, 32768, 65536))

    # 1. Representable vllm-ascend config round-trips bit-exact (+ short flags & suffixes).
    script = codec.render_launch_config({
        "model_name": model.name, "hardware_sku": REPRESENTATIVE_SKUS["a5"],
        "tp": tp, "dp": dp, "pp": pp, "ep": ep, "dcp": dcp, "pcp": pcp,
        "max_model_len": max_len, "has_mla": (attention == "mla"),
    })
    parsed = codec.parse_launch_config(script)
    ppar = parsed.parallel
    assert (ppar.tp, ppar.dp, ppar.pp, ppar.ep, ppar.cp, ppar.pcp) == (tp, dp, pp, ep, dcp, pcp)
    assert int(dict(parsed.runtime_limits)["max_model_len"]) == max_len

    suffix, mult = rng.choice((("K", 1024), ("k", 1000), ("M", 1024 * 1024)))
    base_val = rng.choice((4, 8, 16))
    short_cmd = (
        f"vllm serve synthetic -tp {tp} -dp {dp} -pp {pp} -dcp {dcp} -pcp {pcp} "
        f"--max-model-len {base_val}{suffix}"
    )
    parsed_short = codec.parse_launch_config(short_cmd)
    assert parsed_short.parallel.cp == dcp and parsed_short.parallel.pcp == pcp
    assert int(dict(parsed_short.runtime_limits)["max_model_len"]) == base_val * mult

    # 2. Partial EP (1 < ep < dp*tp*pcp) raises ModelingError instead of silent doubling.
    partial_eps = [e for e in (2, 4, 8) if 1 < e < ep_full and ep_full % e == 0]
    if partial_eps:
        with pytest.raises(ModelingError, match="vllm_ascend_partial_ep_unrepresentable_in_cli"):
            codec.render_launch_config({
                "model_name": model.name, "hardware_sku": REPRESENTATIVE_SKUS["a5"],
                "tp": tp, "dp": dp, "pp": pp, "ep": rng.choice(partial_eps), "dcp": 1, "pcp": pcp,
            })

    # 3. Single canonical DCP legality table == validator codomain + unpinned PP > 2 enumeration.
    for cand_dcp in (1, 2, 4, 8):
        if (tp * pcp) % cand_dcp == 0:
            ok, _ = validate_vllm_ascend_dcp_codomain(model, ParallelSpec(tp=tp, pcp=pcp, cp=cand_dcp))
            assert ok == (cand_dcp in legal_dcps)
    cands = _build_pd_stage_candidates(
        role="decode", max_dies=8, is_moe=True, layers=8, prompt_tokens=4096,
        rec_topo=None, pinned={}, model_spec=model,
    )
    assert any(c["pp"] >= 4 for c in cands) and any(c["tp"] == 8 and c["pp"] == 1 for c in cands)
    assert any(c["dp"] >= 2 and c["tp"] >= 2 for c in cands)
    if attention == "mla":
        assert any(c["dcp"] > 1 for c in cands)
    assert all(c["dcp"] in legal_vllm_ascend_dcp_values(model, tp=c["tp"], pcp=c["pcp"]) for c in cands)

    # 4. Closed-form PD Pareto (output_tokens == 1 & > 1) matches exhaustive enumeration.
    p_dies, d_dies = rng.choice((1, 2, 4)), rng.choice((1, 2, 4))
    total = p_dies * rng.randint(2, 4) + d_dies * rng.randint(1, 3)
    output_toks = rng.choice((1, 1, 16, 64))
    p_rps, d_rps = rng.uniform(2.0, 20.0), rng.uniform(3.0, 30.0)
    st = {
        "infeasible": None, "rps_p": p_rps, "p_kv_rps_per_inst": p_rps * 4.0,
        "rps_d": d_rps, "raw_rps_d": d_rps, "d_kv_rps_per_inst": d_rps * 4.0,
        "decode_budget": 1.0, "decode_tps_per_die": d_rps * output_toks / d_dies,
        "decode_step": 12.0, "effective_tpot": 12.0, "first_itl": 12.0,
    }
    closed = _solve_pareto_pd_ratios_closed_form(
        st, total=total, p_dies=p_dies, d_dies=d_dies, output=output_toks, mode="direct_blocking_d2d",
    )
    best_by_used: dict[int, float] = {}
    for n_p in range(1, total // p_dies + 1):
        rem = total - n_p * p_dies
        for n_d in range(1, rem // d_dies + 1):
            used = n_p * p_dies + n_d * d_dies
            rps = n_p * p_rps if output_toks == 1 else min(n_p * p_rps, n_d * d_rps)
            best_by_used[used] = max(best_by_used.get(used, -1.0), rps * output_toks)
    expected_front: list[tuple[int, float]] = []
    best_so_far = -1.0
    for used in sorted(best_by_used):
        if best_by_used[used] > best_so_far + 1e-12:
            expected_front.append((used, best_by_used[used]))
            best_so_far = best_by_used[used]
    got_front = sorted((pt["used_dies"], pt["sustainable_tps"]) for pt in closed)
    assert len(got_front) == len(expected_front)
    for (gu, gt), (eu, et) in zip(got_front, expected_front, strict=True):
        assert gu == eu and gt == pytest.approx(et, rel=1e-9)


@pytest.mark.parametrize("seed", range(100))
def test_law3_usable_l2_upper_bound_and_write_allocate(seed: int) -> None:
    rng = random.Random(30_000 + seed)
    line_bytes = 128
    cap_bytes = rng.randint(4, 32) * line_bytes
    full_reserve = (seed % 2 == 0)
    reserved = cap_bytes if full_reserve else rng.randrange(0, cap_bytes + 1, line_bytes)

    # 1. resolve_usable_l2_bytes never falls back to raw when effective L2 == 0.
    view = AdmissionMemoryView({0: cap_bytes})
    if reserved:
        view.grant_reservation(0, "peer", reserved)
    assert int(resolve_usable_l2_bytes(view, 0, fallback_raw_l2=RawDieL2Bytes(cap_bytes))) == max(0, cap_bytes - reserved)

    eng = Engine([Resource("l2@r0:s0", 1e9, units=1)])
    eng.attach_admission_memory_view(view)
    assert resolve_engine_admission_view(eng, die=0, raw_l2=RawDieL2Bytes(cap_bytes)) is view

    # 2. When UsableL2Bytes(0), governed L2 consumers credit 0 L2 hits & bill full cold HBM.
    if full_reserve:
        g_a2 = _base_sku_geometry("a2")
        zero_a2 = AdmissionMemoryView({0: g_a2.buffers["l2"]}, arch_id=str(g_a2.datapath.compute_arch.arch_id))
        zero_a2.grant_reservation(0, "full", g_a2.buffers["l2"])
        idx_op = sparse_indexer.make_indexer_op(
            abi="aclnnLightningIndexer", query_tokens=1, scanned_rows=128,
            index_heads=64, index_dim=128, topk_items=2048, native_split_core=True,
        )
        ctx_a2 = OperatorPlanningContext(
            OpOccurrence(0, 0, 0, idx_op.op_id), g_a2, g_a2.datapath, admission_memory_view=zero_a2,
        )
        plan_v1 = sparse_indexer.build(idx_op, ctx_a2)
        assert int(plan_v1.work.resources["hbm"]) == (64 * 128 * 2 + 128 * 128 * 2 + 2048 * 4) + 2 * (128 * 64 * 4) + 128

        g_a3 = _base_sku_geometry("a3")
        zero_a3 = AdmissionMemoryView({0: g_a3.buffers["l2"]})
        zero_a3.grant_reservation(0, "full", g_a3.buffers["l2"])
        mla_op = paged_mla.make_op(
            sequences=((1, 2048),), logical_heads=64, physical_heads=64,
            abi=paged_mla.arch22.UPSTREAM_ABI, s2_base_size=1024,
        )
        g_res = replace(g_a3, admission_memory_view=zero_a3, requests=(RequestView("r", 1, 2047, False, "decode"),))
        s_read = next(ph for ph in paged_mla.build(mla_op, g_res).pipeline_spec.phases if ph.phase_id == "score_read")
        w_map = {item.resource: item.amount for item in s_read.work}
        assert w_map.get("hbm", 0.0) == pytest.approx(w_map["mte2_aiv"])

        cfg_zero = Mxfp4Config(k=4096, n=512, rows=(128,))
        assert activation_l2_fraction(cfg_zero, 32, UsableL2Bytes(0)) == 0.0
        assert weight_l2_fraction(cfg_zero, 32, UsableL2Bytes(0)) == 0.0

        pool_model = _synthetic_moe_model(layers=1, num_experts=4)
        pool_plan = replace(pool_model.layer_plans[0], index_kpool=4, index_heads=4, index_dim=64, index_topk=32)
        pv = (RequestView("r", 8, 256, False, "prefill"),)
        b = Builder("s0.l0", pool_plan, 0, pv, pool_model, "prefill")
        pool_selection(b, query="q", weights="w", cache="k", views=pv, plan=pool_plan,
                       quant=pool_model.quant, padded_tokens=8, l2_bytes=UsableL2Bytes(0))
        pop = next(o for o in b.ops if o.recipe == "index_pool_scores")
        assert int(dict(pop.metadata)["score_l2_hit_bytes"]) == 0 and int(pop.hbm_bytes) == int(pop.mte2_bytes)

        g_a5 = _base_sku_geometry("a5")
        zero_a5 = AdmissionMemoryView({0: g_a5.buffers["l2"]}, arch_id=str(g_a5.datapath.compute_arch.arch_id))
        zero_a5.grant_reservation(0, "full", g_a5.buffers["l2"])
        reg_op = sparse_indexer.make_indexer_op(
            abi=sparse_indexer.common.REGION_ABI, query_tokens=2, scanned_rows=64,
            index_heads=8, index_dim=256, compress_ratio=4, shared_prefix=True,
        )
        reg_plan = sparse_indexer.build(
            reg_op, OperatorPlanningContext(OpOccurrence(0, 0, 0, reg_op.op_id), g_a5, g_a5.datapath, admission_memory_view=zero_a5),
        )
        assert not bool(reg_plan.metadata.get("l2_key_reuse"))

    # 3. LineSegmentLruCache sequential misses == independent functools.lru_cache misses.
    lru_cap = rng.randint(2, 16)
    seg_cache = LineSegmentLruCache(lru_cap * line_bytes, line_bytes)

    @lru_cache(maxsize=lru_cap)
    def oracle_line(stream_id: int, line_idx: int) -> tuple[int, int]:
        return stream_id, line_idx

    seg_misses = 0
    for _ in range(18):
        sid, lo = rng.randint(0, 2), rng.randint(0, 20)
        hi = lo + rng.randint(1, 12)
        seg_misses += sum(e - s for s, e in seg_cache.access_lines(sid, lo, hi))
        for ln in range(lo, hi):
            oracle_line(sid, ln)
    assert seg_misses == oracle_line.cache_info().misses

    # 4. Write-allocate commit & preview provenance + PD fallback policy propagation.
    wa_flag = rng.choice((False, True))
    sz = rng.choice((2048, 4096))
    prov_view = AdmissionMemoryView({0: 64 * 1024}, write_allocate=wa_flag, pool_namespace="prefill")
    w_key, r_key = ReuseKey(0, "w_tensor", 0), ReuseKey(0, "r_tensor", 0)
    runs = AddressRunPlan((AddressRun(0, 1, sz, sz),))
    w_pat = MemoryAccessPattern("w_tensor", 0, 0, 0, "write", "mte3", sz, runs, w_key,
                                visibility=VisibilityContract(private_workspace=False, live_out=True))
    r_from_w = MemoryAccessPattern("w_tensor", 0, 0, 0, "read", "mte2_aic", sz, runs, w_key)
    r_pat = MemoryAccessPattern("r_tensor", 0, 0, 0, "read", "mte2_aic", sz, runs, r_key)
    assert prov_view.admit_access(w_pat, sz) == (0, sz)
    assert prov_view.admit_access(r_pat, sz) == (0, sz)
    assert prov_view.preview_access(r_from_w, sz) == ((sz, 0) if wa_flag else (0, sz))
    assert prov_view.admit_access(r_from_w, sz) == ((sz, 0) if wa_flag else (0, sz))
    assert prov_view.admit_access(r_from_w, sz) == (sz, 0)
    assert prov_view.preview_access(r_pat, sz, write_allocate=False) == (sz, 0)
    eng_wa = Engine([Resource("prefill:l2@r0:s0", 1e9, units=1)])
    eng_wa.attach_admission_memory_view(prov_view)
    fb_view = resolve_engine_admission_view(eng_wa, die=0, raw_l2=RawDieL2Bytes(64 * 1024), pool_namespace="decode")
    assert fb_view.write_allocate_for(0) is wa_flag
    if not wa_flag:
        assert activation_l2_fraction(Mxfp4Config(k=4096, n=64, rows=(128,)), 32, UsableL2Bytes(8 << 20)) == 0.0


@pytest.mark.parametrize("seed", range(100))
def test_law4_periodic_literal_engine_and_grant_conservation(seed: int) -> None:
    rng = random.Random(40_000 + seed)
    units = rng.choice((2, 4, 8))
    tiles = rng.randint(14, 24)
    slots = rng.choice((2, 3))
    load_w = rng.uniform(10.0, 40.0)
    comp_w = rng.uniform(15.0, 50.0)
    prologue_mult = rng.uniform(1.5, 3.0)

    def tile_fn(idx: int) -> tuple[Stage, ...]:
        scale = prologue_mult if idx == 0 else 1.0
        return (
            Stage("load", {"mte2": load_w * scale}),
            Stage("comp", {"cube": comp_w * scale}, after=("load",), previous=("comp",)),
        )

    # 1. Compact periodic pipeline == literal uncompacted Engine replay.
    dummy_ctx = RecipeContext(
        op=None, work={}, resource_names={"mte2": "mte2", "cube": "cube"},
        rates={"mte2": 100.0, "cube": 100.0}, core_count=units, resource_units={"mte2": units, "cube": units},
    )
    compact_prog = compact_periodic_pipeline(dummy_ctx, tiles, slots, tile_fn, head=1, tail=1, threshold=8)
    lit_eng = Engine([Resource("mte2", 100.0, units=units), Resource("cube", 100.0, units=units)])
    lit_eng.advance(until=(PipelineProgram(tiles, slots, tile_fn).start(lit_eng, "p", lit_eng.submit("in")),))
    per_eng = Engine([Resource("mte2", 100.0, units=units), Resource("cube", 100.0, units=units)])
    per_eng.advance(until=(compact_prog.start(per_eng, "p", per_eng.submit("in")),))
    assert per_eng.now == pytest.approx(lit_eng.now, rel=1e-9) and per_eng.serviced == pytest.approx(lit_eng.serviced, rel=1e-9)

    # 2. ServiceGrantObserver conserves per-(resource, owner) delivered work across periodic skipping.
    long_tiles = tiles + 20
    long_compact = compact_periodic_pipeline(dummy_ctx, long_tiles, slots, tile_fn, head=1, tail=1, threshold=8)
    grant_maps = []
    for prog in (PipelineProgram(long_tiles, slots, tile_fn), long_compact):
        obs_eng = Engine([Resource("mte2", 100.0, units=units), Resource("cube", 100.0, units=units)])
        obs = ServiceGrantObserver()
        obs.current_node_idx = 0
        obs_eng.attach_wave_recorder(obs)
        obs_eng.advance(until=(prog.start(obs_eng, "obs", obs_eng.submit("in")),))
        by_ro: dict[tuple[str, int], float] = {}
        for gr in obs.to_observed_grants(("op0",)):
            by_ro[(gr.resource, gr.owner)] = by_ro.get((gr.resource, gr.owner), 0.0) + gr.delivered
        assert sum(by_ro.values()) == pytest.approx(sum(obs_eng.serviced.values()), rel=1e-9)
        grant_maps.append(by_ro)
    assert grant_maps[0].keys() == grant_maps[1].keys()
    assert all(grant_maps[0][k] == pytest.approx(grant_maps[1][k], rel=1e-9) for k in grant_maps[0])

    # 3. Multi-unit placement=None matches explicit tuple(range(units)) identically (grants + exposed_ns_by_op).
    w_multi = rng.uniform(20.0, 80.0)
    per_owner_totals = []
    for pl in (None, {"cube": tuple(range(units))}):
        e_pl = Engine([Resource("cube", 100.0, units=units)])
        o_pl = ServiceGrantObserver()
        o_pl.current_node_idx = 0
        e_pl.attach_wave_recorder(o_pl)
        e_pl.advance(until=(e_pl.submit("u", {"cube": w_multi}, placement=pl),))
        g_pl = o_pl.to_observed_grants(("u",))
        end_ns = max(1, round(e_pl.now * 1_000_000.0))
        exp = attribute_stream0_intervals((ObservedInterval("u", "main", 0, 0, end_ns),), start_ns=0, terminal_ns=end_ns, grants=g_pl)
        by_owner: dict[int, float] = {}
        for gr in g_pl:
            by_owner[gr.owner] = by_owner.get(gr.owner, 0.0) + gr.delivered
        per_owner_totals.append((e_pl.now, by_owner, exp.exposed_ns_by_op))
    assert per_owner_totals[0][0] == pytest.approx(per_owner_totals[1][0], rel=1e-11)
    assert per_owner_totals[0][1] == pytest.approx(per_owner_totals[1][1], rel=1e-9)
    assert per_owner_totals[0][2] == per_owner_totals[1][2]

    # 4. coalesce_execution_program preserves timing under concurrent FIFO peer DMA.
    peer_w = rng.uniform(5.0, 25.0)
    kprog = KernelProgram((Stage("s0", {"dma": load_w}), Stage("s1", {"dma": comp_w}, after=("s0",))))
    coal_prog = coalesce_execution_program(kprog, fair_resources=frozenset())
    times = []
    for prog in (kprog, coal_prog):
        e_fifo = Engine([Resource("dma", 100.0, policy="fifo")])
        d1 = prog.start(e_fifo, "k", e_fifo.submit("in"))
        d2 = e_fifo.submit("peer", {"dma": peer_w}, release=0.05)
        e_fifo.advance(until=(d1, d2), first=False)
        times.append((d1.time, d2.time))
    assert times[0][0] == pytest.approx(times[1][0], rel=1e-9)
    assert times[0][1] == pytest.approx(times[1][1], rel=1e-9)

    # 5. Unrelated-resource priority change causes exact 0.0 timing delta on primary resource.
    prio_times = []
    for aux_prio in (0, 3, 10):
        e_pr = Engine([Resource("cube", 100.0, units=2), Resource("aux", 50.0, units=1)])
        ev_c = e_pr.submit("primary", {"cube": comp_w}, priority=0)
        ev_a = e_pr.submit("disjoint", {"aux": load_w}, priority=aux_prio)
        e_pr.advance(until=(ev_c, ev_a), first=False)
        prio_times.append(ev_c.time)
    assert prio_times[0] == prio_times[1] == prio_times[2]


@pytest.mark.parametrize("seed", range(100))
def test_law5_four_stage_envelope_budget_and_receipt_conservation(seed: int) -> None:
    rng = random.Random(50_000 + seed)
    tier = rng.choice(("a3", "a5"))
    sku = REPRESENTATIVE_SKUS[tier]
    kv_dt = rng.choice(("bf16", "int8")) if tier == "a3" else rng.choice(("bf16", "fp8_e4m3"))
    chunked = rng.choice((False, True))
    async_sched = rng.choice((False, True))
    draft_eager = rng.choice((False, True))
    prefix_mode = rng.choice(("private", "shared"))
    acc_len = rng.choice((1.0, 1.5, 2.2))

    # 1. Explicit compose_case knobs survive resolve_case_contracts identically.
    raw = compose_case(
        "qwen3-32b", sku, layers=1, tp=1, dp=2, ep=1,
        prompt_tokens=64, output_tokens=8, kv_cache_dtype=kv_dt,
        enable_chunked_prefill=chunked, async_scheduling=async_sched,
        draft_enforce_eager=draft_eager, prefix_cache_mode=prefix_mode,
        spec_method="ngram", mtp_proposal_tokens=2, mtp_acceptance_length=acc_len,
    )
    sc, rq, _ = resolve_case_contracts(raw)
    assert sc is not None and rq is not None
    req_mechs = dict(sc.stack_options.requested_mechanisms)
    eff_mechs = dict(sc.stack_options.effective_mechanisms)
    assert eff_mechs["kv_cache_dtype"] == kv_dt
    assert rq.kv_cache_abi and rq.kv_cache_abi[0].planes[0].dtype == kv_dt
    assert eff_mechs["chunked_prefill"] is chunked
    assert eff_mechs["async_scheduling"] is async_sched
    assert req_mechs["draft_enforce_eager"] is draft_eager
    assert raw["runtime"]["prefix_cache_mode"] == prefix_mode
    assert float(raw["runtime"]["mtp_acceptance_length"]) == pytest.approx(acc_len)

    # 2. Asymmetric DP stage receipt matches Memory.resident_memory_receipt die-by-die.
    case = parse_case(raw)
    b0, b1 = rng.randint(1, 3), rng.randint(4, 6)
    req = StepRequest("decode", tokens=1, context=63, sequences=(b0, b1))
    receipt = solve_target_step_receipt(
        case, req, stack_track="shipping", comparison_policy="simulation_only",
    )
    expected_mem = Memory(case).resident_memory_receipt(
        phase="decode", sequence_tokens=64, batch_per_dp=(b0, b1), query_tokens=1,
    )
    st0 = receipt.stages[0]
    assert st0.span_ns > 0 and sum(ns for _, ns in st0.exposed_ns_by_occ) + st0.idle_or_wait_ns == st0.span_ns
    got_kv = [d.kv_cache_allocated_bytes for d in st0.memory_receipt.dies]
    exp_kv = [d.kv_cache_allocated_bytes for d in expected_mem.dies]
    assert got_kv == exp_kv and got_kv[0] < got_kv[1]

    # 3. Export bundle -> trace active HBM peak <= memory_ledger.max_stage_hbm_bytes.
    bundle = _base_export_bundle(tier)
    active_hbm_peaks = [
        float(ev["args"]["Active_Total_GiB"]) * (1024 ** 3)
        for ev in iter_msprof_trace_events(bundle) if ev.get("ph") == "C" and "Active_Total_GiB" in (ev.get("args") or {})
    ]
    assert active_hbm_peaks and max(active_hbm_peaks) <= bundle.memory_ledger.max_stage_hbm_bytes * (1.0 + 1e-4)

    # 4. agent_brief agrees on unclamped fractional TPOT (0 <= acc < 1) & 0 #DIV/0!/bad formulas in XLSX sheets.
    frac_acc = rng.choice((0.0, 0.4, 0.75, 1.0))
    frac_bundle = replace(bundle, avg_accepted_tokens=frac_acc)
    brief = build_agent_brief(frac_bundle)
    after_kpi = brief["kpis_before_vs_after"]["after"]
    step_ms = frac_bundle.step_span_us / 1000.0
    expected_tpot = (step_ms / frac_acc if frac_acc > 0.0 else step_ms) + float(getattr(frac_bundle, "handoff_tpot_ms", 0.0) or 0.0)
    assert after_kpi["effective_tpot_ms"] == pytest.approx(expected_tpot, rel=1e-6)
    assert after_kpi["step_ms"] == pytest.approx(step_ms, rel=1e-6) and brief["kpis_before_vs_after"]["reconciliation_error_ns"] == 0

    wb = Workbook()
    ws_sum, ws_mem, ws_pend = wb.create_sheet("Summary"), wb.create_sheet("Memory"), wb.create_sheet("Pending")
    populate_summary_sheet(ws_sum, frac_bundle, compare_bundle=bundle)
    populate_memory_sheet(ws_mem, frac_bundle)
    populate_pending_kernel_details_sheet(ws_pend, (), frac_bundle, before_bundle=bundle)
    for ws in (ws_sum, ws_mem, ws_pend):
        for cell in ws._cells.values():
            txt = f"{getattr(cell, 'value', '')}{getattr(cell, 'formula', '')}"
            assert not any(bad in txt for bad in ("#DIV/0!", "#VALUE!", "#NAME?", "/0)", "/0 "))

def test_spine_modules_have_zero_type_ignore() -> None:
    """Every strictly typed Spine contract module must contain zero '# type: ignore' comments."""
    offenders = [rel for rel in SPINE_STRICT_TYPED_MODULES if "type: ignore" in (REPO_ROOT / rel).read_text(encoding="utf-8")]
    assert offenders == [], f"forbidden 'type: ignore' in Spine modules: {offenders}"
