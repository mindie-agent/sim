"""Sequence and context parallel: SP, PCP/DCP and ultra-long-context scaling.

[Role] Prove that source-selected sequence parallelism and decode/prefill
  context parallelism lower one physical declaration: SP wire/shape conservation,
  padded row domains, AttnRes boundaries, dual-plane PCP/DCP composition, and
  the ultra-long-context/synthetic-MoE operator domains that feed them.
[Invariants] Per-rank wire bytes derive from the declared TP width; every rank
  keeps its exact valid row; negative structural paths raise their own codes.
[Timeouts & Locks] Tier-0 in-memory simulation with bounded monotonic-clock guards.
[Side-Effect Policy] Read-only example loads and in-memory execution.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import math
import time

import pytest
from _support.captures import REPO_ROOT
from _support.memory_cases import case as memory_case
from _support.moe_hc_mc2_fixtures import mxfp8_case, mxfp8_graph
from _support.operator_cases import (PAGED_MLA_BUFFERS, Engine as MlaEngine, Geometry as MlaGeometry,
                                     OpSpec as MlaOpSpec, RecipeContext as MlaRecipeContext,
                                     Resource as MlaResource, gqa_decode_hardware, gqa_decode_op,
                                     gqa_decode_run, paged_mla_attention, paged_mla_case,
                                     paged_mla_execute, paged_mla_membase_run)
from mindie_sim import graph_costs, operator_costs, predict, target_graphs, target_operators
from mindie_sim.catalog.ops import comm_op, vector_op
from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import derive_expert_pruned, load_model
from mindie_sim.prediction.operators.attention.attention_decode import StagedGQADecode
from mindie_sim.prediction.sharding.routing import cohort_padded_tokens
from mindie_sim.prediction.body_defaults import effective_body_declaration
from mindie_sim.prediction.execution.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.operators.attention.fai_decode_mapping import host_partitions, partitions
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operators.grouped_moe.grouped_dequant import Config, assignments
from mindie_sim.prediction.operators.sparse_indexer.index_pool_native import IMPLEMENTATION as KEYPOOL_IMPL
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.moe_routing.moe_init_mxfp8 import RECIPE as MXFP8_INIT_RECIPE
from mindie_sim.prediction.operators.moe_transport.moe_callers import RoutingPolicy
from mindie_sim.prediction.operators.paged_mla.paged_mla import build as build_paged_mla
from mindie_sim.prediction.collectives.peer_traffic import PeerTraffic
from mindie_sim.prediction.program import KernelProgram, PipelinePhase, PipelineProgram, PipelineTemplateProgram, RecipeContext, Stage
from mindie_sim.prediction.execution.rank_ledger import rank_ledger_for
from mindie_sim.prediction.sharding.sequence_parallel import DECLARATION, allgather, close_output, input_shard, reduce_scatter, select
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0
CONTEXTS = (409600, 1048576)
EXPERTS = ((16, 2), (32, 4), (48, 6), (64, 8), (96, 4), (128, 8))
GROUPS = (((0, 2), (1, 3)), ((3, 0), (2, 1)), ((1, 0), (3, 2)))
PRESETS = (("qwen3.6-27b", 4, {"gqa_decode", "gdn_decode"}), ("qwen3.5-35b-a3b", 4, {"gqa_decode", "gdn_decode"}), ("glm-5.3-flash", 4, {"indexed_mla"}), ("glm-5.3", 2, {"indexed_mla"}))
from _support.axes import A2_SKUS  # noqa: E402
SP_CASE_PATH = "examples/cases/kimi-k3-a5-current-checkpoint-conditional.json"
def _recipes(case, *, phase: str, tokens: int, context: int):
    graph = target_graphs(case, phase=phase, tokens=tokens, context=context, sequences=1)[0]
    return graph, {op.recipe for op in graph}
@contextmanager
def _exact_pipeline(module):
    """Temporarily unroll the compact periodic pipeline into an exact PipelineProgram."""
    original = module.compact_periodic_pipeline
    module.compact_periodic_pipeline = lambda ctx, count, slots, fn, **kw: PipelineProgram(count, slots, fn)
    try: yield
    finally: module.compact_periodic_pipeline = original
def _source_case(*, tp=2, ep=None, dp=1):
    ep = tp if ep is None else ep
    raw = memory_case(dp=dp, tp=tp, ep=ep, prompt=8); raw["model"].update(ffn="moe", num_experts=8, top_k=2, moe_intermediate_size=32, num_heads=tp, num_kv_heads=max(1, tp // 2), head_dim=8)
    parsed = parse_case(raw); model = replace(parsed.model, layer_plans=tuple(replace(layer, residual_block_size=2, hyper_connections=False, hc_mult=0) for layer in parsed.model.layer_plans))
    return replace(parsed, model=model, runtime=replace(parsed.runtime, kernels={"sequence_parallel": dict(DECLARATION)}))
def _views(*lengths, phase="prefill"):
    return tuple(RequestView(f"r{index}", length, 0, False, phase) for index, length in enumerate(lengths))
def _plan_for(*, tp=2, lengths=(8,), ep=None, dp=1, global_tokens=None, phase="prefill"):
    parsed = _source_case(tp=tp, ep=ep, dp=dp); batch = _views(*lengths, phase=phase)
    return parsed, batch, select(parsed.runtime, parsed.model, batch, tokens=sum(lengths), global_tokens=global_tokens)
def _one(ops, predicate):
    found = [op for op in ops if predicate(op)]; assert len(found) == 1, [op.op_id for op in found]; return found[0]
def _combined_case(*, layers=None, bounded=False):
    raw = json.loads((REPO_ROOT / "examples/cases/kimi-k3-a5-sp-attnres-conditional.json").read_text())
    if bounded:
        kernels = raw["runtime"]["kernels"]; kernels.setdefault("programs", {})["matmul"] = {"kind": "steady_matmul"}
        kernels.setdefault("families", {}).update({"grouped_gemm": {"kind": "steady_matmul"}, "alltoall": {"kind": "pairwise_collective"}, "allgather": {"kind": "ring_collective"}, "reduce_scatter": {"kind": "ring_collective"}})
    case = parse_case(raw)
    if layers is not None: case = replace(case, model=replace(case.model, layers=layers, body_layers=layers, layer_plans=case.model.layer_plans[:layers]), runtime=replace(case.runtime, pp_layer_partition=(layers,)))
    return case
def _unequal_graphs(case, rows):
    graphs = {}
    for replica, count in enumerate(rows):
        views = (RequestView(f"r{replica}", count, 0, False, "prefill"),) if count else ()
        ops, _ = Frontend().body(case.model, case.runtime, views, ready="analysis_input", tokens=count, mode="prefill", global_tokens=sum(rows), machine=case.machine)
        graphs[replica] = lower_implementations(ops, case.runtime, case.model.activation_dtype, local_tokens=count)
    return bind_cohort_traffic(case, graphs)
def _check_structure(case, graphs, rows, *, phase="prefill"):
    """Recompute local, global and unique expert rows from the declaration."""
    parallel, model, width = case.runtime.parallel, case.model, case.model.hidden_size
    mean_rows = sum(rows) * model.layer_plans[1].top_k / parallel.ep
    expected, _ = RoutingPolicy(**dict(case.runtime.routing)).local(model, model.layer_plans[1], parallel, sum(rows), runtime=case.runtime)
    assert expected >= mean_rows; result = []
    for replica, ops in graphs.items():
        local = rows[replica] // parallel.tp; points = [op for op in ops if op.recipe == "attn_res_fwd"]; by_id = {op.op_id: op for op in ops}
        if not local: assert not points and not any(dict(op.metadata).get("group_axis") == "tp" for op in ops); assert not any(op.recipe in {"latent_projection", "router"} for op in ops) and not any(op.op_id in {"sp_shard", "sp_output_gather"} for op in ops)
        else:
            assert len(points) == 2 * model.layers + 1 and all(op.local_shape == (local, width) for op in points)
            assert all(dict(op.metadata)["attn_res_view"]["tokens_local"] == local and dict(op.metadata)["attn_res_view"]["optimize_prefill"] == (phase == "prefill") for op in points)
            assert by_id["sp_shard"].deps == ("analysis_input",) and points[0].deps == ("sp_shard",) and by_id["sp_output_gather"].deps == ("final.attn_res_fwd",) and by_id["sp_output_gather"].local_shape == (rows[replica], width)
            # Every residual point supplies the local input to its global attention gather; the attention projection retains full rows.
            for layer in model.layer_plans:
                prefix = f"s0.l{layer.index}"; gather = by_id[prefix + ".attention.sp_gather"]
                assert gather.deps == (prefix + ".attention.attn_res_fwd",) and gather.local_shape == (rows[replica], width)
                projections = [op for op in ops if op.layer == layer.index and dict(op.metadata).get("structure_block") == "attention" and (op.recipe in {"linear_q", "q_a"} or str(op.recipe).startswith("linear_qkvbfg_a"))]
                assert projections and all(op.local_shape[0] == rows[replica] for op in projections)
                scatter = _one(ops, lambda op: op.layer == layer.index and dict(op.metadata).get("structure_block") == "attention" and op.family == "reduce_scatter"); assert scatter.op_id in by_id[prefix + ".ffn.attn_res_fwd"].deps
            latent = [op for op in ops if op.recipe == "latent_projection"]
            assert latent and all(op.local_shape[0] == local for op in latent) and all(dict(op.metadata)["matrix_wrapper"]["group_list_values"] == (local,) for op in latent)
        gates = [op for op in ops if op.recipe == "grouped_gate_up"]
        assert len(gates) == sum(layer.ffn == "moe" for layer in model.layer_plans)
        for gate in gates:
            _, n, k = gate.local_shape; assert gate.flops / (2 * n * k) == pytest.approx(expected)
        dispatch = _one(ops, lambda op: op.layer == 1 and op.recipe == "expert_dispatch"); combine = _one(ops, lambda op: op.layer == 1 and op.recipe == "expert_combine")
        assert dict(dispatch.metadata)["source_assignment_partitions"] == 1
        forward = PeerTraffic.from_mapping(dict(dispatch.metadata)["peer_traffic"], parallel.ep); backward = PeerTraffic.from_mapping(dict(combine.metadata)["peer_traffic"], parallel.ep)
        assert all(forward.payload(source, destination) == backward.payload(destination, source) for source in range(parallel.ep) for destination in range(parallel.ep))
        if not local: assert dispatch.link_bytes == 0 < combine.link_bytes and expected > 0
        result.append({"replica": replica, "global_rows": rows[replica], "local_rows": local, "attnres_points": len(points), "expert_rows_per_layer": expected, "mean_expert_rows_per_layer": mean_rows, "operators": len(ops), "dispatch_bytes": dispatch.link_bytes, "combine_bytes": combine.link_bytes})
    return result
def _execute(case, graphs):
    report = graph_costs(case, graphs, external_inputs={"analysis_input": 13}, input_ready_ms=7, isolated=False, trace_limit=512)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0 and all(abs(amount) < 1e-6 for amount in report["execution"]["live_allocated"].values())
    assert all(row["ready_ms"] >= 13 for row in report["timeline"]) and all(event["start_ms"] >= 13 for event in report["events"] if event["demand"])
    timeline = {row["id"]: row for row in report["timeline"]}
    for replica, ops in graphs.items():
        for op in ops: assert all(timeline[f"r{replica}:{op.op_id}"]["ready_ms"] >= timeline[f"r{replica}:{dep}"]["complete_ms"] - 1e-10 for dep in op.deps if dep != "analysis_input")
    topology, wire = Topology(case.machine, case.runtime.parallel), {}
    for replica, ops in graphs.items():
        for op in ops:
            for key, amount in (topology.accounting(op, replica).wire_by_resource.items() if op.link_bytes else ()): wire[key] = wire.get(key, 0) + amount
    actual = report["execution"]["serviced"]
    for key, amount in wire.items(): assert math.isclose(actual[key], amount, rel_tol=1e-10, abs_tol=1e-5)
    return {"duration_ms": report["duration_ms"], "operators": len(report["timeline"]), "completed_tasks": report["execution"]["completed_tasks"], "live_tasks": report["execution"]["live_tasks"], "live_programs": report["execution"]["live_programs"], "remaining_buffers": report["execution"]["live_allocated"]}
def _contended(prog, plan, comp_aiv, units, rates):
    engine = Engine([Resource(r, v, units=units[r]) for r, v in rates.items()]); ready = engine.submit("ready")
    if comp_aiv is not None: engine.submit("competitor", {"vector": 3 * plan.work.resources["vector"]}, after=(ready,), placement={"vector": (comp_aiv,)})
    done = prog.start(engine, "indexed", ready); engine.advance(until=(done,)); return done.time
def _sp_case(batch):
    return json.loads((REPO_ROOT / "examples/cases/high-concurrency-20261030/kimi-source-paged-host" / f"a5_kimi_tp8_e16_bf16_c{batch}_p4k_source.json").read_text())
def _source_frontend_case():
    raw = memory_case(dp=2, tp=2, ep=4)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32, num_shared_experts=1, layer_plans=[
        {"index": 0, "attention": "gqa", "ffn": "dense", "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "intermediate_size": 64, "residual_block_size": 2},
        {"index": 1, "attention": "gqa", "ffn": "moe", "num_experts": 4, "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "top_k": 2, "moe_intermediate_size": 32, "num_shared_experts": 1, "residual_block_size": 2}])
    raw["runtime"].update(graph_mode="none", routing={"policy": "balanced"}, kernels={"sequence_parallel": dict(DECLARATION)})
    return parse_case(raw)
def _padding_case(*, graph_bucket=None):
    raw = json.loads((REPO_ROOT / SP_CASE_PATH).read_text()); runtime = raw["runtime"]
    runtime["routing"] = {"policy": "balanced"}; runtime.setdefault("kernels", {})["sequence_parallel"] = dict(DECLARATION, padding="right_to_tp"); runtime["kernels"]["programs"] = {"matmul": {"kind": "steady_matmul"}}; runtime["graph_mode"] = "eager" if graph_bucket is None else "auto"
    if graph_bucket is not None: runtime["graph_buckets"] = [graph_bucket]; runtime["graph_capture_keys"] = [graph_bucket]
    return parse_case(raw)
def _padding_views(rows, request="r"):
    return (RequestView(request, rows, 4096, False, "decode"),) if rows else ()
@pytest.fixture(scope="module")
def full_body_graphs():
    parsed = _padding_case(); return parsed, {q: target_graphs(parsed, phase="decode", tokens=q, context=4096, scope="body", resource_fusion=False) for q in (1, 4)}
def test_hybrid_pcp_prefill_with_ple_host_offload_and_dcp_decode():
    for pcp in (2, 4):
        with pytest.raises(ModelingError, match="shared_slab_requires_dcp_1_pcp_1"):
            compose_case("qwen3.8-flash-next", hardware="a5", tp=4, ep=4, pcp=pcp, layers=4, engram_cpu_offload=True)
    offload = compose_case("qwen3.8-flash-next", hardware="a5", tp=4, ep=4, pcp=1, layers=4, engram_cpu_offload=True)
    assert offload["runtime"]["engram_cpu_offload"] is True
    rates = offload["machine"]["device"]["rates"]; assert rates["host_memory"] > 0.0 and rates["host_vector"] > 0.0 and rates["h2d"] > 0.0
    prefill = compose_case("kimi-k3.1", hardware="a5", tp=4, ep=4, pcp=4, layers=4); assert int(prefill["runtime"]["parallel"]["pcp"]) == 4; _graph, recipes = _recipes(prefill, phase="prefill", tokens=16, context=128); assert "pcp_index_gather" in recipes and {"kda_pcp_gather", "kda_pcp_restore"} <= recipes
    decode = compose_case("kimi-k3.1", hardware="a5", tp=4, ep=4, dcp=4, layers=4); assert int(decode["runtime"]["parallel"]["cp"]) == 4; _graph, recipes = _recipes(decode, phase="decode", tokens=1, context=1024); assert {"dcp_lse_merge", "catalog_dcp_lse_merge"} & recipes
@pytest.mark.parametrize("model", ["deepseek-v3.2", "glm-5.3-flash"])
def test_pooled_indexer_and_hybrid_kda_decode_context_parallelism(model):
    if model == "glm-5.3-flash":
        with pytest.raises(ModelingError, match="kpool_indexer_requires_dcp_1"):
            compose_case(model, hardware="a5", tp=4, ep=4, dcp=2, layers=4)
        case = compose_case(model, hardware="a5", tp=4, ep=4, dcp=1, layers=4)
        assert int(case["runtime"]["parallel"]["cp"]) == 1
        _graph, recipes = _recipes(case, phase="decode", tokens=1, context=1024); assert recipes
        return
    case = compose_case(model, hardware="a5", tp=4, ep=4, dcp=2, layers=4); assert int(case["runtime"]["parallel"]["cp"]) == 2
    _graph, recipes = _recipes(case, phase="decode", tokens=1, context=1024); assert {"dcp_lse_merge", "catalog_dcp_lse_merge"} & recipes
    mtp_case = compose_case(model, hardware="a5", tp=4, ep=4, layers=4, mtp_proposal_tokens=3); auxiliary = mtp_case["model"]["auxiliary"]; assert auxiliary is not None; mtp = auxiliary["mtp"]
    assert mtp["kind"] == "mtp" and int(mtp["layers"]) == 1 and mtp["attention_kind"] == "mla"
    assert mtp_case["runtime"]["metadata"]["draft_mode"] == "mtp" and int(mtp_case["runtime"]["metadata"]["draft_proposal_tokens"]) == 3
def test_periodic_pipeline_compaction_matches_exact_pipeline_program():
    import mindie_sim.prediction.operators.attention.attention_pipeline as ad
    op = gqa_decode_op([(1, 131072)] * 8, heads=16, kv_heads=2, dim=128); recipe = StagedGQADecode(key_tile=128, query_tile=16)
    with _exact_pipeline(ad): eng_exact, end_exact, _, _, _ = gqa_decode_run(op, recipe, units=24)
    started = time.perf_counter(); eng_compact, end_compact, _, _, _ = gqa_decode_run(op, recipe, units=24); elapsed = time.perf_counter() - started
    assert elapsed < 5.0 and end_compact.time == pytest.approx(end_exact.time, rel=1e-11)
    assert eng_compact.serviced == pytest.approx(eng_exact.serviced, rel=1e-12) and eng_compact.peak_allocated == pytest.approx(eng_exact.peak_allocated, rel=1e-12)
    short = gqa_decode_op([(1, 50 * 128 - 1)], heads=2, kv_heads=1, dim=32); compact = ad.compact_periodic_pipeline
    def unrolled(context, count, slots, tile, **kwargs): return PipelineTemplateProgram(ad._AttentionPipelineTemplate(0, slots, (PipelinePhase("full", tuple(tile(0)), count, residue_stages=tuple(tuple(tile(i)) for i in range(slots))),)))
    with pytest.MonkeyPatch.context() as control:
        control.setattr(ad, "compact_periodic_pipeline", unrolled); exact, exact_end, *_ = gqa_decode_run(short, StagedGQADecode(key_tile=128, query_tile=16), units=1)
    selected = []
    def checked_compact(*args, **kwargs):
        program = compact(*args, **kwargs); selected.append(program); return program
    def no_eager_template(*args, **kwargs): pytest.fail("compact selection eagerly constructed a finite template")
    with pytest.MonkeyPatch.context() as control:
        control.setattr(ad, "_attention_pipeline_template", no_eager_template); control.setattr(ad, "compact_periodic_pipeline", checked_compact)
        folded, folded_end, *_ = gqa_decode_run(short, StagedGQADecode(key_tile=128, query_tile=16), units=1)
    assert selected and all(isinstance(program, KernelProgram) for program in selected)
    # Floating accumulation order differs; these are numerical, not bitwise, equivalences.
    assert folded_end.time == pytest.approx(exact_end.time, rel=1e-12) and folded.serviced == pytest.approx(exact.serviced, rel=1e-12) and folded.peak_allocated == exact.peak_allocated
@pytest.mark.parametrize("context_len", CONTEXTS)
def test_ultra_long_context_gqa_and_paged_mla_execute_in_o1_time(context_len):
    started = time.perf_counter(); op = gqa_decode_op([(1, context_len)] * 8, heads=16, kv_heads=2, dim=128)
    eng, end, _, _, _ = gqa_decode_run(op, StagedGQADecode(key_tile=128, query_tile=16), units=24); gqa_elapsed = time.perf_counter() - started
    assert gqa_elapsed < 15.0 and eng.completed_count < 1000 and end.time > 0 and eng.summary()["live_tasks"] == 0 and all(v == 0 for v in eng.allocated.values())
    started = time.perf_counter(); raw_mla = paged_mla_case(splits=8); raw_mla["runtime"]["max_model_len"] = max(1048576, context_len + 16)
    mla_op, _ = paged_mla_attention(raw_mla, batch=4, context=context_len); row = paged_mla_execute(raw_mla, mla_op)
    assert time.perf_counter() - started < 15.0 and row["duration_ms"] > 0
def test_fai_decode_mapping_supports_single_query_high_batch_and_core_domains():
    mapped = host_partitions([(1, 409600)] * 32, 16, 2, 32, key_tile=128, query_tile=64)
    assert len(mapped) == 64 and {owner for spans in mapped.values() for owner, _, _ in spans} == set(range(32))
    # With tail_tasks = 2 <= 32 // 2 and kv_block_tile = 8, SplitCoreDecodeBS1GN2 unpacks the 2 tail batch tasks into single-KV-head tasks.
    mapped_tail = host_partitions([(1, 409600)] * 34, 16, 8, 32, key_tile=128, query_tile=64)
    assert len({mapped_tail[(32, h)][0][0] for h in range(8)}) == 8
    flash_mapped = partitions([(1, 409600)] * 4, 16, 2, 32, key_tile=128, query_tile=64)
    assert len(flash_mapped) == 8 and max(owner for spans in flash_mapped.values() for owner, _, _ in spans) < 32
def test_attention_decode_compaction_conserves_causal_tail_and_ragged_multi_request_schedule():
    import mindie_sim.prediction.operators.attention.attention_decode as ad
    causal = gqa_decode_op([(12, 512)], heads=8, kv_heads=2, dim=128); causal_recipe = StagedGQADecode(key_tile=8, query_tile=16)
    with _exact_pipeline(ad): eng_exact, end_exact, _, _, _ = gqa_decode_run(causal, causal_recipe, units=1)
    eng_comp, end_comp, _, _, _ = gqa_decode_run(causal, causal_recipe, units=1)
    assert eng_comp.serviced == pytest.approx(eng_exact.serviced, rel=1e-12) and end_comp.time == pytest.approx(end_exact.time, rel=1e-11)
    ragged = gqa_decode_op([(1, 2048), (1, 1024)], heads=8, kv_heads=2, dim=128); ragged_recipe = StagedGQADecode(key_tile=16, query_tile=16)
    with _exact_pipeline(ad): eng_exact_r, end_exact_r, _, _, _ = gqa_decode_run(ragged, ragged_recipe, units=1)
    eng_comp_r, end_comp_r, _, _, _ = gqa_decode_run(ragged, ragged_recipe, units=1)
    assert end_comp_r.time == pytest.approx(end_exact_r.time, rel=1e-12) and eng_comp_r.peak_allocated == pytest.approx(eng_exact_r.peak_allocated, rel=1e-12)
    from _support import operator_cases as td; raw_slow = td.gqa_decode_hardware(units=1); raw_slow["machine"]["device"]["rates"] = {"memory": 15e9, "vector": 26e6, "mte1": 4.5e6, "mte2": 31e6, "mte3": 1.5e6}
    raw_slow["machine"]["device"]["compute_peaks"] = {"bf16": 1e11, "fp32": 1e12}; recipe_slow = StagedGQADecode(key_tile=32, query_tile=64, slots=2); previous = 0.0
    for ctx_len in (32800, 32832, 409632, 409664):
        op_slow = gqa_decode_op([(12, ctx_len)], heads=2, kv_heads=1, dim=32)
        with _exact_pipeline(ad): _, end_unrolled, *_ = gqa_decode_run(op_slow, recipe_slow, units=1, raw_case=raw_slow)
        _, end_compact, *_ = gqa_decode_run(op_slow, recipe_slow, units=1, raw_case=raw_slow)
        assert end_compact.time == pytest.approx(end_unrolled.time, rel=1e-11) and end_compact.time > previous; previous = end_compact.time
def test_paged_mla_membase_multi_job_owner_and_refill_steady_subblock_isolation():
    from mindie_sim.prediction.operators.paged_mla.paged_mla_membase_dispatch import source_dispatch; from mindie_sim.prediction.operators.matrix.refill_steady import _extract_stage; import mindie_sim.prediction.operators.paged_mla.paged_mla_membase as pma22
    raw = source_dispatch((32768, 32768), 16, 24)
    with _exact_pipeline(pma22): _, eng_exact, dur_exact = paged_mla_membase_run(raw)
    _, eng_comp, dur_comp = paged_mla_membase_run(raw)
    assert dur_comp == pytest.approx(dur_exact, rel=1e-12) and eng_comp.serviced == pytest.approx(eng_exact.serviced, rel=1e-12)
    with pytest.raises(ValueError, match="unsupported_nested_stage_program:PipelineProgram"): _extract_stage(Stage("bad", program=PipelineProgram(2, 1, lambda i: (Stage("s"),))))
    units = {r: (2 if r in ("vector", "mte2_aiv", "mte3") else 1) for r in ("hbm", "vector", "cube", "mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe")}; rates = {r: (2 if r == "vector" else 1e20) for r in units}; rates["cube"] = 1
    pools = {p: max(n, 1 << 20) for p, n in PAGED_MLA_BUFFERS.items()}; pu = {p: (2 if p == "ub" else 1) for p in pools}
    for heads in (1, 2, 16, 31, 32, 33, 48, 63, 64):
        op = MlaOpSpec("mla", "attention", "decode", 0, 0, (), (), (), (), (), metadata=(("paged_mla_geometry", {"contexts": (8192,), "heads": heads, "kv_splits": 1, "page_size": 128}),))
        plan = build_paged_mla(op, MlaGeometry(1, units, pools, pu)); engine = MlaEngine([MlaResource(r, v, units=units[r]) for r, v in rates.items()], pools=pools, pool_units=pu)
        ctx = MlaRecipeContext(plan.work.apply(op), {}, {r: r for r in rates}, rates, 1, resource_units=units, compute_peaks={"bf16": 1e20}, pool_names={p: p for p in pools}, pool_capacities=pools, pool_units=pu)
        ready = engine.submit("input", release=7); competitor = engine.submit("external", {"vector": plan.work.resources["vector"] * 3}, after=(ready,), placement={"vector": (0,)})
        done = plan.lower(ctx).start(engine, "mla", ready); engine.advance(until=(done,))
        assert done.time - 7 == pytest.approx(plan.work.resources["vector"] * ((heads + 1) // 2) / heads * 2, rel=1e-12)
        engine.advance(until=(competitor,)); assert not any(engine.allocated.values())
def test_native_keypool_caller_scales_block_table_columns_above_max_model_len():
    raw = {"machine": {"backend": "ascend", "sku": "Ascend950DT_9581", "num_devices": 8}, "model": {"preset": "glm-5.3-flash", "layers": 4, "vocab_size": 128, "auxiliary": None}, "workload": {"requests": [{"id": "r0", "prompt_tokens": 1024, "output_tokens": 1}]}, "runtime": {"tp": 8, "max_model_len": 131072, "kernels": {"lowerings": {"pool_key_indexer": {"implementation": KEYPOOL_IMPL, "options": {"tail_capacity": 4}}}}}}
    ops = target_operators(raw, scope="body", phase="decode", tokens=1, context=409600, sequences=4); meta = dict(next(op for op in ops if op.recipe == "cann_keypool_nonempty_inputs").metadata)
    expected_cols = math.ceil((409600 + 1) / 128); assert meta["block_table_shape"] == (5, expected_cols) and expected_cols > math.ceil(131072 / 128)
@pytest.mark.parametrize("experts,topk", EXPERTS)
def test_derive_expert_pruned_preserves_dense_layers_and_sanitizes_router_groups(experts, topk):
    for base_key in ("glm-5.3", "deepseek-v4.1-flash", "qwen3.5-35b-a3b", "deepseek-v4-10t"):
        loaded = load_model(base_key); base = replace(loaded, expert_routing=tuple(1.0 / 256 for _ in range(256)), expert_routing_source="measured", layer_plans=tuple(replace(p, router_groups=8, router_group_topk=4) if p.ffn == "moe" else p for p in loaded.layer_plans))
        pruned = derive_expert_pruned(base, num_experts=experts, top_k=topk, model_id=f"{base_key}-e{experts}-k{topk}")
        assert pruned.num_experts == experts and pruned.top_k == topk
        if len(base.expert_routing) != experts: assert pruned.expert_routing == () and pruned.expert_routing_source == "uniform"
        for orig_plan, new_plan in zip(base.layer_plans, pruned.layer_plans):
            if orig_plan.ffn == "dense": assert new_plan.num_experts == 0 and new_plan.top_k == 0
            else:
                assert new_plan.num_experts == experts and new_plan.top_k == topk
                if orig_plan.router_groups > 0 and experts % orig_plan.router_groups == 0 and (experts // orig_plan.router_groups) * orig_plan.router_group_topk >= topk:
                    assert new_plan.router_groups == orig_plan.router_groups and new_plan.router_group_topk == orig_plan.router_group_topk
def test_moe_init_mxfp8_supports_sparse_topk_and_synthetic_expert_counts():
    raw = mxfp8_case(16); raw["model"]["num_experts"] = 16; raw["model"]["top_k"] = 2
    op = next(o for o in mxfp8_graph(raw, 16) if o.recipe == MXFP8_INIT_RECIPE); row = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]; meta = row["operator"]["metadata"]["program_execution"]
    # For topk <= 3, useGatherCopy is False (ScatterCopyExpandedXandMXQuant): each sorted assignment is loaded and quantized individually.
    assert meta["assignments"] == 16 * 4 * 2 and meta["quantized_input_rows"] == 16 * 4 * 2 and meta["output_data_bytes"] == 16 * 4 * 2 * 4096
    assert meta["source_branch"] == "onecore_sort_GATHER_count_mode_full_range_scatterCopy" and row["duration_ms"] > 0 and row["live_tasks"] == 0
def test_grouped_dequant_handles_sparse_zero_expert_groups_in_special_mode():
    rows = [0] * 64; rows[32] = 5; rows[33] = 7; rows[63] = 4; cfg = Config(tuple(rows), 512, 64, 64, 253952)
    by_bank = [sum(n for owner in [assignments(cfg, owner) for owner in range(cfg.blocks)] for g, n in owner if g == bank) for bank in range(64)]
    assert cfg.special and by_bank == rows
    # For E = 128 > 64 AIVs, advancing cursor before skipping empty groups lets the second wave execute cleanly.
    stalled = [0] * 128; stalled[64] = 8
    assert assignments(Config(tuple(stalled), 512, 64, 64, 253952), 0) == ((64, 8),)
    trailing = [0] * 128; trailing[0] = 8; assert assignments(Config(tuple(trailing), 512, 64, 64, 253952), 0) == ((0, 8),)
@pytest.mark.parametrize("sku", A2_SKUS)
def test_arch2201_skus_declare_complete_pipeline_and_precision_rates(sku):
    sku_json = json.loads((REPO_ROOT / "mindie_sim/backends/profiles/ascend.json").read_text(encoding="utf-8"))[sku]
    assert all(sku_json["device"]["kernel_invocations"][kind]["provenance"]["basis"] == "arch2201_baseline_extrapolation" for kind in ("vector", "cube"))
    assert sku_json["source"]["pipeline"]["fields"]["hbm"]["rate_bytes_per_s"] == pytest.approx(A2_SKUS[sku][4])
    assert sku_json["source"]["pipeline"]["provenance"]["architecture_inference"].startswith("A2 npu_arch_2201")
    clock, aic, aiv, memory, hbm = A2_SKUS[sku]
    case = Case.from_mapping({"machine": {"backend": "ascend", "sku": sku}, "model": {"preset": "qwen3.6-27b", "layers": 1, "vocab_size": 128, "auxiliary": None}, "workload": {"requests": [{"id": "r0", "prompt_tokens": 1024, "output_tokens": 1}]}, "runtime": {}})
    die = case.machine.dies[0]; rates, modes, peaks = dict(die.rates), dict(die.service_modes), dict(die.compute_peaks)
    assert die.memory_bytes == memory and die.compute_units == aic and rates["hbm"] == pytest.approx(hbm) and rates["mte1"] == pytest.approx(128 * aic * clock)
    assert rates["mte2_aic"] == pytest.approx(256 * aic * clock) and rates["mte2_aiv"] == pytest.approx(128 * aiv * clock) and rates["mte3"] == pytest.approx(64 * aiv * clock) and rates["fixpipe"] == pytest.approx(128 * aic * clock)
    assert modes["mte1_l0a"] == ("mte1", pytest.approx(256 * aic * clock)) and modes["mte1_l0b"] == ("mte1", pytest.approx(128 * aic * clock)) and peaks["fp32"] == pytest.approx(peaks["bf16"] / 4) and peaks["hf32"] == pytest.approx(peaks["bf16"] / 2) and peaks["int4"] == pytest.approx(peaks["bf16"] * 4)
def test_indexed_attention_gt16_heads_splits_softmax_and_output_across_both_aivs():
    from mindie_sim.prediction.engine import Allocation, Engine, Resource
    from mindie_sim.prediction.operators.paged_mla.indexed_attention import build as build_indexed, build_pipeline
    rates = {"hbm": 1.6e12, "vector": 5e10, "cube": 300e12, "mte1": 5e12, "mte2_aic": 1.4e12, "mte2_aiv": 2.8e12, "mte3": 5e12, "fixpipe": 5e12}; res_units = {"hbm": 1, "vector": 40, "mte2_aiv": 40, "mte3": 40, "mte2_aic": 20, "mte1": 20, "fixpipe": 20, "cube": 20}
    buffers = {"ub": 196608, "l2": 200_000_000}; buf_units = {"ub": 40, "l2": 1}; geom = Geometry(compute_units=20, resource_units=res_units, buffers=buffers, buffer_units=buf_units)
    def run(plan):
        mapped = {}; engine = Engine([Resource(r, v, units=res_units[r]) for r, v in rates.items()], pools=buffers, pool_units=buf_units)
        ctx = RecipeContext(OpSpec("idx", "attention", "decode", 0, 0, (), (), (), (), ()), {}, {r: r for r in rates}, rates, 20, resource_units=res_units, compute_peaks={"bf16": 300e12}, pool_names={p: p for p in buffers}, pool_capacities=buffers, pool_units=buf_units, record_mapping=mapped.update)
        done = plan.lower(ctx).start(engine, "idx", engine.submit("ready")); engine.advance(until=(done,)); return done.time, mapped
    for heads, ownership in ((16, "two_gather_AIVs; one_softmax_and_output_AIV"), (64, "two_gather_AIVs; two_softmax_and_output_AIVs"), (128, "two_gather_AIVs; two_softmax_and_output_AIVs")):
        shape = {"sequences": ((1, 4096),), "topk": 2048, "heads": heads, "latent": 512, "rope": 64, "window": 0, "compression_ratio": 1, "cp": 1}
        dur_compact, mapped = run(build_pipeline(shape, geom, scratch_policy="offchip", compact=True)); dur_exact, _ = run(build_pipeline(shape, geom, scratch_policy="offchip", compact=False))
        assert mapped["vector_ownership"] == ownership and dur_compact == pytest.approx(dur_exact, rel=0.05)
    # Uneven heads=17 (m0=16 on AIV0, m1=1 on AIV1) subblock contention control.
    units = {r: (2 if r in ("vector", "mte2_aiv", "mte3") else 1) for r in ("hbm", "vector", "cube", "mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe")}; rates_1 = {r: (2 if r == "vector" else 1e20) for r in units}; rates_1["cube"] = 1
    shape_17 = dict(query=1, context=127, topk=2048, heads=17, latent=512, rope=64, query_dtype="bf16", cache_dtype="bf16", window=0, compression_ratio=1, cp=1); op_17 = OpSpec("idx", "attention", "decode", 0, 0, (), (), (), (), (), metadata=(("indexed_attention_geometry", shape_17),))
    for compact_mode in (False, True):
        plan = build_indexed(op_17, Geometry(1, units, {}, {}), scratch_policy="offchip", compact=compact_mode)
        ctx = RecipeContext(plan.work.apply(op_17), {}, {r: r for r in rates_1}, rates_1, 1, resource_units=units, compute_peaks={"bf16": 1e20}); prog = plan.lower(ctx); times = {}
        times = {comp_aiv: _contended(prog, plan, comp_aiv, units, rates_1) for comp_aiv in (None, 0, 1)}
        assert times[None] == pytest.approx(65952.0, rel=1e-12) and times[0] == pytest.approx(131904.0, rel=1e-12)
        assert times[1] < times[0] if compact_mode else times[1] == pytest.approx(74144.0, rel=1e-12)
    # Balanced heads=32 (m0=16 on AIV0, m1=16 on AIV1) subblock contention control.
    op_32 = OpSpec("idx32", "attention", "decode", 0, 0, (), (), (), (), (), metadata=(("indexed_attention_geometry", dict(shape_17, heads=32)),))
    for compact_mode in (False, True):
        plan = build_indexed(op_32, Geometry(1, units, {}, {}), scratch_policy="offchip", compact=compact_mode)
        ctx = RecipeContext(plan.work.apply(op_32), {}, {r: r for r in rates_1}, rates_1, 1, resource_units=units, compute_peaks={"bf16": 1e20}); prog = plan.lower(ctx)
        for comp_aiv, expected in ((None, 65952.0), (0, 131904.0), (1, 131904.0)):
            assert _contended(prog, plan, comp_aiv, units, rates_1) == pytest.approx(expected, rel=1e-12)
@pytest.mark.parametrize("preset,tp,expected_families", PRESETS)
def test_arch2201_default_body_pipelines_execute_ultra_long_decode(preset, tp, expected_families):
    raw = {"machine": {"backend": "ascend", "sku": "Ascend910B2C", "num_devices": tp}, "model": {"preset": preset, "quantization": "bf16", "layers": 4, "vocab_size": 128, "auxiliary": None}, "workload": {"requests": [{"id": "r0", "prompt_tokens": 409600, "output_tokens": 1}]}, "runtime": {"tp": tp, "max_model_len": 524288}}
    parsed = parse_case(raw); _, selection = effective_body_declaration(parsed, "decode", tokens=1, scope="body")
    assert expected_families <= {row["family"] for row in selection["selections"] if row["selection"] == "default_installed"}
    assert sum(row["duration_ms"] for row in operator_costs(raw, target_operators(raw, scope="body", phase="decode", tokens=1, context=409600, sequences=4))["operators"]) > 0
def test_execution_prepare_and_program_cache_normalize_layer_prefixed_metadata():
    from mindie_sim.prediction.execution import _meta_diff, _meta_patch, _op_meta_key
    def meta_op(layer):
        return OpSpec(f"s0.l{layer}.mc2_dispatch.send_read_quantize", "mc2_protocol", "decode", 0, layer, (), (1, 5120), (1, 5120), (1, 5120), (1, 5120), recipe="arch35_fullmesh_resource_phase", metadata=(("device_kernel_group", {"id": f"s0.l{layer}.mc2_dispatch", "entry": False, "exit": False, "kernel": "MoeDistributeDispatchV3"}), ("fusion_group_key", f"s0.l{layer}.fusion"), ("fusion_semantic_ids", [f"s0.l{layer}.a", f"s0.l{layer}.b"]), ("structure_subgraphs", ({"key": f"s0.l{layer}.sub"},))))
    first, second = meta_op(1), meta_op(2); assert _op_meta_key(first) == _op_meta_key(second) and not hasattr(first, "_frozen_meta_key")
    # Deletions (e.g. Work.apply removing resource_work when zeroed) are replayed on a cache hit.
    diff_del = _meta_diff({"resource_work": {"vector": 100.0}, "fusion_group_key": "s0.l0.g"}, {"fusion_group_key": "s0.l0.g", "program_execution": {"mode": "compact"}}); patched = _meta_patch({"resource_work": {"vector": 100.0}, "fusion_group_key": "s0.l1.g"}, diff_del)
    assert "resource_work" not in patched and patched["fusion_group_key"] == "s0.l1.g" and patched["program_execution"] == {"mode": "compact"}
    # Nested dictionaries preserve layer-specific sibling keys like fusion_semantic_ids and external_outputs.
    orig = {"fusion_boundary": {"fusion_semantic_ids": ["s0.l0.q_a", "s0.l0.q_b"], "external_outputs": ["s0.l0.q_b"], "source_program": {"program": {"name": "prolog"}}}}; result = {"fusion_boundary": {"fusion_semantic_ids": ["s0.l0.q_a", "s0.l0.q_b"], "external_outputs": ["s0.l0.q_b"], "source_program": {"program": {"name": "prolog", "work_binding": "physical_device_geometry"}}}}
    patched = _meta_patch({"fusion_boundary": {"fusion_semantic_ids": ["s0.l1.q_a", "s0.l1.q_b"], "external_outputs": ["s0.l1.q_b"], "source_program": {"program": {"name": "prolog"}}}}, _meta_diff(orig, result))
    assert patched["fusion_boundary"]["fusion_semantic_ids"] == ["s0.l1.q_a", "s0.l1.q_b"] and patched["fusion_boundary"]["external_outputs"] == ["s0.l1.q_b"] and patched["fusion_boundary"]["source_program"]["program"]["work_binding"] == "physical_device_geometry"
@pytest.mark.parametrize("tp,dtype", [(2, "bf16"), (4, "fp32"), (8, "int8")])
def test_independent_ag_and_rs_wire_recompute_across_tp_widths(tp, dtype):
    parsed, batch, plan = _plan_for(tp=tp, lengths=(2 * tp,)); width = 32; expected = 2 * tp * width * dtype_bytes(dtype) * (tp - 1) / tp; requests = tuple(view.request_id for view in batch)
    gathered = allgather(plan=plan, op_id="ag", deps=("shard",), stage=0, layer=0, requests=requests, phase="prefill", width=width, dtype=dtype); scattered = reduce_scatter(plan=plan, op_id="rs", deps=("attention",), stage=0, layer=0, requests=requests, phase="prefill", width=width, dtype=dtype)
    # Each physical rank sends this payload in both directions of the path.
    assert gathered.link_bytes == pytest.approx(expected) and scattered.link_bytes == pytest.approx(expected)
    ag_meta, rs_meta = dict(gathered.metadata), dict(scattered.metadata)
    assert ag_meta["input_shape"] == rs_meta["output_shape"] == (2, width) and ag_meta["output_shape"] == rs_meta["input_shape"] == (2 * tp, width)
    assert ag_meta["group_axis"] == rs_meta["group_axis"] == "tp" and ag_meta["steps"] == rs_meta["steps"] == tp - 1 and ag_meta["source_partitions"] == 1
    assert gathered.local_shape == (2 * tp, width) and scattered.local_shape == (2, width) and gathered.dtype == scattered.dtype == dtype
    # The ordinary topology accounts the same per-rank payload independently.
    topo = Topology(parsed.machine, parsed.runtime.parallel); ag_work = topo.accounting(gathered, 0); rs_work = topo.accounting(scattered, 0)
    assert set(ag_work.useful_by_rank) == set(rs_work.useful_by_rank) == set(range(tp))
    assert all(value == pytest.approx(expected) for value in ag_work.useful_by_rank.values()) and all(value == pytest.approx(expected) for value in rs_work.useful_by_rank.values())
def test_plan_dimensions_follow_tp_and_global_token_cohort():
    _, _, plan = _plan_for(tp=4, lengths=(16,)); parsed = _source_case(tp=4)
    dense_layer = replace(parsed.model.layer_plans[0], ffn="dense"); moe_layer = replace(parsed.model.layer_plans[1], ffn="moe")
    assert plan.ffn_tokens(dense_layer) == plan.tokens_global == 16 and plan.ffn_tokens(moe_layer) == plan.tokens_local == 4
    assert plan.attention_tokens == 16 and plan.normalize_tokens == plan.attn_res_tokens == plan.router_tokens == 4 and plan.cohort_tokens == 16 and plan.source_partitions == 1
    _, batch, plan = _plan_for(tp=2, lengths=(3, 3, 1, 1))
    assert [view.q_len for view in batch] == [3, 3, 1, 1] and all(view.q_len % 2 for view in batch) and plan.tokens_global == 8 and plan.tokens_local == 4
    _, _, decode_plan = _plan_for(tp=2, lengths=(1, 1, 1, 1), phase="decode")
    assert decode_plan.phases == ("decode",) and decode_plan.tokens_global == 4 and decode_plan.tokens_local == 2
    with pytest.raises(ModelingError, match="tokens_divisible_by_tp"): _plan_for(tp=2, lengths=(1, 1, 1), phase="decode")
def test_public_serving_non_draft_decode_keeps_the_single_committed_token():
    parsed = _source_case(tp=2); parsed = replace(parsed, runtime=replace(parsed.runtime, kernels={"sequence_parallel": {**DECLARATION, "padding": "right_to_tp"}}))
    result = predict(parsed, trace_limit=4); row = result["requests"][0]
    assert row["status"] == "served" and row["generated_tokens"] == 3 and row["iterations"] == 3
    assert row["itl"]["count"] == 2 and row["finish_ms"] > row["first_token_abs_ms"] and [step["valid_tokens"] for step in result["traces"]] == [8, 1, 1]
    assert row["speculative"]["proposed_tokens"] == 0 and result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
def test_cross_dp_and_idle_cohorts_keep_unique_totals_and_reject_incomplete_totals():
    _, batch, plan = _plan_for(tp=2, ep=4, dp=2, lengths=(6,), global_tokens=10)
    assert plan.tokens_global == 6 and plan.tokens_local == 3 and plan.cohort_tokens == 10 and plan.ep == 4 and plan.dp == 2
    parsed = _source_case(tp=2, ep=4, dp=2)
    with pytest.raises(ModelingError, match="cross_dp_requires_complete_cohort_tokens"): select(parsed.runtime, parsed.model, batch, tokens=6)
    with pytest.raises(ModelingError, match="cohort_tokens_below_local_full"): select(parsed.runtime, parsed.model, batch, tokens=6, global_tokens=4)
    with pytest.raises(ModelingError, match="cohort_tokens_not_divisible_by_tp"): select(parsed.runtime, parsed.model, batch, tokens=6, global_tokens=9)
    with pytest.raises(ModelingError, match="outside_ep_cohort"): select(_source_case(tp=2, ep=2).runtime, _source_case(tp=2, ep=2).model, batch, tokens=6, global_tokens=8)
    active = select(parsed.runtime, parsed.model, _views(8), tokens=8, global_tokens=8); idle = select(parsed.runtime, parsed.model, (), tokens=0, global_tokens=8)
    assert idle.tokens_global == idle.tokens_local == 0 and active.cohort_tokens == idle.cohort_tokens == 8
    assert sum(candidate.tokens_local * candidate.tp for candidate in (active, idle)) == 8 and active.source_partitions == idle.source_partitions == 1
    for builder in (allgather, reduce_scatter):
        with pytest.raises(ModelingError, match="empty_replica_has_no_tp_collective"): builder(plan=idle, op_id="empty", deps=("input",), stage=0, layer=0, requests=(), phase="prefill", width=32, dtype="bf16")
def test_real_source_case_retains_tp_shards_and_unique_cross_dp_rows():
    raw = json.loads((REPO_ROOT / SP_CASE_PATH).read_text()); raw["runtime"].setdefault("kernels", {})["sequence_parallel"] = dict(DECLARATION); parsed = parse_case(raw)
    plans = tuple(select(parsed.runtime, parsed.model, _views(rows) if rows else (), tokens=rows, global_tokens=32768) for rows in (8192, 16384, 8192, 0))
    assert (plans[0].tp, plans[0].ep, plans[0].dp) == (8, 32, 4) and [plan.tokens_local for plan in plans] == [1024, 2048, 1024, 0]
    assert sum(plan.tokens_local * plan.tp for plan in plans) == 32768 and {plan.cohort_tokens for plan in plans} == {32768} and {plan.source_partitions for plan in plans} == {1}
def test_source_selected_branch_rejects_context_parallel_and_accepts_pipeline_and_prefill_context_parallel():
    parsed = _source_case(tp=2); runtime = replace(parsed.runtime, parallel=replace(parsed.runtime.parallel, cp=2, cp_plan="dcp"))
    with pytest.raises(ModelingError, match="requires_cp1"): select(runtime, parsed.model, _views(8), tokens=8)
    plan = select(replace(parsed.runtime, parallel=replace(parsed.runtime.parallel, pp=2)), parsed.model, _views(8), tokens=8)
    assert plan.tp == 2 and plan.tokens_local == 4 and plan.tokens_global == 8
    pcp_plan = select(replace(parsed.runtime, parallel=replace(parsed.runtime.parallel, pcp=2, cp_plan="pcp")), parsed.model, _views(8), tokens=8)
    assert pcp_plan.tokens_local == 4 and pcp_plan.tokens_global == 8
def test_declaration_and_structural_domain_reject_unsupported_rows():
    parsed = _source_case(tp=2)
    with pytest.raises(ModelingError, match="option_unsupported:padding"): select(replace(parsed.runtime, kernels={"sequence_parallel": dict(DECLARATION, padding="pad")}), parsed.model, _views(8), tokens=8)
    with pytest.raises(ModelingError, match="declaration_unknown:extra"): select(replace(parsed.runtime, kernels={"sequence_parallel": dict(DECLARATION, extra=True)}), parsed.model, _views(8), tokens=8)
    no_residual = replace(parsed.model, layer_plans=tuple(replace(layer, residual_block_size=0) for layer in parsed.model.layer_plans))
    with pytest.raises(ModelingError, match="attn_residual_block_size"): select(parsed.runtime, no_residual, _views(8), tokens=8)
    hyper = replace(parsed.model, layer_plans=tuple(replace(layer, hyper_connections=True) for layer in parsed.model.layer_plans))
    with pytest.raises(ModelingError, match="hyperconnections_unsupported"): select(parsed.runtime, hyper, _views(8), tokens=8)
    draft = replace(parsed.model, layer_plans=(replace(parsed.model.layer_plans[0], scope="draft"),))
    assert select(parsed.runtime, draft, _views(8), tokens=8) is not None
    with pytest.raises(ModelingError, match="requires_body_scope"): select(parsed.runtime, draft, _views(8), tokens=8, allow_draft_scope=False)
    with pytest.raises(ModelingError, match="no_padding_domain"): select(parsed.runtime, parsed.model, (replace(_views(8)[0], padding=True),), tokens=8)
    with pytest.raises(ModelingError, match="speculative_field_unsupported"): select(parsed.runtime, parsed.model, (replace(_views(8)[0], draft_proposal_tokens=1),), tokens=8)
    with pytest.raises(ModelingError, match="tokens_must_equal_sum_q_len"): select(parsed.runtime, parsed.model, _views(8), tokens=9)
    with pytest.raises(ModelingError, match="requires_positive_q_len"): select(parsed.runtime, parsed.model, _views(0), tokens=8)
    with pytest.raises(ModelingError, match="tokens_divisible_by_tp"): select(parsed.runtime, parsed.model, _views(7), tokens=7)
def test_input_shard_and_aux_channels_conserve_payload_and_true_shape():
    _, batch, plan = _plan_for(tp=2, lengths=(8,)); shard = input_shard(plan=plan, op_id="shard", deps=("full_embedding",), stage=0, layer=0, requests=tuple(view.request_id for view in batch), phase="prefill", width=32, dtype="bf16")
    assert shard.family == "barrier" and shard.link_bytes == 0; metadata = dict(shard.metadata)
    assert metadata["shard_materialization"] == "view" and metadata["physical_payload_bytes"] == 0 and metadata["zero_payload_reason"] == "declared_view_no_padding" and metadata["input_shape"] == (8, 32) and metadata["output_shape"] == (4, 32) and shard.deps == ("full_embedding",)
    assert metadata["input_dependency"] == "full_embedding" and metadata["source_assignment_partitions"] == 1
    with pytest.raises(ModelingError, match="requires_full_embedding_dependency"): input_shard(plan=plan, op_id="bad", deps=(), stage=0, layer=0, requests=("r0",), phase="prefill", width=32, dtype="bf16")
    _, batch, plan = _plan_for(tp=4, lengths=(16,)); packed = allgather(plan=plan, op_id="final_packed", deps=("final",), stage=0, layer=-1, requests=tuple(view.request_id for view in batch), phase="prefill", width=32, dtype="bf16", channel_multiplier=3)
    assert dict(packed.metadata)["input_shape"] == (4, 96) and dict(packed.metadata)["output_shape"] == (16, 96) and packed.link_bytes == pytest.approx(16 * 96 * dtype_bytes("bf16") * 3 / 4)
def test_close_output_replaces_only_the_exact_tp_allreduce_tail():
    _, batch, plan = _plan_for(tp=4, lengths=(16,)); requests = tuple(view.request_id for view in batch)
    ag = allgather(plan=plan, op_id="mlp_ag", deps=("mlp_shard",), stage=0, layer=0, requests=requests, phase="prefill", width=32, dtype="bf16")
    allreduce = replace(comm_op(op_id="mlp_reduce", deps=(ag.op_id,), stage=0, layer=0, requests=requests, family="allreduce", phase="prefill", recipe="ring_allreduce", link_bytes=2 * 3 / 4 * 16 * 32 * 2, shape=(16, 32), metadata=(("group_axis", "tp"), ("steps", 6))), stream="aux")
    ops, last = close_output((ag, allreduce), allreduce.op_id, plan=plan, width=32, dtype="bf16")
    assert last != allreduce.op_id and [op.family for op in ops] == ["allgather", "reduce_scatter"] and ops[-1].deps == (ag.op_id,) and ops[-1].stream == "aux"
    assert ops[-1].link_bytes == pytest.approx(ag.link_bytes) and not any(op.family == "allreduce" for op in ops)
    with pytest.raises(ModelingError, match="requires_exact_tail"): close_output((allreduce, ag), allreduce.op_id, plan=plan, width=32, dtype="bf16")
    non_tp = replace(allreduce, metadata=(("group_axis", "ep"),))
    with pytest.raises(ModelingError, match="requires_tp_allreduce_tail"): close_output((non_tp,), non_tp.op_id, plan=plan, width=32, dtype="bf16")
    two_deps = replace(allreduce, deps=(ag.op_id, "other"))
    with pytest.raises(ModelingError, match="requires_sole_producer"): close_output((two_deps,), two_deps.op_id, plan=plan, width=32, dtype="bf16")
def test_ordinary_graph_executes_dependencies_and_releases_input_ready_state():
    parsed, batch, plan = _plan_for(tp=4, lengths=(16,)); requests = tuple(view.request_id for view in batch)
    shard = input_shard(plan=plan, op_id="shard", deps=("full_embedding",), stage=0, layer=0, requests=requests, phase="prefill", width=32, dtype="bf16"); ag = allgather(plan=plan, op_id="attention_ag", deps=(shard.op_id,), stage=0, layer=0, requests=requests, phase="prefill", width=32, dtype="bf16")
    attention = vector_op(op_id="attention", deps=(ag.op_id,), stage=0, layer=0, requests=requests, elements=16 * 32, ops_per_element=2, dtype="bf16", family="rmsnorm", phase="prefill", recipe="rmsnorm", shape=(16, 32))
    rs = reduce_scatter(plan=plan, op_id="attention_rs", deps=(attention.op_id,), stage=0, layer=0, requests=requests, phase="prefill", width=32, dtype="bf16"); consumer = vector_op(op_id="consumer", deps=(rs.op_id,), stage=0, layer=0, requests=requests, elements=4 * 32, ops_per_element=1, dtype="bf16", family="rmsnorm", phase="prefill", recipe="rmsnorm", shape=(4, 32))
    ring = replace(parsed.runtime, kernels={"sequence_parallel": dict(DECLARATION), "families": {"allgather": {"kind": "ring_collective"}, "reduce_scatter": {"kind": "ring_collective"}}})
    report = graph_costs(replace(parsed, runtime=ring), (shard, ag, attention, rs, consumer), external_inputs={"full_embedding": 3}, trace_limit=200); rows = {row["id"]: row for row in report["timeline"]}
    assert rows["r0:shard"]["ready_ms"] >= 3 and rows["r0:attention_ag"]["ready_ms"] >= rows["r0:shard"]["complete_ms"] and rows["r0:attention"]["ready_ms"] >= rows["r0:attention_ag"]["complete_ms"]
    assert rows["r0:attention_rs"]["ready_ms"] >= rows["r0:attention"]["complete_ms"] and rows["r0:consumer"]["ready_ms"] >= rows["r0:attention_rs"]["complete_ms"]
    assert report["execution"]["live_tasks"] == 0 and report["execution"]["live_programs"] == 0
    topo = Topology(parsed.machine, parsed.runtime.parallel); expected = sum(topo.accounting(ag, 0).wire_by_resource.values()) + sum(topo.accounting(rs, 0).wire_by_resource.values())
    assert report["execution"]["serviced"]["link:fabric"] == pytest.approx(expected) and report["execution"]["work"]["link_bytes"] == pytest.approx(ag.link_bytes + rs.link_bytes)
@pytest.mark.parametrize("counts", [(4, 4), (8, 0)])
def test_full_cohort_work_survives_sequence_shards_and_idle_source(counts):
    parsed = _source_frontend_case(); views = tuple((RequestView(f"r{i}", count, 0, True, "prefill"),) if count else () for i, count in enumerate(counts))
    global_views = tuple(view for group in views for view in group); graphs = {}
    for replica, local in enumerate(views):
        work = Frontend()(IterationPlan(local), parsed.model, parsed.runtime, parsed.machine, global_tokens=sum(counts), global_views=global_views); graphs[replica] = tuple(op for op in work.ops if op.family not in {"sampling", "token_commit", "acceptance"})
    graphs = bind_cohort_traffic(parsed, graphs)
    for replica, ops in graphs.items():
        full, local = counts[replica], counts[replica] // 2; grouped = [op for op in ops if op.family == "grouped_gemm"]
        assert len(grouped) == 2 and all(op.local_shape[0] == sum(counts) * 2 // 4 for op in grouped)
        dispatch = _one(ops, lambda op: op.recipe == "expert_dispatch"); combine = _one(ops, lambda op: op.recipe == "expert_combine"); assert dict(dispatch.metadata)["source_assignment_partitions"] == 1 and dispatch.link_bytes == local * 2 * 32 * 2 * 3 / 4 and dict(combine.metadata)["peer_traffic"]["direction"] == "transpose"
        if not full:
            assert combine.link_bytes > 0 and not any(op.family in {"allgather", "reduce_scatter"} for op in ops) and not any(dict(op.metadata).get("ffn_category") == "shared_expert" for op in ops); continue
        assert {op.op_id for op in ops if op.family == "allreduce"} == {"embedding_allreduce"}
        assert not any(op.family == "allreduce" and op.layer >= 0 for op in ops)
        assert all(op.local_shape[0] == full for op in ops if dict(op.metadata).get("ffn_category") == "shared_expert")
        assert _one(ops, lambda op: op.recipe == "router").local_shape[0] == local
        assert all(op.local_shape[0] == full for op in ops if op.recipe == "dense_gate_up") and sum(op.recipe == "dense_gate_up" for op in ops) == 2
        source = {op.op_id: op for op in ops}
        for op in ops:
            if op.family == "qkv_projection": assert op.local_shape[0] == full and source[op.deps[0]].family == "allgather"
        assert sum(op.family == "reduce_scatter" for op in ops) == 4
    result = graph_costs(parsed, graphs)
    assert result["duration_ms"] > 0 and result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0 and not any(result["execution"]["live_allocated"].values())
def test_real_source_prefill_wrapper_uses_local_rows_without_shrinking_attention_or_experts():
    raw = json.loads((REPO_ROOT / SP_CASE_PATH).read_text()); raw["runtime"]["graph_mode"] = "eager"
    raw["runtime"]["kernels"] = {"sequence_parallel": dict(DECLARATION), "lowerings": {"matrix_wrapper": {"implementation": "vllm_ascend_mxfp8_single_group_eager", "options": {"projection_recipes": ["latent_projection"]}}}}
    parsed = parse_case(raw)
    expected_rows = math.ceil(RoutingPolicy(**dict(parsed.runtime.routing)).local(parsed.model, parsed.model.layer_plans[1], parsed.runtime.parallel, 8192 * parsed.runtime.parallel.dp, runtime=parsed.runtime)[0])
    assert expected_rows >= 16384
    graphs = target_graphs(parsed, phase="prefill", tokens=8192, scope="model"); assert len(graphs) == 4
    for ops in graphs.values():
        latent = [op for op in ops if op.recipe == "latent_projection"]
        assert len(latent) == 184 and all(op.local_shape[0] == 1024 for op in latent) and all("matrix_wrapper" in dict(op.metadata) for op in latent)
        grouped = [op for op in ops if op.family == "grouped_gemm"]
        assert grouped and all(op.local_shape[0] == expected_rows for op in grouped)
        assert all(op.local_shape[0] == 8192 for op in ops if op.family == "attention") and any(op.family == "attention" for op in ops)
        shared = [op for op in ops if dict(op.metadata).get("ffn_category") == "shared_expert"]; assert shared and all(op.local_shape[0] == 8192 for op in shared)
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 32])
def test_complete_source_graph_slices_attention_and_preserves_padded_ffn(batch):
    graph = target_graphs(_sp_case(batch), scope="body", phase="decode", tokens=1, sequences=batch, context=4096, apply_body_defaults=True)[0]
    by_id = {op.op_id: op for op in graph}; physical = max(8, batch); assert {op.layer for op in graph if op.layer >= 0} == set(range(93))
    for layer in range(93):
        prefix = f"s0.l{layer}"; ops = [op for op in graph if op.layer == layer]; is_mla = any(op.recipe == "q_a" for op in ops)
        # The K3 KDA input projections are packed into one declared matrix, so the shared producer replaces the separate Q projection.
        first = by_id[prefix + (".q_a" if is_mla else ".linear_q" if prefix + ".linear_q" in by_id else ".linear_qkvbfg_a")]; assert first.local_shape[0] == batch
        if not is_mla: assert by_id[prefix + ".short_convolution"].local_shape == (batch * 4608,) and dict(by_id[prefix + ".recurrent_scan0"].metadata)["kda_recurrent_params"]["lengths"] == (1,) * batch
        gather = by_id[prefix + ".attention.sp_gather"]; assert gather.local_shape == (physical, 7168) and gather.link_bytes == physical * 7168 * 2 * 7 / 8
        sliced = by_id.get(prefix + ".attention.sp_input_view"); pad = [op for op in ops if op.recipe == "sequence_parallel_cat_zeros_pad"]
        if batch < physical:
            assert sliced.local_shape == (batch, 7168) and sliced.deps == (gather.op_id,) and first.deps == (sliced.op_id,) and sliced.hbm_bytes == sliced.link_bytes == 0
            has_shared_scatter = any(op.op_id == prefix + ".shared.sp_scatter" for op in ops)
            expected_pads = 2 if has_shared_scatter else 1
            assert len(pad) == expected_pads
            for pad_op in pad:
                assert pad_op.mte2_bytes == physical * 7168 * 2 and pad_op.mte3_bytes == (2 * physical - batch) * 7168 * 2 and pad_op.vector_ops == (physical - batch) * 7168
                scatter = next(op for op in ops if op.deps == (pad_op.op_id,)); assert scatter.family == "reduce_scatter" and scatter.link_bytes == gather.link_bytes
        else: assert sliced is None and pad == [] and first.deps == (gather.op_id,)
    # FFN and residual tensors are still SP-local, while shared/dense TP matrices execute their physical allgather rows.
    assert by_id["s0.l0.gate_up"].local_shape[0] == physical and by_id["s0.l1.router"].local_shape[0] == physical // 8
    for op in graph:
        if op.family == "grouped_gemm":
            route = dict(op.metadata)["routing_realization"]; assert route["global_tokens"] == physical and route["valid_local_tokens"] == batch and sum(route["global_expert_rows"]) == physical * 8
        if op.recipe == "attn_res_fwd": assert op.local_shape[0] == physical // 8
def test_graph_bucket_rows_are_preserved_but_sp_rows_do_not_reach_attention():
    parsed = parse_case(_sp_case(1)); views = (RequestView("r", 1, 4096, False, "decode"),); plan = select(parsed.runtime, parsed.model, views, tokens=4, global_tokens=8)
    assert (plan.valid_global, plan.attention_tokens, plan.physical_global) == (1, 4, 8)
    graph, _ = Frontend().body(parsed.model, parsed.runtime, views, ready="input", tokens=4, mode="decode", global_tokens=8, machine=parsed.machine); by_id = {op.op_id: op for op in graph}
    assert (by_id.get("s0.l0.linear_q") or by_id["s0.l0.linear_qkvbfg_a"]).local_shape[0] == 4 and by_id["s0.l3.q_a"].local_shape[0] == 4 and by_id["s0.l0.attention.sp_input_view"].local_shape == (4, 7168)
    pad = by_id["s0.l0.linear_reduce.reduce_scatter.pad"]; assert (pad.mte2_bytes, pad.mte3_bytes) == (8 * 7168 * 2, 12 * 7168 * 2) and dict(by_id["s0.l0.recurrent_scan0"].metadata)["sequence_tokens"] == (1,)
@pytest.mark.parametrize("batch", [1, 8])
def test_real_source_view_and_repad_wait_for_input_and_release_engine_state(batch):
    graph = target_graphs(_sp_case(batch), scope="body", phase="decode", tokens=1, sequences=batch, context=4096, apply_body_defaults=True)[0]
    # Actual first dense/KDA and routed/KDA blocks; complete depth is checked above.
    result = graph_costs(_sp_case(batch), {0: tuple(op for op in graph if op.layer in (0, 1) or op.op_id == "sp_shard")}, external_inputs={"analysis_input": 7}); execution = result["execution"]
    assert min(row["ready_ms"] for row in result["timeline"]) >= 7 and result["duration_ms"] > 7 and execution["live_tasks"] == execution["live_programs"] == 0 and all(value == 0 for value in execution["live_allocated"].values())
def test_full_depth_public_cohort_keeps_local_attnres_global_attention_and_experts():
    case = _combined_case(); assert case.model.layers == 93
    evidence = _check_structure(case, target_graphs(case, phase="prefill", tokens=8192, scope="body"), (8192,) * 4)
    straggler_rows = {row["expert_rows_per_layer"] for row in evidence}
    assert {row["attnres_points"] for row in evidence} == {187} and {row["local_rows"] for row in evidence} == {1024} and {row["mean_expert_rows_per_layer"] for row in evidence} == {16384} and len(straggler_rows) == 1 and next(iter(straggler_rows)) > 16384
    rows = (8192, 16384, 8192, 0); unequal = _check_structure(case, _unequal_graphs(case, rows), rows)
    assert [row["attnres_points"] for row in unequal] == [187, 187, 187, 0] and [row["local_rows"] for row in unequal] == [1024, 2048, 1024, 0] and {row["mean_expert_rows_per_layer"] for row in unequal} == {16384} and {row["expert_rows_per_layer"] for row in unequal} == straggler_rows
@pytest.mark.parametrize("unequal", [False, True])
def test_actual_complete_small_graph_preserves_bank_shard_and_transport_dependencies(unequal):
    case = _combined_case(layers=2, bounded=True); rows = (8, 16, 8, 0) if unequal else (8,) * 4
    graphs = _unequal_graphs(case, rows) if unequal else target_graphs(case, phase="prefill", tokens=8, scope="body")
    _check_structure(case, graphs, rows); evidence = _execute(case, graphs); assert evidence["completed_tasks"] > evidence["operators"]
def test_decode_keeps_sp_branch_but_does_not_claim_prefill_cache():
    case = _combined_case(layers=2); graphs = target_graphs(case, phase="decode", tokens=1, sequences=8, context=64, scope="body")
    assert {row["local_rows"] for row in _check_structure(case, graphs, (8,) * 4, phase="decode")} == {1}
def test_idle_exception_does_not_hide_nonidle_unmatched_source_selection():
    case = _combined_case(layers=2); ops, _ = Frontend().body(case.model, case.runtime, (), ready="analysis_input", tokens=0, global_tokens=32, mode="prefill", machine=case.machine); original = deepcopy(ops)
    lower_implementations(ops, case.runtime, case.model.activation_dtype, local_tokens=0); assert ops == original
    with pytest.raises(ModelingError, match="matrix_wrapper_projection_recipes_unmatched"): lower_implementations(ops, case.runtime, case.model.activation_dtype, local_tokens=8)
@pytest.mark.parametrize("rows,physical,valid_per_tp", [(1, 8, (1, 0, 0, 0, 0, 0, 0, 0)), (4, 8, (1, 1, 1, 1, 0, 0, 0, 0)), (9, 16, (2, 2, 2, 2, 1, 0, 0, 0)), (16, 16, (2, 2, 2, 2, 2, 2, 2, 2))])
def test_real_catalog_padding_geometry_does_not_change_valid_queries(rows, physical, valid_per_tp):
    parsed = _padding_case(); assert parsed.model.layers == 93 and {layer.num_experts for layer in parsed.model.layer_plans if layer.ffn == "moe"} == {896}
    batch = _padding_views(rows); plan = select(parsed.runtime, parsed.model, batch, tokens=rows, global_tokens=physical * 4)
    assert plan.valid_global == plan.input_rows == rows and plan.physical_global == plan.tokens_global == physical
    assert plan.tokens_local == physical // 8 and plan.valid_per_tp == valid_per_tp and sum(plan.valid_per_tp) == rows and batch[0].q_len == rows and plan.cohort_tokens == physical * 4
def test_full_depth_graph_keeps_sp_padding_outside_recurrent_query_loop(full_body_graphs):
    parsed, by_query = full_body_graphs
    for q, graphs in by_query.items():
        assert set(graphs) == {0, 1, 2, 3}
        for ops in graphs.values():
            by_id = {op.op_id: op for op in ops}; pad = by_id["sp_shard"]; metadata = dict(pad.metadata)
            assert pad.mte2_bytes == q * parsed.model.hidden_size * 2 and pad.mte3_bytes == 8 * parsed.model.hidden_size * 2 and metadata["valid_global"] == q and metadata["physical_global"] == 8 and metadata["valid_per_tp"] == (1,) * q + (0,) * (8 - q)
            packed = by_id.get("s0.l0.linear_q") or next(op for op in ops if str(op.recipe).startswith("linear_qkvbfg_a"))
            norm_op = by_id.get("s0.l0.norm1") or by_id["s0.l0.attention.attn_res_fwd"]
            assert packed.local_shape[0] == q and by_id["s0.l0.recurrent_scan0"].local_shape[0] == q and norm_op.local_shape[0] == 1
            assert by_id["s0.l0.attention.sp_gather"].link_bytes == 7 * parsed.model.hidden_size * 2
            trim = by_id["sp_output_trim"]; assert trim.local_shape == (q, parsed.model.hidden_size) and trim.deps == ("sp_output_gather",) and trim.hbm_bytes == trim.link_bytes == 0
            assert {op.layer for op in ops if op.layer >= 0} == set(range(93))
def test_padding_executes_alltoall_router_rows_but_is_not_valid_token_count(full_body_graphs):
    _, by_query = full_body_graphs; contracts = {}
    for q, graphs in by_query.items():
        ops = graphs[0]; grouped = [op for op in ops if op.family == "grouped_gemm"]
        for op in grouped:  # per-invocation routing contract
            routing = dict(op.metadata)["routing_realization"]
            assert routing["global_tokens"] == 4 * 8 and sum(routing["global_expert_rows"]) == 4 * 8 * 16 and routing["valid_local_tokens"] == q
            assert routing["padded_local_tokens"] == 8 and routing["source_partitions"] == 1 and sum(routing["rank_assignment_rows"]) == 4 * 8 * 16
        dispatch = next(op for op in ops if op.recipe == "expert_dispatch")
        # Each source physical rank has one actual padded router row.
        assert dispatch.local_shape == (16, 3584) and dispatch.link_bytes == 16 * 3584 * 2 * 31 / 32
        contracts[q] = [(op.recipe, op.flops, op.mte2_bytes, op.mte3_bytes) for op in grouped]
    assert contracts[1] == contracts[4]
def test_unequal_and_idle_dp_keep_per_replica_padding_and_inverse_traffic():
    parsed = _padding_case(); populations = (1, 4, 9, 0); groups = tuple(_padding_views(n, f"r{i}") for i, n in enumerate(populations)); assert cohort_padded_tokens(groups, parsed.runtime) == 8 + 8 + 16; graphs = {}
    for replica, group in enumerate(groups):
        ops, _ = Frontend().body(parsed.model, parsed.runtime, group, ready="input", tokens=populations[replica], mode="decode", global_tokens=28, machine=parsed.machine); graphs[replica] = lower_implementations(tuple(op for op in ops if op.layer in (0, 1) or op.op_id == "sp_shard"), parsed.runtime, parsed.model.activation_dtype, resource_fusion=False)
    graphs = bind_cohort_traffic(parsed, graphs); row_bytes = 16 * 3584 * 2 * 31 / 32
    for replica, ops in graphs.items():
        assert next(op for op in ops if op.recipe == "expert_dispatch").link_bytes == (1, 1, 2, 0)[replica] * row_bytes; routing = dict(next(op for op in ops if op.family == "grouped_gemm").metadata)["routing_realization"]
        assert routing["global_tokens"] == 32 and routing["valid_local_tokens"] == populations[replica] and routing["padded_local_tokens"] == (8, 8, 16, 0)[replica] and sum(routing["global_expert_rows"]) == 512
        traffic = dict(next(op for op in ops if dict(op.metadata).get("reverse_of_collective")).metadata)["peer_traffic"]; assert traffic["direction"] == "transpose" and traffic["source_bytes"] == [row_bytes] * 16 + [2 * row_bytes] * 8 + [0] * 8
    assert not [op for op in graphs[3] if dict(op.metadata).get("sequence_parallel_role")]
    report = graph_costs(parsed, graphs, external_inputs={"input": 7})
    assert report["duration_ms"] > 7 and report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0 and min(row["ready_ms"] for row in report["timeline"]) >= 7
def test_graph_capture_rows_sp_rows_and_cohort_padding_are_separate_from_valid_rows():
    parsed = _padding_case(graph_bucket=4); plan = select(parsed.runtime, parsed.model, _padding_views(1), tokens=4, global_tokens=32)
    assert (plan.valid_global, plan.input_rows, plan.physical_global, plan.tokens_local) == (1, 4, 8, 1)
    assert cohort_padded_tokens((_padding_views(1), _padding_views(9), ()), parsed.runtime) == 8 + 16
    pad = input_shard(plan=plan, op_id="pad", deps=("input",), stage=0, layer=-1, requests=("r",), phase="decode", width=7168, dtype="bf16")
    assert pad.mte2_bytes == 4 * 7168 * 2 and pad.mte3_bytes == 8 * 7168 * 2 and pad.vector_ops == 4 * 7168
    # Padding describes the physical source tensor even when the caller chooses probability moments rather than an integer balanced scenario.
    parsed = replace(parsed, runtime=replace(parsed.runtime, routing={})); graphs = target_graphs(parsed, phase="decode", tokens=1, context=32, scope="body", resource_fusion=False); assert set(graphs) == {0, 1, 2, 3}
    for ops in graphs.values():
        pad = next(op for op in ops if op.op_id == "sp_shard")
        assert dict(pad.metadata)["cohort_tokens"] == 32 and dict(pad.metadata)["valid_global"] == 1 and next(op for op in ops if op.recipe == "expert_dispatch").local_shape[0] == 16
def test_real_sp_bucket_preserves_every_rank_and_valid_row():
    raw = json.loads((REPO_ROOT / "examples/cases/kimi-k3-a5-sp-right-pad-conditional.json").read_text()); case = parse_case(raw); counts = (10, 12, 14, 16)
    peers = {d: tuple(RequestView(f"{d}-{i}", 1, 128, False, "decode") for i in range(n)) for d, n in enumerate(counts)}
    ledger = rank_ledger_for(Frontend(), case.model, case.runtime, peers[0], 16, peers, 0, case.machine, 0)
    assert ledger is not None and ledger.capacity == 16 and ledger.allocated == (2,) * 32 and ledger.physical == 64 and sum(ledger.valid) == sum(counts)
@pytest.mark.parametrize("groups", GROUPS)
def test_sp_source_is_the_tp_coordinate_even_if_ep_groups_cross_dp(groups):
    from types import SimpleNamespace
    p = SimpleNamespace(tp=2, dp=2, pcp=1, pp=1, ep=2, rank_table=groups, pcp_lane_policy="parked"); runtime = SimpleNamespace(parallel=p, routing={}, kernels={"sequence_parallel": {"padding": "right_to_tp"}}); frontend = SimpleNamespace(graph=SimpleNamespace(select=lambda views, rt: (None, "none")))
    peers = {d: tuple(RequestView(f"{d}-{i}", 1, 128, False, "decode") for i in range(n)) for d, n in enumerate((4, 8))}
    for group in groups:
        rank = group[0]; d = rank // 2; ledger = rank_ledger_for(frontend, None, runtime, peers[d], len(peers[d]), peers, d, None, rank)
        expected = tuple((2, 4)[r // 2] for r in group); assert ledger is not None and ledger.valid == ledger.allocated == expected and ledger.physical == sum(expected)
