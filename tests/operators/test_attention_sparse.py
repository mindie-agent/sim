"""Sparse attention selection, gather accounting, and cache consumers.

[Role] Check indexed MLA, packed gathers, and source-specific sparse ABIs.
[Invariants] Conserve issued accesses, bytes, work, and per-query ownership.
[Timeouts & Locks] Bounded CPU graphs and local retained case reads.
[Side-Effect Policy] No device context or persistent writes.
"""

from dataclasses import replace
import json
from math import ceil
from types import SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.builders import dense_case
from _support.captures import REPO_ROOT
from _support.hardware import sku_case
from mindie_sim import compose_case, graph_costs, target_graphs, target_operators
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes
from mindie_sim.prediction.call_builders import build_aclnn_call
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy, gqa
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.operators.mla.generic_latent_attention import owned_store_rows, sparse_over_parts
from mindie_sim.prediction.operators.context_parallel.graph import PACK_LSE_OPS
from mindie_sim.prediction.local_projection_tp import _local_dimension, _packed_local_width
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import attention as governed
from mindie_sim.prediction.operators.attention.common import MINDIE_V1_ABI, MINDIE_V2_ABI, MINDIE_V3_ABI
from mindie_sim.prediction.program import AddressRun, AddressRunPlan, AdmissionMemoryView, MemoryAccessPattern, RecipeContext, ReuseKey
from mindie_sim.prediction.operators.sparse_indexer.sparse_mla_metadata import IMPLEMENTATION as METADATA_RECIPE, NATIVE
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.compute_arch import NativeComputeMode
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView

REGBASE_SKU = REPRESENTATIVE_SKUS["a5"]
VECTOR_SKU = REPRESENTATIVE_SKUS["a3"]

pytestmark = pytest.mark.tier0


def geometry():
    return Geometry.from_devices(sku_case(REGBASE_SKU).machine.dies)


def stages(plan):
    return {p.name: p for p in plan.program.phases}


@pytest.mark.parametrize("head_dim,v_head_dim,expected", ((128, 32, (32768, 8192)), (32, 128, (8192, 32768))))
def test_standalone_fused_attention_conserves_each_matrix_product(head_dim, v_head_dim, expected):
    call = build_aclnn_call({"api": "aclnnFusedInferAttentionScore", "heads": 4,
                            "query_tokens": 1, "key_tokens": 32, "head_dim": head_dim, "v_head_dim": v_head_dim})
    op, = call.ops
    recipe = call.recipes.select(op)
    for variant in (op, replace(op, metadata=tuple((k, v) for k, v in op.metadata if k != "attention_geometry"))):
        ctx = RecipeContext(variant, {"cube": op.flops}, {"cube": "cube"}, {"cube": 1.0}, 1)
        work = {stage.name: stage.work.get("cube", 0) for stage in recipe.lower(ctx).stages}
        assert (work["qk"], work["pv"]) == expected and sum(work.values()) == op.flops


@pytest.mark.parametrize("mode,row,sector_count", [(1, 656, 21), (5, 656, 21), (3, 386, 13)])
def test_packed_gather_preserves_independent_row_accesses(mode, row, sector_count):
    gather = governed.c8_gather_accounting(quant_mode=mode, selected_tokens=2048)
    assert gather.rows == 2048 and gather.row_bytes == row
    assert gather.logical_bytes == 2048 * row
    assert gather.sector_bytes == gather.dram_sector_bytes == 2048 * sector_count * 32
    assert gather.l2_hit_bytes == 0
    if row == 656:
        assert (gather.logical_bytes, gather.sector_bytes) == (1_343_488, 1_376_256)
    for offset in (0, 1, 16, 17, 31):
        ledger = governed.c8_gather_accounting(((offset,),), quant_mode=mode)
        assert ledger.dram_sector_bytes == ceil(((offset % 32) + row) / 32) * 32


@pytest.mark.parametrize("mode,row", [(1, 656), (3, 386)])
def test_contiguous_rows_share_sectors_and_verify_retains_repeated_accesses(mode, row):
    sectors = ceil(row / 32)
    contiguous = governed.c8_gather_accounting(((0, row),), quant_mode=mode)
    assert contiguous.rows == 2 and contiguous.sector_bytes == ceil(2 * row / 32) * 32
    assert contiguous.aligned_transfer_bytes == 2 * sectors * 32
    k_spec = 3
    selections = ((0, sectors * 32, 0),) * (1 + k_spec)
    cold = governed.c8_gather_accounting(selections, quant_mode=mode)
    warm = governed.c8_gather_accounting(selections, quant_mode=mode, l2_capacity_bytes=32 * 1024)
    assert cold.rows == warm.rows == (1 + k_spec) * 3
    assert cold.logical_bytes == warm.logical_bytes == (1 + k_spec) * 3 * row
    issued = (1 + k_spec) * 3 * sectors * 32
    assert cold.sector_bytes == warm.sector_bytes == cold.dram_sector_bytes == issued
    assert cold.aligned_transfer_bytes == warm.aligned_transfer_bytes == issued
    assert warm.dram_sector_bytes == 3 * sectors * 32 and warm.l2_hit_bytes == issued - warm.dram_sector_bytes
    op = governed.make_kv_quant_sfa_c8_op(op_id="verify", query_tokens=1 + k_spec, num_heads=8,
        rope_dim=64, quant_mode=mode, selection_lists=selections)
    if mode == 3:
        with pytest.raises(UnmodeledCapabilityError, match="attention_arch35_tq4_unsupported"):
            governed.build(op, geometry())
        return
    plan = governed.build(op, geometry())
    phase = stages(plan)
    assert plan.metadata["kv_logical_bytes"] == 12 * row and plan.metadata["kv_gather_rows"] == 12
    assert phase["kv_dequant"].after == ("packed_kv_gather",) and "kv_to_l1" in phase["qk"].after
    assert plan.work.matrix == {"bf16": 2 * 12 * 8 * (576 + 512)}


@pytest.mark.parametrize("mode", [0, 1, 5, 6])
@pytest.mark.parametrize("block", [8, 16])
def test_block_sparse_modes_and_metadata_reuse_conserve_compute(mode, block):
    geom = geometry()
    opts = dict(op_id="gbsa", query_tokens=2, num_heads=4, num_kv_heads=2, head_dim=128,
                v_head_dim=64, selected_tokens=17, block_size=block, quant_mode=mode)
    op = governed.make_gbsa_op(**opts)
    plan = governed.build(op, geom)
    changed = stages(governed.build(governed.make_gbsa_op(**opts, ssmax=True, metadata_reuse=True), geom))
    phase = stages(plan)
    count = ceil(17 / block) * 2 * 2
    assert phase["sparse_index"].work.resources["hbm"] == count * 4
    assert changed["sparse_index"].work.resources["hbm"] == 0
    assert phase["offset_workspace_write"].work.resources["hbm"] == count * 2 * 8
    assert changed["online_softmax"].work.resources["vector"] - phase["online_softmax"].work.resources["vector"] == 2 * 17 * 2 * 4
    assert ("key_dequant" in phase) == ("value_dequant" in phase) == (mode in {1, 5})
    assert plan.work.matrix == {("fp8_e4m3" if mode == 6 else "bf16"): 2 * 17 * 2 * 4 * (128 + 64)}
    bound = governed.recipes()[op.recipe].build(op, geom)
    assert bound.work == plan.work and bound.pipeline_spec == plan.pipeline_spec


@pytest.mark.parametrize("abi", [MINDIE_V1_ABI, MINDIE_V2_ABI, MINDIE_V3_ABI])
def test_mindie_sparse_abi_preserves_source_and_precision_boundary(abi):
    geom = geometry()
    contract = governed.AttentionContract(2, 4, 2, 128, 64, selected_tokens=17, kv_layout="TND", abi=abi)
    op = governed.make_attention_op(contract, op_id="mindie")
    plan = governed.build(op, geom)
    assert plan.pipeline_spec.implementation == plan.metadata["abi"] == abi
    assert plan.work.matrix == {"bf16": 2 * 2 * 17 * 4 * (128 + 64)}
    fp8 = governed.make_attention_op(replace(contract, quant_mode=1), op_id="mindie")
    if abi == MINDIE_V1_ABI:
        with pytest.raises(UnmodeledCapabilityError, match="supported layout and storage format"):
            governed.build(fp8, geom)
    else:
        assert governed.build(fp8, geom).work.matrix == {"fp8_e4m3": op.flops}
    with pytest.raises(UnmodeledCapabilityError, match="consistent ABI"):
        governed.build(replace(op, recipe=governed.GBSA_RECIPE), geom)


def latent_case(rope):
    return {
        "model": {"catalog_key": "glm-5.3", "quantization": "w4a8", "layers": 1,
                  "layer_overrides": {"0": {"qk_rope_head_dim": rope}}},
        "machine": {"backend": "ascend", "sku": VECTOR_SKU if rope else REGBASE_SKU,
                    "num_devices": 16 if rope else 8},
        "runtime": {"tp": 16 if rope else 8, "kernels": {}},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
    }


def metadata_case():
    raw = latent_case(0)
    raw["runtime"]["kernels"]["lowerings"] = {"sparse_mla_metadata": {
        "implementation": METADATA_RECIPE, "options": {"layer_groups": [[0]]}}}
    return raw


def selected_graph(raw, batch):
    graph = target_graphs(raw, phase='decode', tokens=4, sequences=batch,
                          context=4096, scope='body')[0]
    metadata = next(op for op in graph if op.recipe == METADATA_RECIPE)
    publish = next(op for op in graph if op.recipe == 'sparse_mla_metadata_publish')
    attention = next(op for op in graph if op.family == 'attention')
    assert len([op for op in graph if op.recipe == METADATA_RECIPE]) == 1
    assert publish.deps == (metadata.op_id,)
    assert publish.op_id in attention.deps
    return metadata, publish, attention


@pytest.mark.parametrize('batch', (1, 8))
def test_one_cpu_invocation_per_batch_is_a_real_body_attention_dependency(batch):
    raw = metadata_case()
    nodes = selected_graph(raw, batch)
    meta = dict(nodes[0].metadata)
    assert meta['execution_domain'] == 'AI_CPU'
    assert meta['metadata_call']['rows'] == batch * 4
    assert not any(op.vector_ops or op.flops for op in nodes[:2])
    raw['runtime']['kernels']['service_regimes'] = [{
        'id': 'typed-native-control', 'match': {'kernel': NATIVE},
        'response': {'fixed_us': 100}, 'provenance': {'kind': 'synthetic_control'}}]
    keys = {op.op_id for op in nodes}
    report = graph_costs(raw, nodes, isolated=False, input_ready_ms=7,
        external_inputs={d: 7 for op in nodes for d in op.deps if d not in keys})
    rows = {row['op_id']: row for row in report['timeline']}
    assert rows[nodes[0].op_id]['elapsed_ms'] == pytest.approx(.1)
    assert rows[nodes[2].op_id]['ready_ms'] >= rows[nodes[1].op_id]['complete_ms']
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert report['parameters']['service_regimes'][0]['queries'] == {'observable_response': 1}


def test_sparse_metadata_grouping_retains_padding_and_index_dependency():
    raw = metadata_case()
    raw["model"]["layers"] = 3
    raw["model"]["layer_overrides"] = {str(i): {"qk_rope_head_dim": 0} for i in range(3)}
    raw["runtime"].update(graph_mode="auto", graph_buckets=[16])
    options = raw["runtime"]["kernels"]["lowerings"]["sparse_mla_metadata"]["options"]
    options["layer_groups"] = [[0, 1, 2]]
    ops = target_graphs(raw, phase="decode", tokens=4, sequences=3, context=4096, scope="body")[0]
    (controller,) = [op for op in ops if op.recipe == METADATA_RECIPE]
    call = dict(controller.metadata)["metadata_call"]
    assert call["rows"] == 16 and call["valid_rows"] == 12
    (publish,) = [op for op in ops if op.recipe == "sparse_mla_metadata_publish"]
    assert publish.deps == (controller.op_id,)
    assert all(publish.op_id in op.deps for op in ops if op.family == "attention")
    options.update(layer_groups=[[0], [1], [2]], schedule="layer_ordered")
    ops = target_graphs(raw, phase="decode", tokens=4, sequences=3, context=4096, scope="body")[0]
    controllers = [op for op in ops if op.recipe == METADATA_RECIPE]
    assert len(controllers) == 3
    by_id = {op.op_id: op for op in ops}
    assert all(by_id[op.deps[0]].family == "indexer" for op in controllers)


def test_local_projection_pads_the_split_axis_to_the_weight_scale_block():
    quant = QuantSpec(name="block128", weight_format="fp8_e4m3",
                      activation_format="fp8_e4m3", weight_block=(128, 128))
    assert _local_dimension(512, 8, quant) == 128
    packed = {"packed_projection_slices": ({"shape": (2, 512)},)}
    assert _packed_local_width(packed, 8, 512, quant) == 64
    parallel = ParallelSpec(linear_proj_tp=8)
    term = WeightTerm("kda_q", 512, 256, "kda.q", "out_tp", quant=quant,
                      placement_role="linear_input")
    assert term_bytes(term, quant, parallel) == quant.matrix_bytes(64, 256, "kda.q")


def test_dense_noncausal_topk_keeps_the_smaller_window_cap():
    plan = SimpleNamespace(
        index_topk=16, compress_ratio=1, window=4, index_kpool=1,
        index_kpool_always_select_tail=False, noncausal_attention=True,
        cache_kind="dense")
    traffic = sparse_over_parts(
        ((2, 100),), plan, 128, dense=True, kv_len=102, topk=4)
    assert traffic.total_cache_rows == 2 * 4

def test_batched_context_parallel_selection_conserves_rank_local_work():
    case = dense_case()
    plan = replace(case.model.layer_plans[0], window=8, head_dim=8)
    views = (RequestView("a", 1, 0, False, "decode"), RequestView("b", 3, 11, False, "decode"))
    parallel = replace(case.runtime.parallel, cp=2, cp_plan="dcp")
    ops = gqa(prefix="probe", plan=plan, views=views, stage=0, deps=("input",),
              model=case.model, parallel=parallel, quant=case.model.quant, mode="decode",
              padded_tokens=4, policy=AttentionPolicy(), cp_interleave_size=4,
              runtime=case.runtime, cache_ids={}, stage_of=lambda _: 0).ops
    meta = dict(next(op for op in ops if op.family == "attention").metadata)
    assert meta["cp_per_rank_pairs"] == (13.0, 12.0)
    assert meta["cp_per_rank_sweeps"] == (5, 6) and meta["kv_sweeps"] == 6
    assert meta["cp_bottleneck_pairs_rank"] == 0 and meta["cp_bottleneck_sweeps_rank"] == 1
    sharded = ParallelSpec(tp=8, cp=4, pcp=2, cp_plan="pcp_dcp")
    hybrid = (SimpleNamespace(context=0, phase="prefill", q_len=8),
              SimpleNamespace(context=100, phase="decode", q_len=1))
    assert owned_store_rows([8, 1], hybrid, sharded, 1536) == 9

def test_sparse_prefill_frontend_accounts_independent_selections_and_resident_kv():
    raw = latent_case(64)
    op = next(op for op in target_operators(raw, phase="prefill", tokens=4096)
              if op.recipe == "latent_attention0")
    meta = dict(op.metadata)
    rows = sum(min(q, 2048) for q in range(1, 4097))
    assert meta["kv_sweeps"] == meta["indexed_gather_rows"] == meta["logical_pairs"] == rows
    traffic = meta["attention_traffic_plan"]
    assert traffic["logical_read_bytes"] == rows * 576 * 2
    assert traffic["logical_read_bytes"] == traffic["hbm_read_bytes"] + traffic["l2_hit_bytes"]
    assert traffic["private_refill_bytes"] == rows * 576 * 2
    assert op.mte3_bytes == 4096 * 4 * 512 * 2 and op.flops == 2 * rows * 4 * (576 + 512)

def test_paired_sparse_decode_keeps_rank_owned_index_and_lse_pack():
    config = json.loads((REPO_ROOT / "mindie_sim/model_configs/kimi-k3.1.json").read_text())
    quant = dict(name="packed", weight_format="bf16", activation_format="bf16", expert_weight_format="bf16",
                 kv_dtype="fp8_e4m3", index_dtype="fp8_e4m3", index_scale_dtype="fp16")
    raw = compose_case(config, hardware=REGBASE_SKU, tp=8, dp=4, ep=32, dcp=8, num_devices=32,
                       block_size=1536, max_model_len=140000, quant=quant, prompt_tokens=131072,
                       output_tokens=1, graph_capture_mode="none")
    ops = target_graphs(raw, phase="decode", tokens=1, context=100, scope="body", resource_fusion=False)[0]
    index = next(op for op in ops if op.family == "indexer_quant_cache")
    attention, pack, exchange = [next(op for op in ops if op.recipe == recipe)
        for recipe in ("paired_fp8_qsfa_analytic", "dcp_output_pack", "dcp_output_alltoall")]
    assert index.local_shape[0] == 1 and index.mte3_bytes == 128 + 2
    assert dict(attention.metadata)["qsfa_statistics_bytes"] == 96 * 2 * 4
    assert attention.mte3_bytes == pack.mte2_bytes == 96 * (512 * 2 + 2 * 4)
    assert pack.mte3_bytes == 96 * (512 + 4) * 2 and pack.vector_ops == 96 * PACK_LSE_OPS
    assert exchange.link_bytes == 12 * (512 + 4) * 2 * (8 - 1)


@pytest.mark.parametrize("selection,cold_reads", [((0, 0), (672, 672)), ((0, 656), (1312,))])
def test_packed_gather_charges_cold_runs_and_reuses_them_across_queries(selection, cold_reads):
    geom = geometry()
    op = governed.make_kv_quant_sfa_c8_op(
        op_id="gather", query_tokens=1, num_heads=1, selection_lists=(selection,), kv_tensor_id="cache",
    )
    plan = governed.build(op, geom)
    phases = stages(plan)
    gather = phases["packed_kv_gather"]
    cold_bytes = sum(cold_reads)
    assert gather.work.resources["mte2_aiv"] == 1344
    assert gather.work.resources["hbm"] == plan.metadata["kv_sector_bytes"] == cold_bytes
    other_hbm = sum(phase.work.resources.get("hbm", 0) for phase in phases.values() if phase is not gather)
    assert plan.work.resources["hbm"] == other_hbm + cold_bytes
    assert plan.metadata["kv_logical_bytes"] == 1312

    engine, context = engine_binding(op, geom)
    program = plan.lower(context)
    cold = program.start(engine, "cold", engine.submit("input"))
    engine.advance(until=(cold,))
    assert engine.serviced["hbm"] == pytest.approx(plan.work.resources["hbm"])
    warm = program.start(engine, "warm", cold)
    engine.advance(until=(warm,))
    assert engine.serviced["hbm"] == pytest.approx(plan.work.resources["hbm"] + other_hbm)
    reads = [row["demand"] for row in engine.trace if ":packed_kv_gather:program:read" in row["task"]]
    assert tuple(row["hbm"] for row in reads[:len(cold_reads)]) == cold_reads
    assert [row["l2"] for row in reads[len(cold_reads):]] == [1344 // len(cold_reads)] * len(cold_reads)
    audit = verify_operator(sku_case(REGBASE_SKU), op, governed.build)
    assert audit["status"] == "passed"


def test_verify_geometry_without_addresses_has_no_residency_credit():
    op = governed.make_kv_quant_sfa_c8_op(
        op_id="verify", query_tokens=4, num_heads=1, selected_tokens=2048, batch_size=1, shared_prefix=True,
    )
    plan = governed.build(op, geometry())
    gather = stages(plan)["packed_kv_gather"]
    assert plan.metadata["l2_shared_kv_bytes"] == gather.work.resources["l2"] == 0
    assert gather.work.resources["hbm"] == gather.work.resources["mte2_aiv"] == 4 * 2048 * 672
    assert not governed.c8_gather_accounting(query_tokens=4, selected_tokens=2048,
                                            l2_capacity_bytes=32 * 1024**2).l2_hit_bytes
    audit = verify_operator(sku_case(REGBASE_SKU), op, governed.build)
    assert audit["status"] == "passed"


def engine_binding(op, geom, trace_limit=128):
    rates = {**{role: rate / 1000 for role, rate in geom.rates.items()}, "cube": 1}
    context = RecipeContext(op, {}, {role: role for role in rates}, rates, geom.compute_units,
                            resource_units=geom.resource_units, compute_peaks=geom.compute_peaks)
    engine = Engine([Resource(role, rate, units=geom.resource_units[role]) for role, rate in rates.items()],
                    trace_limit=trace_limit)
    return engine, context

def test_admission_rebinds_packed_reads_by_tensor_and_producer_version():
    geom = geometry()
    arguments = dict(op_id="decode", query_tokens=1, num_heads=1, selection_lists=((0,),),
                     kv_tensor_id="layer-cache", kv_version=7)
    op = governed.make_kv_quant_sfa_c8_op(**arguments)
    engine, context = engine_binding(op, geom)
    program = governed.build(op, geom).lower(context)
    ready = engine.submit("ready", release=5)
    first = program.start(engine, "first", ready)
    second = program.start(engine, "second", first)
    changed = governed.make_kv_quant_sfa_c8_op(**{**arguments, "kv_version": 8})
    third = governed.build(changed, geom).lower(context).start(engine, "third", second)
    other = governed.make_kv_quant_sfa_c8_op(**{**arguments, "kv_tensor_id": "other-cache"})
    fourth = governed.build(other, geom).lower(context).start(engine, "fourth", third)
    fifth = program.start(engine, "old-version", fourth)
    engine.advance(until=(fifth,))

    reads = [row for row in engine.trace if row["task"].endswith(":read:program") and "packed_kv_gather" in row["task"]]
    assert len(reads) == 5
    assert all(row["demand"]["mte2_aiv"] == 672 for row in reads)
    assert [row["demand"].get("hbm", 0) for row in reads] == [672, 0, 672, 672, 672]
    assert [row["demand"].get("l2", 0) for row in reads] == [0, 672, 0, 0, 0]
    assert reads[0]["start_ms"] >= 5 and reads[1]["start_ms"] >= first.time
    assert engine.delivered["mte2_aiv"] == 5 * 672 and engine.delivered["l2"] == 672

def test_packed_split_waits_for_each_owners_dma_and_conserves_rows():
    geom = geometry()
    contract = governed.AttentionContract(
        1, 1, 1, 576, 512, abi=governed.C8_ABI, quant_mode=5, splits=2,
        kv_layout="PA_BLK_BSND", selection_lists=((0, 1344),), kv_tensor_id="cache",
    )
    op = governed.make_attention_op(contract, op_id="decode")
    engine, context = engine_binding(op, geom)
    plan = governed.build(op, geom)
    end = plan.lower(context).start(engine, "decode", engine.submit("input", release=3))
    engine.advance(until=(end,))
    suffix = ":packed_kv_gather:program:read:program"
    reads = sorted((row for row in engine.trace if row["task"].endswith(suffix)),
                   key=lambda row: row["placement"]["mte2_aiv"])
    assert len(reads) == 2
    assert len({tuple(row["placement"]["mte2_aiv"]) for row in reads}) == 2
    split_ends = []
    for read in reads:
        assert read["demand"]["hbm"] == read["demand"]["mte2_aiv"] == 672
        branch = read["task"].removesuffix(suffix)
        names = ("kv_dequant", "kv_to_l1", "qk", "pv", "split_write")
        dequant, transfer, qk, pv, write = steps = [
            next(row for row in engine.trace if row["task"] == f"{branch}:{name}") for name in names
        ]
        assert all(left["end_ms"] <= right["start_ms"] for left, right in zip((read, *steps[:-1]), steps, strict=True))
        assert dequant["demand"]["vector"] == 1024 and transfer["demand"]["mte3"] == 1152
        assert qk["demand"]["mte1"] == 2304 and pv["demand"]["mte1"] == 1026
        split_ends.append(write["end_ms"])
    merge_read = next(row for row in engine.trace if row["task"].endswith(":split_read"))
    assert max(split_ends) <= merge_read["start_ms"]
    assert plan.work.matrix == {"bf16": 2 * 2 * (576 + 512)}
    assert {role: amount for role, amount in engine.serviced.items() if amount} == pytest.approx(context.bind(plan.work))

    # Warming only the first offset identifies which physical row each split reads.
    engine, context = engine_binding(op, geom)
    warm = governed.make_attention_op(replace(contract, splits=1, selection_lists=((0,),)), op_id="warm")
    ready = governed.build(warm, geom).lower(context).start(engine, "warm", engine.submit("input"))
    end = plan.lower(context).start(engine, "decode", ready)
    engine.advance(until=(end,))
    rereads = {row["task"]: row["demand"] for row in engine.trace
               if row["task"].startswith("decode:") and row["task"].endswith(suffix)}
    assert rereads[reads[0]["task"]].get("l2") == 672
    assert rereads[reads[1]["task"]].get("hbm") == 672


@pytest.mark.parametrize("selections", [(), ((0,), (1344,))])
def test_shared_prefix_needs_matching_physical_addresses(selections):
    geom = geometry()
    op = governed.make_kv_quant_sfa_c8_op(
        op_id="decode", query_tokens=2, num_heads=1, selected_tokens=1, shared_prefix=True,
        selection_lists=selections, kv_tensor_id="cache",
    )
    engine, context = engine_binding(op, geom)
    end = governed.build(op, geom).lower(context).start(engine, "decode", engine.submit("input"))
    engine.advance(until=(end,))
    reads = [row["demand"] for row in engine.trace if ":packed_kv_gather" in row["task"] and row["demand"]]
    assert sum(read["hbm"] for read in reads) == 1344
    assert sum(read.get("l2", 0) for read in reads) == 0

def test_packed_residency_starts_after_the_dma_completes():
    geom = geometry()
    op = governed.make_kv_quant_sfa_c8_op(
        op_id="decode", query_tokens=2, num_heads=1, selection_lists=((0,), (0,)), kv_tensor_id="cache",
    )
    engine, context = engine_binding(op, geom)
    program = governed.build(op, geom).lower(context)
    ready = engine.submit("input")
    first = program.start(engine, "first", ready)
    second = program.start(engine, "reuse", first)
    engine.advance(until=(second,))
    reads = [row for row in engine.trace if row["task"].endswith(":packed_kv_gather:program:read:program")]
    assert len(reads) == 4
    assert reads[0]["start_ms"] < reads[1]["end_ms"] and reads[1]["start_ms"] < reads[0]["end_ms"]
    assert [row["demand"].get("hbm", 0) for row in reads] == [672, 672, 0, 0]
    assert [row["demand"].get("l2", 0) for row in reads] == [0, 0, 672, 672]

def test_packed_read_reuses_shared_die_cache_without_requests():
    die, row_bytes = 1, 672
    view = AdmissionMemoryView({die: 4 * row_bytes})
    geom = replace(geometry(), rank=die, admission_memory_view=view)
    key = ReuseKey(die, "layer-cache", 0)
    pattern = MemoryAccessPattern("layer-cache", 0, die, 0, "read", "mte2_aiv", row_bytes,
                                  AddressRunPlan((AddressRun(0, 1, row_bytes, row_bytes),)), key)
    assert view.admit_access(pattern, row_bytes) == (0, row_bytes)
    op = governed.make_kv_quant_sfa_c8_op(
        op_id="decode", query_tokens=1, num_heads=1, selection_lists=((0,),), kv_tensor_id="layer-cache",
    )
    engine, context = engine_binding(op, geom)
    done = governed.build(op, geom).lower(context).start(engine, "decode", engine.submit("input"))
    engine.advance(until=(done,))
    assert engine.delivered["l2"] == row_bytes and engine.delivered.get("hbm", 0) == 0


@pytest.mark.parametrize("known_page", [False, True])
def test_paged_attention_reuses_only_proven_backing_addresses(known_page):
    view = AdmissionMemoryView({0: 1024**2})
    geom = Geometry.from_devices(sku_case(VECTOR_SKU).machine.dies)
    geom = replace(geom, admission_memory_view=view, rates={**geom.rates, "l2": geom.rates["hbm"]},
                   resource_units={**geom.resource_units, "l2": 1})
    contract = governed.AttentionContract(1, 1, 1, 128, 128, selected_tokens=32, kv_layout="PA_BLK_BSND",
                                          kv_tensor_id="layer-cache", block_table=((0,),) if known_page else ())
    op = governed.make_attention_op(contract, op_id="decode")
    engine, context = engine_binding(op, geom)
    ready = engine.submit("input")
    for name in ("first", "second"):
        request_geom = replace(geom, requests=(RequestView(name, 1, 31, True, "decode"),))
        ready = governed.build(op, request_geom).lower(context).start(engine, name, ready)
        engine.advance(until=(ready,))
    reads = [row["demand"] for row in engine.trace if ":k_panel_" in row["task"] and row["demand"]]
    assert len(reads) == 2 and reads[0]["hbm"] > 0
    if known_page:
        assert reads[1].get("l2", 0) > 0 and reads[1].get("hbm", 0) == 0
    else:
        assert reads[1] == reads[0]


@pytest.mark.parametrize("distinct_allocations", [False, True])
@pytest.mark.parametrize("capacity,hit", [(1024**2, 672), (1024, 0)])
def test_packed_residency_evicts_by_bytes(capacity, hit, distinct_allocations):
    base = geometry()
    geom = replace(base, buffers={**base.buffers, "l2": capacity})

    def request(i):
        return governed.make_kv_quant_sfa_c8_op(
            op_id="decode", query_tokens=1, num_heads=1,
            selection_lists=((0 if distinct_allocations else i * 672,),),
            kv_tensor_id=f"allocation-{i}" if distinct_allocations else "allocation",
        )

    first = request(0)
    engine, context = engine_binding(first, geom, trace_limit=4096)
    done = engine.submit("input")
    for i in range(129):
        op = request(i)
        done = governed.build(op, geom).lower(context).start(engine, f"row-{i}", done)
        engine.advance(until=(done,))
    done = governed.build(first, geom).lower(context).start(engine, "reread", done)
    engine.advance(until=(done,))
    reads = [row["demand"] for row in engine.trace if row["task"].endswith(":packed_kv_gather:program:read:program")]
    assert len(reads) == 130
    assert sum(row["hbm"] for row in reads[:-1]) == 86_688
    assert sum(row.get("l2", 0) for row in reads[:-1]) == 0
    assert reads[-1].get("l2", 0) == hit and reads[-1].get("hbm", 0) == 672 - hit


def test_packed_residency_evicts_oldest_sectors_within_one_allocation():
    base = geometry()
    geom = replace(base, buffers={**base.buffers, "l2": 1024**2})
    row_bytes, warm_rows = 672, 1560

    def request(offsets):
        return governed.make_kv_quant_sfa_c8_op(
            op_id="decode", query_tokens=1, num_heads=1, selection_lists=(offsets,), kv_tensor_id="cache",
        )

    warm = request(tuple(range(row_bytes, (warm_rows + 1) * row_bytes, row_bytes)))
    engine, context = engine_binding(warm, geom)
    ready = governed.build(warm, geom).lower(context).start(engine, "warm", engine.submit("input"))
    engine.advance(until=(ready,))
    assert engine.delivered["hbm"] == pytest.approx(warm_rows * row_bytes)
    # Hitting row 2 must also refresh it before the next fill evicts from the warm range.
    for i, row in enumerate((0, 0, 2, warm_rows + 1, 2, 3)):
        op = request((row * row_bytes,))
        ready = governed.build(op, geom).lower(context).start(engine, f"row-{i}", ready)
    engine.advance(until=(ready,))
    suffix = ":packed_kv_gather:program:read:program"
    rereads = [row["demand"] for row in engine.trace if row["task"].startswith("row-") and row["task"].endswith(suffix)]
    assert [(row.get("l2", 0), row.get("hbm", 0)) for row in rereads] == [
        (0, 672), (672, 0), (672, 0), (0, 672), (672, 0), (256, 416),
    ]


@pytest.mark.parametrize("heads,block,selected,q_scale,kv_scale", [(2, 128, 129, 8, 16), (1, 384, 384, 4, 12)])
def test_mindie_v3_fp8_consumes_fp32_scales_and_rejects_unknown_mode(heads, block, selected, q_scale, kv_scale):
    contract = governed.AttentionContract(1, heads, 1, 128, 128, selected_tokens=selected, kv_layout="BNSD",
                                          abi=MINDIE_V3_ABI, block_size=block, quant_mode=1)
    op = governed.make_attention_op(contract, op_id="block-sparse")
    plan = governed.build(op, geometry())
    phase = stages(plan)
    assert plan.metadata["scale_dtype"] == "fp32"
    assert plan.metadata["scale_bytes"] == {"query": q_scale, "key": kv_scale, "value": kv_scale}
    for tensor, consumer, size in (("query", "qk", q_scale), ("key", "qk", kv_scale), ("value", "pv", kv_scale)):
        scale = f"{tensor}_scale_load"
        assert phase[scale].work.resources["hbm"] == phase[scale].work.resources["mte2_aic"] == size
        assert scale in phase[consumer].after
    assert verify_operator(sku_case(REGBASE_SKU), op, governed.build)["status"] == "passed"
    with pytest.raises(UnmodeledCapabilityError, match="quantization 5"):
        replace(contract, quant_mode=5)

@pytest.mark.parametrize("mode,dst_type_max", [(2, 0.0), (3, 4.0)])
def test_mindie_v3_binds_packed_operands_and_eight_bit_scale_planes(mode, dst_type_max):
    contract = governed.AttentionContract(2, 4, 2, 128, 128, selected_tokens=64, kv_layout="BNSD", abi=MINDIE_V3_ABI,
                                          block_size=64, query_block_size=128, quant_mode=mode, dst_type_max=dst_type_max)
    op = governed.make_attention_op(contract, op_id="block-sparse")
    plan = governed.build(op, geometry())
    phase = stages(plan)
    assert (plan.metadata["quant_mode"], plan.metadata["dst_type_max"], plan.metadata["block_shape"]) == (mode, dst_type_max, (128, 64))
    assert plan.metadata["packed_dtype"] == "fp4_e2m1" and plan.metadata["scale_dtype"] == "fp8_e8m0"
    q_rows, kv_rows = 2 * 4, 2 * 64 * 2
    assert plan.metadata["operand_bytes"] == {"query": q_rows * 64, "key": kv_rows * 64, "value": kv_rows * 64}
    assert plan.metadata["scale_bytes"] == {"query": q_rows * 4, "key": kv_rows * 4, "value": 2 * 2 * 128 * 2}
    for tensor, compute in (("query", "qk"), ("key", "qk"), ("value", "pv")):
        load, scale = f"{tensor}_load", f"{tensor}_scale_load"
        assert phase[load].work.resources["hbm"] == plan.metadata["operand_bytes"][tensor]
        assert phase[scale].work.resources["hbm"] == plan.metadata["scale_bytes"][tensor]
        assert {load, scale} <= set(phase[compute].after)
    assert plan.work.matrix == {"mxfp4": 2 * 2 * 64 * 4 * (128 + 128)} and phase["output"].work.resources["hbm"] == q_rows * 128 * 2
    case = sku_case(REGBASE_SKU)
    (die,) = case.machine.dies
    native = NativeComputeMode("mxfp4_x_mxfp4_acc_fp32", "matrix", ("mxfp4",), "cube", ("fp32",))
    die = replace(die, compute_arch=replace(die.compute_arch, native_modes=(*die.compute_arch.native_modes, native)),
                  compute_peaks=(*die.compute_peaks, ("mxfp4", 1e12)))
    assert verify_operator(replace(case, machine=replace(case.machine, dies=(die,))), op, governed.build)["status"] == "passed"
    for change, message in (({"return_lse": True}, "no LSE"), ({"block_size": 32}, "64-multiple"),
                            ({"query_block_size": 32}, "64-multiple"), ({"abi": MINDIE_V2_ABI}, "V3")):
        with pytest.raises(UnmodeledCapabilityError, match=message):
            governed.build(governed.make_attention_op(replace(contract, **change), op_id="bad"), geometry())

def test_c8_gather_l2_admission_uses_unique_sector_union_not_duplicate_copy_bytes():
    accounting = governed.c8_gather_accounting(query_tokens=2, selected_tokens=1, selection_lists=((0, 0), (0,)), l2_capacity_bytes=1024)
    assert accounting.aligned_transfer_bytes == 3 * 672 and accounting.l2_hit_bytes == 672 and accounting.dram_sector_bytes == 2 * 672
