"""Sparse index scoring, placement and producer contracts.

[Role] Exercise governed scoring ABIs and the finite matrix reader.
[Invariants] Logical work, physical transfers and slot ownership are conserved.
[Timeouts & Locks] Bounded CPU-only simulation with no external locks.
[Side-Effect Policy] No runtime device access or file writes.
"""
from collections import Counter
from dataclasses import replace
import pickle

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.hardware import hardware_case, run, sku_case
from mindie_sim import graph_costs, operator_costs, target_operators
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction import scheduling
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.implementations import produced_values
from mindie_sim.prediction.operators.matrix.matrix import TiledMatmul
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import mla_prolog, sparse_indexer as scorer
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelinePhase, PipelineProgram,
    PipelineTemplate, PipelineTemplateProgram, RecipeBook, RecipeContext, Stage,
)
from mindie_sim.prediction.streams import StreamCoreBudget, StreamLane, StreamPlan, bind_stream_owners
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

MEMBASE = REPRESENTATIVE_SKUS["a3"]
REGBASE = REPRESENTATIVE_SKUS["a5"]
LIGHTNING = "aclnnLightningIndexerV2"
QLI = "aclnnQuantLightningIndexerV2"
VLLM = "VllmQuantLightningIndexer"
VLLM_EXPORT = "vllm_quant_lightning_indexer_v2"
DSL = "tilelang_dsl.quant_sparse_lightning_indexer"
MSA = "aclnnMsaIndexScore"
MSA_SELECT = "aclnnMsaIndexScoreV2"
REGION = "npu_indexer_region_score"


def _geometry(hardware=REGBASE):
    return Geometry.from_devices(sku_case(hardware).machine.dies)


def _context(op, geometry):
    roles = set(geometry.resource_units) | {"cube"}
    return RecipeContext(op, {}, {role: role for role in roles}, {}, geometry.compute_units,
                         resource_units=geometry.resource_units,
                         compute_peaks=dict.fromkeys(dict(op.matrix_flops), 1),
                         pool_names={pool: pool for pool in geometry.buffers})


def _stages(program):
    if isinstance(program, ParallelProgram):
        for child in program.programs:
            yield from _stages(child)
        return
    stages = (program.stages if isinstance(program, KernelProgram) else
              (stage for i in range(program.count) for stage in program.tile(i)))
    for stage in stages:
        yield stage
        if stage.program is not None:
            yield from _stages(stage.program)


def _conserved(plan, op, geometry=None):
    context = _context(op, geometry if geometry is not None else _geometry())
    stages = tuple(_stages(plan.program(context)))
    issued = Counter()
    for stage in stages:
        issued.update(stage.work)
    assert +issued == pytest.approx(context.bind(plan.work))
    phases = Counter()
    for phase in plan.pipeline_spec.phases:
        phases.update({(item.resource, item.dtype): item.amount for item in phase.work})
    assert {dtype: amount for (role, dtype), amount in phases.items() if role == "cube"} == plan.work.matrix
    assert {role: amount for (role, _), amount in phases.items() if role != "cube"} == plan.work.resources
    return stages


def _scorer(abi, rows=417, **options):
    fields = dict(query_tokens=1, scanned_rows=rows, index_heads=4, index_dim=128, topk_items=128)
    return scorer.make_indexer_op(abi=abi, **(fields | options))


def _verified(case, op):
    report = verify_operator(case, op, scorer.build, input_ready_ms=7)
    assert report["status"] == "passed"
    assert report["result"]["declared_service"] == pytest.approx(report["result"]["serviced"])
    return report["result"]


@pytest.mark.parametrize("hardware,abi,cache_dtype,receivers", [
    (MEMBASE, LIGHTNING, "bf16", (14,)),
    (REGBASE, QLI, "int8", (14, 15)),
])
@pytest.mark.parametrize("lane_cores", (1, 2))
def test_disjoint_lane_preserves_physical_matrix_and_vector_owners(hardware, abi, cache_dtype, receivers, lane_cores):
    geometry = _geometry(hardware)
    stream = StreamPlan(0, (
        StreamLane("main", StreamCoreBudget(aic=7, aiv=14, isolation="disjoint")),
        StreamLane("score", StreamCoreBudget(aic=lane_cores, aiv=2 * lane_cores, isolation="disjoint")),
    ))
    owners = bind_stream_owners(stream, geometry.datapath)["score"]
    op = replace(_scorer(abi, rows=65, cache_dtype=cache_dtype), stream="score")
    placed = replace(geometry, stream_owners=owners)
    plan = scorer.build(op, placed)
    stages = {stage.name: stage for stage in _conserved(plan, op, placed)}
    assert stages["cube_scores"].placement["cube"] == (7,)
    assert stages["rectification"].placement["vector"] == receivers
    assert stages["key_load"].placement["mte2_aic"] == (7,)
    assert stages["index_store"].placement["mte3"] == receivers
    result = run(sku_case(hardware), (op,), scorer.recipes(), stream_plans=(stream,), audit_allocations=True)
    assert result["execution"]["work"]["flops"] == op.flops
    if cache_dtype == "int8":
        handoff = stages["score_handoff"]
        assert tuple(a.units for a in handoff.acquire) == ((14,), (15,))
        peak = result["execution"]["peak_allocated_per_unit"]
        assert {key for key, value in peak.items() if key.startswith("ub@") and value} == {
            "ub@r0:s0:unit14", "ub@r0:s0:unit15",
        }


def test_paired_handoff_rejects_lane_without_both_reachable_receivers():
    geometry = _geometry()
    stream = StreamPlan(0, (
        StreamLane("main", StreamCoreBudget(aic=7, aiv=14, isolation="disjoint")),
        StreamLane("score", StreamCoreBudget(aic=1, aiv=1, isolation="disjoint")),
    ))
    owners = bind_stream_owners(stream, geometry.datapath)["score"]
    with pytest.raises(UnmodeledCapabilityError, match="missing_owners"):
        scorer.build(_scorer(QLI, cache_dtype="int8"), replace(geometry, stream_owners=owners))


@pytest.mark.parametrize("abi", (QLI, VLLM, VLLM_EXPORT))
@pytest.mark.parametrize("rows,columns", [(1, 8), (64, 64), (65, 128), (128, 128)])
def test_quant_fixpipe_aligns_m_and_switches_nd_width(abi, rows, columns):
    op = _scorer(abi, rows=rows, cache_dtype="int8", index_heads=64, topk_items=16)
    plan = scorer.build(op, _geometry())
    handoff, = (stage for stage in _conserved(plan, op) if stage.name == "score_handoff")
    assert plan.work.matrix == {"int8": 2 * rows * 64 * 128}
    assert handoff.work["fixpipe"] == plan.work.resources["fixpipe"] == 128 * columns * 4
    assert [alloc.size for alloc in handoff.acquire] == [65536, 65536]
    if rows == 65:
        assert plan.work.resources["fixpipe"] == 65536
        _verified(sku_case(REGBASE), op)


@pytest.mark.parametrize("queries,batches,topk,padded_queries", [
    (2, 1, 64, (2,)), (2, 2, 64, (2, 2)), (5, 1, 64, (4, 2)),
    (5, 2, 64, (4, 2)), (4, 1, 8192, (2, 2)),
])
def test_quant_fixpipe_aligns_live_queries_inside_each_request(queries, batches, topk, padded_queries):
    op = _scorer(QLI, rows=65, cache_dtype="int8", index_heads=64,
                 query_tokens=queries, batch_size=batches, topk_items=topk)
    plan = scorer.build(op, _geometry())
    transfers = [stage.work["fixpipe"] for stage in _conserved(plan, op) if stage.name == "score_handoff"]
    assert sorted(transfers) == sorted(rows * 64 * 128 * 4 for rows in padded_queries)
    assert plan.work.resources["fixpipe"] == sum(transfers)
    assert plan.work.matrix == {"int8": 2 * queries * 64 * 65 * 128}
    result = _verified(sku_case(REGBASE), op)
    assert result["serviced"]["fixpipe@r0:s0"] == pytest.approx(sum(transfers))
    if queries == 2 and batches == 1:
        assert plan.work.resources["fixpipe"] == 65536


def test_quant_ub_checks_both_source_sized_parity_slots():
    op = _scorer(QLI, rows=65, cache_dtype="int8", index_heads=64, topk_items=16)
    geometry = _geometry()

    def with_ub(capacity):
        buffers = tuple((name, capacity if name == "ub" else size) for name, size in geometry.datapath.buffers)
        return replace(geometry, datapath=replace(geometry.datapath, buffers=buffers))

    needed = 2 * 65536
    assert scorer.build(op, with_ub(needed)).work.resources["fixpipe"] == 65536
    with pytest.raises(UnmodeledCapabilityError, match="score_slot_capacity"):
        scorer.build(op, with_ub(needed - 1))


def test_quant_fixpipe_ping_pong_releases_each_slot_before_reuse():
    op = _scorer(QLI, cache_dtype="int8")
    plan = scorer.build(op, _geometry())
    tiles = next(s.program for s in _conserved(plan, op) if s.name == "score_tiles")
    assert (tiles.count, tiles.slots) == (4, 2)
    for i, columns in enumerate((128, 128, 128, 40)):
        by_name = {stage.name: stage for stage in tiles.tile(i)}
        handoff = by_name["score_handoff"]
        slot = i % 2
        release = by_name[f"score_release_{slot}"]
        assert handoff.metadata == {
            "route": "l0c->ub", "score_bytes": 2 * 4 * columns * 4,
            "score_slot": slot, "score_layout": "row_major",
            "dual_dst_ctl": 1, "dual_destination": True,
        }
        assert handoff.previous == (release.name,)
        assert by_name["score_publish"].after == (handoff.name,)
        assert by_name["rectification"].after == ("score_publish",)
        assert release.after == ("head_reduction",)
        assert release.free == handoff.acquire and len({item.units for item in handoff.acquire}) == 2
    result = run(sku_case(REGBASE), (op,), scorer.recipes(), audit_allocations=True)
    slots = {}
    for row in result["allocation_trace"]:
        if "index_score_" in row["name"]:
            slots.setdefault(row["name"], []).append(row["action"])
    assert len(slots) == 4
    assert all(events == ["acquire", "release"] * 2 for events in slots.values())


@pytest.mark.parametrize("abi,tiling,trunk,repository", [
    (QLI, "cann_qli_v2_tiler", 16384, "cann/ops-transformer"),
    (VLLM, "vllm_candidate_tiler", 8704, "vllm-project/vllm-ascend"),
])
def test_quant_caller_owns_its_tiler_and_source(abi, tiling, trunk, repository):
    op = _scorer(abi, rows=17408, cache_dtype="int8")
    plan = scorer.build(op, _geometry())
    assert plan.metadata["tiling"] == tiling
    assert {entry.repository for entry in plan.pipeline_spec.sources} == {repository}
    stage, = (s for s in _conserved(plan, op) if s.name == "qli_radix_topk")
    assert (stage.metadata["trunk_items"], stage.metadata["trunks"]) == (trunk, 2)


@pytest.mark.parametrize("abi", (LIGHTNING, QLI, VLLM, VLLM_EXPORT, "npu_quant_lightning_indexer_v2"))
def test_weighted_v2_contract_rejects_missing_weight_input(abi):
    with pytest.raises(ModelingError, match="requires_head_weights"):
        _scorer(abi, has_head_weights=False)
    op = _scorer(abi)
    metadata = dict(op.metadata)
    metadata["indexer_contract"]["has_head_weights"] = False
    with pytest.raises(ModelingError, match="requires_head_weights"):
        scorer.build(replace(op, metadata=tuple(metadata.items())), _geometry())


@pytest.mark.parametrize("hardware", (MEMBASE, REGBASE))
def test_qsa_native_weights_and_weight_free_overlay_have_distinct_sources(hardware):
    fields = dict(op_id="scorer", query_tokens=2, context_tokens=260, index_heads=4)
    native = scorer.make_lightning_indexer_qsa_op(**fields)
    overlay = scorer.make_lightning_indexer_qsa_op(**fields, has_head_weights=False)
    geometry = _geometry(hardware)
    weighted, unweighted = (scorer.build(op, geometry) for op in (native, overlay))
    for op, plan, repository, path in (
        (native, weighted, "vllm-project/vllm-ascend", "vllm_ascend/models/qwen4_exp/lightning_indexer.py"),
        (overlay, unweighted, "mindie-agent/sim", "mindie_sim/stacks/vllm_ascend/manifest.json"),
    ):
        source, = plan.pipeline_spec.sources
        assert (source.repository, source.relative_path) == (repository, path)
        assert len(bytes.fromhex(source.revision)) == 20
        assert len(bytes.fromhex(source.source_sha256)) == 32
        assert dict(op.metadata)["native_source"] == source.to_dict()
        restored = pickle.loads(pickle.dumps(plan))
        assert restored == plan
        _verified(sku_case(hardware), op)
    assert weighted.work.matrix == unweighted.work.matrix == {"bf16": 2 * 2 * 65 * 4 * 128}
    stages = _conserved(weighted, native, geometry)
    weights = 2 * 4 * 4
    assert sum(s.metadata.get("head_weight_bytes", 0) for s in stages) == weights
    assert sum(s.work["vector"] for s in stages if s.name == "weighting") == 2 * 65 * 4
    for role in ("hbm", "mte2_aiv"):
        assert weighted.work.resources[role] - unweighted.work.resources.get(role, 0) == weights
    assert not any(s.metadata.get("head_weight_bytes") or s.name == "weighting"
                   for s in _conserved(unweighted, overlay, geometry))


@pytest.mark.parametrize("context,groups,tail", [(259, 64, 3), (3, 0, 3), (256, 64, 0)])
def test_qsa_scores_complete_groups_and_packs_residual_tokens(context, groups, tail):
    op = scorer.make_lightning_indexer_qsa_op(
        op_id="scorer", query_tokens=1, context_tokens=context, compress_ratio=4, block_topk=128,
    )
    contract = dict(op.metadata)["indexer_contract"]
    assert (contract["scanned_rows"], contract["residual_tokens"]) == (groups, tail)
    plan = scorer.build(op, _geometry())
    assert plan.metadata["residual_tokens"] == tail
    assert op.flops == plan.work.matrix.get("bf16", 0) == 2 * groups * 32 * 128
    pack, = (s for s in _conserved(plan, op) if s.name == "index_pack")
    assert pack.after == ("index_store",)
    assert pack.metadata == {"valid_groups": groups, "residual_tokens": tail}
    assert pack.work.get("mte2_aiv", 0) == groups * 4
    assert pack.work["mte3"] == (128 + 4 - 1) * 4
    _verified(sku_case(REGBASE), op)


@pytest.mark.parametrize("cache_dtype", ("bf16", "mxfp4"))
@pytest.mark.parametrize("choice", (0, 1, 2))
def test_long_blocks_keep_independent_head_maxima_and_topk_ports(cache_dtype, choice):
    op = scorer.make_msa_index_score_op(op_id="blocks", query_tokens=1, context_tokens=129,
                                        index_heads=16, block_size=128, topk_blocks=16,
                                        cache_dtype=cache_dtype, return_choice=choice)
    plan = scorer.build(op, _geometry())
    row_bytes = 256 if cache_dtype == "bf16" else 128 // 2 + 128 // 32
    assert plan.work.matrix == {cache_dtype: 2 * 129 * 16 * 128}
    stages = _conserved(plan, op)
    assert sum(s.work["hbm"] for s in stages if s.name == "key_load") == 129 * row_bytes
    assert not any(s.name in {"head_maxima", "head_reduction", "rectification"} for s in stages)
    assert sum(s.work.get("vector", 0) for s in stages if s.name == "block_maxima") == 16 * (129 - 2)
    score_writes = [s for s in stages if s.name == "score_row_store"]
    assert sum(s.metadata["valid_items_per_head"] for s in score_writes) == 2
    assert all(s.metadata["heads"] == s.metadata["score_stride"] == 16 for s in score_writes)
    by_phase = Counter()
    for stage in stages:
        by_phase[stage.name] += stage.work.get("hbm", 0)
    assert by_phase["score_row_store"] == 1024
    assert by_phase["score_row_load"] == (16 * 8 * 4 if choice else 0)
    assert by_phase["index_store"] == (1024 if choice else 0)
    assert by_phase["values_store"] == (1024 if choice == 2 else 0)
    assert plan.metadata.get("score_workspace_bytes", 0) == (1024 if choice else 0)
    if choice:
        topk, = (s for s in stages if s.name == "score_row_topk")
        assert (topk.metadata["heads"], topk.metadata["valid_items_per_head"]) == (16, 2)
        assert topk.after == ("score_row_load",)
    outputs = produced_values(op)
    expected = [("output", "int32" if choice else "fp32")]
    if choice == 2:
        expected.append(("values", "fp32"))
    assert [(v["output"], v["dtype"]) for v in outputs] == expected
    assert all(v["logical_shape"] == [16, 1, 16] and v["write_bytes"] == 1024 for v in outputs)
    if cache_dtype == "mxfp4":
        # The profile has no MXFP4 rate. Use unit-rate service to check work
        # conservation without assigning throughput to the silicon profile.
        context = _context(op, _geometry())
        demand = context.bind(plan.work)
        engine = Engine([Resource(role, 1, units=context.resource_units.get(role, 1)) for role in demand])
        done = plan.program(context).start(engine, "selection", engine.submit("ready", latency=7))
        engine.advance(until=(done,))
        assert done.time > 7 and engine.serviced == pytest.approx(demand)
    else:
        _verified(sku_case(REGBASE), op)


@pytest.mark.parametrize("size", (16, 128))
def test_block_score_stride_and_head_pooling_are_independent_of_s2_partitions(size):
    contract = scorer.SparseIndexerContract(3, 129, 16, 128, abi=MSA, block_size=size,
                                            return_choice=0, s2_splits=2)
    op = scorer.make_indexer_op(contract)
    stages = _conserved(scorer.build(op, _geometry()), op)
    assert sum(s.work.get("vector", 0) for s in stages if s.name == "head_maxima") == (
        3 * 129 * (16 - 1) if size == 16 else 0)
    heads = 1 if size == 16 else 16
    assert op.padded_shape == (heads, 3, 16)
    assert sum(s.work.get("mte3", 0) for s in stages) == heads * 3 * 16 * 4
    _verified(sku_case(REGBASE), op)


@pytest.mark.parametrize("choice", (-1, 3, True, 1.0))
def test_msa_rejects_invalid_return_choices(choice):
    with pytest.raises(ModelingError, match="invalid_return_choice"):
        scorer.make_msa_index_score_op(op_id="blocks", query_tokens=1, context_tokens=129, return_choice=choice)


@pytest.mark.parametrize("choice", (1, 2))
def test_pooled_heads_do_not_offer_fused_selection(choice):
    with pytest.raises(ModelingError, match="block16_disallows_fused_topk"):
        scorer.make_msa_index_score_op(op_id="blocks", query_tokens=1, context_tokens=129,
                                       block_size=16, return_choice=choice)


def test_radix_histogram_uses_fixed_trunks_and_retains_candidates():
    op = _scorer(DSL, rows=16385, cache_dtype="fp8_e4m3")
    stage, = (s for s in _conserved(scorer.build(op, _geometry()), op) if s.name == "radix_histogram_topk")
    assert stage.metadata == {"trunks": 2, "trunk_items": 16384, "histogram_bins": 256,
                              "histogram_bytes": 2 * 256, "counter_dtype": "uint16",
                              "pair_bytes": 16385 * 8}
    assert stage.work["vector"] == 2 * (16385 + 256 + 2 * 256)


@pytest.mark.parametrize("abi", (QLI, VLLM, DSL, MSA, MSA_SELECT, REGION))
def test_membase_datapath_rejects_register_only_abis(abi):
    with pytest.raises(UnmodeledCapabilityError, match="arch22_abi"):
        scorer.build(_scorer(abi), _geometry(MEMBASE))


@pytest.mark.parametrize("rows,context", [(1, 0), (3, 0), (4, 0), (257, 1), (512, 4096), (8192, 122880)])
@pytest.mark.parametrize("ratio", (2, 4, 8))
def test_compressed_causal_partitions_conserve_matrix_and_selection_work(rows, context, ratio):
    lengths = Counter((context + q + 1) // ratio for q in range(rows))
    ops = [_scorer(LIGHTNING, count, query_tokens=queries) for count, queries in lengths.items()]
    expected_pairs = sum((context + q + 1) // ratio for q in range(rows))
    assert sum(op.flops for op in ops) == 2 * expected_pairs * 4 * 128
    for op in ops[:1] + ops[-1:]:
        plan = scorer.build(op, _geometry())
        assert plan.work.matrix.get("bf16", 0) == op.flops
        assert plan.work.resources["mte3"] == op.local_shape[0] * 128 * 4


@pytest.mark.parametrize("changes", [{"cache_dtype": "int8"}, {"cache_dtype": "fp8_e4m3"}])
def test_membase_lightning_rejects_unmodeled_score_representations(changes):
    with pytest.raises(UnmodeledCapabilityError, match="arch22_abi"):
        scorer.build(_scorer(LIGHTNING, **changes), _geometry(MEMBASE))


def test_quant_source_is_causal_and_services_both_operand_pipes():
    op = replace(_scorer(QLI, 4097, cache_dtype="int8"), deps=("query", "cache", "weights"))
    case = sku_case(REGBASE)
    result = graph_costs(case, (op,), recipes=RecipeBook(scorer.recipes()), input_ready_ms=7,
                         external_inputs=dict.fromkeys(op.deps, 7), isolated=False)
    assert result["complete_ms"] > 7 and all(row["ready_ms"] >= 7 for row in result["timeline"])
    prepared, plan = scorer.recipes()[op.recipe].prepare(op, _geometry())
    assert result["execution"]["work"]["hbm_bytes"] == prepared.hbm_bytes
    service = result["execution"]["serviced"]
    for role in ("mte2_aic", "mte2_aiv", "fixpipe", "mte3"):
        assert service[f"{role}@r0:s0"] == pytest.approx(plan.work.resources[role])
    for role in ("mte2_aic", "mte2_aiv"):
        slow = sku_case(REGBASE, device={"rates": {role: 1e6}})
        assert _verified(slow, replace(op, deps=()))["duration_ms"] > result["duration_ms"]


def test_quant_weight_preparation_is_reused_across_s2_tiles():
    for rows in (65, 513):
        op = _scorer(QLI, rows, cache_dtype="int8", index_heads=64, topk_items=16)
        plan = scorer.build(op, _geometry())
        stages = _conserved(plan, op)
        for name in ("head_weight_load", "query_scale_load"):
            load, = (stage for stage in stages if stage.name == name)
            assert load.work["mte2_aiv"] == 64 * 2
        prescale, = (stage for stage in stages if stage.name == "prescale_head_weights")
        assert prescale.work["vector"] == 64
        assert sum(stage.work["mte2_aiv"] for stage in stages if stage.name == "key_scale_load") == rows * 2


@pytest.mark.parametrize("queries", (1, 2, 3))
def test_empty_quant_scan_only_services_sentinel_outputs(queries):
    op = _scorer(QLI, 0, cache_dtype="int8", query_tokens=queries)
    plan = scorer.build(op, _geometry())
    assert not plan.work.matrix
    assert plan.work.resources == {"vector": queries * 128, "hbm": queries * 128 * 4,
                                   "mte3": queries * 128 * 4}
    _verified(sku_case(REGBASE), op)


def test_packed_prolog_cache_retains_group_scale_and_rope_payload():
    op = mla_prolog.make_op(op_id="projection", tokens=8, hidden_size=4096, num_heads=8,
                            q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, tile_size=128)
    contract = mla_prolog.validate_contract(op)
    assert contract.kv_row_bytes == 512 + 2 * 64 + 4 * (512 // 128)
    plan = mla_prolog.build(op, _geometry())
    absorbed_flops = 2 * 8 * 8 * contract.qk_nope_head_dim * 512
    assert dict(op.matrix_flops)["bf16"] == absorbed_flops
    assert plan.work.matrix["bf16"] >= absorbed_flops
    assert verify_operator(sku_case(REGBASE), op, mla_prolog.build)["status"] == "passed"
    with pytest.raises(UnmodeledCapabilityError, match="unsupported_abi_or_weight_mode"):
        mla_prolog.build(op, _geometry(MEMBASE))


def _partition_case():
    raw = hardware_case()
    raw["machine"]["num_devices"] = 8
    raw["model"] = {"catalog_key": "GLM-5.3-W4A8", "layers": 1}
    raw["runtime"].update(tp=8, graph_mode="eager", kernels={"lowerings": {
        "native_fusion": {"implementation": "declared_native_fusion",
                          "options": {"roles": ["lightning_indexer"]}}}})
    device = raw["machine"]["device"]
    device["rates"]["native_call_service"] = 1e9
    device["resource_units"] = {"native_call_service": 1}
    device["work_units"] = {"native_call_service": "busy_ns"}
    return raw

def _partition_op(ops):
    return next(op for op in ops if dict(op.metadata).get("fusion_role") == "lightning_indexer")

def test_native_service_matches_sequence_partition_and_active_rows():
    raw = _partition_case()
    ops = [_partition_op(target_operators(raw, phase="prefill", tokens=q, context=c, sequences=s))
           for q, c, s in ((8, 0, 1), (4, 2, 2), (2, 3, 4))]
    view = dict(ops[0].metadata)["fusion_tensor_view"]
    assert {op.local_shape for op in ops} == {ops[0].local_shape}
    assert {op.flops for op in ops} == {2 * 36 * view["heads"] * view["width"]}
    raw["runtime"]["kernels"]["service_regimes"] = [{
        "id": "one-sequence", "match": {"recipe": "declared_native_fusion",
                                        "metadata.fusion_tensor_view.sequence_partition": view["sequence_partition"]},
        "execution": {"kind": "joint_native", "resource": "native_call_service",
                      "reference_rate": 1e9, "native_kernel": "LightningIndexer"},
        "response": {"fixed_us": 123},
    }]
    rows = [operator_costs(raw, (op,))["operators"][0] for op in ops]
    assert rows[0]["program"] == "JointNativeService" and rows[0]["duration_ms"] == .123
    assert all(row["program"] != "JointNativeService" for row in rows[1:])
    raw["runtime"].update(graph_mode="full", graph_buckets=[4], graph_capture_keys=[4])
    op = _partition_op(target_operators(raw, phase="decode", tokens=1, context=8, sequences=3))
    view = dict(op.metadata)["fusion_tensor_view"]
    assert (view["tokens"], view["physical_tokens"], op.local_shape[0]) == (3, 4, 3)
    assert view["sequence_partition"] == ({"query": 1, "context": 8, "kv_length": 9},) * 3

def test_partition_order_and_request_renaming():
    case = parse_case(_partition_case())
    def lower(requests):
        work = Frontend()(IterationPlan(requests), case.model, case.runtime, case.machine)
        return dict(_partition_op(work.ops).metadata)["fusion_tensor_view"]
    requests = (RequestView("first", 2, 7, False, "prefill"), RequestView("second", 6, 1, False, "prefill"))
    view = lower(requests)
    assert view == lower(tuple(replace(v, request_id=str(i)) for i, v in enumerate(requests)))
    assert view["sequence_partition"] == ({"query": 2, "context": 7, "kv_length": 9},
                                          {"query": 6, "context": 1, "kv_length": 7})
    assert lower(requests[::-1])["sequence_partition"] == view["sequence_partition"][::-1]


def _matrix_context(m=64, n=64, k=160):
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(), m=m, n=n, k=k,
                 weight_count=1, quant=QuantSpec("bf16"), category="dense_mlp", phase="prefill")
    rates = {"cube": 128, "hbm": 1024, "mte2": 16, "mte1": 1024, "mte3": 1024}
    pools = {"l1": 4096, "l0c": 4096}
    work = {"cube": op.flops, "hbm": op.hbm_bytes, "mte2": op.mte2_bytes, "mte3": op.mte3_bytes}
    return RecipeContext(op, work, {r: r for r in rates}, rates, 2,
                         pool_names={p: p for p in pools}, pool_capacities=pools,
                         pool_units=dict.fromkeys(pools, 2),
                         resource_units={r: 1 if r == "hbm" else 2 for r in rates})
@pytest.mark.parametrize("slots", (1, 2, 3))
@pytest.mark.parametrize("shape,panels", [((64, 64, 160), 20), ((128, 128, 160), 80), ((70, 33, 17), 6)])
def test_indexed_reader_matches_expanded_matrix_program(shape, panels, slots):
    ctx = _matrix_context(*shape)
    def execute(indexed):
        recipe = TiledMatmul(32, 32, 32, slots=slots, template_execution=indexed)
        program = recipe.lower(ctx)
        cores = next(stage.program for stage in program.stages if stage.name == "cores")
        engine = Engine([Resource(r, rate, units=ctx.resource_units[r]) for r, rate in ctx.rates.items()],
                        pools=ctx.pool_capacities, pool_units=ctx.pool_units)
        done = program.start(engine, "matrix", engine.submit("ready"))
        engine.advance(until=(done,))
        return (done.time, engine.serviced, engine.peak_allocated, engine.peak_unit_allocated), cores
    detailed, expanded = execute(False)
    indexed, cores = execute(True)
    assert indexed == detailed
    templates = [item.template for item in cores.programs]
    assert sum(template.indexed_panels for template in templates) == panels
    if shape[2] == 160:
        assert tuple(sum(getattr(t, field) for t in templates)
                     for field in ("phase_objects", "stage_objects", "tile_factory_calls")) == (6, 18, 6)
        for owner, (template, pipe) in enumerate(zip(templates, expanded.programs, strict=True)):
            reference = pipe.template(owner=owner, panels_per_output=5, k_tail=False)
            for i in range(template.tiles):
                phase, end, wave, panel, slot = template.index_of(i)
                assert (wave, panel, slot) == (i // 5, i % 5, i % slots)
                assert end == (i + 1 if panel in (0, 4) else wave * 5 + 4)
                assert template.stage_for(i) == template.phases[phase].stages_for(i)
                for actual, expected in zip(template.stage_for(i), reference.stage_for(i), strict=True):
                    assert replace(actual, acquire=(), free=()) == replace(expected, acquire=(), free=())
                    footprint = lambda items: [(a.pool, a.size, a.units) for a in items]
                    assert footprint(actual.acquire) == footprint(expected.acquire)
                    assert footprint(actual.free) == footprint(expected.free)

def test_indexed_reader_previous_and_current_panel_edges(monkeypatch):
    captured = {}
    submit = Stage.submit
    def record(self, engine, name, predecessors, scoped, owner, prepared=None):
        captured[name] = tuple(event.name for event in predecessors)
        return submit(self, engine, name, predecessors, scoped, owner, prepared=prepared)
    monkeypatch.setattr(Stage, "submit", record)
    body = (Stage("producer", {"r": 1}), Stage("consumer", previous=("producer",)),
            Stage("local", after=("producer",)))
    template = PipelineTemplate(0, 2, (PipelinePhase("body", body, 1),),
                                indexed_panels=2, index_of=lambda i: (0, i + 1, 0, i, i % 2))
    engine = Engine([Resource("r", 1)])
    done = scheduling.start_pipeline_template(engine, template, "p", engine.submit("ready"))
    engine.advance(until=(done,))
    assert captured["p:tile1:consumer"] == ("p:tile0:producer", "p:tile0:consumer")
    assert captured["p:tile1:local"] == ("p:tile1:producer", "p:tile0:local")

def test_indexed_reader_isolates_allocation_identity_and_drains_empty_catalog():
    allocation = Allocation("w", "p", 1, (0,))
    work = Stage("work", {"r": 1}, acquire=(allocation,))
    retire = Stage("retire", {"r": 1}, free=(allocation,), after=("work",))
    for tiles in (0, 1):
        engine = Engine([Resource("r", 2)], pools={"p": 2}, pool_units={"p": 1}, trace_limit=16)
        ready = engine.submit("ready")
        events = []
        for owner in range(2):
            scope = "output" if tiles else "invocation"
            phases = (PipelinePhase("body", (work, retire), 1),) if tiles else ()
            edges = {} if tiles else dict(setup=(work,), drain=(replace(retire, after=()),), setup_ms=2)
            template = PipelineTemplate(owner, 1, phases, indexed_panels=tiles,
                index_of=lambda i: (0, 1, 0, 0, 0), bindings=({"name": "w", "scope": scope, "owner": owner},), **edges)
            events.append(scheduling.start_pipeline_template(engine, template, "same", ready, owner="same"))
        engine.advance(until=tuple(events), first=False)
        assert events[0].time == events[1].time == (2 if tiles else 4)
        assert engine.serviced["r"] == 4 and engine.allocated["p"] == 0
        assert engine.peak_allocated["p"] == 2
        names = {row["name"] for row in engine.allocation_trace}
        assert all(any(f"owner{i}:" in name for name in names) for i in range(2))

def test_indexed_reader_requires_capability_and_preserves_ordered_reader(monkeypatch):
    template = TiledMatmul(32, 32, 32, slots=2).finite_templates(_matrix_context())[0]
    engine = Engine([Resource("r", 1)])
    ready = engine.submit("ready")
    monkeypatch.delattr(scheduling.start_pipeline_template, "indexed_panels_abi")
    with pytest.raises(ModelingError, match="pipeline_indexed_consumer_unavailable"):
        PipelineTemplateProgram(template).start(engine, "rejected", ready)
    assert engine.summary()["live_tasks"] == 1 and engine.summary()["live_programs"] == 0
    ordered = PipelineProgram(8, 2, lambda i: (Stage("load", {"r": 1}),
                              Stage("compute", {"r": 1}, after=("load",))), homogeneous=True).template()
    done = scheduling.start_pipeline_template(engine, ordered, "ordered", ready)
    engine.advance(until=(done,))
    assert done.time == engine.serviced["r"] == 16


def test_gqa_indexer_replicates_kv_heads_under_tp_and_shards_under_dcp():
    from dataclasses import fields
    from types import SimpleNamespace
    from _support.builders import dense_case
    from mindie_sim.prediction.operators.sparse_indexer.gqa_indexer import lower_gqa_indexer
    from mindie_sim.schema.parallel import ParallelSpec

    base = dense_case(layers=1)
    base_plan = replace(base.model.layer_plans[0], num_heads=16, num_kv_heads=2, head_dim=128,
                        indexer=True, owns_index_k=True, owns_topk=True,
                        index_heads=8, index_dim=64, index_topk=256)
    plan = SimpleNamespace(**{f.name: getattr(base_plan, f.name) for f in fields(base_plan)}, index_kv_heads=2)
    model = replace(base.model, num_heads=16, num_kv_heads=2, head_dim=128, layer_plans=(base_plan,))
    views = (RequestView("r0", 1, 1024, False, "decode"),)

    def lower(tp=1, cp=1, sharded=False):
        ops = []
        rt = replace(base.runtime, metadata=(("dcp_indexer_sharded", sharded),))
        lower_gqa_indexer(
            ops, prefix="idx", plan=plan, views=views, stage=0, deps="x",
            model=model, parallel=ParallelSpec(tp=tp, cp=cp), quant=model.quant,
            mode="decode", padded_tokens=1, valid=1, pcp_prefill=False,
            runtime=rt, machine=sku_case(REGBASE).machine,
        )
        return {op.op_id.rsplit(".", 1)[-1]: op for op in ops}

    tp1, tp2, tp8 = lower(tp=1), lower(tp=2), lower(tp=8)
    assert tp1["index_wk"].local_shape[1] == 2 * 64
    assert tp2["index_wk"].local_shape[1] == 1 * 64
    assert tp8["index_wk"].local_shape[1] == 1 * 64
    assert tp1["index_store"].mte3_bytes > tp2["index_store"].mte3_bytes == tp8["index_store"].mte3_bytes > 0

    cp_rep = lower(tp=1, cp=4, sharded=False)
    cp_sharded = lower(tp=1, cp=4, sharded=True)
    assert cp_sharded["index_scores"].local_shape[0] < cp_rep["index_scores"].local_shape[0]


def test_index_cache_writer_enforces_int8_default_fp8_guard_and_l2_slot_hbm():
    from mindie_sim.prediction.operators.sparse_indexer import family_quant_indexer
    from mindie_sim.prediction.operators.sparse_indexer.index_cache_writer import (
        build_index_cache_writer,
        emit_index_cache_write,
    )

    a3_machine = sku_case(MEMBASE).machine
    a3_geom = _geometry(MEMBASE)
    ops = emit_index_cache_write(rows=4, dim=128, machine=a3_machine)
    assert len(ops) == 1
    meta = dict(ops[0].metadata)
    assert ops[0].dtype == "int8" and meta["payload_dtype"] == "int8"
    plan_no_l2 = build_index_cache_writer(ops[0], a3_geom)
    assert ops[0].hbm_bytes <= plan_no_l2.work.resources["hbm"]
    l2_geom = replace(a3_geom, rates={**dict(a3_geom.rates), "l2": 1e12})
    plan_with_l2 = build_index_cache_writer(ops[0], l2_geom)
    assert ops[0].hbm_bytes == pytest.approx(plan_with_l2.work.resources["hbm"])

    with pytest.raises(ModelingError, match=r"^index_cache_writer_fp8_unsupported_on_"):
        emit_index_cache_write(rows=4, dim=128, cache_dtype="fp8_e4m3", machine=a3_machine)

    dummy_op = _scorer(QLI, rows=128, cache_dtype="fp8_e4m3")
    with pytest.raises(ModelingError, match=r"^quant_lightning_indexer_v2_.*_requires_"):
        family_quant_indexer._build_arch35(dummy_op, a3_geom)


def test_sparse_kv_offload_uses_packed_c8_row_bytes_read_role_and_exact_hbm():
    from mindie_sim import compose_case
    from mindie_sim.catalog.cache import packed_region_row_bytes
    from mindie_sim.catalog.cache_format import with_packed_mla_cache
    from mindie_sim.prediction.lowering import Builder
    from mindie_sim.prediction.operators.sparse_indexer import sparse_kv_offload

    raw = compose_case("deepseek-v3.2", hardware="a3", tp=1, dp=1, ep=1, num_devices=1,
                       layers=2, quant="int8", prompt_tokens=16, output_tokens=2)
    case = parse_case(raw)
    c8_model = with_packed_mla_cache(replace(case.model, quant=replace(case.model.quant, kv_dtype="int8")), 64)
    budget = sparse_kv_offload.sparse_kv_offload_memory_budget(
        c8_model, case.runtime, case.machine,
        options={"enabled": True, "layerwise_kv_offload": True},
    )
    assert budget["kv_row_bytes"] == int(packed_region_row_bytes(c8_model, "mla_kv", 576)) == 656

    plan = c8_model.layer_plans[0]
    views = (RequestView("r0", 4, 4096, False, "decode"),)
    b_pre = Builder("p", plan, 0, views, c8_model, "prefill")
    sparse_kv_offload.lower_prefill_offload(b_pre, plan=plan, views=views, store_id="kv", budget=budget)
    assert b_pre.ops[-1].hbm_bytes == pytest.approx(4 * 656)

    b_dec = Builder("d", plan, 0, views, c8_model, "decode")
    sparse_kv_offload.lower_decode_offload(b_dec, plan=plan, views=views, store_id="kv", topk_id="tk", budget=budget)
    d2h = next(op for op in b_dec.ops if op.recipe == "memfabric_sparse_copy_d2h")
    assert d2h.hbm_bytes == pytest.approx(4 * 656)

    single_mte2_geom = Geometry(1, {"hbm": 1, "mte2": 2, "mte3": 2, "vector": 2}, {"ub": 262144}, {"ub": 2})
    plan_d2h = sparse_kv_offload.recipes()["memfabric_sparse_copy_d2h"].build(d2h, single_mte2_geom)
    assert "mte2" in plan_d2h.work.resources


def test_bf16_indexer_weight_bytes_ub_head_scaling_and_governed_make_op_io():
    from mindie_sim import compose_case
    from mindie_sim.prediction.operators.sparse_indexer import _bf16_access, index_pool_native

    case = parse_case(compose_case("deepseek-v3.2", hardware="a5", tp=1, dp=1, ep=1, num_devices=1,
                                   layers=2, quant="bf16", prompt_tokens=16, output_tokens=2))
    plan = replace(case.model.layer_plans[0], index_heads=32, index_dim=128, index_kpool=4, index_topk=256)
    bf16_quant = replace(case.model.quant, index_dtype="bf16")
    views = (RequestView("r0", 1, 512, False, "decode"),)
    _, ledger = index_pool_native._pool_ledger(views, plan, bf16_quant, 1, 256, 256, 128)
    assert ledger["weight_bytes"] == float(32 * 128 * 2 + 32 * 4)

    p1 = _bf16_access._key_access_plan(
        ((2, 128),), compression=1, key_row_bytes=256, q_row_bytes=256, q_tile_rows=2,
        topk=32, kv_heads=1, aiv_units=2, ub_capacity=262144, l1_capacity=524288,
        l2_capacity=0, owner=0, identity_prefix="h1",
    )
    p2 = _bf16_access._key_access_plan(
        ((2, 128),), compression=1, key_row_bytes=256, q_row_bytes=256, q_tile_rows=2,
        topk=32, kv_heads=2, aiv_units=2, ub_capacity=262144, l1_capacity=524288,
        l2_capacity=0, owner=0, identity_prefix="h2",
    )
    assert p2["workspace_bytes"] > p1["workspace_bytes"]

    gov = _scorer(LIGHTNING, rows=256, cache_dtype="bf16")
    assert gov.hbm_bytes > 0 and gov.mte2_bytes > 0 and gov.mte3_bytes > 0 and gov.vector_ops > 0


def test_arch22_v1_lightning_indexer_charges_per_group_q_rows_on_heterogeneous_batch():
    from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
    from mindie_sim.prediction.operators.sharedkv.sparse_indexer_v1 import build_arch22_v1_lightning
    from mindie_sim.prediction.operators.sparse_indexer.common import SparseIndexerContract, make_op

    geom = Geometry.from_devices(sku_case(MEMBASE).machine.dies)
    heads, dim, topk = 4, 128, 128
    contract = SparseIndexerContract(
        abi="aclnnLightningIndexer",
        query_tokens=3,
        scanned_rows=8192,
        index_heads=heads,
        index_dim=dim,
        topk_items=topk,
        batch_size=2,
        cache_dtype="bf16",
        has_head_weights=True,
        fused_topk=True,
    )
    op = make_op(contract, op_id="idx_v1")
    req_views = (
        RequestView("long_decode", 1, 8192, False, "decode"),
        RequestView("short_mtp", 2, 510, False, "decode"),
    )
    planning_ctx = OperatorPlanningContext(
        occurrence=None,
        geometry=geom,
        datapath=geom.datapath,
        requests=req_views,
    )
    owners = tuple((aic, aivs) for aic, aivs in geom.datapath.aic_aiv_pairs)
    plan = build_arch22_v1_lightning(op, planning_ctx, contract, owners, ())

    # Exact per-group QK FLOPs and score Fixpipe bytes: 1*8192 + 2*512 = 9216 (NOT 2*(8192+512) = 17408).
    expected_q_kv_pairs = 1 * 8192 + 2 * 512
    assert plan.work.matrix["bf16"] == pytest.approx(2 * expected_q_kv_pairs * heads * dim)
    assert plan.work.resources["fixpipe"] == pytest.approx(4 * expected_q_kv_pairs * heads)

    # Exact split-list write/read conservation: only the split long_decode group (qr=1) writes/reads split lists.
    phases = {phase.phase_id: phase for phase in plan.pipeline_spec.phases}
    store_bytes = sum(item.amount for item in phases["split_list_store"].work if item.resource == "mte3")
    load_bytes = sum(item.amount for item in phases["split_list_load"].work if item.resource == "mte2_aiv")
    assert store_bytes > 0 and store_bytes == pytest.approx(load_bytes)

    # Full Engine replay must conserve declared OperatorPlan.work exactly.
    units = geom.resource_units
    rates = {r: float(geom.rates.get(r, 1e12)) / 1000.0 for r in units}
    rates["cube"] = 1.0
    ctx = RecipeContext(
        op, {}, {r: r for r in units}, rates, geom.compute_units,
        resource_units=units, compute_peaks=dict(geom.compute_peaks),
        pool_names={p: p for p in geom.buffers}, pool_capacities=dict(geom.buffers),
        pool_units=dict(geom.buffer_units), datapath=geom.datapath,
    )
    engine = Engine([Resource(r, rates[r], units=units[r]) for r in units])
    done = plan.program(ctx).start(engine, "idx_v1", engine.submit("in"))
    engine.advance(until=(done,))
    assert {r: v for r, v in engine.serviced.items() if v} == pytest.approx(ctx.bind(plan.work))


def test_internal_indexer_stage_does_not_leak_into_native_binding_or_export():
    """Producer must never emit synthetic stage names as native_kind or child kernels."""
    from _support.axes import REPRESENTATIVE_SKUS
    from mindie_sim.compose import compose_case
    from mindie_sim.export.kernel_catalog import resolve_ascend_kernel_identity
    from mindie_sim.prediction.native_binding import bind_native_call
    from mindie_sim.prediction.operators.sparse_indexer import index_cache_writer
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec

    # 1. step5_csa_summary_update and qsa_dual_page_cache_write carry no synthetic native_kind.
    summary_op = index_cache_writer.emit_index_cache_write(
        mode="step5_summary_update", op_id="l0.csa_summary", deps=(), stage=0, layer=0,
        requests=("r",), phase="decode", rows=4, dim=64, block_size=16, input_dtype="bf16",
    )[0]
    smeta = dict(summary_op.metadata)
    assert "native_kind" not in smeta
    bound = bind_native_call(summary_op)
    assert bound is not None
    assert not any("step5_" in str(k.get("name", "")) for k in bound.kernels)
    assert "step5_" not in str(bound.api.get("entry") or "")

    # 2. _kernel_records never promotes child recipe or family names when undeclared.
    _sh = dict(requests=("r",), semantic_shape=(1, 16), local_shape=(1, 16), padded_shape=(1, 16), launch_shape=(1, 16))
    parent = OpSpec(op_id="l0.parent", family="indexer", recipe="indexer_parent", phase="decode", stage=0, layer=0, dtype="bf16", **_sh)
    child = OpSpec(op_id="l0.child", family="internal_stage_family", recipe="step5_internal_stage", phase="decode", stage=0, layer=0, dtype="bf16", **_sh)
    bound_parent = bind_native_call(parent, children=(child,))
    assert bound_parent is not None
    assert bound_parent.kernels == ()

    # 3. Export projection stays honest empty rather than leaking internal stage names.
    a5_case = parse_case(compose_case("deepseek-v3.2", REPRESENTATIVE_SKUS["a5"], dp=1, tp=2, ep=2, layers=4))
    desc = resolve_ascend_kernel_identity(
        summary_op.op_id, summary_op, summary_op.family, summary_op.recipe,
        summary_op.local_shape, summary_op.dtype, a5_case, None, (),
    )
    assert "step5_" not in desc.ascend_kernel
    assert desc.ascend_kernel not in {"aclnnCustomOp", "aclnnAdd / VectorOp"}
