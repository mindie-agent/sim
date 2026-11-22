"""Sparse selection through governed sort, merge, and radix stages.

[Role] Verify selection arithmetic, local lists, and execution dependencies.
[Invariants] Tail padding and split ownership preserve score and output work.
[Timeouts & Locks] Bounded CPU simulation without external locks.
[Side-Effect Policy] No device access or file writes.
"""
from collections import Counter
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest

from _support import operator_calibration_helpers as source_cases
from _support.axes import REPRESENTATIVE_SKUS
from _support.hardware import run, sku_case
from _support.operator_cases import mqsmla_case
from mindie_sim import compose_case, target_graphs
from mindie_sim.catalog.cache import index_row_bytes
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.operators.sparse_indexer import family_quant_indexer as quant_indexer
from mindie_sim.prediction.operators.sparse_indexer import sparse_indexer_pipeline
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.operators.mla.generic_latent_attention import (
    index_key_window, semantic_index_selection, unique_index_key_rows,
)
from mindie_sim.prediction.operators.sparse_indexer.index_pool import _visited_pool_rows
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import sparse_indexer
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_arch import NativeComputeMode
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

MEMBASE = REPRESENTATIVE_SKUS["a3"]
REGBASE = REPRESENTATIVE_SKUS["a5"]
POOL_CONTEXT = 32768
POOL_SIZE = 4
POOL_ROW_BYTES = 132


def _context(op, geometry):
    roles = set(geometry.resource_units) | {"cube"}
    return RecipeContext(op, {}, {role: role for role in roles}, {},
                         geometry.compute_units, resource_units=geometry.resource_units,
                         compute_peaks=dict.fromkeys(geometry.compute_peaks, 1),
                         pool_names={pool: pool for pool in geometry.buffers},
                         pool_capacities=geometry.buffers)


def _make(rows, **geometry):
    fields = dict(query_tokens=1, scanned_rows=rows, index_heads=4, index_dim=128)
    return sparse_indexer.make_indexer_op(**(fields | geometry))


def _program(op, hardware):
    geometry = Geometry.from_devices(sku_case(hardware).machine.dies)
    plan = sparse_indexer.build(op, geometry)
    return plan, plan.program(_context(op, geometry))


def _query_tiles(program):
    query = program.programs[0].tile(0)
    owners = [lane.tile(0)[0].program for lane in query[0].program.programs]
    tiles = [next(s.program for s in owner.stages if s.name == "score_tiles") for owner in owners]
    return query, tiles


def _membase(rows, **geometry):
    op = _make(rows, **geometry)
    plan, program = _program(op, MEMBASE)
    query, tiles = _query_tiles(program)
    return op, plan, query, tiles


def _verified(op, hardware=MEMBASE, *, case=None):
    report = verify_operator(case or sku_case(hardware), op, sparse_indexer.build, input_ready_ms=7)
    assert report["finite_completion"] and report["buffers_released"] and report["input_dependency_obeyed"]
    assert report["work_mismatches"] == {}
    assert report["result"]["declared_service"] == pytest.approx(report["result"]["serviced"])
    return report["result"]


@pytest.mark.parametrize("n,merges", [
    (32, ()), (64, ((2, 1),)), (128, ((4, 1),)),
    (256, ((4, 2), (2, 1))), (384, ((4, 3), (3, 1))), (512, ((4, 4), (4, 1))),
    (1024, ((4, 8), (4, 2), (2, 1))), (2048, ((4, 16), (4, 4), (4, 1))),
])
def test_sort32_merge_recurrence(n, merges):
    trace = sparse_indexer.trace_sort_all(n)
    assert trace.sort32_repeats == n // 32
    assert trace.mrg_passes == merges and trace.passes == len(merges)
    copyback = len(merges) % 2 == 0
    assert trace.needs_copyback is copyback
    assert trace.barriers == 1 + len(merges) + copyback


@pytest.mark.parametrize("rows,sorts,flushes,merges", [
    (8 * 1024, 16, 4, 3), (32 * 1024, 64, 16, 15),
    (64 * 1024, 128, 32, 31), (128 * 1024, 256, 64, 63),
])
def test_periodic_typed_sort_flush_counts(rows, sorts, flushes, merges):
    trace = sparse_indexer.trace_arch22_topk_flushes(rows)
    assert (trace.block_sorts, trace.flushes, trace.retained_merges) == (sorts, flushes, merges)
    assert (trace.sort32_repeats, trace.mrg_passes) == (0, 0)
    assert trace.barriers == sorts + 1 + 4 * merges
    assert trace.padded_rows == trace.split_merges == 0
    assert trace.flush_pair_bytes == rows * 8


def test_small_query_tail_is_sorted_and_flushed_as_two_full_blocks():
    op, _, _, (tiles,) = _membase(513, topk_items=2048)
    head, tail = ({stage.name: stage for stage in tiles.tile(i)} for i in range(2))
    assert all("typed_sort" in tile and "sort32" not in tile and "sort_copyback" not in tile
               for tile in (head, tail))
    assert "flush_merge" not in head and "retained_topk_merge" not in tail
    assert head["typed_sort"].metadata["pair_bytes"] == tail["typed_sort"].metadata["pair_bytes"] == 512 * 8
    assert tail["prepare_sort_pairs"].metadata == {"pair_bytes": 512 * 8, "padded_items": 511}
    assert tail["flush_merge"].metadata == {"valid_inputs": 2, "pair_bytes": 8192}
    assert tail["flush_merge"].after == ("typed_sort_barrier",)
    trace = sparse_indexer.trace_arch22_topk_flushes(513)
    assert (trace.block_sorts, trace.flushes, trace.retained_merges) == (2, 1, 0)
    assert (trace.padded_rows, trace.padded_tail_rows, trace.flush_pair_bytes) == (511, 512, 8192)
    assert trace.barriers == 3
    _verified(op)


@pytest.mark.parametrize("query_rows,batch,topk,force_explicit", [
    (5, 1, 2048, False), (1, 1, 128, True), (1, 1, 3072, False),
])
def test_explicit_sort_all_merges_the_initial_retained_list(query_rows, batch, topk, force_explicit):
    op, _, _, (tiles,) = _membase(513, query_tokens=query_rows, batch_size=batch,
                                  topk_items=topk, is_sparse_count_over_2k=force_explicit)
    for i in range(tiles.count):
        stages = {stage.name: stage for stage in tiles.tile(i)}
        assert "typed_sort" not in stages and "flush_merge" not in stages
        assert stages["sort32"].metadata["repeats"] == 16
        assert stages["mrg_sort_1"].metadata["repeats"] == 4
        assert stages["mrg_sort_2"].metadata["repeats"] == 1
        assert stages["sort_copyback"].after == ("mrg_sort_barrier_2",)
        assert stages["sort_copyback_barrier"].after == ("sort_copyback",)
        merge = stages["retained_topk_merge"]
        assert merge.after == ("sort_all_complete",)
        assert merge.metadata["input_pair_bytes"] == (topk + 512) * 8
        assert merge.metadata["output_pair_bytes"] == topk * 8
    trace = sparse_indexer.trace_arch22_topk_flushes(513, act_s1_size=query_rows, topk=topk,
                                                    is_sparse_count_over_2k=force_explicit)
    assert (trace.block_sorts, trace.flushes, trace.retained_merges) == (2, 2, 2)
    assert (trace.sort32_repeats, trace.mrg_passes, trace.barriers) == (32, 4, 14)
    assert (trace.padded_rows, trace.padded_tail_rows) == (511, 512)
    _verified(op)


@pytest.mark.parametrize("tile_size,mask", [(64, 0b0011), (384, 0b0111)])
def test_explicit_merge_mask_follows_the_number_of_sorted_runs(tile_size, mask):
    _, _, _, (tiles,) = _membase(tile_size + 1, s2_tile_size=tile_size, query_tokens=5)
    stages = tiles.tile(1)
    assert next(s for s in stages if s.name == "prepare_sort_pairs").metadata["padded_items"] == tile_size - 1
    masks = [s.metadata["valid_mask"] for s in stages if "valid_mask" in s.metadata]
    assert masks[-1] == mask


def test_batch_partition_uses_per_request_query_count_for_caller_branch():
    op, _, _, (tiles,) = _membase(513, query_tokens=5, batch_size=5)
    assert any(s.name == "typed_sort" for s in tiles.tile(0))
    _verified(op)


def test_split_owners_restart_flushes_then_merge_through_gm():
    rows = 8192 + 49
    op, plan, query, owners = _membase(rows, s2_splits=3)
    assert plan.metadata["score_path"] == "l0c->gm->ub"
    assert [tile.count for tile in owners] == [6, 6, 5]
    for tile in owners:
        stages = [[s.name for s in tile.tile(i)] for i in range(tile.count)]
        assert [i for i, names in enumerate(stages) if "flush_merge" in names] == ([3, 5] if tile.count == 6 else [3])
        assert [i for i, names in enumerate(stages) if "retained_topk_merge" in names] == [tile.count - 1]
        body = {s.name: s for s in tile.tile(0)}
        assert body["typed_sort"].metadata["repeats"] == 16
        assert body["score_read"].after == ("score_handoff",)
    trace = sparse_indexer.trace_arch22_topk_flushes(rows, s2_splits=3)
    assert (trace.block_sorts, trace.flushes, trace.retained_merges, trace.split_merges) == (17, 6, 3, 2)
    assert trace.flush_pair_bytes == 17 * 512 * 8 and trace.barriers == 31
    split = {s.name: s for s in query}
    assert split["split_list_load"].after == ("s2_owners",)
    assert split["split_topk_merge"].metadata["merges"] == 2
    assert split["split_topk_merge"].after == ("split_list_load",)
    work = plan.work.resources
    handoff = (16 * 512 + 64) * 4 * 4
    lists = 3 * (2048 * 8 + 128)
    assert work["fixpipe"] == handoff
    assert work["mte3"] == lists + 2048 * 4
    assert work["hbm"] == (rows + 3 * 4) * 256 + 2 * handoff + 17 * 4 * 2 + 2 * lists + 2048 * 4
    assert plan.work.matrix == {"bf16": 2 * rows * 4 * 128}
    _verified(op)


@pytest.mark.parametrize("scanned,split,topk", [(0, 1, 4), (1, 1, 4), (513, 2, 1), (32769, 3, 16)])
def test_selection_shape_and_split_work_follow_visible_score_counts(scanned, split, topk):
    op = _make(scanned, topk_items=topk, s2_splits=split)
    plan, program = _program(op, REGBASE)
    exit_stages = {s.name: s for s in program.programs[0].tile(0)}
    assert op.local_shape == (1, topk)
    assert exit_stages["index_extract"].metadata["valid_items_per_head"] == min(topk, scanned)
    assert exit_stages["index_store"].work["mte3"] == topk * 4
    assert plan.work.matrix.get("bf16", 0) == 2 * scanned * 4 * 128
    if split > 1:
        store = plan.work.resources["mte3"] - topk * 4
        assert store == split * topk * 8
        assert exit_stages["split_list_load"].work["mte2_aiv"] == store
    _verified(op, REGBASE)


@pytest.mark.parametrize("queries,rows,heads,topk", [
    (1, 640, 1, 2), (3, 8192, 2, 16), (5, 32768, 4, 32),
])
def test_score_workspace_and_vector_work_match_independent_per_head_oracle(queries, rows, heads, topk):
    op = sparse_indexer.make_msa_index_score_op(op_id="blocks", query_tokens=queries,
                                               context_tokens=rows, index_heads=heads,
                                               block_size=128, topk_blocks=topk)
    plan, program = _program(op, REGBASE)
    valid = (rows + 127) // 128
    stride = (valid + 15) // 16 * 16
    assert plan.work.matrix == {"bf16": 2 * queries * rows * heads * 128}
    assert plan.metadata["score_workspace_bytes"] == queries * heads * stride * 4
    assert plan.work.resources["mte3"] == queries * heads * (stride + topk) * 4
    query, _ = _query_tiles(program)
    select = next(stage for stage in query if stage.name == "score_row_topk")
    assert select.work["vector"] == heads * valid * (valid - 1).bit_length()
    _verified(op, REGBASE)


def test_region_mask_and_retained_index_order_preserve_selection_extent():
    op = sparse_indexer.make_step5_region_score_op(op_id="regions", query_tokens=1, context_tokens=65,
                                                   region_block_size=8, topk_regions=16)
    plan, program = _program(op, REGBASE)
    query, (tiles,) = _query_tiles(program)
    mask, = (s for s in tiles.tile(0) if s.name == "mask3")
    assert mask.work["vector"] == 9
    assert plan.work.matrix == {"fp8_e4m3": 2 * 9 * 16 * 128}
    by_name = {s.name: s for s in query}
    assert by_name["region_radix_topk"].metadata["pair_bytes"] == 9 * 8
    assert by_name["index_sort"].after == ("region_radix_topk",)
    assert by_name["index_store"].work["mte3"] == 16 * 4
    _verified(op, REGBASE)


@pytest.mark.parametrize("candidates,tiles", [(512, 1), (513, 2), (520, 2)])
def test_packed_candidate_tiles_cross_gm_before_cube(candidates, tiles):
    op = _make(candidates, abi="cannbot_dsl.quant_sparse_lightning_indexer",
               cache_dtype="mxfp4", index_heads=32, topk_items=64)
    case = sku_case(REGBASE)
    die, = case.machine.dies
    native = NativeComputeMode("mxfp4_x_mxfp4_acc_fp32", "matrix", ("mxfp4",), "cube", ("fp32",))
    architecture = replace(die.compute_arch, native_modes=(*die.compute_arch.native_modes, native))
    die = replace(die, compute_arch=architecture, compute_peaks=(*die.compute_peaks, ("mxfp4", 1e12)))
    case = replace(case, machine=replace(case.machine, dies=(die,)))
    geometry = Geometry.from_devices(case.machine.dies)
    plan = sparse_indexer.build(op, geometry)
    program = plan.program(_context(op, geometry))
    query, (stream,) = _query_tiles(program)
    assert stream.count == tiles
    staged = 512 * (128 // 2 + 128 // 32)
    assert staged == 34816
    for i in range(tiles):
        stages = {s.name: s for s in stream.tile(i)}
        gather = stages["key_gather"]
        store = stages["key_stage_store"]
        load = stages["key_stage_load"]
        assert len(gather.placement["mte2_aiv"]) == store.metadata["aiv_subblocks"] == 2
        assert store.placement["mte3"] == gather.placement["mte2_aiv"]
        assert len(load.placement["mte2_aic"]) == 1
        assert store.after == (gather.name,) and load.after == (store.name,)
        assert stages["load_operands"].after == (load.name,)
        assert stages["cube_scores"].after == ("load_operands",)
        assert stages["rectification"].placement["vector"] == gather.placement["mte2_aiv"]
        assert gather.work["mte2_aiv"] == gather.work["hbm"] == staged
        assert store.work["mte3"] == store.work["hbm"] == staged
        assert load.work["mte2_aic"] == load.work["hbm"] == staged
        assert stages["rectification"].work["vector"] == min(512, candidates - i * 512) * 32
    owner = query[0].program.programs[0].tile(0)[0].program
    radix, = (s for s in owner.stages if s.name == "radix_histogram_topk")
    assert radix.metadata["pair_bytes"] == candidates * 8
    assert plan.work.resources["mte3"] == tiles * staged + 64 * 4
    assert plan.work.resources["mte2_aic"] == tiles * staged + 32 * 68
    assert plan.work.resources["mte2_aiv"] == tiles * staged + 32 * 4
    assert plan.work.matrix == {"mxfp4": 2 * tiles * 512 * 32 * 128}
    assert op.flops == plan.metadata["useful_flops"] == 2 * candidates * 32 * 128
    assert len(next(s for s in query if s.name == "index_store").placement["mte3"]) == 1
    serviced = _verified(op, case=case)["serviced"]
    for role in ("mte3", "mte2_aic", "mte2_aiv"):
        assert serviced[f"{role}@r0:s0"] == pytest.approx(plan.work.resources[role])


@pytest.mark.parametrize("ratio,context,expected", [(4, 511, 128), (4, 512, 128), (4, 515, 129), (8, 31, 4)])
def test_compressed_rows_transport_only_declared_payloads(ratio, context, expected):
    op = _make((context + 1) // ratio, topk_items=512)
    _, program = _program(op, MEMBASE)
    _, tiles = _query_tiles(program)
    assert sum(s.work["hbm"] for stream in tiles for i in range(stream.count)
               for s in stream.tile(i) if s.name == "key_load") == expected * 128 * 2
    assert op.flops == 2 * expected * 4 * 128
    _verified(op)


def test_shared_prefix_reuse_only_changes_where_keys_are_serviced():
    geometry = Geometry.from_devices(sku_case(REGBASE).machine.dies)
    rows, queries, heads, dim = 1025, 3, 4, 128
    options = dict(query_tokens=queries, index_heads=heads, index_dim=dim, topk_items=16)
    base, shared = (_make(rows, shared_prefix=value, **options) for value in (False, True))
    cold, cached = (sparse_indexer.build(op, geometry).work for op in (base, shared))
    reused = (queries - 1) * rows * dim * 2
    assert cold.resources["hbm"] - cached.resources["hbm"] == cached.resources["l2"] == reused
    assert cached.resources["mte2_aic"] == cold.resources["mte2_aic"]
    assert cached.matrix == cold.matrix and cached.resources["mte3"] == cold.resources["mte3"]
    constrained = replace(geometry, physical_mechanisms=None, buffers=geometry.buffers | {"l2": 1})
    assert sparse_indexer.build(shared, constrained).work == cold
    _verified(shared, REGBASE)


def test_long_score_owner_lists_scale_with_queries_and_owner_splits():
    short, many, wide = (_membase(rows, query_tokens=q, s2_splits=splits)[1].work
                         for q, rows, splits in ((2, 32768, 2), (8, 32768, 2), (8, 65536, 4)))
    pair_list = 2048 * 8 + 128
    assert many.matrix["bf16"] == 4 * short.matrix["bf16"]
    assert wide.matrix["bf16"] == 2 * many.matrix["bf16"]
    assert wide.resources["mte3"] - many.resources["mte3"] == 8 * 2 * pair_list
    assert wide.resources["mte2_aiv"] > many.resources["mte2_aiv"] > short.resources["mte2_aiv"]


@pytest.mark.parametrize("overrides,error", [
    ({"abi": "unknown"}, "Unsupported.*ABI"), ({"cache_dtype": "fp16"}, "cache_dtype"),
    ({"index_dim": 0}, "index_dim"), ({"block_size": 0}, "block_size"),
    ({"s2_tile_size": 0}, "s2_tile_size"), ({"s2_splits": 0}, "s2_splits"),
    ({"scanned_rows": -1}, "scanned_rows"), ({"query_tokens": True}, "query_tokens"),
    ({"topk_items": 0}, "topk_items"), ({"batch_size": 0}, "batch_size"),
])
def test_indexer_geometry_rejects_invalid_domains(overrides, error):
    with pytest.raises(ModelingError, match=error):
        _make(512, **overrides)


def test_phase_work_is_independent_of_layer_and_request_identity():
    op = _make(769, query_tokens=2, s2_splits=2)
    renamed = replace(op, op_id="other", layer=97, requests=("different",))
    left, right = (_program(item, MEMBASE)[0] for item in (op, renamed))
    assert left.work == right.work
    assert left.pipeline_spec == right.pipeline_spec
    metadata = dict(op.metadata)
    metadata["indexer_contract"] = metadata["indexer_contract"] | {"query_tokens": -1}
    malformed = replace(op, metadata=tuple(metadata.items()))
    with pytest.raises(ModelingError, match="invalid_query_tokens"):
        _program(malformed, MEMBASE)


def test_large_geometry_builds_finite_owner_streams_without_expanding_tiles(monkeypatch):
    geometry = Geometry.from_devices(sku_case(MEMBASE).machine.dies)
    op = _make(1000000, query_tokens=200000, topk_items=16)
    calls = Counter()
    assert sparse_indexer.trace_arch22_topk_flushes(1000000, act_s1_size=200000).block_sorts == 1954
    plan = sparse_indexer.build(op, geometry)
    tile = type(plan.program).tile

    def count_tiles(self, *args, **kwargs):
        calls["tile"] += 1
        return tile(self, *args, **kwargs)

    monkeypatch.setattr(type(plan.program), "tile", count_tiles)
    prepared = sparse_indexer.build(op, geometry)
    assert calls["tile"] <= 4
    program = prepared.program(_context(op, geometry))
    assert len(program.programs) == geometry.datapath.aic_count
    assert sum(lane.count for lane in program.programs) == 200000
    assert calls["tile"] <= 4


def _execution(ops, scope):
    case = sku_case(MEMBASE, device={"launch_us": 50})
    execution = Execution(case, CostModel(case), trace_limit=2500, recipes=RecipeBook(sparse_indexer.recipes()))
    work = IterationWork(ops, IterationPlan(()), 1, 1, 1, None, "decode", "selection", replay_scope=scope)
    result = execution.add(work, 0, 0)
    execution.engine.advance(until=(result.completion,))
    return execution.engine


def test_capture_preserves_governed_kernel_work_and_downstream_dependency():
    first = _make(513, op_id="first")
    second = _make(1025, op_id="second", deps=(first.op_id,))
    eager, captured = (_execution((first, second), scope) for scope in ("none", "full"))
    for engine, count in ((eager, 2), (captured, 1)):
        assert sum(e["task"].endswith(":submit") for e in engine.trace) == count
        assert engine.serviced["host@r0:s0"] == pytest.approx(count * .05)
        writes = [e for e in engine.trace if e["task"].endswith(":index_store")]
        reads = [e for e in engine.trace if e["task"].endswith(":query_load")]
        assert writes[0]["end_ms"] <= reads[1]["start_ms"]
    assert {k: v for k, v in eager.serviced.items() if "host" not in k} == {
        k: v for k, v in captured.serviced.items() if "host" not in k}


def test_membase_score_load_dot_sort_and_list_retirement_are_ordered():
    op = _make(2049, topk_items=1024)
    engine = _execution((op,), "none")
    names = ("key_load", "cube_scores", "score_handoff", "score_read", "typed_sort")
    trace = {name: [e for e in engine.trace if e["task"].endswith(":" + name)] for name in names}
    assert all(len(events) == 5 for events in trace.values())
    for events in zip(*trace.values(), strict=True):
        assert all(a["end_ms"] <= b["start_ms"] + 1e-12 for a, b in zip(events, events[1:]))
    assert all(a["end_ms"] <= b["start_ms"] + 1e-12 for a, b in zip(trace["typed_sort"], trace["key_load"][1:]))
    loads = [e["demand"]["memory@r0:s0"] for e in trace["key_load"]]
    assert loads == [512 * 256] * 4 + [256]


@pytest.mark.parametrize("pipe", ("vector", "mte2_aic", "memory"))
def test_declared_pipe_capacity_changes_time_without_changing_plan(pipe):
    op = _make(513)
    base = run(sku_case(MEMBASE), (op,), sparse_indexer.recipes())
    slow = run(sku_case(MEMBASE, device={"rates": {pipe: 1e5}}), (op,), sparse_indexer.recipes())
    assert slow["duration_ms"] > 10 * base["duration_ms"]
    assert slow["execution"]["work"] == base["execution"]["work"]


def test_plan_requires_declared_hardware_routes():
    geometry = Geometry.from_devices(sku_case(REGBASE).machine.dies)
    datapath = replace(geometry.datapath, usable_paths=())
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_unsupported"):
        sparse_indexer.build(_make(513), replace(geometry, datapath=datapath))


@pytest.mark.parametrize("context,ratio,completed,flops", [(1023, 1, 1024, 8388608), (5, 4, 1, 8192)])
def test_quant_indexer_completed_causal_keys_determine_flops(context, ratio, completed, flops):
    view = SimpleNamespace(context=context, q_len=1)
    assert quant_indexer._view_completed_keys(view, ratio) == completed
    assert quant_indexer._candidate_keys((view,), ratio)[0] == completed
    ledger = {
        "query_rows": 1, "wave_rows": 32, "waves": 1, "blocks": 1,
        "heads": 32, "width": 128, "keys": 1024, "completed_keys": completed,
        "payload_keys": completed, "key_row_bytes": 256, "output_bytes": 0.0,
    }
    work = sparse_indexer_pipeline.arch35_indexer_work(ledger)
    assert work.matrix["fp8_e4m3"] == 2 * completed * 32 * 128 == flops


def test_quant_indexer_empty_selection_still_stores_indices():
    op = source_cases._arch35_op_for(4, (RequestView("r", 1, 0, False, "decode"),))
    ledger = dict(op.metadata)[quant_indexer.QUANT_LEDGER]
    assert ledger["completed_keys"] == ledger["blocks"] == ledger["keys"] == 0
    assert ledger["row_completed_keys"] == ((1, 0),)
    assert op.flops == 0
    assert op.mte3_bytes == 8192
    plan = quant_indexer.Arch35QuantIndexerRecipe().build(op, None)
    program = source_cases._conserved(plan, op, source_cases.DT)
    names = [stage.name for stage in source_cases._stages(program)]
    assert "blocks" not in names
    assert "indices" in names


@pytest.mark.parametrize("views,ratio,heads,expected,flops", [
    pytest.param((RequestView("p", 2, 0, False, "prefill", query_ranges=((0, 1), (3, 4)),
                              global_q_len=4),), 4, 64, ((1, 0), (1, 1)), 16384, id="query-ranges"),
    pytest.param((RequestView("a", 1, 1, False, "decode"), RequestView("b", 1, 3, False, "decode")),
                 1, 32, ((1, 2), (1, 4)), 49152, id="unequal-contexts"),
])
def test_quant_indexer_preserves_causal_keys_per_query(views, ratio, heads, expected, flops):
    op = source_cases._arch35_op_for(ratio, views, heads=heads)
    ledger = dict(op.metadata)[quant_indexer.QUANT_LEDGER]
    assert ledger["row_completed_keys"] == expected
    assert ledger["completed_keys"] == max(keys for _, keys in expected)
    completed = sum(rows * keys for rows, keys in expected)
    assert op.flops == 2 * completed * heads * 128 == flops


@pytest.mark.parametrize("compression,cp,query,noncausal", tuple(product(
    (1, 2, 4, 128), (1, 8), (1, 6, 128, 129), (False, True),
)))
def test_compressed_cp_index_reads_are_per_query_tile(compression, cp, query, noncausal):
    owned = []
    for end in (*range(128, query, 128), query):
        horizon = 4096 + (query if noncausal else end)
        counts = [0] * cp
        for row in range(horizon // compression):
            counts[(row // max(1, 128 // compression)) % cp] += 1
        owned.append(max(counts))
    assert index_key_window(((query, 4096),), cp=cp, compression=compression,
                            noncausal=noncausal, kv_len=4096 + query) == (max(owned), sum(owned))


def _shared_prefix_views():
    return tuple(RequestView(str(i), 6, 256, True, "decode",
                             prefix_tokens=128, prefix_cache_key="shared") for i in range(2))


@pytest.mark.parametrize("options,same_key,unique", [({}, True, 98),
                                                      ({"share_prefix": False}, True, 130),
                                                      ({}, False, 130)])
def test_compressed_key_union_respects_sharing_and_cache_identity(options, same_key, unique):
    views = _shared_prefix_views()
    if not same_key:
        views = (views[0], replace(views[1], prefix_cache_key="other"))
    assert unique_index_key_rows(views, compression=4, **options) == (unique, 130)


@pytest.mark.parametrize("mode,key_rows", [("batch_shared", 98), ("per_sequence", 130)])
def test_generic_indexer_precharge_deduplicates_only_external_keys(mode, key_rows):
    model = load_model("deepseek-v4-pro-0813")
    plan = next(p for p in model.layer_plans if p.indexer)
    views = _shared_prefix_views()
    parallel = ParallelSpec()
    row = index_row_bytes(model, plan.index_dim)
    score_rows = 2 * sum((256 + i) // 4 for i in range(1, 7))
    runtime = replace(mqsmla_case().runtime, parallel=parallel, kernels={},
                      prefix_cache_mode="shared", prefix_read_mode=mode, kv_block_size=128)
    builder = Builder("s0.l1", plan, 0, views, model, "decode")
    semantic_index_selection(
        builder, deps=("input",), views=views, plan=plan, model=model, quant=model.quant,
        parallel=parallel, shard=1, prefix="s0.l1", padded_tokens=12, runtime=runtime)
    scores = next(op for op in builder.ops if op.family == "indexer")
    assert scores.flops == 2 * score_rows * plan.index_heads * plan.index_dim
    assert scores.mte2_bytes == 130 * row
    assert scores.hbm_bytes == key_rows * row + scores.mte3_bytes
    assert not any(op.recipe in {"index_scores", "index_topk"} for op in builder.ops)
    assert scores.mte3_bytes == 12 * plan.index_topk * 4


def _pool_view(index, key, ranges=()):
    return SimpleNamespace(
        q_len=2, context=8192, kv_len=8192, prefix_tokens=4096,
        prefix_cache_key=key, request_id=f"req-{index}", query_ranges=ranges)


@pytest.mark.parametrize("mode,second_key,expected", [("private", "b", 4096),
                                                      ("shared", "b", 4096), ("shared", "a", 3072)])
def test_visited_pool_rows_counts_private_suffixes_and_prefix_union(mode, second_key, expected):
    plan = SimpleNamespace(index_kpool=4, noncausal_attention=False)
    runtime = SimpleNamespace(prefix_cache_mode=mode, prefix_read_mode="auto")
    assert _visited_pool_rows([_pool_view(0, "a"), _pool_view(1, second_key)],
                              plan, runtime=runtime) == expected


def test_visited_pool_rows_honors_pcp_query_ranges_and_noncausal_extent():
    causal = SimpleNamespace(index_kpool=4, noncausal_attention=False)
    noncausal = SimpleNamespace(index_kpool=4, noncausal_attention=True)
    shared = SimpleNamespace(prefix_cache_mode="shared", prefix_read_mode="auto")
    lates = _pool_view(0, "a", ((4000, 4100),))
    base = _pool_view(0, "a")
    assert _visited_pool_rows([lates], causal, runtime=shared) > _visited_pool_rows(
        [base], causal, runtime=shared)
    exposed = SimpleNamespace(q_len=2, context=8192, kv_len=16384, prefix_tokens=0,
                              prefix_cache_key="", request_id="r", query_ranges=())
    assert _visited_pool_rows([exposed], noncausal, runtime=shared) == 4096


def test_native_pool_indexer_charges_cold_scan_and_fused_score_work():
    case = parse_case(compose_case(
        "glm-5.3-flash", hardware="a5", tp=8, ep=8, num_devices=8,
        ranks_per_node=8, prompt_tokens=POOL_CONTEXT, output_tokens=2, concurrency=1))
    ops = target_graphs(case, phase="decode", tokens=1, context=POOL_CONTEXT,
                       sequences=1, scope="body")[0]
    scores = next(op for op in ops if op.recipe == "cann_pool_key_indexer")
    metadata = dict(scores.metadata)
    assert metadata["scanned_pools"] == POOL_CONTEXT // POOL_SIZE
    assert metadata["pool_row_bytes"] == POOL_ROW_BYTES
    scanned = metadata["safe_reference_scanned_pools"]
    pool_read = scanned * POOL_ROW_BYTES
    assert scores.mte2_bytes >= pool_read
    assert scores.hbm_bytes - scores.mte3_bytes >= pool_read
    plan = next(p for p in case.model.layer_plans if p.index_topk)
    assert scores.flops == pytest.approx(2 * scanned * plan.index_heads * plan.index_dim)
    assert scores.vector_ops > 0.0
    assert not any(op.recipe in {"index_pool_scores", "index_pool_topk"} for op in ops)
    topk = next(op for op in ops if op.recipe == "index_topk")
    assert topk.mte2_bytes > 0.0


def test_arch22_block_run_assignment_tracks_exact_group_owner_counts():
    from mindie_sim.prediction.operators.sharedkv.l0_sync import p10_sort_to_alu_op_ratio
    from mindie_sim.prediction.operators.sharedkv.sparse_indexer_v1 import _assign_block_runs

    straggler = [(gid, 1, 512) for gid in range(7)] + [(7, 128, 512)]
    _, _, _, owners_20 = _assign_block_runs(straggler, 20, 8)
    assert owners_20[:7] == (1,) * 7 and owners_20[7] == 19
    _, _, _, owners_24 = _assign_block_runs(straggler, 24, 8)
    assert owners_24[7] == 23
    uniform = [(gid, 17, 512) for gid in range(8)]
    _, _, _, u_owners = _assign_block_runs(uniform, 20, 8)
    assert min(u_owners) >= 3 and max(u_owners) == 4
    assert p10_sort_to_alu_op_ratio(SimpleNamespace(primitive_table=None)) == 1.0
