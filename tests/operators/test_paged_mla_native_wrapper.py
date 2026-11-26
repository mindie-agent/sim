"""Native paged MLA declarations, dispatch, and execution boundaries.

[Role] Verify governed plans through source dispatch and ordinary execution.
[Invariants] Preserve active work, owner-local tiles, and readiness dependencies.
[Timeouts & Locks] Bounded local simulation. No locks.
[Side-Effect Policy] Read checked-in workload and capture evidence. No device access.
"""
import json
from collections import Counter
from dataclasses import replace
from math import ceil
from pathlib import Path

import numpy as np
import pytest

from _support import operator_calibration_helpers as source_cases
from _support.axes import REPRESENTATIVE_SKUS
from _support.hardware import hardware_case, sku_case
from mindie_sim import graph_costs
from mindie_sim.prediction.operators.mla_prolog import family_mla_prolog as prolog; from mindie_sim.prediction.operators.sparse_indexer import index_pool_native
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operators.sparse_indexer.index_pool import native_shard_sizes, row_shard_partition
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe, Work
from mindie_sim.prediction.operators import FamilyRegistry, paged_mla as governed
from mindie_sim.prediction.operators.paged_mla import DSL_FLASH_ABI, FIA_ABI, FLASH_ABI
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.streams import StreamOwners
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure.physical import node_physical, subgraph_physical

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"
LOCAL_DIE = REPRESENTATIVE_SKUS["a3"]
DIRECT_DIE = REPRESENTATIVE_SKUS["a5"]
LATENT, ROPE, DQK, RECORD, PAGE = 512, 64, 576, 1152, 128
READY = 7


def _recipe_book(*, source_kernel=False):
    spec = governed.register_family()
    bindings = {spec.name: PlannedRecipe(governed.build)} if source_kernel else None
    return RecipeBook(bindings, family_registry=FamilyRegistry((spec,)))


def _case_and_geometry(hardware, owner_count=None):
    case = sku_case(hardware)
    geometry = Geometry.from_devices(case.machine.dies)
    if owner_count is not None:
        pairs = geometry.datapath.aic_aiv_pairs[:owner_count]
        aivs = tuple(aiv for _, siblings in pairs for aiv in siblings)
        owners = StreamOwners(tuple(aic for aic, _ in pairs), aivs, len(pairs), len(aivs))
        geometry = replace(geometry, stream_owners=owners)
    return case, geometry


def _make_operator(lengths, *, heads=8, abi=FLASH_ABI, **contract):
    return governed.make_op(sequences=tuple((1, length) for length in lengths),
                            logical_heads=heads, physical_heads=heads, abi=abi, **contract)


def _prepare(case, geometry, operator, *, trace_limit=0):
    execution = Execution(case, CostModel(case), recipes=_recipe_book(), trace_limit=trace_limit)
    prepared, plan = execution.recipes.prepare(operator, geometry)
    context = execution._context(prepared, 0, "none")
    return execution, plan, context


def _simulate(case, geometry, operator, *, trace_limit=16000):
    execution, plan, context = _prepare(case, geometry, operator, trace_limit=trace_limit)
    engine = execution.engine
    ready = engine.submit("input", release=READY)
    done = plan.lower(context).start(engine, "attention", ready)
    engine.advance(horizon=READY)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert {r: n for r, n in engine.serviced.items() if n} == pytest.approx(context.bind(plan.work))
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return plan, tuple(engine.trace), done.time - READY


def _queries(pipeline):
    return (stage.program for i in range(pipeline.count) for stage in pipeline.tile(i))


def _splits(query):
    waves = query.stages[0].program
    return (stage.program for i in range(waves.count) for stage in waves.tile(i))


def _stages(split, phase):
    return (stage for i in range(split.count) for stage in split.tile(i)
            if stage.name.partition(":")[0] == phase)


def _split_keys(query):
    return [tuple((s.metadata["kv_tile"], s.metadata["key_logical_bytes"] // RECORD)
                  for s in _stages(split, "kv_load")) for split in _splits(query)]


def _case_rows(filename):
    return json.loads((FIXTURES / filename).read_text())["cases"]


@pytest.mark.parametrize("lengths,heads,splits", [
    ((1, 33, 129), 24, 8), ((17, 65), 8, 2), ((128, 129, 4097), 16, 4),
    ((513, 1025, 2061), 4, 2), ((0, 1, 0), 2, 8),
])
@pytest.mark.parametrize("hardware,abi", [(LOCAL_DIE, FIA_ABI), (DIRECT_DIE, FLASH_ABI)])
def test_split_grid_tails_and_ledgers_conserve_source_work(hardware, abi, lengths, heads, splits):
    case, geometry = _case_and_geometry(hardware)
    operator = _make_operator(lengths, abi=abi, heads=heads, s2_base_size=32, return_lse=True, splits=splits)
    _, plan, context = _prepare(case, geometry, operator)
    tile = 32 if abi == FIA_ABI else 112
    assert plan.metadata["key_tile"] == tile
    groups_per_request = ceil(heads / 16) if abi == FIA_ABI else 1
    program = plan.program(context)
    fa_program = program.stages[0].program if hasattr(program, "stages") else program
    reduce_stages = (
        [s for sub in program.stages[-1].program.programs for s in sub.stages]
        if hasattr(program, "stages") and program.stages[-1].name == "fd_reduce"
        else []
    )
    queries = tuple(_queries(fa_program))
    assert len(queries) == len(lengths) * groups_per_request
    for length, group in zip((n for n in lengths for _ in range(groups_per_request)), queries, strict=True):
        ntiles = ceil(length / tile)
        parts = min(ntiles, splits)
        intervals = _split_keys(group)
        assert len(intervals) == parts
        assert [pair for part in intervals for pair in part] == [
            (i, min(tile, length - i * tile)) for i in range(ntiles)
        ]
        if parts:
            assert max(map(len, intervals)) - min(map(len, intervals)) <= 1
        per_group_heads = min(heads, 16) if abi == FIA_ABI else heads
        output = group.stages[-1]
        if "output_logical_bytes" in output.metadata:
            assert output.metadata["output_logical_bytes"] <= per_group_heads * LATENT * 2
            assert output.metadata["lse_logical_bytes"] == output.metadata["output_logical_bytes"] // (LATENT // 2)
    if reduce_stages:
        stores = [s for s in reduce_stages if s.name.startswith("store")]
        assert stores
        assert all(s.metadata["lse_logical_bytes"] == s.metadata["output_logical_bytes"] // (LATENT // 2)
                   for s in stores)
        local_out = sum(q.stages[-1].metadata.get("output_logical_bytes", 0) for q in queries)
        reduce_out = sum(s.metadata["output_logical_bytes"] for s in stores)
        assert local_out + reduce_out == len(lengths) * heads * LATENT * 2
    assert plan.work.matrix["bf16"] == 2 * sum(lengths) * heads * (DQK + LATENT)
    report = verify_operator(case, operator, governed.build, input_ready_ms=READY)
    assert report["status"] == "passed", report
    assert report["buffers_released"] and report["input_dependency_obeyed"] and report["finite_completion"]
    assert not report["work_mismatches"]
    result = report["result"]
    assert result["operator"]["flops"] == plan.work.matrix["bf16"]
    assert result["serviced"] == pytest.approx(result["declared_service"])


@pytest.mark.parametrize("abi", [FLASH_ABI, DSL_FLASH_ABI])
@pytest.mark.parametrize("keys,widths", [(112, (112,)), (113, (112, 1))])
def test_flash_mla_uses_fixed_key_tile_and_trims_only_the_tail(abi, keys, widths):
    case, geometry = _case_and_geometry(DIRECT_DIE)
    operator = _make_operator((keys,), abi=abi, s2_base_size=32)
    _, plan, context = _prepare(case, geometry, operator)
    (group,) = _queries(plan.program(context))
    (split,) = _splits(group)
    assert _split_keys(group) == [tuple(enumerate(widths))]
    qk = tuple(_stages(split, "qk"))
    assert [stage.metadata["kv_tile"] for stage in qk] == list(range(len(widths)))
    assert [stage.work for stage in qk] == [context.bind(Work({"bf16": 2 * 8 * width * DQK}))
                                            for width in widths]


@pytest.mark.parametrize("fixture,hardware,abi", [
    ("paged_mla_host_v2.json", DIRECT_DIE, FLASH_ABI),
    ("paged_mla_arch22_dispatch.json", LOCAL_DIE, FIA_ABI),
])
def test_host_workloads_cover_each_active_tile_and_merge_head(fixture, hardware, abi):
    case = sku_case(hardware)
    rows = _case_rows(fixture)
    # Long context and ragged cases exercise the boundaries without an exhaustive catalog sweep.
    for row in (rows[0], next(item for item in rows if len(set(item["contexts"])) > 1)):
        lengths = tuple(c + 1 for c in row["contexts"])
        heads, cores = row["heads"], row["cores"]
        _, geometry = _case_and_geometry(hardware, cores)
        operator = _make_operator(lengths, heads=heads, abi=abi, splits=cores)
        _, plan, ctx = _prepare(case, geometry, operator)
        size = plan.metadata["key_tile"]
        expected = Counter({
            (seq, i): ceil(heads / 16) if abi == FIA_ABI else 1
            for seq, length in enumerate(lengths) for i in range(ceil(length / size))
        })
        program = plan.program(ctx)
        fa_program = program.stages[0].program if hasattr(program, "stages") else program
        reduce_stages = (
            [s for sub in program.stages[-1].program.programs for s in sub.stages]
            if hasattr(program, "stages") and program.stages[-1].name == "fd_reduce"
            else []
        )
        queries = list(_queries(fa_program))
        groups_per_sequence = len(queries) // len(lengths)
        covered = Counter((idx // groups_per_sequence, tile) for idx, query in enumerate(queries)
                          for split in _split_keys(query) for tile, _ in split)
        assert covered == expected
        expected_merge = Counter()
        for seq, length in enumerate(lengths):
            group = queries[seq * groups_per_sequence]
            parts = min(ceil(length / size), cores)
            finish = {s.name: s for s in group.stages}
            group_heads = min(heads, 16) if abi == FIA_ABI else heads
            if "store" in finish:
                assert finish["store"].metadata["nonempty_splits"] == parts
                if parts > 1:
                    assert finish["merge_read"].metadata["vec2_split_state_logical_bytes"] == (
                        parts * group_heads * (LATENT + 16) * 4
                    )
            elif parts > 1:
                expected_merge[parts] += groups_per_sequence
        if reduce_stages:
            merge_reads = [s for s in reduce_stages if s.name.startswith("merge_read")]
            assert sum(expected_merge.values()) == len(merge_reads)
            for stage in merge_reads:
                parts = stage.metadata["nonempty_splits"]
                group_heads = min(heads, 16) if abi == FIA_ABI else heads
                assert stage.metadata["vec2_split_state_logical_bytes"] == (
                    parts * group_heads * (LATENT + 16) * 4
                )


@pytest.mark.parametrize("row", _case_rows("paged_mla_arch22_query_capacity.json")[::3])
def test_native_bucket_capacity_changes_storage_without_adding_work(row):
    capture = row["host_result"]
    batch, capacity = row["active_sequence_count"], row["native_query_capacity"]
    lengths = tuple(c + 1 for c in capture["contexts"][:batch])
    case, geometry = _case_and_geometry(LOCAL_DIE)
    operator = _make_operator(lengths, heads=capture["heads"], abi=FIA_ABI, splits=2)
    captured_shape = (capacity, *operator.local_shape[1:])
    padded = replace(operator, padded_shape=captured_shape, launch_shape=captured_shape)
    prepared, plan = _recipe_book().prepare(padded, geometry)
    assert prepared.local_shape == operator.local_shape
    assert prepared.padded_shape == prepared.launch_shape == captured_shape
    assert plan.work == governed.build(operator, geometry).work
    assert plan.work.matrix["bf16"] == 2 * sum(lengths) * capture["heads"] * (DQK + LATENT)
    assert verify_operator(case, padded, governed.build)["status"] == "passed"


@pytest.mark.parametrize("length,heads,page", [(1, 12, 128), (4097, 12, 128), (4097, 12, 64),
                                               (8193, 32, 128)])
def test_direct_traffic_follows_context_head_and_page_axes(length, heads, page):
    case, geometry = _case_and_geometry(DIRECT_DIE, 1)
    operator = _make_operator((length,), heads=heads, page_size=page)
    _, plan, ctx = _prepare(case, geometry, operator)
    (query,) = _queries(plan.program(ctx))
    (split,) = _splits(query)
    loads = tuple(_stages(split, "kv_load"))
    page_reads = sum(ceil(((idx * 112) % page + keys) / page) for idx, keys in _split_keys(query)[0])
    assert sum(s.metadata["key_pages_logical_bytes"] for s in loads) == page_reads * 4
    assert sum(s.metadata["key_logical_bytes"] for s in loads) == length * RECORD
    assert sum(s.metadata.get("query_logical_bytes", 0) for s in _stages(split, "q_load")) == heads * RECORD
    assert query.stages[-1].metadata["output_logical_bytes"] == heads * LATENT * 2
    assert not any(s.work for s in _stages(split, "pv_load"))


def test_native_execution_preserves_page_table_and_cache_producer_dependencies():
    raw = hardware_case()
    raw["machine"] = {"backend": "ascend", "sku": DIRECT_DIE, "num_devices": 1}
    raw["model"].update(layers=1, hidden_size=32, attention="mla", num_heads=4, num_kv_heads=1,
                        head_dim=192, mla=dict(q_lora_rank=32, kv_lora_rank=LATENT, qk_nope_head_dim=128,
                                               qk_rope_head_dim=ROPE, v_head_dim=128))
    case = parse_case(raw)
    views = (RequestView("first", 1, 16, True, "decode"), RequestView("second", 1, 32, True, "decode"))
    work = Frontend()(IterationPlan(views), case.model, case.runtime, case.machine)
    base = next(op for op in work.ops if op.family == "attention")
    consumer = next(op for op in work.ops if op.recipe == "v_expand")
    assert base.op_id in consumer.deps
    assert any("kv_store" in dep for dep in base.deps)
    native = governed.make_op(sequences=tuple((v.q_len, v.context + v.q_len) for v in views),
                              logical_heads=4, physical_heads=8, abi=FLASH_ABI, op_id=base.op_id,
                              deps=(*base.deps, "page_table"), layer=base.layer, stage=base.stage)
    native = replace(native, source_boundary="kernel")
    graph = tuple(native if op is base else op for op in work.ops)
    ready = READY + 4
    report = graph_costs(case, graph, recipes=_recipe_book(source_kernel=True), external_inputs={"page_table": ready},
                         input_ready_ms=READY, trace_limit=4000, isolated=False)
    timeline = {r["op_id"]: r for r in report["timeline"]}
    row = timeline[native.op_id]
    assert row["source_boundary"] == "kernel" and row["ready_ms"] >= ready
    assert all(timeline[d]["complete_ms"] <= row["ready_ms"] for d in base.deps)
    assert row["complete_ms"] <= timeline[consumer.op_id]["ready_ms"]
    assert all(e["start_ms"] >= READY for e in report["events"] if e["demand"])
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert not any(report["execution"]["live_allocated"].values())


@pytest.mark.parametrize("replay,submits", [("none", 2), ("full", 1)])
def test_governed_kernel_boundaries_and_graph_submission(replay, submits):
    case, _ = _case_and_geometry(DIRECT_DIE)
    launch_us = 50
    dies = tuple(replace(die, launch_us=launch_us) for die in case.machine.dies)
    case = replace(case, machine=replace(case.machine, dies=dies))
    first = replace(_make_operator((17,), op_id="first"), source_boundary="kernel")
    second = replace(_make_operator((33,), op_id="second", deps=(first.op_id,), splits=2, s2_base_size=32),
                     source_boundary="kernel")
    plan = IterationPlan((RequestView("r", 1, 0, True, "decode"),))
    iteration = IterationWork((first, second), plan, 1, 1, 1, None, "decode", "source", replay_scope=replay)
    execution = Execution(case, CostModel(case), recipes=_recipe_book(source_kernel=True), trace_limit=4000)
    result = execution.add(iteration, 0, READY)
    execution.engine.advance(until=(result.completion,))
    trace = tuple(execution.engine.trace)
    assert result.operators == 2
    assert sum(row["task"].endswith(":submit") for row in trace) == submits
    assert execution.engine.serviced["host@r0:s0"] == pytest.approx(submits * launch_us / 1000)
    assert execution.engine.summary()["live_tasks"] == execution.engine.summary()["live_programs"] == 0


@pytest.mark.parametrize("batch", [1, 4])
def test_source_parallelism_only_uses_owners_with_intervals(batch):
    case, serial = _case_and_geometry(DIRECT_DIE, 1)
    _, parallel = _case_and_geometry(DIRECT_DIE, 4)
    operator = _make_operator((129,) * batch)
    timings = []
    for geometry in (serial, parallel):
        _, trace, duration = _simulate(case, geometry, operator)
        active = {aic for row in trace for resource, owners in row["placement"].items()
                  if resource.partition("@")[0] == "fixpipe" for aic in owners}
        assert len(active) == min(batch, geometry.stream_owners.aic_limit)
        timings.append(duration)
    if batch == 1:
        assert timings[0] == pytest.approx(timings[1])
    else:
        assert timings[0] > timings[1]


def test_owner_limit_and_rates_affect_timing_without_changing_prepared_work():
    case, geometry = _case_and_geometry(DIRECT_DIE)
    operator = _make_operator((4097,), splits=3)
    baseline = verify_operator(case, operator, governed.build)["result"]
    dies = tuple(replace(die, rates=tuple((r, v / 2) for r, v in die.rates),
                         compute_peaks=tuple((r, v / 2) for r, v in die.compute_peaks))
                 for die in case.machine.dies)
    slow_case = replace(case, machine=replace(case.machine, dies=dies))
    slow = verify_operator(slow_case, operator, governed.build)["result"]
    assert baseline["duration_ms"] < slow["duration_ms"]
    assert baseline["operator"]["metadata"]["resource_work"] == slow["operator"]["metadata"]["resource_work"]
    assert governed.build(operator, geometry).work.matrix["bf16"] == baseline["operator"]["flops"]


def test_large_split_grid_and_long_context_lower_lazily():
    case, geometry = _case_and_geometry(DIRECT_DIE, 4)
    operator = _make_operator((131073, 1), s2_base_size=96, splits=8)
    _, plan, context = _prepare(case, geometry, operator)
    pipeline = plan.program(context)
    assert pipeline.slots == 1 and pipeline.count == 2
    full, tail = _queries(pipeline)
    assert full.stages[0].program.count == 2
    programs = tuple(_splits(full))
    assert len(programs) == 8 and all(p.slots == 3 for p in programs)
    assert sum(p.count - (p.slots - 1) for p in programs) == ceil(131073 / 112)
    assert tail.stages[0].program.count == 1 and len(tuple(_splits(tail))) == 1
    _, trace, _ = _simulate(case, geometry, _make_operator((769,), splits=4))
    fixpipe_stages = [event for event in trace if any(
        resource.partition("@")[0] == "fixpipe" for resource in event["demand"])]
    assert len(fixpipe_stages) == 3 * ceil(769 / 112)


@pytest.mark.parametrize("changes,match", [
    ({"sequences": ((1, -1),)}, "invalid_kv_len"),
    ({"sequences": ((-1, 4),)}, "invalid_q_len"),
    ({"logical_heads": 0}, "invalid_logical_heads"),
    ({"physical_heads": 1}, "physical_heads_below_logical"),
    ({"page_size": 0}, "invalid_page_size"), ({"splits": 0}, "invalid_splits"),
    ({"block_table_len": 0}, "block_table_coverage"),
    ({"block_table": ((3,), (4,))}, "invalid_block_table_batch"),
    ({"n_ori": 1}, "sparse_attributes_require_mqsmla"),
])
def test_native_contract_rejects_invalid_active_geometry(changes, match):
    params = dict(sequences=((1, 17),), logical_heads=4, physical_heads=4, abi=FLASH_ABI)
    with pytest.raises(ModelingError, match=match):
        governed.make_op(**(params | changes))


def test_native_dispatch_requires_abi_dtype_and_owner_geometry():
    _, direct = _case_and_geometry(DIRECT_DIE)
    _, workspace = _case_and_geometry(LOCAL_DIE)
    operator = _make_operator((17,))
    with pytest.raises(UnmodeledCapabilityError, match="paged_mla_datapath_required"):
        governed.build(operator, replace(direct, datapath=None))
    with pytest.raises(UnmodeledCapabilityError, match="paged_mla_unsupported_abi"):
        governed.build(operator, workspace)
    with pytest.raises(UnmodeledCapabilityError, match="paged_mla_unsupported_dtype"):
        governed.build(replace(operator, dtype="fp8"), direct)
    aic, (aiv, _) = direct.datapath.aic_aiv_pairs[0]
    truncated = replace(direct, stream_owners=StreamOwners((aic,), (aiv,), 1, 1))
    with pytest.raises(UnmodeledCapabilityError, match="paged_mla_requires_two_aivs"):
        governed.build(operator, truncated)
    with pytest.raises(UnmodeledCapabilityError, match="paged_mla_abi_conflict"):
        governed.build(replace(operator, metadata=(*operator.metadata, ("requested_abi", FIA_ABI))), direct)


@pytest.mark.requires_profiler
def test_structure_work_and_layout_follow_the_captured_tensor_extents(monkeypatch, profiler_api):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.tensor_contracts import observed_contract

    rel = Path("tests/fixtures/attention_cache_contracts/a5_kimi_mla31.csv")
    installed = Path(mindie_profiler.__file__).resolve().parents[1] / rel
    capture = installed if installed.is_file() else ROOT.parent.parent / "profiler" / rel
    rows = [observed_contract(row) for row in kernel_rows(capture)]
    assert [row[-1] for row in rows] == [1, 1, 32]
    case, geometry = _case_and_geometry(DIRECT_DIE)
    book = _recipe_book()
    nodes = []
    for row in rows:
        _, _, (query, rotary), (key, value, _), output, capacity = row
        (_, heads, _, latent), dtype, _ = query
        assert rotary[0][-1] == ROPE and key == value and output[0][0] == heads
        page = key[0][1]
        operator = _make_operator((4097,), heads=heads, abi=FIA_ABI, page_size=page)
        assert (latent, page, operator.dtype) == (LATENT, PAGE, dtype)
        shape = (capacity, heads, DQK)
        active = replace(operator, padded_shape=shape, launch_shape=shape)
        actual = verify_operator(case, active, governed.build)["result"]["operator"]
        with monkeypatch.context() as patch:
            def forbid_timing(*args, **kwargs):
                pytest.fail("Structure inspection entered the timing engine")
            patch.setattr(Engine, "advance", forbid_timing)
            prepared, _ = book.prepare(active, geometry)
            node = node_physical(prepared, book=book)
        physical = node["physical_work"]
        assert all(physical[r] == actual[r] for r in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops"))
        assert physical["resource_work"] == actual["metadata"]["resource_work"]
        assert {"mte1", "fixpipe", "mte2_aic"} <= physical["resource_work"].keys()
        nodes.append({**node, "deps": []})
    assert nodes[0] == nodes[1]
    assert nodes[0]["physical_signature"] != nodes[-1]["physical_signature"]
    assert nodes[0]["physical_work"] == nodes[-1]["physical_work"]
    pair = {"entry": nodes[0], "second": {**nodes[1], "deps": ["entry"]}}
    group = {"node_ids": list(pair), "external_outputs": ["second"], "implementation": {}, "coverage": "complete"}
    summed = subgraph_physical(pair, group)["physical_work"]
    expected = nodes[0]["physical_work"]["resource_work"]
    assert summed["resource_work"] == {name: 2 * amount for name, amount in expected.items()}
    assert summed["launches"] == 2


@pytest.mark.parametrize("length,latent,rope,splits", [(449, 12, 4, 5), (113, LATENT, ROPE, 7)])
def test_normalized_split_outputs_merge_by_logsumexp(length, latent, rope, splits):
    case, geometry = _case_and_geometry(DIRECT_DIE)
    operator = _make_operator((length,), heads=4, abi=DSL_FLASH_ABI, latent_dim=latent, rope_dim=rope,
                              s2_base_size=3, splits=splits)
    _, plan, context = _prepare(case, geometry, operator)
    (query_group,) = _queries(plan.program(context))
    intervals = [tuple(s.metadata["kv_tile"] for s in _stages(split, "kv_load")) for split in _splits(query_group)]
    spans = [np.arange(tiles[0] * 112, min(length, (tiles[-1] + 1) * 112)) for tiles in intervals]
    assert [i for span in spans for i in span] == list(range(length))
    rng = np.random.default_rng(7301)
    query, key = rng.normal(size=(4, latent + rope)), rng.normal(size=(length, latent + rope))
    scores, values = (query @ key.T) * 0.37, key[:, :latent]
    dense_lse = np.logaddexp.reduce(scores, axis=1)
    dense = np.exp(scores - dense_lse[:, None]) @ values
    local_lse = [np.logaddexp.reduce(scores[:, span], axis=1) for span in spans]
    local_out = [np.exp(scores[:, span] - lse[:, None]) @ values[span] for span, lse in zip(spans, local_lse)]
    total_lse = np.logaddexp.reduce(local_lse, axis=0)
    merged = sum(np.exp(lse - total_lse)[:, None] * out for lse, out in zip(local_lse, local_out))
    np.testing.assert_allclose(total_lse, dense_lse, atol=1e-12)
    np.testing.assert_allclose(merged, dense, atol=1e-12)
    assert not np.allclose(merged, np.exp(scores - dense_lse[:, None]) @ rng.normal(size=values.shape))


@pytest.fixture
def keypool_selection():
    plan, model = source_cases._case_model(pool=16)
    model = replace(model, quant=replace(model.quant, index_scale_dtype="fp32"))
    runtime = source_cases._Runtime(o_proj_tp=8)

    def select(views, *, padded_tokens=8, **shard):
        builder = Builder("s0.l0", plan, 0, views, model, "decode")
        index_pool_native.selection(
            builder, query="q", weights="w", cache="cache", views=views, plan=plan,
            quant=builder.quant, padded_tokens=padded_tokens, runtime=runtime, **shard)
        return {op.op_id.rsplit(".", 1)[-1]: op for op in builder.ops}

    return select


def test_keypool_shards_request_rows_and_packs_local_indices_and_tails(keypool_selection):
    single_long = (RequestView("r0", 8, 131072, False, "decode"),)
    eight_decode = tuple(RequestView(f"r{i}", 1, 131072, False, "decode") for i in range(8))
    assert [view.q_len for view in row_shard_partition(eight_decode, 8)] == [1]
    owned = row_shard_partition(single_long, 8)
    assert len(owned) == 1 and owned[0].q_len == 1

    ops = keypool_selection(single_long, row_shard_tp=8)
    gather = next(op for op in ops.values() if op.recipe == index_pool_native.ROW_SHARD_GATHER)
    meta = dict(gather.metadata)
    assert meta["local_rows"] == 1
    assert meta["local_payload_bytes"] == 2048 * 4 == 8192
    assert gather.link_bytes == 57344
    assert meta["assembled_bytes"] == 65536
    assert meta["assembled_shape"] == (8, 2048)
    pack = next(op for op in ops.values() if op.recipe == "index_topk")
    assert pack.mte2_bytes == 8192 + 8 + 8
    assert pack.mte3_bytes == 8192


def test_keypool_gates_packing_before_exchange_and_counts_indices_once(keypool_selection):
    views = tuple(RequestView(str(i), 1, 131072, False, "decode") for i in range(8192))
    assert native_shard_sizes(views[:8], 16, 8) == (8,)
    shallow = (replace(views[0], q_len=8192, context=0),)
    assert native_shard_sizes(shallow, 16, 8) == (8192,)
    sizes = native_shard_sizes(views, 16, 8)
    assert sizes == (1023, 1024, 1024, 1024, 1024, 1024, 1024, 1025)
    count = 0
    for rank, owned in enumerate(sizes):
        ops = keypool_selection(views, padded_tokens=8192, row_shard_rank=rank)
        pack, staged = ops["index_topk"], ops["index_exchange_stage"]
        wire, restore = ops["index_kpool_row_shard_gather"], ops["index_exchange_restore"]
        assert pack.deps == (ops["cann_pool_key_indexer"].op_id,)
        assert staged.deps == (pack.op_id,)
        assert wire.deps == (staged.op_id,)
        assert restore.deps == (wire.op_id,)
        assert wire.link_bytes == 7 * 1025 * 2063 * 4
        assert pack.mte3_bytes == owned * 2063 * 4
        assert pack.mte2_bytes == owned * (2048 * 4 + 2 * 8)
        count += pack.mte2_bytes - owned * 2 * 8
    assert count == 67108864


@pytest.mark.parametrize("shards,expected", [({}, 0), ({"row_shard_tp": 8}, 1)])
def test_keypool_small_batch_only_gathers_with_explicit_override(keypool_selection, shards, expected):
    views = (RequestView("r", 8, 131072, False, "decode"),)
    ops = keypool_selection(views, **shards)
    wires = [op for op in ops.values() if op.recipe == index_pool_native.ROW_SHARD_GATHER]
    assert len(wires) == expected
    assert sum(op.link_bytes for op in wires) == expected * 57344


def test_mla_prolog_without_query_norm_output_has_no_scale_shape():
    plan, model = source_cases._prolog_case()
    plan = replace(plan, indexer=False, owns_topk=False,
                   index_query_per_layer=False)
    ops = [
        source_cases._op("s0.l0.q_a", "gemm",
                   (8, source_cases.PROLOG_QUERY_LORA_RANK, source_cases.HIDDEN)),
        source_cases._op("s0.l0.q_norm", "rmsnorm",
                   (8, source_cases.PROLOG_QUERY_LORA_RANK)),
        source_cases._op("s0.l0.q_b", "gemm",
                   (8, 8 * source_cases.HEAD_WIDTH, source_cases.KV_RANK)),
        source_cases._op("s0.l0.kv_a", "gemm",
                   (8, source_cases.HEAD_WIDTH, source_cases.HIDDEN)),
        source_cases._op("s0.l0.kv_norm", "rmsnorm", (8, source_cases.KV_RANK)),
        source_cases._op("s0.l0.q_absorb", "gemm",
                   (64, source_cases.KV_RANK, source_cases.KV_RANK)),
    ]
    prolog.declare(
        ops, prefix="s0.l0", plan=plan, model=model,
        parallel=ParallelSpec(tp=8, cp=8, pcp=1),
        quant=model.quant, tokens=8,
        members=tuple(op.op_id for op in ops),
        q_a="s0.l0.q_a", q_b="s0.l0.q_b", kv_a="s0.l0.kv_a",
        q_absorb="s0.l0.q_absorb", cache_layout=source_cases.LAYOUT)
    view = source_cases._declaration(ops, "s0.l0.q_absorb")["tensor_view"]

    assert view["query_norm_output"] is False
    assert view["query_norm_scale_dtype"] == "fp8_e8m0"
    assert view["query_norm_scale_shape"] == ()


def test_indexed_attention_cast_ratio_and_paired_sync_helpers():
    from types import SimpleNamespace
    from mindie_sim.prediction.operators.paged_mla.indexed_attention import (
        _dequant_cast_scale, _indexed_phase_sync_ms,
    )
    from mindie_sim.prediction.operators.sharedkv.l0_sync import p09_cast_to_alu_op_ratio
    from mindie_sim.schema.common import ModelingError

    uncal = SimpleNamespace(primitive_table=None)
    assert p09_cast_to_alu_op_ratio(uncal) == 1.0
    assert _dequant_cast_scale(uncal) == 1.0
    assert _indexed_phase_sync_ms(uncal, "scores", 512) == 0.0

    cells = {
        "P07_vec_alu": {"vec_alu_ops_per_s_per_aiv": 34e9},
        "P09_vec_reduce_cast": {"cast_bytes_per_s_per_aiv": 388e9},
        "P03_fixpipe": {"setup_ns": 170.0},
        "P13_crosscore_sync": {"mode2_paired_ns": 560.0},
    }

    def _cell(cid):
        if cid not in cells:
            raise ModelingError(cid)
        return SimpleNamespace(band=lambda m: SimpleNamespace(p50=cells[cid].get(m)))

    cal = SimpleNamespace(primitive_table=SimpleNamespace(cell=_cell))
    ratio = p09_cast_to_alu_op_ratio(cal)
    assert 0.3 < ratio < 0.4
    assert 0.55 < _dequant_cast_scale(cal) < 0.65
    assert _indexed_phase_sync_ms(cal, "scores", 512) == pytest.approx((560.0 + 4 * 170.0) / 1e6)
    assert _indexed_phase_sync_ms(cal, "partial", 512) == pytest.approx((560.0 + 4 * 170.0) / 1e6)
    assert _indexed_phase_sync_ms(cal, "gather", 512) == 0.0


def test_keypool_row_shard_propagates_physical_rank_and_noncausal_horizon():
    from mindie_sim import compose_case, target_graphs

    raw = compose_case(
        "glm-5.3-flash", "a5", tp=2, dp=1, ep=1, layers=4,
        prompt_tokens=128, output_tokens=2,
    )
    lowerings = raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})
    lowerings["pool_key_indexer"] = "vllm_ascend_cann_keypool_caller"
    raw["runtime"].setdefault("metadata", {})["keypool_row_shard_tp"] = 2
    causal_case = parse_case(raw)
    for rank in range(2):
        ops = target_graphs(
            causal_case, scope="body", phase="prefill", tokens=128,
            sequences=1, context=0, physical_rank=rank,
        )[0]
        meta = dict(next(o for o in ops if o.recipe == "cann_pool_key_indexer").metadata)
        expected = sum(pos // meta["pool_size"] for pos in range(64 * rank + 1, 64 * (rank + 1) + 1))
        assert meta["scanned_pools"] == expected

    raw["model"]["layer_plans"][3]["noncausal_attention"] = True
    noncausal_case = parse_case(raw)
    for rank in range(2):
        ops = target_graphs(
            noncausal_case, scope="body", phase="prefill", tokens=128,
            sequences=1, context=0, physical_rank=rank,
        )[0]
        meta = dict(next(o for o in ops if o.recipe == "cann_pool_key_indexer").metadata)
        assert meta["scanned_pools"] == meta["valid_query_rows"] * (128 // meta["pool_size"]) == 2048

    dp_raw = compose_case(
        "glm-5.3-flash", "a5", tp=1, dp=2, ep=1, layers=4,
        prompt_tokens=128, output_tokens=2,
    )
    dp_raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
        "pool_key_indexer"
    ] = "vllm_ascend_cann_keypool_caller"
    dp_raw["runtime"].setdefault("metadata", {})["keypool_row_shard_tp"] = 2
    dp_case = parse_case(dp_raw)
    dp_scans = [
        dict(next(
            o for o in target_graphs(
                dp_case, scope="body", phase="prefill", tokens=128,
                sequences=1, context=0, physical_rank=rank,
            )[0]
            if o.recipe == "cann_pool_key_indexer"
        ).metadata)["scanned_pools"]
        for rank in range(2)
    ]
    assert dp_scans == [496, 1520]

    from mindie_sim.prediction.operators.sparse_indexer.index_pool_native import resolve_row_shard_rank
    from mindie_sim.prediction.sharding.local_tp import local_tp_groups

    pcp_raw = compose_case(
        "glm-5.3-flash", "a5", tp=1, dp=4, ep=1, layers=4,
        prompt_tokens=128, output_tokens=2,
    )
    pcp_raw["runtime"]["parallel"].update(dp=2, pcp=2)
    pcp_case = parse_case(pcp_raw)
    groups = local_tp_groups(pcp_case.machine, pcp_case.runtime.parallel, stage=0, width=2)
    assert groups == ((0, 2), (1, 3))
    for grp in groups:
        assert tuple(
            resolve_row_shard_rank({}, 2, physical_rank=r, parallel=pcp_case.runtime.parallel, machine=pcp_case.machine)
            for r in grp
        ) == (0, 1)


def test_analytic_index_pool_zero_l2_charges_full_hbm_reads():
    from mindie_sim.model_catalog import load_model
    from mindie_sim.prediction.operators.sparse_indexer.index_pool import selection
    from mindie_sim.prediction.runtime.l2_residency import UsableL2Bytes

    model = load_model("glm-5.3-flash")
    plan = next(p for p in model.layer_plans if p.index_kpool > 1)
    views = (RequestView("r", 32, 8192, False, "prefill"),)
    builder = Builder("s0.l3", plan, 0, views, model, "prefill")
    selection(
        builder, query="q", weights="w", cache="k", views=views, plan=plan,
        quant=model.quant, padded_tokens=32, l2_bytes=UsableL2Bytes(0),
    )
    op = next(o for o in builder.ops if o.recipe == "index_pool_scores")
    assert int(dict(op.metadata)["score_l2_hit_bytes"]) == 0
    assert int(op.hbm_bytes) == int(op.mte2_bytes) == 16_824_320
