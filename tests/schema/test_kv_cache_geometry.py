"""Paged KV layouts, causal index pools, and allocator byte conservation.

[Role] Exercise cache geometry through schema declarations, lowering, and the
  allocation and transfer ledgers.
[Invariants] Cache pages cover owned rows. Physical packing retains scale
  bytes, distinct planes, private tails, and persistent state identities.
[Timeouts & Locks] Deterministic local graph and capacity checks. No locks.
[Side-Effect Policy] Read retained case JSON and simulate in memory.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from itertools import combinations, product
import json
from pathlib import Path
import tracemalloc
from types import SimpleNamespace

import pytest

from _support.builders import dense_case, linear_layer, view
from _support.hardware import machine
from _support.memory_cases import SPARSE_OFFLOAD_SKU, compressed_case, index_pool_case, index_pool_layer, linear_work
from _support.memory_cases import case as memory_case
from mindie_sim import compose_case, graph_costs, predict, predict_pd, target_graphs
from mindie_sim.catalog.cache import cache_row_split, layer_cache_components, owned_token_rows, request_cache_breakdown
from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.catalog.cache_ownership import batch_cache_footprint
from mindie_sim.catalog.cache_pages import CachePageGroup
from mindie_sim.catalog.index_pool import pool_selection
from mindie_sim.catalog.mqsmla_layout import packed_row_parts
from mindie_sim.catalog.paired_sfa import page_groups
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import cache_row_bytes_by_layer, layer_weight_terms
from mindie_sim.prediction.operators.sparse_indexer import family_quant_indexer, index_pool; from mindie_sim.prediction.sharding import context_parallel as cp_graph
from mindie_sim.prediction.operators.paged_mla import indexed_quant_attention as c8
from mindie_sim.prediction.operators.attention.attention import _CacheProducer
from mindie_sim.prediction.operators.attention.attention_decode import StagedGQADecode
from mindie_sim.prediction.operators.attention.batch_prefix_attention import nested_prefix_union
from mindie_sim.prediction.body_defaults import effective_body_declaration
from mindie_sim.prediction.cache_representation import store
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.capacity.features import batch_cache_bytes
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.execution.state import bind_op_accesses
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.native_binding import bind_native_call
from mindie_sim.prediction.offload import OffloadPolicy, _accesses_from_ops, project_state_tiers
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.disaggregated.ranges import transfers_for_runs
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, RecipeContext
from mindie_sim.prediction.operators.sharedkv.sharedkv_metadata import query_partitions
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
BLOCK = 128

def paged_case(name="qwen38-a5", *, layers=4, **policy):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    raw["model"].update({} if layers is None else {"layers": layers})
    return parse_case(raw | {"runtime": raw["runtime"] | {"attention_policy": policy}})

def attention_ops(case, **overrides):
    return target_operators(case, **(dict(phase="decode", tokens=1, context=1028) | overrides))

@pytest.mark.parametrize("name,phase,q,heads,kv_heads", [
    ("qwen35-a3-tp4", "decode", 2, 4, 1), ("qwen35-a3-tp4", "prefill", 8192, 4, 1),
    ("qwen36-a3-tp4", "decode", 1, 6, 1), ("qwen36-a3-tp4", "decode", 4, 6, 1),
    ("qwen38-a5", "decode", 1, 24, 4), ("qwen38-a5", "decode", 4, 24, 4),
])
def test_paged_view_preserves_logical_attention_work(name, phase, q, heads, kv_heads):
    kwargs = dict(phase=phase, tokens=q, context=16384, sequences=2)
    original = target_operators(paged_case(name), **kwargs)
    selected = target_operators(paged_case(name, kv_storage="paged"), **kwargs)
    assert [replace(op, metadata=()) for op in selected] == [replace(op, metadata=()) for op in original]
    tensors = [dict(op.metadata)["attention_tensor_view"] for op in selected if op.family == "attention"]
    assert tensors
    for tensor in tensors:
        assert tensor["query_shape"] == tensor["output_shape"] == (2 * q, heads, 256)
        assert tensor["key_tail_shape"] == tensor["value_tail_shape"] == (BLOCK, kv_heads * 256)
    if name == "qwen35-a3-tp4" and phase == "decode":
        op = next(op for op in selected if op.family == "attention")
        native = bind_native_call(op).typed["attrs"]["native_paged_fia"]
        assert native["ports"]["lse"] is None and not native["state"]["writes"]
        assert native["work"]["head_token_pairs"] == 262168

def test_paged_view_omits_an_unmodeled_context_partition():
    case = paged_case(kv_storage="paged")
    runtime = replace(case.runtime, parallel=replace(case.runtime.parallel, tp=2, cp=2))
    die = case.machine.dies[0]
    ops = attention_ops(replace(case, runtime=runtime, machine=replace(case.machine, dies=(die, replace(die, rank=1)))))
    attention = [op for op in ops if op.family == "attention"]
    assert attention and all("attention_tensor_view" not in dict(op.metadata) for op in attention)

@pytest.mark.parametrize("layout", ("native", "packed", "split", "suffix"))
def test_paged_storage_conserves_typed_reader_and_writer_traffic(layout):
    parsed = paged_case(kv_storage="paged", fused_batch=False)
    packed = {"dtype": "int8", "group_size": 64, "scale_dtype": "fp32"}
    formats = {"native": {}, "packed": {"gqa_kv": packed},
               "split": {"gqa_kv_key": {"dtype": "bf16"}, "gqa_kv_value": packed},
               "suffix": {"gqa_kv": packed | {"suffix_elements": 64, "suffix_dtype": "bf16"}}}[layout]
    model = replace(parsed.model, metadata=tuple(dict(parsed.model.metadata, cache_representations=formats).items()))
    if layout == "suffix":
        ops = []
        store(_CacheProducer(ops, prefix="p", stage=0, layer=0, requests=("r",),
                            phase="decode", model=model), "cache", (), 1, 256, "gqa_kv")
        row, = ops
        assert (row.mte2_bytes, row.mte3_bytes, row.hbm_bytes) == (512, 332, 844)
        return
    ops = attention_ops(replace(parsed, model=model))
    attention = next(op for op in ops if op.family == "attention")
    recipe = StagedGQADecode()
    access = recipe.kv_access(attention)
    assert access.paged and access.page_size == BLOCK and access.page_meta_bytes == 4
    stores = [op for op in ops if op.family == "cache_store" and op.layer == attention.layer]
    if layout == "native":
        contiguous = next(op for op in attention_ops(paged_case(fused_batch=False)) if op.family == "attention")
        before, after = recipe.work(contiguous), recipe.work(attention)
        assert (before.matrix, before.resources["vector"]) == (after.matrix, after.resources["vector"])
        assert after.resources["hbm"] == before.resources["hbm"] + 4 * 9 * 4
    else:
        assert [op.mte3_bytes for op in stores] == ([2176] if layout == "packed" else [2048, 1088])
        assert (access.key_format.dtype == "int8") == (layout == "packed")
        assert access.value_format.dtype == "int8" and access.value_conversion_ops(BLOCK) == BLOCK * 256 * 2
        if layout == "packed":
            assert (access.key_payload_row, access.key_scale_row) == (256, 16)
            assert stores[0].hbm_bytes == 4096 + 2176

@pytest.mark.parametrize("mode", ("admission", "transfer", "duplicate", "pipeline"))
def test_independent_cache_pages_preserve_payload_and_pipeline_ownership(mode):
    raw = memory_case(prompt=3 if mode == "transfer" else 4, output=3 if mode == "transfer" else 1,
                      pp=2 if mode == "pipeline" else 1)
    groups = [dict(layers=[i], tokens_per_page=rows, bytes_per_layer_page=rows * 64)
              for i, rows in enumerate((3, 5))]
    raw["runtime"].update(kv_block_size=8, gpu_memory_utilization=1, memory={
        "derive_workspace": False, "cache_page_groups": groups})
    if mode in {"duplicate", "pipeline"}:
        if mode == "duplicate":
            groups.append(groups[0])
        else:
            groups[:] = [{"layers": [0, 1], "tokens_per_page": 4, "bytes_per_group_page": 512}]
        reason = {"duplicate": "duplicate_owner", "pipeline": "spans_pipeline_stages"}[mode]
        with pytest.raises(ModelingError, match=reason):
            Memory(parse_case(raw))
    elif mode == "transfer":
        raw["pools"] = {side: {key: deepcopy(raw[key]) for key in ("machine", "runtime")}
                        for side in ("prefill", "decode")}
        raw["pools"]["decode"]["runtime"]["memory"]["cache_page_groups"] = [
            {"layers": [0, 1], "tokens_per_page": 7, "bytes_per_layer_page": 448}]
        raw["connector"] = {"bytes_per_s": 128000, "latency_us": 0}
        result = predict(raw, lower=linear_work)
        assert result["execution"]["handoff_bytes"] == 3 * 2 * 64 and result["requests"][0]["finish_ms"] == 8
    else:
        raw["machine"]["device"]["memory_bytes"] = Memory(parse_case(raw)).weights[0] + 704
        memory = Memory(parse_case(raw))
        assert memory.cache_bytes(4) == (704,) and memory.cache_bytes(4, padded=False) == (512,)
        assert memory.resize(0, "r", 4) and not memory.resize(0, "r", 7)
        memory.release(0, "r")
        assert memory.live == [[0]] and predict(raw, lower=linear_work)["status"] == "modeled"

@pytest.mark.parametrize("rows,allocated,payload", [(1, 256, 96), (9, 384, 320)])
def test_group_pages_account_for_window_compression_and_recurrent_state(rows, allocated, payload):
    case = compressed_case()
    groups = [
        {"members": [{"layer": rank, "component": "window"} for rank in (0, 1)],
         "tokens_per_page": 3, "bytes_per_group_page": 96, "resident_tokens": 5},
        {"layers": [0], "components": ["compressor_state"], "fixed_pages": 1,
         "bytes_per_group_page": 160},
    ]
    runtime = replace(case.runtime, kv_block_size=8, memory={"derive_workspace": False, "cache_page_groups": groups})
    memory = Memory(replace(case, runtime=runtime))
    assert (memory.cache_bytes(rows), memory.cache_bytes(rows, padded=False)) == ((allocated,), (payload,))

@pytest.mark.parametrize("tokens,flight,policy,fixed,expected", [
    (1, 8192, "max_model_len", None, 66), (1, 16384, "max_model_len", None, 129),
    (16384, 8192, "max_model_len", 1, 1), (0, 8192, "max_model_len", None, 0),
    (1, 0, "live", None, 2),
])
def test_window_pages_include_flight_and_recycling_extents(tokens, flight, policy, fixed, expected):
    group = CachePageGroup(layers=(0,), tokens_per_page=128, bytes_per_group_page=462848,
                          resident_tokens=128, max_in_flight_tokens=flight,
                          recycling_pages=1, allocation=policy, fixed_pages=fixed or 0)
    assert group.pages(tokens, 16384) == expected

POOL_WIDTH = 128
HISTORY_ROWS = 2048
POOL_SIZE = 4
LONG_CONTEXT = 131201

def test_causal_pool_scan_and_expansion_match_scalar_selections():
    for context, query in product((0, 1, 3, 7, 2046, 4096, LONG_CONTEXT), (0, 1, 3, 6, 9)):
        result = pool_selection(query, context, HISTORY_ROWS, POOL_SIZE, always_select_tail=True)
        visible = list(range(context + 1, context + query + 1))
        complete, tail = zip(*(divmod(n, POOL_SIZE) for n in visible)) if visible else ((), ())
        selected = [min(n, HISTORY_ROWS // POOL_SIZE) for n in complete]
        assert (result.scanned_pools, result.selected_pools, result.selected_tokens) == (
            sum(complete), sum(selected), sum(selected) * POOL_SIZE + sum(tail))
        assert (result.causal_tail_tokens, result.completed_pools) == (sum(tail), tail.count(0))

@pytest.mark.parametrize("batch,eager,context,per_head", [
    (batch, eager, LONG_CONTEXT, False) for batch, eager in product((1, 3, 4, 8), (False, True))
] + [(1, False, 4096, False), (1, False, LONG_CONTEXT, True)])
def test_index_pool_graph_preserves_valid_scan_and_cache_write_geometry(batch, eager, context, per_head):
    raw = index_pool_case()
    raw["runtime"].setdefault("metadata", {})["raw_eager"] = eager
    if per_head:
        raw["runtime"]["kernels"].setdefault("lowerings", {})["pool_key_indexer"] = {
            "implementation": "causal_pool_indexer_analytic", "options": {"score_mode": "per_head_relu"}}
    ops = {op.recipe: op for op in index_pool_layer(raw, batch, context=context)}
    pools = [ctx // POOL_SIZE for ctx in range(context + 1, context + 7)]
    scanned, unique = batch * sum(pools), batch * pools[-1]
    scores, packed = ops["index_pool_scores"], ops["index_topk"]
    metadata = dict(scores.metadata)
    assert metadata["scanned_pools"] == scanned and metadata["valid_query_rows"] == batch * 6
    assert scores.vector_ops == scanned * (32 if per_head else POOL_WIDTH) * 2
    if not per_head:
        assert metadata["score_l2_hit_bytes"] == (scanned - unique) * POOL_WIDTH * 2
        score_bytes = scanned * 4 if eager else 0
        assert scores.hbm_bytes == scores.mte2_bytes - (scanned - unique) * POOL_WIDTH * 2 + score_bytes
        assert ops["index_pool_topk"].mte2_bytes == score_bytes
    padded = metadata["padded_query_rows"]
    assert packed.local_shape == (padded, HISTORY_ROWS + POOL_SIZE - 1)
    attention = next(op for op in ops.values() if op.family == "attention")
    selected = batch * sum(HISTORY_ROWS + n % POOL_SIZE for n in range(context + 1, context + 7))
    assert dict(attention.metadata)["indexed_gather_rows"] == dict(packed.metadata)["selected_token_rows"] == selected

def test_index_pool_residency_and_projection_formats_follow_checkpoint_ownership():
    case = parse_case(index_pool_case())
    plan = case.model.layer_plans[3]
    cache = layer_cache_components(case.model, plan, case.runtime.parallel, LONG_CONTEXT, capacity_padded=False)
    assert cache["index_k"] == LONG_CONTEXT // POOL_SIZE * POOL_WIDTH * 2
    assert cache["index_pool_tail"] == POOL_SIZE * 2 * POOL_WIDTH * 4
    terms = {term.name: term for term in layer_weight_terms(plan, case.model)}
    ops = {op.recipe: op for op in index_pool_layer(index_pool_case())}
    for recipe, name in (("index_k", "indexer_wk"), ("index_pool_gate", "indexer_pool_gate")):
        prepared = terms[name + "_prepared_fp32"]
        assert dict(ops[recipe].metadata)["packed_weight_bytes"] == term_bytes(prepared, case.model.quant, case.runtime.parallel)
        assert term_bytes(prepared, case.model.quant, case.runtime.parallel, source=True) == 0
    assert ops["index_weights"].dtype == "bf16" and "index_k_weights" not in ops

@pytest.mark.parametrize("query,context,ranges,span", [
    (0, 4, (), ()), (1, 4, (), ((0, 1),)), (1, 5, (), ((1, 2),)),
    (2, 3, (), ((0, 1), (3, 4))), (6, LONG_CONTEXT, (), ((0, 4),)),
    (2, 4, ((2, 3), (0, 1)), ((0, 1), (2, 3))),
])
def test_causal_tail_writes_only_owned_ring_slots(query, context, ranges, span):
    raw = index_pool_case()
    if ranges:
        raw["runtime"].setdefault("metadata", {})["offload"] = {"kvc_ddr_ratio": .5, "kvc_ssd_ratio": .25, "ssd_bw_gbps": 14.0}
        raw["model"].setdefault("quant", {}).update(index_dtype="fp8_e4m3", index_scale_dtype="fp16")
        raw["model"].setdefault("metadata", {})["cache_representations"] = {
            "index_k": {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": "fp16"}}
    case = parse_case(raw)
    views = (RequestView("a", query, context, False, "decode", query_ranges=ranges),) if query else ()
    builder = Builder("s0.l3", case.model.layer_plans[3], 0, views, case.model, "decode")
    index_pool.producer(builder, source="hidden", views=views, plan=builder.plan, model=case.model,
                        quant=builder.quant, padded_tokens=query, state_versions={"a": 7},
                        state_owners={"a": 0} if ranges else None)
    tail = next(op for op in builder.ops if op.recipe == "index_pool_tail_store")
    writer = next(op for op in builder.ops if op.recipe == "index_store")
    assert index_pool.packed_parent(writer)["scale"]["row_bytes"] == (8 if ranges else 0)
    work = IterationWork((tail,), IterationPlan(views), 1, query, query, None, "decode", "pool_tail")
    accesses = bind_op_accesses(work, tail)
    expected = sum(hi - lo for lo, hi in span) * POOL_WIDTH * 2 * 4
    assert dict(tail.metadata)["state_access_total_bytes"] == expected and len(accesses) == bool(span)
    if span:
        access, = (binding.access for binding in accesses)
        assert (access.span, access.bytes, access.state_version) == (span, expected, 7)
        tier, = project_state_tiers(index_pool.state_accesses(tail),
                                    OffloadPolicy(kvc_ddr_ratio=.5, kvc_ssd_ratio=.25, ssd_bw_gbps=14.0), expected_bytes=expected)
        assert (tier["hbm_bytes"], sum(row.bytes for row in _accesses_from_ops(work))) == (expected * .75, expected)
    if ranges:
        execution = Execution(case, CostModel(case), include_serving_io=False)
        inputs = {dep: execution.engine.submit(dep) for dep in tail.deps}
        step = execution.add(work, 0, 0, external_inputs=inputs)
        execution.engine.advance(until=(step.completion,))
        for tier, factor in (("ddr", .5), ("ssd", .25)):
            assert execution.engine.serviced[f"offload_{tier}@r0:s0"] == expected * factor

def test_sparse_pipeline_compaction_preserves_issued_work_and_readiness():
    ledgers = []
    for compact in (False, True):
        raw = index_pool_case()
        raw["runtime"]["kernels"]["programs"]["indexed_attention"]["compact"] = compact
        ops = [node for node in index_pool_layer(raw, 2, 3, 2047) if node.family == "attention"]
        result = graph_costs(raw, ops, input_ready_ms=5, isolated=False,
                             external_inputs={dep: 7 for op in ops for dep in op.deps}, trace_limit=1000)
        assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
        assert all(event["start_ms"] >= 7 for event in result["events"] if event["demand"])
        ledgers.append(result["execution"]["serviced"])
    assert ledgers[0] == pytest.approx(ledgers[1])

def scalar_partitions(query, context, window, ratio, topk, heads, cores):
    costs = []
    for n in range(context + 1, context + query + 1):
        sizes = (min(n, window), min(n // ratio, topk))
        blocks = [6 * ((heads + 15) // 16) + 10 * ((min(size - i, 512) + 63) // 64)
                  for size in sizes for i in range(0, size, 512)]
        costs.append((sum(blocks), blocks[-1]))
    remaining, maximum = sum(row[0] for row in costs), max(row[0] for row in costs)
    owners, cursor = [], 0
    for core in range(cores):
        if cursor == query:
            break
        limit = max(remaining // (cores - core), maximum)
        start, spent = cursor, 0
        if remaining <= limit + costs[-1][1] // 2:
            owners.append((start, query))
            break
        for cost, tail in costs[start:]:
            if spent + cost > limit + tail // 2:
                break
            spent += cost
            cursor += 1
        remaining -= spent
        owners.append((start, cursor))
    return tuple(owners)

@pytest.mark.parametrize("args", [
    (8, 0, 128, 4, 512, 8, 4), (8, 0, 128, 4, 512, 8, 20), (3, 0, 128, 128, 512, 4, 20),
    (65, 0, 65, 4, 512, 8, 4), (257, 0, 512, 4, 512, 12, 20), (65, 2047, 128, 4, 1024, 16, 7),
    (129, 4095, 128, 4, 1024, 8, 20), (513, 0, 1, 128, 512, 8, 3), (513, 8191, 128, 128, 512, 8, 20),
    (4096, 0, 128, 4, 512, 8, 20), (1024, 0, 128, 4, 1024, 8, 20), (1024, 3072, 128, 4, 1024, 8, 20),
])
def test_shared_cache_partitions_cover_each_query_once(args):
    actual = query_partitions(*args)
    assert actual == scalar_partitions(*args)
    assert [i for lo, hi in actual for i in range(lo, hi)] == list(range(args[0]))

def test_query_partition_memory_scales_with_cost_bands():
    query_partitions.cache_clear()
    tracemalloc.start()
    try:
        partitions = query_partitions(10**12, 0, 512, 128, 1024, 8, 32)
        assert tracemalloc.get_traced_memory()[1] < 64 * 1024
    finally:
        tracemalloc.stop()
    assert len(partitions) == 32 and sum(hi - lo for lo, hi in partitions) == 10**12

def catalog_case(model, **overrides):
    settings = dict(hardware=SPARSE_OFFLOAD_SKU, graph_capture_mode="none",
                    prompt_tokens=128, output_tokens=2)
    return parse_case(compose_case(model, **(settings | overrides)))

@pytest.mark.parametrize("model,query,total,draft", [
    ("deepseek-v4.1-flash", 1, 9_830_400, 208896), ("glm-5.3-flash", 4, 41_431_040, None),
    ("step-5-preview", 1, None, None),
])
def test_draft_cache_allocation_and_batch_ledger_conserve_bytes(model, query, total, draft):
    case = catalog_case(model, tp=4, ep=4, layers=4, mtp_proposal_tokens=3, mtp_acceptance_length=4)
    memory = Memory(case)
    allocated = memory.cache_bytes(8192)
    assert memory.allocation_report(8192)["cache"]["allocated"]["total_bytes_per_stage_rank"] == list(allocated)
    if total is not None:
        ledger = request_cache_breakdown(case.model, case.runtime.parallel, 8192, draft_kind="mtp", runtime=case.runtime)
        batch = batch_cache_bytes(case, sequences=1, context=8192 - query, tokens_per_seq=query, phase="decode", padded=True)
        assert tuple(batch.total_bytes) == allocated == (total,) and allocated[0] >= ledger["total_bytes"]
        if draft is not None:
            assert ledger["by_component"]["draft_kv"] == draft

@pytest.mark.parametrize("layers,tp", [(2, 1), (4, 1), (4, 4)])
def test_packed_window_rows_survive_layer_slicing_and_select_mixed_attention(layers, tp):
    case = catalog_case("deepseek-v4.1-flash", tp=tp, ep=tp, layers=layers)
    assert tuple(dict(cache_row_bytes_by_layer(case.model))[i] for i in (0, 1)) == (544, 544)
    graph = target_graphs(case, phase="decode", tokens=1, context=2048, scope="model")[0]
    assert any("mixed_quant_sparse" in op.recipe for op in graph if op.layer == 0 and op.family == "attention")

@pytest.mark.parametrize("model,region,elements,row_bytes,layer", [
    ("glm-5.3-flash", "mla_kv", 512, 528, 3), ("gemma-4-31b", "gqa_kv", None, 8192, 5),
])
def test_cache_row_geometry_is_independent_of_execution_phase(model, region, elements, row_bytes, layer):
    options = {"quant": {"name": "w8a8", "weight_format": "int8", "activation_format": "int8", "kv_dtype": "int8"}} if elements else {}
    case = catalog_case(model, tp=8 if elements else 1, ep=8 if elements else 1, layers=layer + 1, **options)
    for phase in ("prefill", "decode"):
        effective, _ = effective_body_declaration(case, phase)
        assert dict(cache_row_bytes_by_layer(effective.model))[layer] == row_bytes
        width = elements or effective.model.layer_plans[layer].cache_row_elements
        assert cache_format(effective.model, region).row_bytes(width) == row_bytes

def test_shared_prefix_footprint_is_independent_of_request_order():
    case = catalog_case("kimi-k3.1", tp=8, ep=8, layers=4, block_size=1536, prefix_cache_mode="shared")
    requests = tuple(RequestView(str(i), 1, 8000, True, "decode", prefix_tokens=prefix,
                                 prefix_cache_key="doc") for i, prefix in enumerate((1536, 3072)))
    a, b = (batch_cache_footprint(case, requests=order, replica=0, phase="decode", padded=True)
            for order in (requests, requests[::-1]))
    assert a.total == b.total and a.shared_once_bytes == b.shared_once_bytes and a.shared_once_total > 0

@pytest.mark.parametrize("dtype,parts", [("bf16", (576, 14, 18)), ("uint8", (576, 7, 1)),
                                       ("fp8_e8m0", (576, 7, 1))])
def test_packed_cache_scale_size_and_aligned_extent(dtype, parts):
    case = paged_case("longctx-decode-20261030/v41-a5-tp1-dp8-ep8", layers=None)
    region = {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": dtype}
    model = replace(case.model, metadata=tuple(dict(case.model.metadata,
                    cache_representations=dict(window=region, compressed_kv=region)).items()))
    assert packed_row_parts(model, "compressed_kv", 512) == parts
    assert cache_row_split(model, "compressed_kv", 512) == (parts[0], sum(parts[1:]))

def test_nested_prefix_upper_bound_matches_a_maximum_distinct_selection():
    draws = [(extent, count) for extent in range(4) for count in range(extent + 1)]
    for spec in product(draws, repeat=3):
        selections = [tuple(combinations(range(extent), count)) for extent, count in spec]
        exact = max(len(set().union(*options)) for options in product(*selections))
        assert nested_prefix_union(spec) == exact

def issued_stages(program):
    """Yield the stages of every call, owner branch, and pipeline tile."""
    if isinstance(program, ParallelProgram):
        for branch in program.programs:
            yield from issued_stages(branch)
        return
    if isinstance(program, KernelProgram):
        batches = (program.stages,)
    else:
        assert isinstance(program, PipelineProgram)
        batches = (program.tile(i) for i in range(program.count))
    for stages in batches:
        for stage in stages:
            yield stage
            if stage.program is not None:
                yield from issued_stages(stage.program)

@pytest.mark.parametrize("contexts,prefix,keys,topk,shared,private", [
    ((128, 256), (128, 128), ("A", "B"), 128, 0, 256),
    ((127, 127, 127), (127, 127, 127), ("A", "A", "B"), 128, 127, 130),
    ((1, 1, 1024), (1, 1, 1024), ("A", "A", "A"), 512, 513, 2),
    ((63, 63), (63, 63), ("A", "A"), 128, 63, 2),
    ((128, 128), (128, 128), ("A", "A"), 128, 128, 0),
])
def test_paired_attention_separates_shared_prefix_fill_from_private_tile_reads(
        contexts, prefix, keys, topk, shared, private):
    seqs = tuple((1, ctx) for ctx in contexts)
    shape = dict(sequences=seqs, latent=512, rope=64, heads=64, cp=1, topk=topk,
                 window=0, compression_ratio=1, query_dtype="bf16", cache_dtype="int8",
                 prefix_cache_mode="shared", prefix_tokens=prefix, prefix_cache_keys=keys)
    assert c8.prefix_partition(shape, seqs, topk) == (shared, private, min(shared, topk))
    geometry, statistics = Geometry.from_devices(machine(SPARSE_OFFLOAD_SKU).dies), 96 * 2 * 4
    plan = c8.arch35_plan(shape | {"statistics_bytes": statistics}, geometry)
    schedule = plan.metadata["c8_schedule"]
    assert schedule["external_fill_bytes"] == shared * 656 and schedule["private_rows"] == private
    assert plan.work.resources["hbm"] == len(seqs) * 64 * (576 + 512) * 2 + (shared + private) * 656 + statistics
    op = OpSpec("attention", "attention", "decode", 0, 0, (), *((len(seqs), 64, 576),) * 4)
    names = {role: f"{role}@r0:s0" for role in geometry.rates.keys() | {"cube"}}
    context = RecipeContext(plan.work.apply(op), {}, names, {}, geometry.compute_units,
        resource_units=geometry.resource_units, compute_peaks=geometry.compute_peaks, pool_units=geometry.buffer_units,
        pool_capacities=geometry.buffers, pool_names={key: key for key in geometry.buffers})
    program = plan.program(context)
    issued = Counter()
    for stage in issued_stages(program):
        issued.update(stage.work)
    assert dict(issued) == pytest.approx(context.bind(plan.work))
    stage = program.stages[-1]
    assert (stage.name, stage.after, sorted(stage.work.values())) == ("lse_statistics", ("call0",), [statistics] * 2)

@pytest.mark.parametrize("fmt,bytes_per_row,eligible", [
    (CacheFormat("int8"), 576, False), (CacheFormat("int8", 128, "fp32", 64, "bf16"), 656, True),
    (CacheFormat("int8", 128, "fp32", 64, "bf16", packed_elements=2), 400, False),
])
def test_latent_attention_requires_the_exact_packed_cache_geometry(fmt, bytes_per_row, eligible):
    plan = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64, cache_row_elements=576,
                           index_topk=2048, indexer=True)
    views = (RequestView("r", 1, 256, False, "decode"),)
    result = cp_graph.packed_sfa_geometry({"attention_tensor_view": {"cache_dtype": "int8"}},
                                          8, plan, views, 8, cache_dtype="int8", cache_fmt=fmt)
    assert (fmt.row_bytes(576), None if result is None else (result["latent"], result["rope"])) == (bytes_per_row, (512, 64) if eligible else None)

def test_quantized_indexer_prices_payload_scale_and_cold_fill_separately():
    model = parse_case(index_pool_case()).model
    layer = replace(model.layer_plans[3], index_kpool=1, index_heads=32, compress_ratio=1)
    quant = QuantSpec("index", index_dtype="fp8_e4m3", index_scale_dtype="fp32")
    model = replace(model, quant=quant, metadata=())
    views = tuple(RequestView(key, 1, ctx, False, "decode") for key, ctx in (("a", 1), ("b", 3)))
    builder = Builder("index", layer, 0, views, model, "decode")
    family_quant_indexer.declare_arch35(builder, deps=("q", "w", "cache"), views=views,
        plan=layer, quant=quant, parallel=ParallelSpec(tp=8))
    op, = builder.ops
    work = family_quant_indexer.Arch35QuantIndexerRecipe().build(op, None).work.resources
    assert (work["mte2_aic"], work["mte2_aiv"]) == ((2 * 32 + 6) * 128, 2 * 32 * 8 + 6 * 4)
    assert (op.flops, op.hbm_bytes) == (2 * 6 * 32 * 128, 8704 + 792 + 2 * 8192)

def test_context_parallel_statistics_are_in_the_immediate_consumer_read():
    case = catalog_case("kimi-k3.1", tp=8, dcp=8, ep=8, layers=4, quant="int8")
    model = replace(case.model, quant=replace(case.model.quant, kv_dtype="int8"))
    views = (RequestView("r", 1, 127, False, "decode"),)
    builder = Builder("s0.l0", model.layer_plans[3], 0, views, model, "decode")
    dims, payload = (1, 96, 512), 96 * 512 * 2
    source = OpSpec("s0.l0.attention", "attention", "decode", 0, 0, ("r",),
                    dims, dims, dims, dims, hbm_bytes=payload, mte3_bytes=payload,
                    metadata=(("program_kind", "attention"), ("native_kernel", "mla_attention")))
    builder.add(source)
    cp_graph.paired_sfa_attention_batch(builder, [source.op_id], views, plan=builder.plan, heads=96, local_heads=12, cp=8, rows=1)
    by_name = {op.op_id.rsplit(".", 1)[-1]: op for op in builder.ops}
    assert by_name["latent_attention_batch"].mte3_bytes == by_name["dcp_output_pack"].mte2_bytes == payload + 96 * 2 * 4

def test_interleaved_transfer_offsets_follow_packed_source_ownership():
    sources = [(9, ((0, 2), (4, 6)), 40), (7, ((2, 4), (6, 8)), 40)]
    spans = list(transfers_for_runs(sources, ((1, 7),)))
    assert [(rank, lo, hi, offset) for rank, lo, hi, size, offset, unit in spans] == [
        (9, 1, 2, 10), (7, 2, 4, 0), (9, 4, 6, 20), (7, 6, 7, 20)]
    assert sum((hi - lo) * unit for _, lo, hi, _, _, unit in spans) == 60

def test_windowed_kv_intersects_each_owners_live_tail():
    case = dense_case(layers=1)
    plan = replace(case.model.layer_plans[0], window=4, num_kv_heads=1)
    parallel = ParallelSpec(tp=2, cp=2, dcp=2, cp_plan="dcp")
    owners = ((0, 5), (5, 10))
    for rank, rows in enumerate((0, 4)):
        assert owned_token_rows(4, owners, rank, start=6) == rows
        assert layer_cache_components(case.model, plan, parallel, 10, capacity_padded=False,
                                      rank=rank, owner_spans=owners)["gqa_kv"] == rows * 64

def paired_case(**overrides):
    base = dense_case()
    layer = replace(base.model.layer_plans[0], attention="mla", kv_lora_rank=64, qk_rope_head_dim=8,
                    kv_cache_elements=72)
    draft = DraftSpec(kind="dspark", layers=2, hidden_size=32, intermediate_size=64,
                      num_heads=4, num_kv_heads=2, head_dim=16, q_width=64, kv_width=32,
                      attention_kind="gqa", block_size=5, markov_rank=16, vocab_size=128)
    model = replace(base.model, attention="mla", layer_plans=(layer, linear_layer(1, "kda")),
                    metadata=(("cache_allocator", "paired_sfa"),), auxiliary=AuxiliarySpec(dspark=draft))
    return replace(base, model=model, runtime=replace(base.runtime, **(
        {"prefix_cache_mode": "shared", "metadata": ()} | overrides)))

@pytest.mark.parametrize("mode,query,extra,recycling", [
    ({"dspark": DsparkSpec(4, (1.,) * 4)}, 1, 32768, 6),
    ({"metadata": (("draft_proposal_tokens", 3), ("draft_mode", "dspark"))}, 4, 32768, 5),
    ({"metadata": (("draft_proposal_tokens", 0),)}, 1, 0, 2),
])
def test_paired_pages_include_enabled_draft_planes_and_own_shared_pages_once(mode, query, extra, recycling):
    case = paired_case(**mode)
    group = page_groups(case, {i: 0 for i in range(2)}, query)[0]
    page = BLOCK * (64 + 8) * 2 + extra
    assert (group["bytes_per_group_page"], group["recycling_pages"]) == (page, recycling)
    requests = tuple(view(key, 1, 383, prefix=256) for key in ("a", "b"))
    shared = batch_cache_footprint(case, requests=requests, replica=0, phase="decode", padded=True)
    private = batch_cache_footprint(
        replace(case, runtime=replace(case.runtime, prefix_cache_mode="private")),
        requests=requests, replica=0, phase="decode", padded=True)
    assert (shared.shared_once_total, shared.private_total, shared.state_total, shared.total) == (
        2 * page, 2 * (1 + recycling) * page, 2 * recycling * page, 2 * (2 + recycling) * page)
    assert private.shared_once_total == 0 and private.total == 2 * (3 + recycling) * page

def test_dcp_shards_compressed_index_rows_in_complete_interleave_blocks():
    case = paired_case()
    layer = replace(case.model.layer_plans[0], owns_long_kv=False, compress_ratio=32, owns_index_k=True,
                    index_dim=16, index_heads=4, index_topk=8, indexer=True, indexer_type="full")
    model = replace(case.model, metadata=(("attention_execution", {"dcp_indexer_sharded": True}),))
    components = layer_cache_components(model, layer, ParallelSpec(cp=2), 100, interleave=BLOCK)
    assert components["index_k"] == 4 * (16 + 2)

@pytest.mark.parametrize("contiguous", (False, True))
def test_draft_prolog_boundary_depends_on_cache_contiguity(contiguous):
    quant = quant_preset("w8a8_mxfp8").to_dict() | {"kv_dtype": "int8"}
    raw = compose_case("kimi-k3.1", hardware=SPARSE_OFFLOAD_SKU, tp=8, ep=8, dcp=8, layers=4,
                       quant=quant, mtp_proposal_tokens=5, prompt_tokens=8, output_tokens=2)
    lowerings = raw["runtime"]["kernels"]["lowerings"]
    target = lowerings["mla_prolog"]
    cases = [{"domain": {"model_scopes": ["draft"]}, "binding": "vllm_ascend_mla_prolog_v3"}]
    if contiguous:
        raw["model"].setdefault("metadata", {})["draft_cache_representations"] = {
            key: {"dtype": "bf16", "contiguous": True} for key in ("mla_nope", "mla_rope")}
    else:
        cases.insert(0, {"domain": {"model_scopes": ["target"]}, "binding": target})
    lowerings["mla_prolog"] = {"cases": cases, "otherwise": target if contiguous else "analytical"}
    lowerings["native_fusion"] = {"implementation": "declared_native_fusion", "options": {"roles": ["mla_prolog"]}}
    graph = target_graphs(parse_case(raw), phase="decode", tokens=6, context=4, scope="model")[0]
    assert sum(op.phase == "draft" and dict(op.metadata).get("fusion_role") == "mla_prolog" for op in graph) == 5
    context_recipe = "draft_context_qkv_a" if contiguous else "draft_context_mla_prolog_v3"
    assert sum(op.recipe == context_recipe for op in graph) == 5

def test_two_receivers_read_one_named_source_extent():
    raw = compose_case("deepseek-v3.2", hardware=SPARSE_OFFLOAD_SKU, tp=1, dp=1, ep=1,
                       num_devices=3, layers=1, prompt_tokens=10, output_tokens=2)
    raw["pools"] = {side: deepcopy({k: raw[k] for k in ("machine", "runtime")}) for side in ("prefill", "decode")}
    raw["connector"] = {"link": {"capacity_bytes_per_ms": 64, "semantics": "per_pair"}, "chunk_bytes": 64}
    for side, ranks in (("prefill", [0, 1]), ("decode", [2, 3, 4, 5])):
        cfg = raw["pools"][side]
        cfg["machine"].update(num_devices=len(ranks), devices=[{"rank": r} for r in ranks])
        cfg["machine"]["links"][0]["members"] = ranks
        cfg["runtime"]["parallel"].update(tp=len(ranks), dcp=2, cp=2, cp_plan="dcp")
    raw["model"]["layer_plans"][0].update(num_kv_heads=0, window=8, owns_index_k=False)
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    transfers = result["execution"]["pd_handoffs"][0]["transfers"]
    rows = [t for t in transfers if t["component"] == "mla_kv" and t["source_rank"] == 1]
    assert {t["destination_rank"] for t in rows} == {3, 5}
    assert len({(t["source_storage"], t["source_version"]) for t in rows}) == 1
    assert {(tuple(t["source_byte_range"]), tuple(t["source_token_span"])) for t in rows} == {((0, 5760), (5, 10))}
    assert len({(t["destination_rank"], t["destination_storage"]) for t in rows}) == 2 == len(rows)
    assert sum(t["payload_bytes"] for t in rows) == 11520


def test_paired_sfa_caps_draft_kv_pages_at_draft_max_model_len() -> None:
    from mindie_sim.catalog.cache_pages import CachePages
    from mindie_sim.catalog.paired_sfa import page_groups

    uncapped = parse_case(compose_case(
        "kimi-k3.1", hardware=SPARSE_OFFLOAD_SKU, tp=8, ep=8, dcp=8, layers=4,
        mtp_proposal_tokens=5, max_model_len=131072,
        prompt_tokens=1024, output_tokens=16,
    ))
    capped = parse_case(compose_case(
        "kimi-k3.1", hardware=SPARSE_OFFLOAD_SKU, tp=8, ep=8, dcp=8, layers=4,
        mtp_proposal_tokens=5, max_model_len=131072, draft_max_model_len=4096,
        prompt_tokens=1024, output_tokens=16,
    ))
    layer_stage = [0] * len(uncapped.model.layer_plans)
    groups_uncapped = page_groups(uncapped, layer_stage, query_tokens=6)
    groups_capped = page_groups(capped, layer_stage, query_tokens=6)

    assert len(groups_uncapped) == 1
    assert len(groups_capped) == 2
    main_g, aux_g = groups_capped
    assert aux_g["auxiliary"] is True
    assert aux_g["resident_tokens"] == 4096
    assert aux_g["recycling_pages"] == 0
    assert main_g["bytes_per_group_page"] < groups_uncapped[0]["bytes_per_group_page"]

    comp_names = {
        p.index: ("mla_kv", "index_k", "conv_state", "recurrent_state", "replay_records", "state_transient")
        for p in capped.model.layer_plans
    }
    pages_uncapped = CachePages(groups_uncapped, comp_names, layer_stage)
    pages_capped = CachePages(groups_capped, comp_names, layer_stage)
    assert pages_capped.group_components(pages_capped.groups[1]) == {}
    bytes_uncapped = pages_uncapped.stage_bytes(131072, 131072, 1)[0]
    bytes_capped = pages_capped.stage_bytes(131072, 131072, 1)[0]
    assert bytes_capped < bytes_uncapped
    # At 4096 tokens, capped draft group holds the exact same draft pages as 131072 tokens.
    assert pages_capped.groups[1].pages(4096, 131072) == pages_capped.groups[1].pages(131072, 131072)
