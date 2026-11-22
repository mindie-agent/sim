"""Decode attention tiling, storage, collective ownership, and source dispatch.

[Role] Check staged GQA, governed FIA, and indexed shared-KV execution.
[Invariants] Preserve per-request work, finite buffers, and source ABI ownership.
[Timeouts & Locks] Bounded local graph and operator simulations.
[Side-Effect Policy] Read-only fixtures and in-memory declarations.
"""

from dataclasses import replace
from itertools import islice
import json
import pickle
from types import SimpleNamespace

import pytest

from _support import operator_calibration_helpers as source_cases
from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS, SKUS
from _support.builders import dense_case, dense_model
from _support.captures import REPO_ROOT, fixture_json
from _support.hardware import isa_of, sku_case
from mindie_sim import compose_case, graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.catalog.cache import cache_row_split, packed_region_row_bytes
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.observed_contracts.attention import observed_attention_contract
from mindie_sim.prediction.operators.paged_mla import family_mqsmla, mqsmla; from mindie_sim.prediction.sharding import context_parallel as cp_graph
from mindie_sim.prediction.operators.sharedkv import sharedkv_attention as scfa
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.operators.attention.batch_prefix_attention import BatchPrefixAttention
from mindie_sim.prediction.cache_path import CacheGeometry, CacheState
from mindie_sim.prediction.cache_representation import CacheLowering, cache_identity, store
from mindie_sim.prediction.operators.mla_prolog.draft_context import prepare
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.mla.mla_epilogs import KV_COMPRESS_EPILOG_FAMILY, kv_cache_epilog
from mindie_sim.prediction.operators import attention as governed
from mindie_sim.prediction.operators.attention._cp_tiling import HISTORY_DECODE_SITE, VLLM_REVISION
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.streams import StreamOwners
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure_contracts import predicted_contract

KIB = 1024
PACKED_MLA_CASE = REPO_ROOT / "examples/cases/longctx-decode-20261030/v41-a5-tp1-dp8-ep8.json"
MODE1_ROW_BYTES = 608
MODE2_ROW_BYTES = 584

pytestmark = pytest.mark.tier0


def geometry(arch_id):
    sku = next(sku for sku in REPRESENTATIVE_SKUS.values() if isa_of(sku) == arch_id)
    return Geometry.from_devices(sku_case(sku).machine.dies)


@pytest.mark.parametrize("arch_id", ARCH_IDS)
def test_wide_head_candidate_reloads_two_panels_with_one_slot(arch_id):
    geom = geometry(arch_id)
    contract = governed.AttentionContract(1, 4, 2, 512, 512, selected_tokens=128,
                                          panel_mode="tile_solver_candidate")
    solved = governed.solve_wide_head_tiling(contract, geom)
    assert solved.depth == 1 and solved.tile_shape.k == 256
    assert {(r.pool, r.required_bytes, r.available_bytes) for r in solved.rejections} == {
        ("l1", 640 * KIB, 512 * KIB), ("l0b", 128 * KIB, 64 * KIB),
    }
    candidate = governed.build(governed.make_attention_op(contract, op_id="wide"), geom)
    qk = [phase.work.matrix["bf16"] for phase in candidate.program.phases if phase.name.startswith("qk_panel")]
    assert qk == [2 * 4 * 128 * 256] * 2 and candidate.metadata["v_panels"] == 2
    assert candidate.metadata["l0b_slots"] == 1
    if arch_id == governed.ARCH35:
        shipping = governed.shipping_fia_panel_plan(replace(contract, panel_mode="shipping"), arch_id)
        assert shipping.k_panels == shipping.v_panels == (128, 128, 128, 128)


@pytest.mark.parametrize("heads,span", [(1, 256), (2, 512)])
def test_bsh_token_stride_and_head_major_contiguity(heads, span):
    bsh = governed.bsh_kv_access_plan(2, 128, 2, "BSH", heads, rows=3)
    block = governed.bsh_kv_access_plan(2, 128, 2, "PA_BLK_BHSD", heads, rows=3, block_size=4)
    (run,) = bsh.runs
    assert (run.rows, run.row_bytes, run.stride_bytes) == (3, span, 512)
    assert len(block.runs) == heads
    assert [(run.base_byte, run.row_bytes, run.stride_bytes) for run in block.runs] == [
        (head * 4 * 256, 256, 256) for head in range(heads)
    ]


@pytest.mark.parametrize("hardware", SKUS)
def test_governed_decode_conserves_split_work_and_survives_pickle(hardware):
    case = sku_case(hardware)
    geom = Geometry.from_devices(case.machine.dies)
    contract = governed.AttentionContract(2, 4, 2, 128, 64, sequences=((2, 129),), causal=True, splits=2)
    op = governed.make_attention_op(contract, op_id="decode")
    plan = governed.build(op, geom)
    restored = pickle.loads(pickle.dumps(plan))
    assert restored == plan
    assert plan.work.matrix == {"bf16": 2 * 4 * (2 * 129) * (128 + 64)}
    phases = {p.name: p for p in plan.program.phases}
    assert phases["split_write"].work.resources["mte3"] == 2 * 4 * 2 * (64 + 2) * 4
    assert phases["output"].after == ("split_merge",)
    row = operator_costs(case, (op,), recipes=RecipeBook(governed.recipes()))["operators"][0]
    assert row["serviced"] == pytest.approx(row["declared_service"])
    if geom.compute_arch.preset == governed.ARCH22:
        for abi in (governed.GBSA_ABI, governed.C8_ABI, governed.NOPE_ABI, *sorted(governed.MINDIE_ABIS)):
            sparse = replace(contract, abi=abi, selected_tokens=8)
            with pytest.raises(UnmodeledCapabilityError, match="Arch22 attention requires"):
                governed.build(governed.make_attention_op(sparse, op_id="sparse"), geom)


def _shape(query=32, context=0, window=128, ratio=4, topk=512, **changes):
    shape = dict(query=query, context=context, window=window,
                 cmp_ratio=ratio, topk=topk, heads=8, width=512, cp=1, pcp=1,
                 branch="SCFA", query_layout="TND", cache_layout="PA_ND",
                 sequences=1, views=1, kv_heads=1, query_dtype="bf16",
                 cache_dtype="bf16", output_dtype="bf16", sinks_dtype="fp32",
                 sparse_block_size=1, owner_schedule="metadata_cost_balanced",
                 metadata_s2_split=False, return_softmax_lse=False,
                 selected_rows="valid_prefix")
    return {**shape, **changes}


def _op(**kwargs):
    return OpSpec("sharedkv", "attention", "prefill", 0, 0, (), (), (), (), (),
                  metadata=(("sharedkv_attention_geometry", _shape(**kwargs)),))


def _run(op, *, ready_at=0, cores=4, policy="resident", trace_limit=0):
    rates = {"hbm": 1e8, "vector": 1e8, "mte2_aic": 1e8,
             "mte2_aiv": 1e8, "mte1": 1e8, "mte3": 1e8,
             "fixpipe": 1e8, "l2": 1e9, "cube": 1e12}
    units = {"hbm": 1, "l2": 1, "cube": cores, "vector": 2 * cores,
             "mte2_aic": cores, "mte2_aiv": 2 * cores, "mte1": cores,
             "mte3": 2 * cores, "fixpipe": cores}
    pools, pool_units = {"l2": 64 * 1024**2}, {"l2": 1}
    service = {role + "_cached": (role, 2e8)
               for role in ("mte2_aic", "mte2_aiv")}
    service.update(l2_read=("l2", 2e9),
                   l2_write=("l2", 1e9))
    geometry = Geometry(cores, units, pools, pool_units)
    plan = scfa.build(op, geometry, scratch_policy=policy)
    engine = Engine([Resource(role, rate, units=units[role])
                     for role, rate in rates.items()],
                    pools=pools, pool_units=pool_units, trace_limit=trace_limit)
    context = RecipeContext(
        plan.work.apply(op), {}, {role: role for role in rates}, rates, cores,
        resource_units=units, compute_peaks={"bf16": 1e13},
        pool_names={"l2": "l2"}, pool_capacities=pools,
        pool_units=pool_units, service_modes=service)
    ready = engine.submit("ready", release=ready_at)
    done = plan.lower(context).start(engine, "sharedkv", ready)
    return plan, engine, context, done, ready


@pytest.mark.parametrize("query,context,topk", [(3, 0, 512), (3, 4096, 512), (3, 8192, 1024)])
def test_sharedkv_workspace_and_output_follow_combined_window_and_selection(query, context, topk):
    op = _op(query=query, context=context, topk=topk)
    resident, engine, bind, done, ready = _run(op, ready_at=7, trace_limit=2000)
    offchip, *_ = _run(op, policy="offchip")
    scratch = sum(resident.work.resources.get(role, 0) for role in ("l2_read", "l2_write"))
    assert offchip.work.resources["hbm"] - resident.work.resources["hbm"] == scratch
    assert resident.work.matrix == offchip.work.matrix
    engine.advance(horizon=7)
    assert ready.done and not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.serviced == pytest.approx(bind.bind(resident.work))
    assert not engine.allocations and engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    stores = [row for row in engine.trace if row["task"].endswith(":accum_store") and "hbm" in row["demand"]]
    assert len(stores) == query and sum(row["demand"]["hbm"] for row in stores) == query * 8 * 512 * 2
    assert all(row["placement"]["mte3"][0] % 2 == 0 for row in stores)
    for row in engine.trace:
        if row["task"].endswith((":publish0", ":publish1")):
            owner = int(row["task"][-1])
            assert row["placement"]["mte3"][0] % 2 == owner
            consumer = next(event for event in engine.trace if event["task"] == row["task"][:-8] + "qk_load")
            assert consumer["start_ms"] >= row["end_ms"]


def test_sharedkv_source_domain_and_long_stream_remain_bounded():
    for overrides in (dict(branch="CFA"), dict(owner_schedule=None), dict(views=2), dict(return_softmax_lse=True)):
        with pytest.raises(ModelingError, match="source_geometry_outside_reviewed_domain"):
            _run(_op(**overrides))
    sample = tuple(islice(scfa.source_tiles(10**12, 4096, 128, 4, 512, 0, 32), 2))
    assert sample == (("ori", 128, True, False), ("cmp", 512, False, True))


def test_sharedkv_frontend_binds_only_indexed_prefill_and_preserves_tensor_boundary():
    raw = fixture_json("verification", "v4flash-sharedkv-case.json")
    raw["runtime"].pop("routing", None)
    before = target_operators(raw, phase="prefill", tokens=4)
    name = "vllm_ascend_sharedkv_scfa"
    raw["runtime"]["kernels"].setdefault("lowerings", {})["sharedkv_attention"] = {
        "implementation": name, "options": {"owner_schedule": "metadata_cost_balanced",
            "selected_rows": "valid_prefix", "sinks_dtype": "fp32", "metadata_s2_split": False,
            "return_softmax_lse": False, "scratch_policy": "offchip"},
    }
    after = target_operators(raw, phase="prefill", tokens=4)
    selected = [(a, b) for a, b in zip(before, after, strict=True) if b.recipe == name]
    assert len(selected) == 21 and all(a.deps == b.deps for a, b in selected)
    assert all(b.source_boundary == "kernel" and predicted_contract(a.to_dict()) ==
               predicted_contract(b.to_dict()) for a, b in selected)
    assert all(o.recipe != name for o in target_operators(raw, phase="decode", tokens=1, context=4096))
    with pytest.raises(ModelingError, match="single_contiguous_prefill"):
        target_operators(raw, phase="prefill", tokens=8, sequences=2)


def test_shipping_score_padding_and_full_width_value_publications():
    geom = geometry(governed.ARCH35)
    contract = governed.AttentionContract(1, 1, 1, 512, 512, selected_tokens=129)
    op = governed.make_attention_op(contract, op_id="decode")
    plan = governed.build(op, geom)
    rates = {**{role: value / 1000 for role, value in geom.rates.items()}, "cube": 1}
    context = RecipeContext(op, {}, {role: role for role in rates}, rates, geom.compute_units,
                            resource_units=geom.resource_units, compute_peaks=geom.compute_peaks)
    engine = Engine([Resource(role, rate, units=geom.resource_units[role]) for role, rate in rates.items()],
                    trace_limit=128)
    ready = engine.submit("ready", release=3)
    done = plan.lower(context).start(engine, "decode", ready)
    engine.advance(until=(done,))
    score = [row for row in engine.trace if row["task"].endswith(":score_fixpipe")]
    softmax = [row for row in engine.trace if row["task"].endswith(":online_softmax")]
    qk = [row for row in engine.trace if "qk_panel_" in row["task"]]
    pv = [row for row in engine.trace if ":pv_panel_" in row["task"]]
    publish = [row for row in engine.trace if row["task"].endswith(":pv_fixpipe")]
    online = [row for row in engine.trace if row["task"].endswith(":online_output")]
    query = [row for row in engine.trace if row["task"].endswith(":query_load")]
    assert len(query) == 1 and query[0]["demand"]["hbm"] == 512 * 2
    assert len(qk) == 8 and len(score) == len(softmax) == 2
    assert [row["demand"]["fixpipe"] for row in score] == [1024, 64]
    assert all(panel["end_ms"] <= score[0]["start_ms"] for panel in qk[:4])
    assert score[0]["end_ms"] <= softmax[0]["start_ms"] <= qk[4]["start_ms"]
    assert qk[-1]["end_ms"] <= score[1]["start_ms"] <= softmax[1]["start_ms"]
    assert len(pv) == 8 and len(publish) == len(online) == 2
    assert qk[4]["start_ms"] < pv[0]["start_ms"] < online[0]["end_ms"]
    assert qk[-1]["end_ms"] <= pv[0]["start_ms"]
    assert [row["demand"]["fixpipe"] for row in publish] == [4096, 4096]
    for i, (publication, output) in enumerate(zip(publish, online, strict=True)):
        assert all(panel["end_ms"] <= publication["start_ms"] for panel in pv[i * 4:(i + 1) * 4])
        assert publication["end_ms"] <= output["start_ms"]
    assert {role: amount for role, amount in engine.serviced.items() if amount} == pytest.approx(context.bind(plan.work))


@pytest.mark.parametrize("stack_pin", [False, True])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_history_decode_exchanges_output_and_lse_after_attention(stack_pin, dtype):
    manifest = REPO_ROOT / "mindie_sim/stack_sources/vllm_ascend_cann93.json"
    revision = json.loads(manifest.read_text())["baseline_revision"] if stack_pin else VLLM_REVISION
    history = governed.ContextParallelPlan(
        revision, HISTORY_DECODE_SITE, "decode", "history_attention",
        (("query", (1, 1, 128), 2),), dcp_size=2,
    )
    contract = governed.AttentionContract(1, 1, 1, 128, 128, selected_tokens=33, cp_plan=history)
    op = governed.make_attention_op(contract, op_id="history", deps=("input",), dtype=dtype)
    book = RecipeBook(governed.recipes())
    gather, attention, pack, exchange, merge, join = book.expand((op,))
    assert attention.returns == ("lse",) and attention.local_shape == (1, 2, 128)
    assert pack.local_shape == (1, 2, 132) and pack.mte2_bytes == 520 and pack.mte3_bytes == 528
    assert gather.link_bytes == 256 and exchange.link_bytes == 264
    assert exchange.family == "alltoall" and exchange.local_shape == (1, 1, 132)
    assert merge.mte2_bytes == 528 and merge.mte3_bytes == 256
    assert dict(merge.metadata)["partial_reduction"] == "logsumexp"
    assert dict(exchange.metadata)["payload_tensors"] == (("output", (1, 1, 128)), ("lse", (1, 1, 1)))
    assert all(before.op_id in after.deps for before, after in zip(
        (gather, attention, pack, exchange, merge), (attention, pack, exchange, merge, join), strict=True,
    ))

    geom = geometry(governed.ARCH35)
    plan = governed.build(op, geom)
    assert any(ref.revision == revision and ref.relative_path.endswith("/attention_cp.py")
               for ref in plan.pipeline_spec.sources)
    assert plan.work.resources["link"] == 520
    rates = {**{role: value / 1000 for role, value in geom.rates.items()}, "cube": 1, "link": 1000}
    units = {**geom.resource_units, "link": 1}
    context = RecipeContext(op, {}, {role: role for role in rates}, rates, geom.compute_units,
                            resource_units=units, compute_peaks=geom.compute_peaks)
    engine = Engine([Resource(role, rate, units=units[role]) for role, rate in rates.items()], trace_limit=128)
    done = plan.lower(context).start(engine, "history", engine.submit("input", release=5))
    engine.advance(until=(done,))
    steps = [next(row for row in engine.trace if row["task"].endswith(":" + name))
             for name in ("query_load", "output", "output_lse_pack", "output_lse_exchange", "output_lse_merge")]
    network = [row for row in engine.trace if row["demand"].get("link")]
    assert [row["demand"]["link"] for row in network] == [256, 264]
    assert network[0]["end_ms"] <= steps[0]["start_ms"]
    assert steps[1]["demand"]["mte3"] == 520
    assert all(left["end_ms"] <= right["start_ms"] for left, right in zip(steps, steps[1:]))
    assert {role: amount for role, amount in engine.serviced.items() if amount} == pytest.approx(context.bind(plan.work))

    case = compose_case("deepseek-v2-lite", hardware=REPRESENTATIVE_SKUS["a5"], tp=2, dcp=2,
                        layers=1, block_size=16, prompt_tokens=16, output_tokens=2, concurrency=1)
    result = graph_costs(case, (op,), recipes=book, external_inputs={"input": 5}, isolated=False)
    timeline = {row["op_id"]: row for row in result["timeline"]}
    for before, after in zip((gather, attention, pack, exchange), (attention, pack, exchange, merge), strict=True):
        assert timeline[before.op_id]["complete_ms"] <= timeline[after.op_id]["ready_ms"]


def test_disjoint_stream_uses_physical_core_ids_for_work_and_allocations():
    geom = replace(geometry(governed.ARCH35), stream_owners=StreamOwners((7,), (14, 15), 1, 2))
    contract = governed.AttentionContract(1, 1, 1, 512, 512, selected_tokens=1,
                                          panel_mode="tile_solver_candidate")
    op = governed.make_attention_op(contract, op_id="decode")
    plan = governed.build(op, geom)
    rates = {**{role: value / 1000 for role, value in geom.rates.items()}, "cube": 1}
    context = RecipeContext(
        op, {}, {role: role for role in rates}, rates, geom.compute_units,
        resource_units=geom.resource_units, compute_peaks=geom.compute_peaks, pool_units=geom.buffer_units,
        pool_names={pool: pool for pool in geom.buffers},
    )
    engine = Engine([Resource(role, rate, units=geom.resource_units[role]) for role, rate in rates.items()],
                    pools=geom.buffers, pool_units=geom.buffer_units, trace_limit=128)
    ready = engine.submit("ready")
    lowered = plan.program(context)
    (rows,) = lowered.programs
    (row,) = rows.tile(0)
    (partial,) = row.program.stages[0].program.programs
    load = partial.stages[0]
    assert load.acquire and all(allocation.units == (7,) for allocation in load.acquire)
    done = plan.lower(context).start(engine, "decode", ready)
    engine.advance(until=(done,))
    qk = next(row for row in engine.trace if "qk_panel_" in row["task"])
    vec = next(row for row in engine.trace if row["task"].endswith(":online_softmax"))
    assert qk["placement"]["cube"] == (7,)
    assert vec["placement"]["vector"] == (14, 15)
    assert not engine.allocations
    assert {role: amount for role, amount in engine.serviced.items() if amount} == pytest.approx(context.bind(plan.work))


def test_frontend_preserves_ragged_decode_requests_across_batching():
    case = dense_case()
    views = (RequestView("a", 1, 7, False, "decode"), RequestView("b", 3, 11, False, "decode"))
    for fused in (True, False):
        frontend = Frontend(attention_policy=AttentionPolicy(fused_batch=fused))
        ops, _ = frontend.body(case.model, case.runtime, views, ready="input", tokens=4, mode="decode", machine=case.machine)
        attention = [op for op in ops if op.family == "attention" and op.layer == 0]
        assert len(attention) == (1 if fused else 2)
        assert sum(op.flops for op in attention) == 2 * (8 + 12 + 13 + 14) * 4 * (16 + 16)


def test_draft_context_attributes_every_layer_op():
    draft = DraftSpec(
        kind="dspark", layers=2, hidden_size=32, intermediate_size=64,
        num_heads=4, num_kv_heads=2, head_dim=16, q_width=64, kv_width=32,
        attention_kind="mla", block_size=5, markov_rank=16, vocab_size=128,
        q_lora_rank=32, kv_lora_rank=64, qk_rope_head_dim=8)
    model = replace(dense_model(layers=1), auxiliary=AuxiliarySpec(dspark=draft))
    projection = OpSpec("proj", "gemm", "draft", 0, -1, (),
                        (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1))
    ops = []
    prepare(ops, projection=projection, model=model, draft=draft, rows=4,
            query_rows=3, prefix="d:", stage=0, requests=())
    attributed = {op.op_id: op.layer for op in ops}
    for layer in range(draft.layers):
        for suffix in ("mla_prolog_v3", "kv_scatter"):
            assert attributed[f"d:context_kv{layer}.{suffix}"] == layer
    assert not any("qkv_a" in op_id or "kv_copy" in op_id for op_id in attributed)
    assert attributed["d:context_norm"] == -1


@pytest.mark.parametrize("shape,qk,pv", [((1, 128, 4, 3, 1), 75, 25), ((1, 128, 4, 16), 50, 50)])
def test_prefix_attention_matrix_work_uses_qk_and_pv_head_widths(shape, qk, pv):
    op = OpSpec("attn", "attention", "decode", 0, 0, (), shape, shape, shape, shape,
                metadata=(("prefix_read_reuse_applied", True),))
    context = SimpleNamespace(op=op, work={"cube": 100.0}, setup_ms=0.0,
                              minimum_ms=0.0, resource_names={})
    work = {stage.name: stage.work for stage in BatchPrefixAttention().lower(context).stages}
    assert work["qk"]["cube"] == pytest.approx(qk)
    assert work["pv"]["cube"] == pytest.approx(pv)


def _packed_mode_model(scale_dtype):
    case = parse_case(json.loads(PACKED_MLA_CASE.read_text(encoding="utf-8")))
    metadata = dict(case.model.metadata)
    regions = dict(metadata.get("cache_representations", {}))
    region = {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": scale_dtype}
    regions.update(window=dict(region), compressed_kv=dict(region))
    return replace(case.model, metadata=tuple({**metadata, "cache_representations": regions}.items()))


def _packed_cache_builder(model, context):
    views = (RequestView("r0", 1, context, False, "decode"),)
    builder = Builder("l0", model.layer_plans[0], 0, views, model, "decode")
    geometry = CacheGeometry(capacity_bytes=1 << 20, line_bytes=1, transfer_bytes=1, staging_lines=1)
    return builder, geometry


@pytest.mark.parametrize("scale_dtype,row_bytes", [("bf16", MODE1_ROW_BYTES), ("uint8", MODE2_ROW_BYTES)])
def test_mqsmla_window_and_compressed_writers_use_packed_row_extents(scale_dtype, row_bytes):
    model = _packed_mode_model(scale_dtype)
    width = model.layer_plans[0].cache_row_elements
    assert width == 512
    assert packed_region_row_bytes(model, "window", width) == row_bytes
    builder, geometry = _packed_cache_builder(model, 100)
    store(builder, "window", ("src",), 3, width, "window", state=CacheState(geometry),
          geometry=geometry, identity=cache_identity("window", layer=0, request="r0"))
    window = builder.ops[-1]
    assert window.mte3_bytes == pytest.approx(3 * row_bytes)
    assert dict(window.metadata)["cache_row_bytes"] == row_bytes
    assert dict(window.metadata)["cache_representation"]["row_bytes"] == row_bytes

    builder, geometry = _packed_cache_builder(model, 127)
    store(builder, "compressed", ("src",), 1, 512, "compressed_kv", state=CacheState(geometry),
          geometry=geometry, identity=cache_identity("compressed_kv", layer=2, request="r0"))
    compressed = builder.ops[-1]
    assert compressed.mte2_bytes == pytest.approx(1024.0)
    assert compressed.mte3_bytes == pytest.approx(row_bytes)
    assert dict(compressed.metadata)["cache_row_bytes"] == row_bytes


def test_mqsmla_epilog_prefers_the_declared_cache_row_width():
    fmt = cache_format(_packed_mode_model("bf16"), "window")
    op = replace(source_cases._c8_op(((1, 128),), 128, 64, 0), dtype="bf16",
                 local_shape=(3, 512), metadata=(("cache_row_bytes", MODE1_ROW_BYTES),))
    epilog = kv_cache_epilog(op, fmt, "window", native=True)
    assert epilog.family == KV_COMPRESS_EPILOG_FAMILY
    assert dict(epilog.metadata)["cache_epilog"]["row_bytes"] == MODE1_ROW_BYTES


def test_mqsmla_byte_scale_lowering_splits_payload_and_padding():
    model = _packed_mode_model("uint8")
    builder, geometry = _packed_cache_builder(model, 127)
    lowering = CacheLowering(builder, CacheState(geometry))
    lowering.store("compressed", ("src",), 1, 512, "compressed_kv", identity="compressed_kv:l2:r0")
    view = lowering.requests[-1][1].view
    assert cache_row_split(model, "compressed_kv", 512) == (576.0, 8.0)
    assert (view.payload_bytes, view.scale_bytes, view.padding_bytes) == (576, 7, 1)
    assert sum((view.payload_bytes, view.scale_bytes, view.padding_bytes)) == MODE2_ROW_BYTES


def test_mqsmla_window_union_advances_across_query_rows():
    shape = {"sequences": ((6, 131072),), "window": 128, "ratio": 0, "topk": 0}
    external, shared = mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8)
    assert shared == 0
    assert external == 133 * mqsmla.quant_row_bytes(mqsmla.KV_QUANT_FP8)
    assert external == 80864


@pytest.mark.parametrize("mode,keys,topk,rows,external_bytes,shared", [
    ("private", ("A", "A"), 0, 262, 159296, 0),
    ("shared", ("A", "A"), 0, 261, 158688, 1),
    ("shared", ("k0", "k1"), 512, 262, 159296, 0),
])
def test_hca_runtime_and_cache_keys_control_prefix_aliasing(mode, keys, topk, rows, external_bytes, shared):
    views = tuple(RequestView(request, 1, 383, False, "decode", prefix_tokens=128, prefix_cache_key=key)
                  for request, key in zip(("a", "b"), keys))
    fields = family_mqsmla._prefix_fields(views, SimpleNamespace(prefix_cache_mode=mode))
    if mode == "shared":
        assert fields["prefix_tokens"] == (128, 128)
        assert fields["prefix_cache_keys"] == keys
    shape = dict(sequences=((1, 383),) * 2, window=128, ratio=128, topk=topk, **fields)
    assert mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8) == (external_bytes, shared)
    assert external_bytes == rows * MODE1_ROW_BYTES


@pytest.mark.parametrize("contexts,prefixes,keys,ratio,topk,rows,external_bytes,shared", [
    ((191, 191), (64, 64), ("k", "k"), 128, 0, 258, 156864, 0),
    ((511, 767, 767), (256, 256, 0), ("k", "k", ""), 4, 512, 832, 505856, 64),
])
def test_mqsmla_shares_only_completed_prefix_rows(contexts, prefixes, keys, ratio, topk,
                                                 rows, external_bytes, shared):
    shape = dict(sequences=tuple((1, context) for context in contexts), window=128, ratio=ratio,
                 topk=topk, prefix_cache_mode="shared", prefix_tokens=prefixes, prefix_cache_keys=keys)
    assert mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8) == (external_bytes, shared)
    assert external_bytes == rows * MODE1_ROW_BYTES


@pytest.mark.parametrize("row_width", (MODE1_ROW_BYTES, MODE2_ROW_BYTES))
def test_dsl_mqsmla_contract_rejects_native_only_row_widths(row_width):
    query = ((4, 64, 512), "bf16", "ND")
    empty = ((), None, None)
    inputs = (query, empty, ((16, 1, 1, row_width), "uint8", "ND"),
              ((1, 16), "int32", "ND"), empty, empty, empty)
    input_shapes, input_dtypes, input_formats = zip(*inputs)
    output_shapes, output_dtypes, output_formats = zip(query, empty)
    row = SimpleNamespace(input_shapes=input_shapes, input_dtypes=input_dtypes, input_formats=input_formats,
                          output_shapes=output_shapes, output_dtypes=output_dtypes, output_formats=output_formats,
                          kind="MqsmlaKernel___call__")
    assert observed_attention_contract(row) is None
    row.kind = "aclnnMixedQuantSparseFlashMla"
    assert observed_attention_contract(row) is not None


@pytest.mark.parametrize("scale_dtype", ("bf16", "fp8_e8m0"))
def test_dsl_mqsmla_lowering_rejects_native_quant_modes(scale_dtype):
    raw = compose_case(
        "deepseek-v4.1-flash", hardware="a5", tp=4, dp=2, ep=8, num_devices=8,
        layers=4, quant="w8a8", prompt_tokens=16, output_tokens=2)
    packed = {"dtype": "fp8_e4m3", "group_size": 64, "scale_dtype": scale_dtype}
    raw["model"].setdefault("metadata", {})["cache_representations"] = {
        "window": dict(packed), "compressed_kv": dict(packed),
    }
    parsed = parse_case(raw)
    decode = dict(phase="decode", tokens=1, context=128, sequences=1, scope="body")
    graph = target_graphs(parsed, **decode)[0]
    native = next(op for op in graph if op.recipe == "aclnn_mixed_quant_sparse_flash_mla")
    metadata = dict(native.metadata)
    geometry = dict(metadata["mqsmla_geometry"])
    geometry.pop("implementation", None)
    metadata["mqsmla_geometry"] = geometry
    metadata.pop("native_kind", None)
    crafted = replace(native, recipe="cannbot_mixed_quant_sparse_flash_mla", metadata=tuple(metadata.items()))
    with pytest.raises(ModelingError, match=r"^mqsmla_legacy_dsl_requires_mixed_fp8_fp4_cache$"):
        operator_costs(parsed, (crafted,), input_ready_ms=0)

    raw["runtime"].setdefault("metadata", {}).pop("automatic_lowerings", None)
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
        "mqsmla_attention"] = "cannbot_mixed_quant_sparse_flash_mla"
    with pytest.raises(ModelingError, match=r"^mqsmla_cache_representation_contract_required:window$"):
        target_graphs(parse_case(raw), **decode, apply_body_defaults=False)


def test_padded_latent_attention_producer_writes_the_entire_pack_input():
    case = parse_case(compose_case(
        "kimi-k3.1", hardware="a5", dp=1, tp=8, dcp=8, ep=8, num_devices=8,
        layers=4, quant="int8", prompt_tokens=8, output_tokens=2))
    model = replace(case.model, quant=replace(case.model.quant, kv_dtype="int8"))
    layer = model.layer_plans[3]
    views = (RequestView("r", 1, 127, False, "decode"),)
    builder = Builder("p", layer, 0, views, model, "decode")
    member = replace(source_cases._c8_op(((1, 127),), 128, 96, 0), recipe="paired_mla_attention_analytic",
                     mte3_bytes=98304, hbm_bytes=98304, metadata=())
    builder.add(member)
    cp_graph.paired_sfa_attention_batch(builder, [member.op_id], views, plan=layer,
                                        heads=96, local_heads=12, cp=8, rows=8)
    ops = {op.op_id.rsplit(".", 1)[-1]: op for op in builder.ops}
    assert ops["latent_attention_batch"].mte3_bytes == ops["dcp_output_pack"].mte2_bytes == 792576
    assert ops["dcp_output_pack"].mte3_bytes == ops["dcp_lse_merge"].mte2_bytes == 792576
    assert ops["dcp_output_alltoall"].link_bytes == 693504
    assert ops["dcp_lse_merge"].mte3_bytes == 98304


def test_cold_mte2_cap_keeps_unsaturated_knee_when_curve_plateaus():
    from types import SimpleNamespace
    from mindie_sim.prediction.operators.sharedkv.split_core import p01_cold_mte2_aic_cap

    p01 = SimpleNamespace(
        status="calibrated",
        parameters={"mte2_aic_cold_scaling_bps": {"1": 160e9, "8": 1280e9, "16": 2480e9, "32": 2720e9}},
        metrics={
            "hbm_read_bytes_per_s": SimpleNamespace(p50=3750e9, certified_upper_capacity=4000e9),
            "mte2_aic_hit_bytes_per_s_per_core": SimpleNamespace(p50=300e9, certified_upper_capacity=320e9),
        },
    )
    pt = SimpleNamespace(warnings=(), cell=lambda cid: p01)
    ctx = SimpleNamespace(
        primitive_table=pt,
        resource_names={"mte2_aic": "mte2_aic@r0", "hbm": "memory@r0"},
        rates={"memory@r0": 3185e6},
    )
    caps_unsat = p01_cold_mte2_aic_cap(ctx, 8, include_hbm=True)
    assert caps_unsat["mte2_aic@r0"] == pytest.approx(160e6)
    assert caps_unsat["memory@r0"] == pytest.approx(160e6)
    caps_sat = p01_cold_mte2_aic_cap(ctx, 32, include_hbm=True)
    assert caps_sat["mte2_aic@r0"] == pytest.approx(2480e9 / 16.0 / 1000.0)
    assert caps_sat["memory@r0"] == pytest.approx(2480e9 / 16.0 / 1000.0)


def test_fia_weighted_contiguous_partition_balances_ragged_kv_tails():
    from mindie_sim.prediction.operators.attention._fia import _FiaRect, _weighted_contiguous_rect_partition

    rects = tuple(
        _FiaRect(seq, 4, head, 65536 if seq == 7 else 512, 0, 0, 256 if seq == 7 else 2)
        for seq in range(8) for head in range(8)
    )
    slices = _weighted_contiguous_rect_partition(rects, 32)
    long_per_core = [sum(1 for i in core_idxs if rects[i].kv == 65536) for core_idxs in slices]
    assert sum(len(s) for s in slices) == len(rects)
    assert max(long_per_core) == 1


def test_draft_kv_cap_bounds_gqa_and_mla_without_leaking_target_limit():
    from mindie_sim.prediction.operators.attention.attention import _cap_draft_views, _draft_kv_cap, gqa
    from mindie_sim.prediction.operators.mla.mla import mla
    from mindie_sim.schema.parallel import ParallelSpec

    base = dense_case(layers=1)
    plan_target = replace(base.model.layer_plans[0], num_heads=8, num_kv_heads=2, head_dim=128)
    gqa_model = replace(base.model, num_heads=8, num_kv_heads=2, head_dim=128, layer_plans=(plan_target,))
    rt_capped = replace(base.runtime, metadata=(("draft_max_model_len", 4096),))
    long_views = (RequestView("r0", 1, 131071, False, "draft"),)
    plan_draft = replace(plan_target, scope="draft")
    assert _draft_kv_cap(plan_target, gqa_model, rt_capped, "decode") == 0
    assert _draft_kv_cap(plan_draft, gqa_model, rt_capped, "draft") == 4096
    assert _cap_draft_views(long_views, 4096)[0].kv_len == 4096
    gqa_uncapped = gqa(prefix="d", plan=plan_draft, views=long_views, stage=0, deps="x",
                       model=gqa_model, parallel=ParallelSpec(), quant=gqa_model.quant,
                       mode="draft", padded_tokens=1, runtime=base.runtime, machine=base.machine)
    gqa_capped = gqa(prefix="d", plan=plan_draft, views=long_views, stage=0, deps="x",
                     model=gqa_model, parallel=ParallelSpec(), quant=gqa_model.quant,
                     mode="draft", padded_tokens=1, runtime=rt_capped, machine=base.machine)
    attn_uncapped = next(op for op in gqa_uncapped.ops if op.family == "attention")
    attn_capped = next(op for op in gqa_capped.ops if op.family == "attention")
    assert attn_capped.flops < attn_uncapped.flops / 16
    assert attn_capped.hbm_bytes < attn_uncapped.hbm_bytes / 16

    mla_case = parse_case(compose_case(
        "deepseek-v3.2", hardware="a5", tp=1, dp=1, ep=1, num_devices=1,
        layers=2, quant="bf16", prompt_tokens=16, output_tokens=2))
    mla_plan = replace(mla_case.model.layer_plans[0], scope="draft", index_topk=0,
                       indexer=False, owns_topk=False, owns_index_k=False,
                       index_query_per_layer=False, topk_owner_layer_id=-1, index_k_owner_layer_id=-1)
    mla_rt = replace(mla_case.runtime, metadata=(("draft_max_model_len", 2048),))
    mla_uncapped = mla(prefix="dm", plan=mla_plan, views=long_views, stage=0, deps="x",
                       model=mla_case.model, parallel=ParallelSpec(), quant=mla_case.model.quant,
                       mode="draft", padded_tokens=1, runtime=mla_case.runtime, machine=mla_case.machine,
                       cache_ids={}, topk_ids={}, index_cache_ids={}, candidate_ids={}, stage_of=lambda _: 0)
    mla_capped = mla(prefix="dm", plan=mla_plan, views=long_views, stage=0, deps="x",
                     model=mla_case.model, parallel=ParallelSpec(), quant=mla_case.model.quant,
                     mode="draft", padded_tokens=1, runtime=mla_rt, machine=mla_case.machine,
                     cache_ids={}, topk_ids={}, index_cache_ids={}, candidate_ids={}, stage_of=lambda _: 0)
    mla_attn_u = next(op for op in mla_uncapped.ops if op.family == "attention")
    mla_attn_c = next(op for op in mla_capped.ops if op.family == "attention")
    assert mla_attn_c.hbm_bytes < mla_attn_u.hbm_bytes / 16


def test_windowed_fia_gqa_and_mla_bound_visible_kv_rows_and_cp_spans():
    from mindie_sim.prediction.operators.attention._fia import _visible_sequences
    from mindie_sim.prediction.operators.attention.attention import gqa
    from mindie_sim.prediction.operators.mla.generic_latent_attention_views import attention_view_record
    from mindie_sim.prediction.operators.mla.mla import mla
    from mindie_sim.schema.parallel import ParallelSpec

    contract = governed.AttentionContract(1, 32, 2, 128, 128, sequences=((1, 131072),), window_size=512)
    assert _visible_sequences(contract, ((1, 131072), (4, 131072))) == ((1, 512), (4, 515))
    geom = geometry(governed.ARCH35)
    win_plan = governed.build(governed.make_attention_op(contract, op_id="fia-win"), geom)
    assert win_plan.work.matrix["bf16"] == 2 * 32 * 512 * 256

    base = dense_case(layers=1)
    full_plan = replace(base.model.layer_plans[0], num_heads=8, num_kv_heads=2, head_dim=128)
    swa_plan = replace(full_plan, window=256)
    gqa_model = replace(base.model, num_heads=8, num_kv_heads=2, head_dim=128, layer_plans=(full_plan,))
    views = (RequestView("r0", 4, 8192, False, "decode"),)
    res_swa = gqa(prefix="w", plan=swa_plan, views=views, stage=0, deps="x",
                  model=gqa_model, parallel=ParallelSpec(), quant=gqa_model.quant,
                  mode="decode", padded_tokens=4, runtime=base.runtime, machine=base.machine)
    res_full = gqa(prefix="f", plan=full_plan, views=views, stage=0, deps="x",
                   model=gqa_model, parallel=ParallelSpec(), quant=gqa_model.quant,
                   mode="decode", padded_tokens=4, runtime=base.runtime, machine=base.machine)
    attn_swa = next(op for op in res_swa.ops if op.family == "attention")
    attn_full = next(op for op in res_full.ops if op.family == "attention")
    assert attn_swa.hbm_bytes < attn_full.hbm_bytes / 10

    mla_case = parse_case(compose_case(
        "deepseek-v3.2", hardware="a5", tp=1, dp=1, ep=1, num_devices=1,
        layers=2, quant="bf16", prompt_tokens=16, output_tokens=2))
    mla_swa = replace(mla_case.model.layer_plans[0], window=256, index_topk=0,
                      indexer=False, owns_topk=False, owns_index_k=False,
                      index_query_per_layer=False, topk_owner_layer_id=-1, index_k_owner_layer_id=-1)
    rec = attention_view_record(
        views[0], plan=mla_swa, model=mla_case.model, quant=mla_case.model.quant,
        query_tile=128, eff_cp=1, block=128, interleave=128, tp=1, dsa=False,
        topk_present=False, heads=mla_swa.num_heads, latent_dim=mla_swa.kv_lora_rank,
    )
    assert rec["v_rows"] == [259]
    mla_cp = mla(prefix="cp", plan=mla_swa, views=views, stage=0, deps="x",
                 model=mla_case.model, parallel=ParallelSpec(cp=4), quant=mla_case.model.quant,
                 mode="decode", padded_tokens=4, runtime=mla_case.runtime, machine=mla_case.machine,
                 cache_ids={}, topk_ids={}, index_cache_ids={}, candidate_ids={}, stage_of=lambda _: 0)
    mla_cp_attn = next(op for op in mla_cp.ops if op.family == "attention")
    assert mla_cp_attn.flops == pytest.approx(2 * 512 * mla_swa.num_heads * (mla_swa.kv_lora_rank + mla_swa.qk_rope_head_dim + mla_swa.kv_lora_rank))
    mla_cp_q1 = mla(prefix="cp1", plan=mla_swa, views=(RequestView("r0", 1, 8192, False, "decode"),),
                    stage=0, deps="x", model=mla_case.model, parallel=ParallelSpec(cp=4),
                    quant=mla_case.model.quant, mode="decode", padded_tokens=1,
                    runtime=mla_case.runtime, machine=mla_case.machine,
                    cache_ids={}, topk_ids={}, index_cache_ids={}, candidate_ids={}, stage_of=lambda _: 0)
    mla_cp_q1_attn = next(op for op in mla_cp_q1.ops if op.family == "attention")
    assert dict(mla_cp_q1_attn.metadata)["latent_attention_schedule"]["keys"] == 128


def test_gqa_asymmetric_value_head_dim_scales_kv_bytes_and_output_gate():
    from mindie_sim.prediction.operators.attention.attention import gqa
    from mindie_sim.schema.parallel import ParallelSpec

    base = dense_case(layers=1)
    asym_plan = replace(base.model.layer_plans[0], num_heads=8, num_kv_heads=2,
                        head_dim=128, v_head_dim=64, q_output_gate=True)
    gqa_model = replace(base.model, num_heads=8, num_kv_heads=2, head_dim=128, layer_plans=(asym_plan,))
    views = (RequestView("r0", 2, 128, False, "decode"),)
    res = gqa(prefix="a", plan=asym_plan, views=views, stage=0, deps="x",
              model=gqa_model, parallel=ParallelSpec(), quant=gqa_model.quant,
              mode="decode", padded_tokens=2, runtime=base.runtime, machine=base.machine)
    by_id = {op.op_id: op for op in res.ops}
    by_suffix = {op.op_id.rsplit(".", 1)[-1]: op for op in res.ops}
    assert by_id[res.cache_store_id].mte3_bytes == 2 * 2 * (128 + 64) * 2
    assert by_suffix["output_gate"].local_shape == (2 * 8 * 64,)
    assert by_suffix["output"].flops == 2 * 2 * gqa_model.hidden_size * (8 * 64)


def test_wide_head_tiling_enforces_fp32_ub_and_l0c_bounds_and_keeps_aic_pools_disjoint():
    from mindie_sim.prediction.runtime.tile_solver import TileInfeasibleError

    geom = geometry(governed.ARCH22)
    contract = governed.AttentionContract(1, 4, 2, 512, 512, selected_tokens=128,
                                          panel_mode="tile_solver_candidate")
    solved = governed.solve_wide_head_tiling(contract, geom, s1=128, s2=128)
    peaks = {peak.pool: peak.peak_bytes for peak in solved.pool_peaks}
    assert peaks["l0c"] == 128 * 128 * 4
    assert peaks["ub"] == (128 * 128 + 128 * 2) * 4
    plan = governed.build(governed.make_attention_op(contract, op_id="wide-fp32"), geom)
    assert plan.metadata["tile_solution"]["ub_peak_bytes"] == peaks["ub"]
    with pytest.raises(TileInfeasibleError, match=r"ub\(aiv\)=264192>"):
        governed.solve_wide_head_tiling(contract, geom, s1=256, s2=256)


def test_generic_latent_attention_scales_multi_query_work_and_flops():
    from mindie_sim.prediction.operators.mla import generic_latent_attention as gla

    geom = geometry(governed.ARCH35)
    meta_base = {
        "heads": 16, "qk_dim": 576, "latent_dim": 512, "dv": 512, "kv_row_bytes": 1152,
    }
    op1 = OpSpec("gla1", "attention", "decode", 0, 0, ("r0",), (1, 16, 512), (1, 16, 512),
                 (1, 16, 512), (1, 16, 512), flops=float(2 * 1 * 1024 * 16 * 1088),
                 metadata=(("latent_attention_schedule", {**meta_base, "sequences": ((1, 1024),)}),))
    op4 = replace(op1, op_id="gla4", flops=float(2 * 4 * 1024 * 16 * 1088),
                  metadata=(("latent_attention_schedule", {**meta_base, "sequences": ((4, 1024),)}),))
    p1, p4 = gla.build(op1, geom), gla.build(op4, geom)
    assert p4.work.matrix["bf16"] >= op4.flops
    assert p4.work.matrix["bf16"] == 4 * p1.work.matrix["bf16"]
    assert p4.work.resources["vector"] == 4 * p1.work.resources["vector"]


def test_indexed_attention_compact_preserves_vector_work_under_calibrated_dq_scale():
    case = sku_case(REPRESENTATIVE_SKUS["a5"])
    geom = Geometry.from_devices(case.machine.dies)
    op = source_cases._c8_op(((1, 512),), 512, 32, 0)
    p07 = SimpleNamespace(status="calibrated", parameters={},
                          metrics={"vec_alu_ops_per_s_per_aiv": SimpleNamespace(p50=100e9)})
    p09 = SimpleNamespace(status="calibrated", parameters={},
                          metrics={"cast_bytes_per_s_per_aiv": SimpleNamespace(p50=2000e9)})
    pt = SimpleNamespace(warnings=(), cell=lambda cid: {"P07_vec_alu": p07, "P09_vec_reduce_cast": p09}.get(cid))

    def run_mode(compact, primitive_table):
        plan = source_cases.build_packed(op, geom, compact=compact)
        die = case.machine.dies[0]
        die_rates = dict(die.rates)
        roles = set(die_rates) | {"cube"}
        names = {r: r for r in roles}
        rates = {r: float(die_rates.get(r, 1e12)) for r in roles}
        units = {r: int(die.units(r)) for r in roles}
        ctx = RecipeContext(op, {}, names, rates, die.compute_units,
                            resource_units=units, compute_peaks=dict(die.compute_peaks),
                            service_modes=dict(die.service_modes),
                            pool_names={p: p for p in dict(die.buffers)},
                            pool_capacities=dict(die.buffers),
                            pool_units={p: die.pool_units(p) for p in dict(die.buffers)},
                            primitive_table=primitive_table)
        engine = Engine([Resource(r, rates[r], units=units[r]) for r in roles],
                        pools=dict(die.buffers), pool_units=ctx.pool_units)
        done = plan.program(ctx).start(engine, "idx", engine.submit("ready"))
        engine.advance(until=(done,))
        return done.time, engine.serviced["vector"], plan.work.resources["vector"]

    _, vec_expanded, declared = run_mode(False, pt)
    t_cal, vec_compact, _ = run_mode(True, pt)
    t_uncal, vec_uncal, _ = run_mode(True, None)
    assert vec_compact == pytest.approx(vec_expanded)
    assert vec_compact == pytest.approx(declared)
    assert vec_uncal == pytest.approx(declared)
    assert t_cal <= t_uncal


def test_mqsmla_quant_plan_avoids_per_tile_window_overcount_and_rejects_non_fp8_arch():
    raw = compose_case(
        "deepseek-v4.1-flash", hardware="a5", tp=4, dp=2, ep=8, num_devices=8,
        layers=4, quant="w8a8", prompt_tokens=16, output_tokens=2)
    packed = {"dtype": "fp8_e4m3", "group_size": 64, "scale_dtype": "bf16"}
    raw["model"].setdefault("metadata", {})["cache_representations"] = {
        "window": dict(packed), "compressed_kv": dict(packed),
    }
    parsed = parse_case(raw)
    graph = target_graphs(parsed, phase="decode", tokens=1, context=1024, sequences=1, scope="body")[0]
    native = next(op for op in graph if op.recipe == "aclnn_mixed_quant_sparse_flash_mla")
    plan_a5 = mqsmla.build(native, geometry(governed.ARCH35))
    shape = dict(native.metadata)["mqsmla_geometry"]
    rows = tuple(mqsmla.query_rows(shape))
    expected_mte2_tiles = sum(
        work.resources.get("mte2_aic", 0)
        for row in rows
        for idx in range(mqsmla.OwnerTiles.create((row,)).count)
        for _n, work, _a in mqsmla.quant_tile_phases_cached(
            *mqsmla.OwnerTiles.create((row,)).key(idx)[1:], int(shape["heads"]), int(shape["kv_quant_mode"]), 0)
    )
    streamed = len(rows) * int(shape["heads"]) * mqsmla.MQSMLA_HEAD_WIDTH * mqsmla.BF16_ELEMENT_BYTES
    assert plan_a5.work.resources["mte2_aic"] == expected_mte2_tiles + streamed
    with pytest.raises(ModelingError, match=r"^mqsmla_fp8_fp4_unsupported_on_"):
        mqsmla.build(native, geometry(governed.ARCH22))
