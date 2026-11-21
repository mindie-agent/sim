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

import pytest

from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS, SKUS
from _support.builders import dense_case
from _support.captures import REPO_ROOT, fixture_json
from _support.hardware import isa_of, sku_case
from mindie_sim import compose_case, graph_costs, operator_costs, target_operators
from mindie_sim.prediction.operators.sharedkv import sharedkv_attention as scfa
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import attention as governed
from mindie_sim.prediction.operators.attention._cp_tiling import HISTORY_DECODE_SITE, VLLM_REVISION
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.streams import StreamOwners
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure_contracts import predicted_contract

KIB = 1024

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
def test_sharedkv_scratch_and_output_follow_combined_window_and_selection(query, context, topk):
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
        ops, _ = frontend.body(case.model, case.runtime, views, ready="input", tokens=4, mode="decode")
        attention = [op for op in ops if op.family == "attention" and op.layer == 0]
        assert len(attention) == (1 if fused else 2)
        assert sum(op.flops for op in attention) == 2 * (8 + 12 + 13 + 14) * 4 * (16 + 16)
