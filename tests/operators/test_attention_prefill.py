"""Dense prefill, online softmax, and fused attention boundaries.

[Role] Exercise staged, governed, and observed prefill through their owning APIs.
[Invariants] Conserve tensor work, traffic, dependencies, and rank-local ownership.
[Timeouts & Locks] Bounded CPU planning and execution only.
[Side-Effect Policy] Read retained fixtures and use in-memory declarations.
"""

from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.builders import dense_case, dense_layer
from _support.captures import REPO_ROOT
from _support.hardware import sku_case
from _support.operator_cases import mqsmla_attention, mqsmla_case
from mindie_sim import compose_case, graph_costs, target_graphs
from mindie_sim.backends.ascend import materialize
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.operators.context_parallel.graph import emit_dcp_output_exchange_combine
from mindie_sim.prediction.operators.paged_mla import family_mqsmla, mqsmla
from mindie_sim.prediction.operators.attention.attention import gqa
from mindie_sim.prediction.call_builders import build_aclnn_call
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operators.matrix.matrix_mapping import matrix_geometry
from mindie_sim.prediction.operators.matrix.matrix_splitk import splitk_program
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators import attention as governed
from mindie_sim.prediction.operators.attention._cp_tiling import VLLM_MLA_REVISION
from mindie_sim.prediction.operators.attention.common import (
    CACHE_GQA_SITE, CACHE_MLA_SITE, COMPLETE_KV_SITE, HISTORY_PREFILL_SITE, VLLM_REVISION,
)
from mindie_sim.prediction.physical_recipe_registry import ControlSyncRecipe, resolve_physical_recipe
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, RecipeBook, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure import export_op_structure
from mindie_sim.structure_contracts import comparison_scope, predicted_contract

pytestmark = pytest.mark.tier0

PREFILL_TOKENS = 4096
INDEX_LAYERS = 61
INDEX_QUERY_TILE = 128
INDEX_ROW_BYTES = 256
INDEX_HEADS = 64
INDEX_DIM = 128
SPLITK_SKU = "Ascend950DT_9582"


@pytest.mark.parametrize("hardware", SKUS)
@pytest.mark.parametrize("causal,window,expected", [(True, 0, 29), (True, 4, 16), (False, 0, 32)])
def test_governed_prefill_preserves_ragged_selection_and_distinct_head_widths(hardware, causal, window, expected):
    geom = Geometry.from_devices(sku_case(hardware).machine.dies)
    contract = governed.AttentionContract(4, 4, 2, 128, 64, sequences=((3, 9), (1, 5)),
                                          phase="prefill", causal=causal, window_size=window)
    op = governed.make_attention_op(contract, op_id="attention")
    plan = governed.build(op, geom)
    phases = {phase.name: phase for phase in plan.program.phases}
    assert sum(contract.row_counts()) == expected
    assert phases["qk_panel_0_accumulate"].work.matrix == {"bf16": 2 * 32 * 4 * 128}
    assert phases["pv_panel_0"].work.matrix == {"bf16": 2 * 32 * 4 * 64}
    assert plan.work.matrix == {"bf16": 2 * 32 * 4 * (128 + 64)}
    assert plan.metadata["useful_flops"] == op.flops == 2 * expected * 4 * (128 + 64)
    assert phases["output"].work.resources["mte3"] == 4 * 4 * 64 * 2
    assert phases["online_softmax"].after == (("score_workspace_read",)
        if geom.compute_arch.preset == governed.ARCH22 else ("score_fixpipe",))


@pytest.mark.parametrize("hardware", SKUS)
def test_governed_prefill_rejects_nondivisible_head_groups(hardware):
    geom = Geometry.from_devices(sku_case(hardware).machine.dies)
    with pytest.raises(ModelingError):
        contract = governed.AttentionContract(2, 7, 2, 128, 64, selected_tokens=8, phase="prefill")
        governed.build(governed.make_attention_op(contract, op_id="attention"), geom)


def test_prefill_collectives_keep_history_and_cache_publication_ownership():
    plan = governed.ContextParallelPlan
    base = dict(stack_revision=VLLM_REVISION, phase="prefill", pcp_size=2, dcp_size=4)
    history = plan(**base, caller_site=HISTORY_PREFILL_SITE, case="history_attention",
                   distinct_head_ranks=2, tensors=(("query", (3, 2, 128), 2),))
    cache = plan(**base, caller_site=CACHE_GQA_SITE, case="cache_publication",
                 tensors=(("cache_key", (3, 2, 128), 2), ("cache_value", (3, 2, 64), 2)))
    complete = plan(**base, caller_site=COMPLETE_KV_SITE, case="complete_kv_sparse_prefill",
                    tensors=(("cache", (3, 1, 512), 2),))
    token, head = governed.lower_cp_collectives(history, deps=("q_ready",))
    (publish,) = governed.lower_cp_collectives(cache, deps=("kv_ready",))
    (gather,) = governed.lower_cp_collectives(complete)
    assert token.deps == ("q_ready",) and head.deps == (token.op_id,) and publish.deps == ("kv_ready",)
    assert token.link_bytes == 3 * 2 * 128 * 2 and head.link_bytes == 6 * 2 * 128 * 2
    assert dict(token.metadata)["output_tensors"] == (("query", (6, 2, 128)),)
    assert dict(head.metadata)["output_tensors"] == (("query", (6, 4, 128)),)
    keys = [key for op in (token, head, publish, gather) for key in dict(op.metadata)["attention_cp_keys"]]
    assert len(set(keys)) == len(keys)
    assert {(key[3], key[5]) for key in keys} == {("pcp", 0), ("tp", 1), ("dcp", 0)}
    assert {key[0] for key in keys} == {VLLM_REVISION}
    assert history.return_lse and history.output_merge
    assert not dict(gather.metadata)["return_lse"] and not dict(gather.metadata)["output_merge"]
    contract = governed.AttentionContract(3, 4, 1, 512, 512, phase="prefill", selected_tokens=8,
                                          abi=governed.NOPE_ABI, cp_plan=complete)
    geom = Geometry.from_devices(sku_case(REPRESENTATIVE_SKUS["a5"]).machine.dies)
    sparse = governed.build(governed.make_attention_op(contract, op_id="sparse"), geom)
    assert sparse.work.resources["mte3"] >= 3 * 4 * 512 * 2
    assert sparse.work.resources["link"] == gather.link_bytes
    with pytest.raises(ModelingError, match="attention_cp_consumer_contract_mismatch"):
        replace(contract, return_lse=True)


@pytest.mark.parametrize("gate,pcp", [(False, 1), (True, 1), (True, 2)])
def test_qkv_fusion_partitions_projection_and_materializes_gather_inputs(gate, pcp):
    case = dense_case()
    plan = dense_layer(0, qk_norm=True, q_output_gate=gate)
    ops = gqa(prefix="attention", plan=plan, views=(RequestView("r", 3, 0, True, "prefill"),),
              stage=0, deps="input", model=case.model, quant=case.model.quant, mode="prefill",
              padded_tokens=3, parallel=replace(case.runtime.parallel, pcp=pcp, cp_plan="auto")).ops
    by_id = {op.op_id: op for op in ops}
    declaration = dict(by_id["attention.rope"].metadata)["structure_subgraphs"][0]
    addresses = [
        {layout["offset"] + sum(i * s for i, s in zip(index, layout["strides"]))
         for index in product(*(range(n) for n in layout["shape"]))}
        for layout in declaration["tensor_view"]["input_views"].values()
    ]
    pitch = (4 * (2 if gate else 1) + 2 * 2) * 16
    assert set.union(*addresses) == set(range(3 * pitch)) and sum(map(len, addresses)) == 3 * pitch
    copies = [op for op in ops if op.recipe == "qkv_split_materialize"]
    assert sum(op.mte2_bytes for op in copies) == sum(op.mte3_bytes for op in copies) == 3 * (2 + 4 * gate) * 32
    assert sum(op.vector_ops + op.flops for op in copies) == 0
    manifest = export_op_structure(ops, model=case.model.to_dict(), execution={"phase": "prefill"})
    group = next(row for row in manifest["subgraphs"] if row["role"] == "qkv_norm_rope")
    assert group["external_inputs"] == ["rank:0/attention.qkv"] and group["contract"] is None
    if pcp > 1:
        gather = by_id["attention.kv_gather"]
        assert set(gather.deps) == {"attention.rope", "attention.qkv.value"}
        assert by_id["attention.cache"].deps == (gather.op_id,) and gather.op_id not in declaration["op_ids"]


def test_partial_update_follows_collective_and_owns_only_local_merge():
    ops = []
    last = emit_dcp_output_exchange_combine(
        ops, prefix="attention", stage=0, layer=0, requests=("r",), mode="decode",
        deps=("partial_attention",), rows=3, local_heads=4, v_dim=8,
        act_dtype="bf16", pcp_factor=1, tp_gather=2, dp_width=1)
    pack, exchange, update = ops
    meta = dict(update.metadata)
    assert last == update.op_id and update.deps == (exchange.op_id,)
    assert pack.deps == ("partial_attention",) and exchange.deps == (pack.op_id,)
    assert exchange.link_bytes > 0 and update.link_bytes == 0 and update.vector_ops > 0
    assert meta["partial_count"] == 2 and meta["lse_wire_bytes"] == 4


@pytest.mark.requires_profiler
def test_paged_fia_projection_matches_observed_shape_and_rejects_call_aliasing(profiler_api):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.tensor_contracts import observed_contract

    capture = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/mixed_step_body/a3_qwen36.csv"
    observed = next(observed_contract(row) for row in kernel_rows(capture)
                    if row.kind == "FusedInferAttentionScore" and row.input_shapes[0][0] == 4
                    and observed_contract(row)[0] == "attention_paged_active_qkv")
    raw = json.loads((REPO_ROOT / "examples/cases/qwen36-a3-tp4.json").read_text())
    raw["runtime"].setdefault("attention_policy", {})["kv_storage"] = "paged"
    ops = target_operators(parse_case(raw), phase="decode", tokens=4, context=8192)
    attention = [op for op in ops if op.family == "attention"]
    assert len(attention) > 1 and all(predicted_contract(op.to_dict()) == observed for op in attention)
    op = attention[0]
    assert predicted_contract(replace(op, source_boundary="call").to_dict()) is None
    assert not comparison_scope(op.to_dict())["occurrence_identity"]
    recurrent = next(op for op in ops if op.family == "recurrent_vector")
    row = next(row for row in kernel_rows(capture)
               if row.kind == "RecurrentGatedDeltaRule" and row.input_shapes[0][0] == 4)
    assert predicted_contract(recurrent.to_dict()) == observed_contract(row)
    wrong_dtype = replace(recurrent, dtype="fp16").to_dict()
    assert predicted_contract(wrong_dtype) is None
    assert comparison_scope(wrong_dtype)["contract_support"] == "unsupported_recurrent_gdn_dtype"
    for key, value in [("batch_count", 2), ("key_tail_shape", (64, 256)), ("output_dtype", "fp16")]:
        changed = deepcopy(op.to_dict())
        changed["metadata"]["attention_tensor_view"][key] = value
        assert predicted_contract(changed) != observed


@pytest.mark.requires_profiler
def test_observed_fusion_with_missing_output_slots_stays_unbound(profiler_api):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.tensor_contracts import observed_contract, observed_contract_diagnostic

    capture = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/fusion_contracts/a5_qwen38.csv"
    row = next(row for row in kernel_rows(capture)
               if row.kind == "split_qkv_rmsnorm_mrope_kernel" and row.input_shapes)
    assert observed_contract(row) is None
    slots = observed_contract_diagnostic(row)
    assert slots["required_output_slots"] == 4 and slots["observed_output_slots"] == 1


def test_shipping_scores_charge_the_tile_rectangle_before_causal_masking():
    case = sku_case(REPRESENTATIVE_SKUS["a5"])
    contract = governed.AttentionContract(
        128, 1, 1, 512, 512, sequences=((128, 128),), phase="prefill", causal=True,
    )
    op = governed.make_attention_op(contract, op_id="prefill")
    plan = governed.build(op, Geometry.from_devices(case.machine.dies))
    issued = sum(
        item.amount for phase in plan.pipeline_spec.phases if phase.phase_id.startswith("qk_panel_")
        for item in phase.work if item.resource == "cube"
    )
    useful = 2 * sum(range(1, 129)) * 512
    assert issued == 16_777_216 and useful == 8_454_144
    assert plan.metadata["useful_flops"] == useful * 2
    assert plan.work.matrix["bf16"] == issued * 2
    phases = {phase.name: phase for phase in plan.program.phases}
    assert phases["score_fixpipe"].work.resources["fixpipe"] == 128 * 128 * 4
    assert phases["pv_fixpipe"].work.resources["fixpipe"] == 128 * 512 * 4
    assert phases["online_output"].after == ("pv_fixpipe",)
    audit = verify_operator(case, op, governed.build, input_ready_ms=7)
    assert audit["status"] == "passed" and audit["work_mismatches"] == {}


def test_shipping_prefill_reuses_three_slots_and_drains_every_tile():
    case = sku_case(REPRESENTATIVE_SKUS["a5"])
    geom = Geometry.from_devices(case.machine.dies)
    contract = governed.AttentionContract(2, 1, 1, 128, 128, sequences=((2, 385),), phase="prefill")
    op = governed.make_attention_op(contract, op_id="prefill")
    plan = governed.build(op, geom)
    rates = {**{role: rate / 1000 for role, rate in geom.rates.items()}, "cube": 1}
    units = geom.resource_units
    context = RecipeContext(op, {}, {role: role for role in rates}, rates, geom.compute_units,
                            resource_units=units, compute_peaks=geom.compute_peaks)
    engine = Engine([Resource(role, rate, units=units[role]) for role, rate in rates.items()], trace_limit=128)
    done = plan.lower(context).start(engine, "prefill", engine.submit("input", release=3))
    engine.advance(until=(done,))

    def tiles(stage):
        return [row for row in engine.trace if row["task"].endswith(":" + stage)]

    qk = tiles("qk_panel_0_accumulate")
    pv = tiles("pv_panel_0")
    vec2 = tiles("online_output")
    assert len(qk) == len(pv) == len(vec2) == 4
    assert qk[0]["start_ms"] >= 3
    assert qk[2]["end_ms"] <= pv[0]["start_ms"]
    assert vec2[0]["end_ms"] <= qk[3]["start_ms"]
    assert qk[3]["end_ms"] <= pv[1]["start_ms"]
    assert vec2[-1]["end_ms"] <= tiles("output")[0]["start_ms"]
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert {role: amount for role, amount in engine.serviced.items() if amount} == pytest.approx(context.bind(plan.work))


def test_frontend_ragged_prefill_roundtrips_to_physical_work():
    case = dense_case()
    views = (RequestView("a", 3, 5, False, "prefill"), RequestView("b", 5, 2, False, "prefill"))
    ops, _ = Frontend().body(case.model, case.runtime, views, ready="input", tokens=8, mode="prefill")
    semantic = next(op for op in ops if op.family == "attention")
    sequences = tuple((view.q_len, view.context + view.q_len) for view in views)
    contract = governed.AttentionContract(8, 4, 2, 16, 16, sequences=sequences, phase="prefill", causal=True)
    op = governed.make_attention_op(contract, op_id=semantic.op_id, deps=semantic.deps)
    plan = governed.build(op, Geometry.from_devices(sku_case(REPRESENTATIVE_SKUS["a5"]).machine.dies))
    assert op.flops == semantic.flops == 2 * (21 + 25) * 4 * (16 + 16)
    assert plan.work.matrix == {"bf16": 2 * (3 * 8 + 5 * 7) * 4 * (16 + 16)}


def test_collectives_run_before_attention_with_already_sliced_mla_publication():
    base = dict(stack_revision=VLLM_REVISION, phase="prefill", pcp_size=2, dcp_size=4)
    publication = governed.ContextParallelPlan(
        **{**base, "stack_revision": VLLM_MLA_REVISION}, caller_site=CACHE_MLA_SITE,
        case="cache_publication", decode_tokens=3,
        tensors=(("kv_no_split", (2, 576), 2), ("cos", (2, 64), 2), ("sin", (2, 64), 2)),
    )
    history = governed.ContextParallelPlan(
        **base, caller_site=HISTORY_PREFILL_SITE, case="history_attention", distinct_head_ranks=2,
        tensors=(("query", (2, 1, 128), 2),),
    )
    (gather,) = governed.lower_cp_collectives(publication)
    meta = dict(gather.metadata)
    assert publication.caller_site == "mla_v1.exec_kv_prefill"
    assert meta["replicated_decode_tokens"] == 3
    assert [shape[0] for _, shape in meta["input_tensors"]] == [2, 2, 2]
    assert [shape[0] for _, shape in meta["output_tensors"]] == [4, 4, 4]
    assert gather.link_bytes == 2 * (576 + 64 + 64) * 2

    geom = Geometry.from_devices(sku_case(REPRESENTATIVE_SKUS["a5"]).machine.dies)
    contract = governed.AttentionContract(
        4, 2, 1, 128, 128, selected_tokens=64, phase="prefill", return_lse=True, cp_plan=(publication, history),
    )
    op = governed.make_attention_op(contract, op_id="prefill", deps=("input",))
    book = RecipeBook(governed.recipes())
    cache, token, head, consumer, join = book.expand((op,))
    assert token.deps == ("input", cache.op_id) and head.deps == ("input", token.op_id)
    assert consumer.deps == ("input", head.op_id) and join.deps == (consumer.op_id,)
    assert [dict(stage.metadata)["group_axis"] for stage in (cache, token, head)] == ["pcp", "pcp", "tp"]
    assert [stage.link_bytes for stage in (cache, token, head)] == [gather.link_bytes, 512, 1024]
    assert book.expand((cache, token, head, consumer, join)) == (cache, token, head, consumer, join)

    plan = governed.build(op, geom)
    assert any(ref.revision == VLLM_MLA_REVISION for ref in plan.pipeline_spec.sources)
    rates = {**{r: value / 1000 for r, value in geom.rates.items()}, "cube": 1, "link": 1000}
    units = {**geom.resource_units, "link": 1}
    context = RecipeContext(op, {}, {role: role for role in rates}, rates, geom.compute_units,
                            resource_units=units, compute_peaks=geom.compute_peaks)
    engine = Engine([Resource(role, rate, units=units[role]) for role, rate in rates.items()], trace_limit=128)
    ready = engine.submit("ready", release=5)
    end = plan.lower(context).start(engine, "prefill", ready)
    engine.advance(until=(end,))
    comms = [row for row in engine.trace if row["demand"].get("link")]
    query = next(row for row in engine.trace if row["task"].endswith(":query_load"))
    assert [row["demand"]["link"] for row in comms] == [gather.link_bytes, 512, 1024]
    assert comms[0]["start_ms"] >= 5
    assert all(left["end_ms"] <= right["start_ms"] for left, right in zip(comms, (*comms[1:], query)))
    assert {role: amount for role, amount in engine.serviced.items() if amount} == pytest.approx(context.bind(plan.work))

    case = compose_case("deepseek-v2-lite", hardware=REPRESENTATIVE_SKUS["a5"], tp=2, pcp=2, dcp=4,
                        layers=1, block_size=16, prompt_tokens=16, output_tokens=2, concurrency=1)
    result = graph_costs(case, (op,), recipes=book, external_inputs={"input": 5}, isolated=False)
    by_id = {row["op_id"]: row for row in result["timeline"]}
    steps = [by_id[node.op_id] for node in (cache, token, head, consumer)]
    assert steps[0]["ready_ms"] == 5
    assert all(step["complete_ms"] > step["ready_ms"] for step in steps)
    assert all(left["complete_ms"] <= right["ready_ms"] for left, right in zip(steps, steps[1:]))


def test_history_head_gather_uses_dcp_without_pcp_copies():
    history = governed.ContextParallelPlan(
        VLLM_REVISION, HISTORY_PREFILL_SITE, "prefill", "history_attention",
        (("query", (2, 1, 128), 2),), dcp_size=4,
    )
    (gather,) = governed.lower_cp_collectives(history)
    assert dict(gather.metadata)["group_axis"] == "cp"
    assert gather.link_bytes == 2 * 1 * 128 * 2 * (4 - 1)
    assert dict(gather.metadata)["output_tensors"] == (("query", (2, 4, 128)),)



@pytest.fixture(scope="module")
def mixed_quant_source():
    case = mqsmla_case()
    graph = target_graphs(case, phase="decode", tokens=1, context=2048)[0]
    return case, mqsmla_attention(graph)


@pytest.mark.parametrize("tp", (1, 2, 4, 8))
def test_hca_local_head_groups_conserve_source_work(mixed_quant_source, tp):
    case, original = mixed_quant_source
    model = load_model("deepseek-v4-pro-0813")
    model = replace(model, metadata=tuple({
        **dict(model.metadata), "cache_representations": family_mqsmla.FORMATS}.items()))
    hca = next(plan for plan in model.layer_plans if plan.compress_ratio == 128)
    view = RequestView("r", 1, 131584, True, "decode")
    parallel = ParallelSpec(tp=tp)
    family_mqsmla.validate(views=(view,), plan=hca, parallel=parallel, model=model,
                           padded_tokens=1, topk_present=False, cores=32)
    ops = [original]
    family_mqsmla.lower_attention(
        ops, [original.op_id], selection=SimpleNamespace(name=family_mqsmla.RECIPE),
        views=(view,), plan=hca, topk_present=False, prefix="s0.l0", parallel=parallel)
    shape = dict(ops[0].metadata)["mqsmla_geometry"]
    local_heads = 128 // tp
    assert shape["heads"] == local_heads and shape["ratio"] == 128
    assert tuple(mqsmla.query_rows(shape)) == ((128, 1028),)
    tiles = mqsmla.OwnerTiles.create(((128, 1028),))
    expected = Work()
    for start in range(0, local_heads, 64):
        heads = min(64, local_heads - start)
        for index in range(tiles.count):
            for _, work in mqsmla.tile_phases(*tiles.key(index), heads=heads):
                expected += work
    assert mqsmla.build(ops[0], Geometry.from_devices(case.machine.dies)).work == expected


def test_sliding_window_layer_has_no_compressed_rows(mixed_quant_source):
    case, _ = mixed_quant_source
    view = RequestView("r", 1, 131584, True, "decode")
    swa = next(plan for plan in case.model.layer_plans if not plan.owns_long_kv and not plan.owns_topk)
    assert family_mqsmla.query_shape((view,), swa)["ratio"] == 0


@pytest.mark.parametrize("heads,supported", [(heads, True) for heads in (8, 16, 32, 48, 64, 80, 96, 128)]
                         + [(4, False)])
def test_mqsmla_source_head_group_support(heads, supported):
    assert mqsmla.supported_heads(heads) is supported


def test_csa_nested_prefix_and_private_window():
    contexts = (4, 4, 4096)
    shape = dict(sequences=tuple((1, c) for c in contexts), prefix_tokens=contexts,
                 prefix_cache_keys=("A",) * 3, prefix_cache_mode="shared",
                 ratio=4, window=128, topk=512)
    assert mqsmla.quant_external_fill(shape, mqsmla.KV_QUANT_FP8) == (395808, 513)


def test_paged_causal_attention_keeps_causal_pair_count():
    base = {
        "api": "aclnnFlashAttentionScore",
        "tensors": {
            "query": {"shape": [1, 8, 64], "dtype": "bf16"},
            "key": {"shape": [1, 8, 64], "dtype": "bf16"},
            "value": {"shape": [1, 8, 32], "dtype": "bf16"},
        },
        "attrs": {"causal": True},
        "phase": "prefill",
    }
    dense = build_aclnn_call(base).ops[0]
    paged = build_aclnn_call({**base, "block_size": 128}).ops[0]
    assert dense.flops == pytest.approx(6912.0)
    assert paged.flops == pytest.approx(dense.flops)
    assert paged.vector_ops == pytest.approx(dense.vector_ops)


def test_long_prefill_index_scores_read_each_key_union_once():
    case = parse_case(compose_case(
        "deepseek-v3.2", hardware=SPLITK_SKU, tp=8, ep=8, num_devices=8,
        ranks_per_node=8, prompt_tokens=PREFILL_TOKENS, output_tokens=2,
        concurrency=1))
    ops = target_graphs(case, phase="prefill", tokens=PREFILL_TOKENS, context=0,
                       sequences=1, scope="body")[0]
    expected_key_bytes = PREFILL_TOKENS * INDEX_ROW_BYTES
    refill_rows = sum(range(INDEX_QUERY_TILE, PREFILL_TOKENS + 1, INDEX_QUERY_TILE))
    scores = [op for op in ops if op.recipe == "quant_lightning_indexer"]
    assert len(scores) == INDEX_LAYERS
    first = scores[0]
    by_id = {op.op_id: op for op in ops}
    operand_bytes = sum(by_id[dep].mte3_bytes for dep in first.deps[:2])
    # The fused native score+TopK call reads Q, the head weights and every
    # per-tile key refill on MTE2; the external HBM union is charged once.
    assert first.mte2_bytes - operand_bytes == pytest.approx(
        refill_rows * INDEX_ROW_BYTES)
    assert first.hbm_bytes - operand_bytes - first.mte3_bytes == pytest.approx(
        expected_key_bytes)
    # Per-query arithmetic is unchanged; only the external key fetch shrinks.
    assert first.flops == pytest.approx(
        2 * (PREFILL_TOKENS * (PREFILL_TOKENS + 1) / 2)
        * INDEX_HEADS * INDEX_DIM)
    assert sum(op.mte2_bytes - sum(by_id[dep].mte3_bytes for dep in op.deps[:2])
               for op in scores) == pytest.approx(
        INDEX_LAYERS * refill_rows * INDEX_ROW_BYTES)
    assert sum(op.hbm_bytes - sum(by_id[dep].mte3_bytes for dep in op.deps[:2])
               - op.mte3_bytes for op in scores) == pytest.approx(
        INDEX_LAYERS * expected_key_bytes)


def test_sampling_uses_streamed_vector_recipe_and_control_rejects_payload():
    sampling = OpSpec(
        "sample", "sampling", "decode", 0, 0, (), (128000,), (128000,),
        (128000,), (128000,), vector_ops=640000.0, hbm_bytes=1024000.0,
        mte2_bytes=512000.0, mte3_bytes=512000.0, recipe="sample_logits")
    recipe = resolve_physical_recipe(sampling)
    assert type(recipe).__name__ == "AIVTiledVectorAndGatherRecipe"
    assert recipe.submode(sampling) == "streaming"
    guard = SimpleNamespace(
        op=OpSpec("c", "barrier", "decode", 0, 0, (), (), (), (), ()),
        native_work={"mte2": 1.0}, work={}, minimum_ms=0.0)
    with pytest.raises(ModelingError, match=r"^unmodeled_operator_mechanism:barrier::mte2=1$"):
        ControlSyncRecipe().lower(guard)


def _splitk_context():
    raw = materialize({"backend": "ascend", "sku": SPLITK_SKU,
                       "num_devices": 1, "ranks_per_node": 1})
    die = Hardware.from_mapping(raw).dies[0]
    op = gemm_op(op_id="probe", deps=(), stage=0, layer=0, requests=(),
                 m=32, n=32, k=7168, weight_count=1, quant=QuantSpec("bf16"),
                 category="dense_mlp", phase="decode")
    roles = {role for role in dict(die.rates) if role != "link"}
    names = {role: role for role in roles}
    names["cube"] = "cube"
    rates = {role: float(dict(die.rates)[role]) / 1000 for role in roles}
    rates["cube"] = 1.0
    units = {role: die.units(role) for role in roles}
    units["cube"] = die.aic_count
    work = {"cube": 1000 * op.flops / die.peak("bf16"),
            "hbm": float(op.hbm_bytes), "mte2_aic": float(op.mte2_bytes),
            "mte3": float(op.mte3_bytes)}
    pools = {pool: int(size) for pool, size in dict(die.buffers).items()}
    context = RecipeContext(
        op, work, names, rates, die.aic_count,
        pool_names={pool: pool for pool in pools}, pool_capacities=pools,
        pool_units={pool: die.pool_units(pool) for pool in pools},
        resource_units=units, compute_peaks=dict(die.compute_peaks),
        service_modes=dict(die.service_modes),
        work_units=dict(die.work_units))
    return die, op, context


def _leaf_stages(program):
    if isinstance(program, KernelProgram):
        for stage in program.stages:
            if stage.program is not None:
                yield from _leaf_stages(stage.program)
            else:
                yield stage
    elif isinstance(program, ParallelProgram):
        for child in program.programs:
            yield from _leaf_stages(child)


def test_splitk_executes_each_flop_once_and_charges_partial_traffic():
    die, op, context = _splitk_context()
    record = {}
    selected, _ = SteadyMatmul().select(context)
    geometry = matrix_geometry(context, selected)
    program = splitk_program(context, 8, recipe=selected, geometry=geometry,
                             record=record)
    issued = {}
    for stage in _leaf_stages(program):
        for key, value in stage.work.items():
            issued[key] = issued.get(key, 0.0) + value
    issued_flops = issued["cube"] * die.peak("bf16") / 1000.0
    assert issued_flops == pytest.approx(op.flops, rel=1e-12)
    traffic = record["split_k_traffic"]
    assert traffic["partial_write_bytes"] == pytest.approx(4 * 8 * 32 * 32)
    assert traffic["partial_read_bytes"] == pytest.approx(4 * 8 * 32 * 32)
    assert traffic["merge_vector_ops"] == pytest.approx((8 - 1) * 32 * 32)
    assert traffic["merge_latency_ms"] == 0.0
    assert record["split_k_tasks"] == 4 * 8
