"""Causal convolution dispatch, source ownership, and history traffic.

[Role] Check governed convolution and public graph bindings for source updates.
[Invariants] Conserve active rows, history payloads, and owner-local service.
[Timeouts & Locks] Bounded planning and local Engine execution.
[Side-Effect Policy] Read-only fixture inputs and in-memory simulation.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.builders import dense_case, dense_layer, dense_model, linear_model
from _support.hardware import sku_case
from _support.linear_cases import (
    CONV, case, conv_graph as graph, execute_one, gdn_decode_source_case as source_case,
    kda_hardware_case, sku_planning_context,
)
from mindie_sim import compose_case, operator_costs, target_graphs
from mindie_sim.catalog.cache_ownership import draft_cache_components
from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
from mindie_sim.catalog.speculative_state import resolve_speculative_query_tokens
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
from mindie_sim.catalog.weights import stage_weight_bytes, stage_weight_terms
from mindie_sim.feature_mechanisms import _apply_recover_ssm
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.implementation_selection import resolve
from mindie_sim.prediction.operators.linear_attention import build, register_family, resolve_operator_plan
from mindie_sim.prediction.operators.recurrent.recurrent import linear
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure import export_structure
from mindie_sim.structure_contracts import short_conv_active_contract

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
COPY_RECIPE = "triton_noncontiguous_conv_state_copy"
DRAFT_PAGE_ROWS = 128


@pytest.mark.parametrize("batch", (1, 4))
def test_update_binds_active_source_rows_and_resident_weights(batch):
    raw, baseline = case(batch), case(batch, native=False)
    parsed, plain = parse_case(raw), parse_case(baseline)
    ops = graph(raw, batch)
    convs = [op for op in ops if op.recipe == CONV]
    layers = {layer.index for layer in parsed.model.layer_plans if layer.attention == "kda"}
    assert {op.layer for op in convs} == layers and len(convs) == len(layers)
    for op in convs:
        meta = dict(op.metadata)
        assert op.local_shape == (batch, 4608) and op.source_boundary == "kernel"
        assert meta["sequence_tokens"] == (1,) * batch and meta["state_write_bytes"] == batch * 3 * 4608 * 2
    row = operator_costs(raw, convs[:1], input_ready_ms=9)["operators"][0]
    assert row["operator"]["hbm_bytes"] == batch * 8 * 4608 * 2
    assert row["operator"]["mte3_bytes"] == batch * 4 * 4608 * 2
    assert not row["service_before_input"] and row["live_tasks"] == 0
    prepared = [term for group in stage_weight_terms(parsed.model, parsed.runtime.parallel)
                for term in group if "kda_prepared_conv" in term.name]
    assert len(prepared) == len(layers) and all(not term.source_owned for term in prepared)
    assert (sum(stage_weight_bytes(parsed.model, parsed.runtime.parallel))
            - sum(stage_weight_bytes(plain.model, plain.runtime.parallel))) == len(layers) * 4 * 4608 * 2


def physical_case():
    raw = kda_hardware_case(units=2, split_copy=True)
    raw["machine"]["device"].update({
        "rates": {"memory": 1e9, "vector": 1e9, "mte2": 1e9,
                  "mte2_aic": 1e9, "mte2_aiv": 4e6, "mte3": 8e6},
        "resource_units": {"vector": 4, "mte2_aic": 4, "mte2_aiv": 4, "mte3": 4},
        "buffers": {"ub": 4096}, "buffer_units": {"ub": 4}})
    raw["runtime"]["kernels"] = {"lowerings": {"conv_state_copy": {
        "implementation": COPY_RECIPE, "options": {"state_slots": 3}}}}
    return parse_case(raw)


def copy_op(direction, active):
    shape = (len(active), 3, 257)
    metadata = dict(direction=direction, active_requests=active, cache_index_dtype="int32",
                    query_start_dtype="int64", invocation_class="vector")
    return OpSpec("copy", "conv_state_copy", "decode", 0, 0, (), shape, shape, shape, shape,
                  dtype="bf16", recipe=COPY_RECIPE, source_boundary="kernel",
                  deps=("input",), metadata=tuple(metadata.items()))


@pytest.mark.parametrize("direction", ("gather", "scatter"))
@pytest.mark.parametrize("active", ((True, True), (False, True), (False, False)))
def test_copy_tail_and_inactive_masks_match_source_memory_direction(direction, active):
    parsed, op = physical_case(), copy_op(direction, active)
    row = operator_costs(parsed, (op,), input_ready_ms=7)["operators"][0]
    reads = (2 if direction == "scatter" else sum(active)) * 771 * 2
    writes = (sum(active) if direction == "scatter" else 2) * 771 * 2
    packed_indices = 8 if direction == "gather" else 0
    result = row["operator"]
    assert result["hbm_bytes"] == reads + writes + packed_indices + 8 * 20
    assert row["serviced"].get("mte2_aiv@r0:s0", 0) == pytest.approx(reads)
    assert result["mte3_bytes"] == writes + packed_indices
    assert not row["service_before_input"] and row["live_tasks"] == 0
    assert row["service_delta"] == pytest.approx(dict.fromkeys(row["service_delta"], 0), abs=1e-8)


@pytest.mark.parametrize("keys,heads,tokens", ((2, 6, 16), (4, 12, 128)))
def test_projection_materialization_copies_only_the_strided_gate_payload(keys, heads, tokens):
    """Live A3/A5 captures confirm GDN decode eliminates the 5 fake TensorMove/copy ops."""
    _, _, ops = source_case((4,) * (tokens // 4), keys=keys, heads=heads)
    removed = {
        "z_norm_materialize", "core_output_zero", "core_output_copy",
        "beta_contiguous", "alpha_contiguous", "convolution_input_contiguous",
    }
    assert not any(op.recipe in removed for op in ops)
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    projection = next(op for op in ops if op.recipe == "linear_qkvz")
    recurrence = next(op for op in ops if op.family == "recurrent_vector")
    assert projection.op_id in norm.deps and recurrence.op_id in norm.deps


@pytest.mark.parametrize("lengths,heads", (((1,), 12), ((4,), 1)))
def test_singleton_compatible_z_reshape_is_a_view(lengths, heads):
    _, _, ops = source_case(lengths=lengths, keys=1, heads=heads)
    assert not any(op.recipe == "z_norm_materialize" for op in ops)
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    projection = next(op for op in ops if op.recipe == "linear_qkvz")
    assert projection.op_id in norm.deps


def test_source_z_reshape_uses_padded_projection_rows_without_padding_recurrence():
    parsed, views, _ = source_case(lengths=(1,), keys=4, heads=12)
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode", tokens=16, machine=parsed.machine)
    assert not any(op.recipe in {"z_norm_materialize", "core_output_copy", "core_output_zero"} for op in ops)
    recurrence = next(op for op in ops if op.family == "recurrent_vector")
    assert dict(recurrence.metadata)["tokens"] == 1
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    assert recurrence.op_id in norm.deps


@pytest.mark.parametrize("model,tp,dim", (("qwen36", 8, 1280), ("qwen38", 4, 2560)))
def test_source_convolution_exports_active_boundaries_and_layer_shard(model, tp, dim):
    path = ROOT / f"examples/cases/high-concurrency-20261030/source-gdn/{model}-a5-tp{tp}-c4.json"
    raw = json.loads(path.read_text())
    nodes = [node for node in export_structure(raw, phase="decode", tokens=4, context=4104,
                                               sequences=4, scope="body")["nodes"]
             if node["family"] == "short_convolution"]
    expected = ["causal_conv1d_active", "CausalConv1d", [[16, dim], "bf16", "ND"],
                [[4, dim], "bf16", "ND"], [[], None, None],
                [[6, dim], "bf16", "ND"], [[16, dim], "bf16", "ND"]]
    assert len(nodes) == 48 and all(row["contract"] == expected for row in nodes)
    op = nodes[0]["operator"]
    changed = deepcopy(op)
    changed["metadata"]["state_history_slots"] = 7
    assert short_conv_active_contract(changed) != short_conv_active_contract(op)


@pytest.fixture(params=SKUS)
def planning(request):
    sku = request.param
    return sku, *sku_planning_context(sku, "conv")


def _governed_op(*, phase="decode", **metadata):
    values = {"lengths": (1, 4), "channels": 16, "width": 4, **metadata}
    return OpSpec("conv", "causal_conv1d", phase, 0, 0, (), (), (), (), (),
                  recipe="causal_conv1d", dtype="bf16", metadata=tuple(values.items()))


def _phase_work(plan):
    return {p.phase_id: {w.resource: w.amount for w in p.work} for p in plan.pipeline_spec.phases}


@pytest.mark.parametrize("phase,mode,lengths,width,bias,activation,write_state", (
    ("prefill", "varlen", (1, 65), 4, True, "silu", True),
    ("prefill", "varlen", (5,), 2, False, "none", True),
    ("prefill", "varlen", (2, 3, 9), 3, True, "none", False),
    ("decode", "incremental", (1, 4), 4, False, "silu", True),
))
def test_governed_convolution_conserves_work_and_waits_for_inputs(
        planning, phase, mode, lengths, width, bias, activation, write_state):
    sku, geometry, ctx = planning
    op = _governed_op(phase=phase, mode=mode, lengths=lengths, width=width, has_bias=bias,
                      activation=activation, write_state=write_state)
    plan = register_family().resolve(op, ctx).recipe
    assert plan.work == build(op, geometry).work
    channels, item = 16, 2
    elements = sum(lengths) * channels
    history = len(lengths) * (width - 1) * channels * item
    expected = {
        "input": ("hbm", elements * item), "history": ("hbm", history),
        "weights": ("hbm", channels * width * item), "convolve": ("vector", 2 * elements * width),
        "output": ("hbm", elements * item),
    }
    if bias:
        expected.update(bias=("hbm", channels * item), add_bias=("vector", elements))
    if activation == "silu":
        expected["silu"] = ("vector", 4 * elements)
    if write_state:
        expected["state_writeback"] = ("hbm", history)
    work = _phase_work(plan)
    assert work.keys() == expected.keys()
    assert {name: work[name][pipe] for name, (pipe, _) in expected.items()} == {
        name: amount for name, (_, amount) in expected.items()
    }
    parents = {p.phase_id: set(p.after) for p in plan.pipeline_spec.phases}
    assert parents["convolve"] == {"input", "history", "weights"}
    if write_state:
        assert parents["state_writeback"] == {"history", "input", "output"}
    report = verify_operator(sku_case(sku), op, build, input_ready_ms=7)
    assert report["status"] == "passed" and report["input_dependency_obeyed"]
    assert report["buffers_released"] and report["work_mismatches"] == {}


@pytest.mark.parametrize("state_len,accepted,written", ((6, 4, 3072), (3, 1, 1536)))
def test_speculative_history_uses_accepted_offset_and_physical_capacity(planning, state_len, accepted, written):
    _, _, ctx = planning
    op = _governed_op(mode="incremental", lengths=(4,), channels=256, q=4,
                      history_shape=(8, state_len, 256), cache_index=(7,), accepted_tokens=(accepted,))
    plan = resolve_operator_plan(op, ctx).recipe
    assert (plan.metadata["state_len"], plan.metadata["state_offsets"]) == (state_len, (accepted - 1,))
    assert plan.metadata["speculative_width"] == 4 and plan.metadata["history_rows"] == 3
    work = _phase_work(plan)
    assert work["history"]["hbm"] == 1536
    assert work["state_writeback"]["mte3"] == work["state_writeback"]["hbm"] == written == state_len * 256 * 2


@pytest.mark.parametrize("invalid,reason", (
    ({"weight_shape": (4, 16)}, "weight_or_history_shape_mismatch"),
    ({"history_shape": (4, 32)}, "weight_or_history_shape_mismatch"),
    ({"history_shape": (2, 16)}, "state_len_unsupported"),
    ({"history_shape": (1, 6, 16)}, "weight_or_history_shape_mismatch"),
    ({"history_shape": (8, 6, 16), "cache_index": (7, 8)}, "cache_index_out_of_bounds"),
    ({"history_shape": (8, 6, 16), "cache_index": (-1, 7)}, "cache_index_out_of_bounds"),
    ({"state_len": 6, "state_offset": (0, 4)}, "state_offset_out_of_bounds"),
    ({"accepted_tokens": (-1, 1)}, "invalid_accepted_tokens"),
    ({"lengths": (0, 2)}, "positive_integer_geometry"),
    ({"width": 0}, "positive_integer_geometry"),
    ({"mode": "batch"}, "mode_or_activation_unsupported"),
    ({"activation": "relu"}, "mode_or_activation_unsupported"),
))
def test_governed_convolution_rejects_unsupported_contracts(planning, invalid, reason):
    _, _, ctx = planning
    with pytest.raises(ModelingError, match=reason):
        resolve_operator_plan(_governed_op(**invalid), ctx)


def test_governed_update_selects_source_abi(planning):
    sku, _, ctx = planning
    op = _governed_op(abi="causal_conv1d_update", weight_shape=(16, 4), history_shape=(2, 3, 16),
                      token_block_count=2)
    if sku == REPRESENTATIVE_SKUS["a5"]:
        result = resolve_operator_plan(op, ctx)
        assert result.implementation == "causal_conv1d_update"
        assert result.recipe.metadata["mode"] == "incremental"
        assert _phase_work(result.recipe)["weights"]["hbm"] == 2 * 16 * 4 * 2
    else:
        with pytest.raises(ModelingError, match="requires Arch35"):
            resolve_operator_plan(op, ctx)


def test_strided_state_copies_bracket_the_original_convolution():
    path = ROOT / "examples/cases/high-concurrency-20261030/glm-flash-ws9/a5_glm53flash_tp8_c16_p4k.json"
    raw = json.loads(path.read_text())
    raw["runtime"]["kernels"].setdefault("lowerings", {})["conv_state_copy"] = {
        "implementation": COPY_RECIPE, "options": {"state_slots": 6}}
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4, context=4096, sequences=16)[0]
    copies = [op for op in ops if op.recipe == COPY_RECIPE]
    convs = {op.layer: op for op in ops if op.recipe == "short_convolution"}
    assert len(copies) == 2 * len(convs) == 68
    for op in copies:
        meta = dict(op.metadata)
        assert op.local_shape == (16, 6, 3072) and meta["state_strides"] == (1, 6)
        conv = convs[op.layer]
        assert (op.op_id in conv.deps if meta["direction"] == "gather" else op.deps == (conv.op_id,))



def test_capacity_query_width_and_windowed_mtp_draft_geometry():
    base = dense_case(layers=1)
    kda = replace(
        base, model=linear_model("kda", layers=1),
        runtime=replace(base.runtime,
                        dspark=DsparkSpec(6, (1.0,) * 6), metadata=()))
    memory = Memory(kda)
    # DSpark verifies K proposals plus the bonus token: one ``[K+1]`` width everywhere.
    assert memory.query_tokens == 7 == resolve_speculative_query_tokens(kda.runtime)
    assert memory.cache_bytes(128) == memory.cache_bytes(128, query_tokens=7) > memory.cache_bytes(128, query_tokens=6)
    truncated = replace(kda, runtime=replace(kda.runtime, dspark=DsparkSpec(
        6, (1.0,) * 6, confidence=(0.9, 0.9, 0.1, 0.9, 0.9, 0.9), confidence_threshold=0.5)))
    assert Memory(truncated).query_tokens == 3 == resolve_speculative_query_tokens(truncated.runtime)
    meta_only = replace(kda, runtime=replace(kda.runtime, dspark=None, metadata=(
        ("draft_mode", "dspark"), ("draft_proposal_tokens", 6))))
    assert resolve_speculative_query_tokens(meta_only.runtime) == 7

    mtp = DraftSpec(
        kind="mtp", layers=1, hidden_size=32, intermediate_size=64,
        num_heads=4, num_kv_heads=2, head_dim=16, q_width=64, kv_width=32,
        attention_kind="gqa", target_layer_ids=(0,))
    plans = (replace(dense_layer(0), window=64),)
    target = replace(dense_model(layers=1), layer_plans=plans)
    mtp = replace(mtp, layer_plans=native_draft_layer_plans(target, mtp))
    model = replace(target, auxiliary=AuxiliarySpec(mtp=mtp))
    drafted = replace(
        base, model=model, runtime=replace(
            base.runtime,
            metadata=(("draft_mode", "mtp"), ("draft_proposal_tokens", 2))))
    plain = replace(
        base, model=replace(model, auxiliary=None),
        runtime=replace(base.runtime, metadata=()))
    expected = draft_cache_components(
        model, drafted.runtime.parallel, draft_kind="mtp",
        sequence_tokens=512, capacity_padded=True, runtime=drafted.runtime,
        query_tokens=Memory(drafted).query_tokens)["draft_kv"]
    delta = (Memory(drafted).cache_bytes(512)[-1]
             - Memory(plain).cache_bytes(512)[-1])
    assert delta == pytest.approx(expected)
    assert delta == (2 * DRAFT_PAGE_ROWS) * (2 * 2 * 16) * 2


def _kda_norm_case():
    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=1, ep=1, num_devices=1,
        layers=4, prompt_tokens=64, output_tokens=2, max_num_seqs=2)
    raw["runtime"]["lowering_passes"] = ["kda_in_kernel_l2norm"]
    return parse_case(raw)


def test_multi_request_chunk_kda_folds_in_kernel_qk_l2norm():
    case = _kda_norm_case()
    ops = target_graphs(case, phase="prefill", tokens=64, context=0,
                              sequences=2, scope="body")[0]
    assert not [op for op in ops if op.recipe == "qk_l2norm"]
    assert [op for op in ops if dict(op.metadata).get("in_kernel_qk_l2norm")]


def test_mixed_prefill_decode_kda_eliminates_standalone_qk_l2norm():
    case = _kda_norm_case()
    plan = next(lp for lp in case.model.layer_plans if lp.attention == "kda")
    views = (RequestView("p", 64, 0, True, "prefill"),
             RequestView("d", 1, 128, True, "decode"))
    resolved_rt, _ = resolve(case.runtime, views, padded_tokens=65, replay_scope="none")
    res = linear(
        prefix="s0.l0", plan=plan, views=views, stage=0, deps="in",
        model=case.model, parallel=resolved_rt.parallel,
        quant=case.model.quant, mode="prefill",
        policy=AttentionPolicy(recurrent_chunk=64), runtime=resolved_rt,
        padded_tokens=65)
    lowered = lower_implementations(res.ops, resolved_rt, case.model.activation_dtype)
    assert not [op for op in lowered if op.recipe == "qk_l2norm"]
    folded = [op for op in lowered if dict(op.metadata).get("in_kernel_qk_l2norm")]
    assert len(folded) == 2


def test_recover_ssm_keeps_native_kda_and_accounts_replay_separately():
    config = json.loads((ROOT / "mindie_sim/model_configs/kimi-k3.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    text["model_type"] = "kimi_k31"
    text["dsa_config"] = {
        "dsa_layers": text["linear_attn_config"]["full_attn_layers"],
        "index_num_heads": 32, "index_head_dim": 128,
        "index_topk": 2048, "index_rope_dim": 0,
    }
    text.update(dspark_attention_kind="mla", dspark_block_size=5)
    c = parse_case(compose_case(
        config, hardware="a5", tp=8, dp=4, ep=32, dcp=8, num_devices=32,
        block_size=1536, max_model_len=140000, graph_capture_mode="none", recover_ssm=False,
        quant={"name": "k31_fp8_cache", "weight_format": "bf16", "activation_format": "bf16",
               "expert_weight_format": "bf16", "kv_dtype": "fp8_e4m3",
               "index_dtype": "fp8_e4m3", "index_scale_dtype": "fp16"},
        mtp_proposal_tokens=3, mtp_acceptance_length=2,
        prompt_tokens=131072, output_tokens=1024))
    rt = {"metadata": {"draft_proposal_tokens": 3}}
    _apply_recover_ssm(rt, SimpleNamespace(caller_explicit=set(), model=c.model))
    assert rt["recover_ssm"]
    recovered = replace(c, runtime=replace(c.runtime, recover_ssm=True))
    before, after = Memory(c), Memory(recovered)
    records = 69 * 12 * (128 * 4 + 2 * 128 * 2) * 4
    page = before.pages.groups[0].bytes_per_group_page
    assert before.cache_bytes(131584)[0] - after.cache_bytes(131584)[0] == 3 * 3 * page - records
    assert all("replay_records" not in names for names in after.pages.covered.values())
    ops = target_graphs(recovered, phase="decode", tokens=4, context=4096,
                        scope="body", resource_fusion=False)[0]
    native = [op for op in ops if op.recipe == "vllm_ascend_recurrent_kda_arch35"]
    assert len(native) == 69
    assert all(dict(op.metadata)["recover_ssm"] for op in native)


def test_causal_conv_and_state_copy_stamp_nonzero_compulsory_work():
    from _support.hardware import machine
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.operators.causal_conv import conv_state_copy, native_causal_conv_prefill, native_causal_conv_spec
    from mindie_sim.prediction.operators.hyper_connections.common import attnres_spec_terms

    # 1. Native prefill CausalConv1d stamps non-zero compulsory work <= build().work and 0 on all-skipped slots.
    geo = Geometry.from_devices(machine(REPRESENTATIVE_SKUS["a5"]).dies[:1])
    op_act = native_causal_conv_spec.make_op({"lengths": (32, 16), "dim": 128, "width": 4, "dtype": "bf16"})
    plan_act = native_causal_conv_prefill.build(op_act, geo)
    assert op_act.hbm_bytes > 0 and op_act.mte2_bytes > 0 and op_act.mte3_bytes > 0
    assert op_act.hbm_bytes <= plan_act.work.resources["hbm"] + 1e-6
    op_skip = native_causal_conv_spec.make_op({
        "lengths": (8,), "dim": 128, "width": 4, "dtype": "bf16",
        "cache_indices": (-1,), "pad_slot_id": -1,
    })
    assert op_skip.hbm_bytes == 0.0 and op_skip.mte2_bytes == 0.0 and op_skip.mte3_bytes == 0.0

    # 2. AttnRes prefix spec terms return non-zero work for active tokens and 0 for empty tokens.
    from mindie_sim.schema.operators import OpSpec
    def _attnres_op(tok: int) -> OpSpec:
        shape = (tok, 128)
        return OpSpec(
            "attn_res_fwd", "attn_res_prefix", "decode", 0, 0, ("r0",),
            shape, shape, shape, shape, dtype="bf16", recipe="attn_res_fwd",
            metadata=(("tokens", tok), ("hidden_size", 128), ("capacity", 4), ("valid_history", 3)),
        )
    terms = attnres_spec_terms(_attnres_op(8))
    assert all(terms[k] > 0 for k in ("hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops"))
    zero_terms = attnres_spec_terms(_attnres_op(0))
    assert all(zero_terms[k] == 0.0 for k in ("hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops"))

    # 3. Conv state copy gather/scatter and causal_conv_update stamp non-zero compulsory work.
    copy_raw = json.loads((ROOT / "examples/cases/high-concurrency-20261030/glm-flash-ws9/a5_glm53flash_tp8_c16_p4k.json").read_text())
    copy_raw["runtime"]["kernels"].setdefault("lowerings", {})["conv_state_copy"] = {
        "implementation": COPY_RECIPE, "options": {"state_slots": 6}}
    ops = target_graphs(copy_raw, phase="decode", tokens=4, context=4096, sequences=16, scope="body")[0]
    copies = [op for op in ops if op.family == "conv_state_copy"]
    assert copies and all(op.hbm_bytes > 0 and op.mte2_bytes > 0 and op.mte3_bytes > 0 for op in copies)
    raw = compose_case("kimi-k3", REPRESENTATIVE_SKUS["a5"], tp=2, ep=2, layers=2, prompt_tokens=256, output_tokens=16)
    raw["runtime"]["kernels"].setdefault("lowerings", {})["causal_conv_update"] = "ascend_causal_conv1d_update"
    u_ops = target_graphs(raw, phase="decode", tokens=1, context=256, sequences=2, scope="body")[0]
    convs = [op for op in u_ops if op.recipe == "ascend_causal_conv1d_update"]
    assert convs and all(op.hbm_bytes > 0 and op.mte2_bytes > 0 and op.mte3_bytes > 0 and op.vector_ops > 0 for op in convs)
    assert conv_state_copy.IMPLEMENTATION == "triton_noncontiguous_conv_state_copy"
