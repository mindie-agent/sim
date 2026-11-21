"""Causal convolution dispatch, source ownership, and history traffic.

[Role] Check governed convolution and public graph bindings for source updates.
[Invariants] Conserve active rows, history payloads, and owner-local service.
[Timeouts & Locks] Bounded planning and local Engine execution.
[Side-Effect Policy] Read-only fixture inputs and in-memory simulation.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.hardware import sku_case
from _support.linear_cases import (
    CONV, case, conv_graph as graph, execute_one, gdn_decode_source_case as source_case,
    kda_hardware_case, sku_planning_context,
)
from mindie_sim import operator_costs, target_graphs
from mindie_sim.catalog.weights import stage_weight_bytes, stage_weight_terms
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operators.linear_attention import build, register_family, resolve_operator_plan
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_structure
from mindie_sim.structure_contracts import short_conv_active_contract

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
COPY_RECIPE = "triton_noncontiguous_conv_state_copy"


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
    parsed, views, ops = source_case((4,) * (tokens // 4), keys=keys, heads=heads)
    copy = next(op for op in ops if op.recipe == "z_norm_materialize")
    meta = dict(copy.metadata)
    width, value = (2 * keys + heads) * 128, heads * 128
    assert meta["source_layout"]["strides"] == (width + value, 1)
    assert meta["source_layout"]["offset"] == width
    assert copy.mte2_bytes == copy.mte3_bytes == tokens * value * 2
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    assert copy.op_id in norm.deps
    engine, _ = execute_one(parsed, views, copy)
    assert engine.serviced["memory@r0:s0"] == pytest.approx(tokens * value * 4)


@pytest.mark.parametrize("lengths,heads", (((1,), 12), ((4,), 1)))
def test_singleton_compatible_z_reshape_is_a_view(lengths, heads):
    _, _, ops = source_case(lengths=lengths, keys=1, heads=heads)
    assert not any(op.recipe == "z_norm_materialize" for op in ops)
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    projection = next(op for op in ops if op.recipe == "linear_qkvz")
    assert projection.op_id in norm.deps


def test_source_z_reshape_uses_padded_projection_rows_without_padding_recurrence():
    parsed, views, _ = source_case(lengths=(1,), keys=4, heads=12)
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode", tokens=16)
    copy = next(op for op in ops if op.recipe == "z_norm_materialize")
    assert copy.hbm_bytes == 16 * 1536 * 4
    recurrence = next(op for op in ops if op.family == "recurrent_vector")
    assert dict(recurrence.metadata)["tokens"] == 1
    core_copy = next(op for op in ops if op.recipe == "core_output_copy")
    assert core_copy.hbm_bytes == 1536 * 4


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
                      history_shape=(1, state_len, 256), accepted_tokens=(accepted,))
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
