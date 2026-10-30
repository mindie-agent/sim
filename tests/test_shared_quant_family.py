"""Source-bound shared-expert W8A8 INT32 -> SwiGLU INT8 boundary.

The assertions recompute the logical source ledger and transfer sizes from the
same [rows, width] pair.  They do not use a captured duration or a model name.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction import family_shared_quant as shared
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.schema.common import ModelingError
from test_prediction_system import case


def _graph(rows=3, hidden=16, width=8, *, weight_format="int8",
           activation_family="swiglu", gate_id="opaque_gate",
           activation_id="opaque_activation", down_id="opaque_down"):
    quant = QuantSpec(
        "w8a8", weight_format=weight_format, activation_format="int8",
        scale_format="fp32")
    gate_meta = (
        ("ffn_category", "shared_expert"), ("ffn_role", "gate_up"),
    )
    gate = gemm_op(
        op_id=gate_id, deps=(), stage=0, layer=0, requests=(),
        m=rows, n=2 * width, k=hidden, quant=quant,
        category="shared_expert", phase="prefill",
        family="gate_up_projection", recipe="dense_gate_up",
        metadata=gate_meta)
    activation = vector_op(
        op_id=activation_id, deps=(gate.op_id,), stage=0, layer=0, requests=(),
        elements=rows * width, ops_per_element=4, dtype="bf16",
        family=activation_family, phase="prefill",
        recipe="swiglu", shape=(rows, width),
        read_bytes_override=2 * rows * width * 2,
        write_bytes_override=rows * width * 2,
        metadata=(("ffn_category", "shared_expert"),
                  ("ffn_role", "activation"),
                  ("activation_kind", "silu" if activation_family == "swiglu" else "situ")))
    down = gemm_op(
        op_id=down_id, deps=(activation.op_id,), stage=0, layer=0, requests=(),
        m=rows, n=hidden, k=width, quant=quant,
        category="shared_expert", phase="prefill",
        family="down_projection", recipe="dense_down",
        metadata=(("ffn_category", "shared_expert"), ("ffn_role", "down")))
    return (gate, activation, down)


def _lower(ops, binding=shared.BINDING):
    parsed = parse_case(case())
    runtime = replace(parsed.runtime, kernels={
        "lowerings": {"shared_expert": binding}})
    return lower_implementations(ops, runtime, "bf16")


def _by_role(ops, role):
    matches = [op for op in ops
               if dict(op.metadata).get("ffn_category") == "shared_expert"
               and dict(op.metadata).get("ffn_role") == role]
    assert len(matches) == 1
    return matches[0]


def _expected_vector_ops(rows, width, clamp=0.0):
    count = rows * width
    # Cast, channel scale and token scale on the 2*width gate/up columns.
    dequant = 3 * rows * 2 * width
    # Gate bias add, SwiGLU expression, Abs, Div and three casts.
    swiglu = 6 * count
    quant = 1 * count + rows * (width - 1) + rows + 8 * rows + count + 3 * count
    clamp_work = 3 * count if clamp > 0 else 0
    return dequant + swiglu + quant + clamp_work


def test_producer_removes_epilogue_and_consumer_keeps_real_scale_producer():
    rows, hidden, width = 3, 16, 8
    lowered = _lower(_graph(rows, hidden, width))
    gate = _by_role(lowered, "gate_up")
    consumer = _by_role(lowered, "activation")
    down = _by_role(lowered, "down")
    gate_meta = dict(gate.metadata)

    assert gate_meta["output_format"] == "int32"
    assert "matrix_epilogue" not in gate_meta
    assert gate.vector_ops == 0
    assert gate.mte3_bytes == 4 * rows * 2 * width
    weight_scale = 2 * 2 * width
    assert gate.mte2_bytes == rows * hidden + 2 * width * hidden + weight_scale
    assert gate_meta["weight_scale_format"] == "bf16"
    assert gate_meta["dequant_weight_scale_format"] == "fp32"
    assert gate_meta["activation_scale_read_by_producer"] is False

    quantizers = [op for op in lowered if op.family == "quantize"]
    assert len(quantizers) == 1  # Only the original input quantize remains.
    source = quantizers[0]
    assert consumer.deps == (gate.op_id, source.op_id)
    assert down.deps == (consumer.op_id,)
    assert consumer.returns == ("activation_scales",)
    assert consumer.source_boundary == "kernel"
    assert consumer.recipe == shared.RECIPE
    assert dict(consumer.metadata)["activation_scale_shape"] == (rows,)
    assert consumer.mte2_bytes == 4 * rows * 2 * width + 4 * 2 * width + 4 * rows
    assert consumer.mte3_bytes == rows * width + 4 * rows
    assert consumer.vector_ops == _expected_vector_ops(rows, width)
    assert dict(consumer.metadata)["vector_ledger"] == shared.vector_ledger(
        rows, width)


def test_selection_uses_structural_roles_not_op_id_prefix_and_multiple_t():
    for rows in (1, 7):
        lowered = _lower(_graph(rows=rows, gate_id="shared.gate_up",
                                activation_id="shared.swiglu",
                                down_id="shared.down"))
        consumer = _by_role(lowered, "activation")
        assert consumer.vector_ops == _expected_vector_ops(rows, 8)
        assert dict(consumer.metadata)["dequant_swiglu_quant"]["rows"] == rows

    # A similarly named operator without the approved structural roles is
    # never selected by textual identity.
    unrelated = replace(
        _graph()[0], op_id="shared.gate_up",
        metadata=tuple((key, value) for key, value in _graph()[0].metadata
                       if key not in {"ffn_category", "ffn_role"}))
    unselected = _lower((unrelated,))
    gate = next(op for op in unselected if op.op_id == "shared.gate_up")
    assert dict(gate.metadata)["output_format"] == "bf16"
    assert "shared_gate_up_int32" not in dict(gate.metadata)


def test_approved_fixture_descriptor_arithmetic():
    rows, hidden, width = 4096, 6144, 256
    lowered = _lower(_graph(rows=rows, hidden=hidden, width=width))
    gate = _by_role(lowered, "gate_up")
    consumer = _by_role(lowered, "activation")
    # x[4096,6144] INT8, w[512,6144] INT8, scale[512] BF16 -> INT32 [4096,512].
    assert gate.mte2_bytes == rows * hidden + 2 * width * hidden + 2 * 2 * width
    assert gate.mte3_bytes == 4 * rows * 2 * width
    assert gate.vector_ops == 0
    assert consumer.mte2_bytes == 4 * rows * 2 * width + 4 * 2 * width + 4 * rows
    assert consumer.mte3_bytes == rows * width + 4 * rows
    assert consumer.vector_ops == _expected_vector_ops(rows, width)


def test_multiple_shared_layers_join_only_their_own_down_matrix():
    first = _graph(rows=2, hidden=16, width=8, gate_id="g0", activation_id="a0",
                   down_id="d0")
    second = _graph(rows=5, hidden=16, width=8, gate_id="g1", activation_id="a1",
                    down_id="d1")
    lowered = _lower(first + second)
    consumers = {op.op_id: op for op in lowered
                 if op.family == "dequant_swiglu_quant"}
    assert set(consumers) == {"a0", "a1"}
    by_id = {op.op_id: op for op in lowered}
    assert by_id["d0"].deps == ("a0",)
    assert by_id["d1"].deps == ("a1",)
    assert len([op for op in lowered if op.family == "quantize"]) == 2


def test_w4_is_unselected_and_situ_is_explicitly_unsupported():
    w4 = _lower(_graph(weight_format="int4"))
    gate = _by_role(w4, "gate_up")
    assert dict(gate.metadata)["output_format"] == "bf16"
    assert "shared_gate_up_int32" not in dict(gate.metadata)
    assert len([op for op in w4 if op.family == "quantize"]) == 2

    with pytest.raises(ModelingError, match="shared_quant_situ_unsupported"):
        _lower(_graph(activation_family="situ"))


def test_source_plan_and_dependencies_execute_from_operator_costs_and_graph():
    raw = case()
    raw["machine"]["device"]["compute_peaks"] = {"int8": 1e9, "bf16": 1e9}
    raw["machine"]["device"]["rates"] = {
        "memory": 1e9, "vector": 1e9, "mte2": 1e9, "mte3": 1e9}
    lowered = _lower(_graph(rows=4, hidden=16, width=8))
    consumer = _by_role(lowered, "activation")
    book = RecipeBook.from_mapping({"lowerings": {"shared_expert": shared.BINDING}})

    isolated = operator_costs(raw, (consumer,), recipes=book, input_ready_ms=5)
    row = isolated["operators"][0]
    assert row["operator"]["vector_ops"] == _expected_vector_ops(4, 8)
    assert row["operator"]["source_boundary"] == "kernel"
    assert row["operator"]["mte2_bytes"] == 4 * 4 * 16 + 4 * 16 + 4 * 4
    assert row["operator"]["mte3_bytes"] == 4 * 8 + 4 * 4
    assert row["live_tasks"] == 0
    assert row["duration_ms"] > 0

    report = graph_costs(raw, lowered, recipes=book, input_ready_ms=5)
    assert report["complete_ms"] >= 5
    assert all(event["start_ms"] >= 5 for event in report["events"]
               if event.get("demand"))
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0
    assert "ascend_shared_dequant_swiglu_quant" in {
        row["operator"]["recipe"] for row in report["isolated"]["operators"]}


def test_source_stream_scale_slots_and_repeated_layer_identity_are_preserved():
    from mindie_sim.structure_physical import node_physical
    from mindie_sim.structure_matrix_contracts import matrix_contract
    source = tuple(replace(op, stream="shared") for op in _graph())
    lowered = _lower(source)
    gate, consumer, down = (_by_role(lowered, role)
                            for role in ("gate_up", "activation", "down"))
    assert gate.stream == consumer.stream == down.stream == "shared"
    assert matrix_contract(gate.to_dict())[-2:] == ((), "")
    assert dict(down.metadata)["payload_producer"] == consumer.op_id
    another = _lower(tuple(replace(op, stream="shared") for op in _graph(
        gate_id="other_g", activation_id="other_a", down_id="other_d")))
    for role in ("gate_up", "activation", "down"):
        assert node_physical(_by_role(lowered, role),
                             recipe_bindings={"lowerings": {"shared_expert": shared.BINDING}}) == \
            node_physical(_by_role(another, role),
                          recipe_bindings={"lowerings": {"shared_expert": shared.BINDING}})


def test_ordinary_glm_frontend_runs_the_complete_shared_chain_with_four_launches():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "tests/fixtures/verification/"
                     "glm53-w4a8-shared-case.json").read_text())
    raw["runtime"]["kernels"]["lowerings"]["shared_expert"] = shared.BINDING
    parsed = parse_case(raw)
    ops = target_operators(parsed, phase="prefill", tokens=2, scope="body")
    gates = [op for op in ops if dict(op.metadata).get("shared_gate_up_int32")]
    assert len(gates) == 75
    gate = gates[0]
    activation = next(op for op in ops
                      if op.recipe == shared.RECIPE and gate.op_id in op.deps)
    down = next(op for op in ops if _by_category(op) == "down"
                and op.deps == (activation.op_id,))
    input_quantizer = next(op for op in ops if op.op_id == gate.deps[0])
    chain = (input_quantizer, gate, activation, down)
    ids = {op.op_id for op in chain}
    external = {dep: 7 for op in chain for dep in op.deps if dep not in ids}
    result = graph_costs(parsed, {0: chain}, external_inputs=external,
                         input_ready_ms=7, include_submission=True, trace_limit=2500)
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    submits = [event for event in result["events"] if event["task"].endswith(":submit")]
    assert len(submits) == 4
    assert all(event["start_ms"] >= 7 for event in result["events"] if event["demand"])
    assert {row["operator"]["recipe"] for row in result["isolated"]["operators"]} >= {
        shared.RECIPE, "dense_gate_up", "dense_down"}


def _by_category(op):
    category, role = shared._roles(op)
    return role if category == "shared_expert" else None


def test_consumer_contract_matches_real_seven_slot_capture(profiler_api):
    from mindie_profiler.body_fusion_contracts import observed_fusion_contract
    from mindie_sim.observations.kernels import kernel_rows
    from mindie_sim.structure_contracts import predicted_contract
    fixture = Path(__file__).parent / "fixtures/observations/shared-dequant-swiglu-kernel.csv"
    row = next(kernel_rows(fixture))
    consumer = _by_role(_lower(_graph(rows=4096, hidden=6144, width=256)),
                        "activation")
    contract = predicted_contract(consumer.to_dict())
    assert contract == observed_fusion_contract(row)
    assert row.input_shapes[3:] == ((), (), (), ())
    # The accumulator cannot be paired to the BF16 fused epilogue, and the
    # producer's BF16 channel-scale ABI is distinct from this FP32 consumer.
    for field, values in (
        ("input_dtypes", ("DT_BF16", *row.input_dtypes[1:])),
        ("input_dtypes", (row.input_dtypes[0], "DT_BF16", *row.input_dtypes[2:])),
        ("input_shapes", (*row.input_shapes[:3], (512,), *row.input_shapes[4:])),
    ):
        assert observed_fusion_contract(replace(row, **{field: values})) != contract
    wrong = consumer.to_dict()
    wrong["metadata"]["quant_scale"] = True
    assert predicted_contract(wrong) is None
