"""T11 typed transform/fusion replacement and implementation-selection slice.

These are synthetic logical fixtures: they establish value-level contracts,
occurrence replacement, capture independence and implementation separation.
They do not claim device kernel accuracy.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.blocks import norm
from mindie_sim.prediction.graph import GraphPolicy
from mindie_sim.prediction.implementation_selection import (LoweringRegion,
                                                            lower_regions, resolve)
from mindie_sim.prediction.implementations import (add_input_ports,
                                                    audit_boundary, input_ports,
                                                    lower_operators,
                                                    preserve_graph,
                                                    produced_values,
                                                    producer_input_values)
from mindie_sim.prediction.operators.hyper_connections.native_fusion import lower_native_fusions
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators.hyper_connections.native_fusion_traffic import apply_traffic_profile
from mindie_sim.prediction.operators.vector import cuda_quant
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import (activation_quantization,
                                              norm_quant, residual_norm)
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_physical import node_physical
from _support.operator_cases import hardware_case


def _parsed(quant="mxfp8"):
    raw = hardware_case()
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 2e6
    case = parse_case(raw)
    return replace(case, model=replace(case.model, quant=quant_preset(quant)))


def _graph(case, *, rows=2, width=8, recipe="dense_gate_up"):
    residual = vector_op(
        op_id="residual", deps=("input", "saved"), stage=0, layer=0,
        requests=("r",), elements=rows * width, ops_per_element=1, dtype="bf16",
        family="elementwise", phase="prefill", recipe="residual_add",
        read_bytes_override=2 * rows * width * 2,
        write_bytes_override=rows * width * 2)
    normalized = norm(
        name="norm", deps=("residual",), stage=0, layer=0, requests=("r",),
        tokens=rows, width=width, model=case.model, mode="prefill")
    matrix = gemm_op(
        op_id="projection", deps=("norm",), stage=0, layer=0, requests=("r",),
        m=rows, n=width, k=width, quant=case.model.quant,
        category="dense_mlp", phase="prefill", recipe=recipe)
    external = replace(residual, op_id="external_norm_user", deps=("norm",))
    consumer = replace(residual, op_id="consumer", deps=("projection", "residual"))
    return residual, normalized, matrix, external, consumer


def _external_runtime(case, *, passes):
    return replace(case.runtime, lowering_passes=tuple(passes), kernels={
        "fusion_domains": {
            "norm_quant": {
                "consumers": [{"recipe": "dense_gate_up"}],
                "external_outputs": "materialize",
            },
        },
    })


def test_activation_materialization_keeps_input_payload_and_scale_separate():
    case = _parsed()
    residual, normalized, matrix, _, _ = _graph(case)
    ops = activation_quantization((residual, normalized, matrix), "bf16")
    quantized = next(op for op in ops if op.family == "quantize")
    audit = audit_boundary(quantized)
    assert audit["kind"] == "activation_materialization"
    outputs = {value["output"]: value for value in audit["typed_outputs"]}
    assert outputs["payload"]["dtype"] == "fp8_e4m3"
    assert outputs["activation_scales"]["dtype"] == "fp8_e8m0"
    assert outputs["activation_scales"]["shape"] == [2, 1, 2]
    assert audit["typed_inputs"][0]["dtype"] == "bf16"
    assert {item["field"]: item["action"]
            for item in audit["field_dispositions"]}["activation.payload"] == "rewrite"
    consumer = next(op for op in ops if op.op_id == "projection")
    assert any(port["op"] == quantized.op_id and port["output"] == "payload"
               for port in input_ports(consumer))
    assert any(port["op"] == quantized.op_id and port["output"] == "activation_scales"
               for port in input_ports(consumer))


def test_generic_replacements_remove_stale_named_ports():
    case = _parsed()
    residual, normalized, projection, external, consumer = _graph(case)
    external = add_input_ports(external, (
        {"op": normalized.op_id, "output": "output", "source_op": normalized.op_id},))
    consumer = add_input_ports(consumer, (
        {"op": residual.op_id, "output": "output", "source_op": residual.op_id},))
    activated = activation_quantization((residual, normalized, projection,
                                         external, consumer), "bf16")
    norm_fused = norm_quant(activated, "bf16",
                            consumers=({"recipe": "dense_gate_up"},),
                            external_output="materialize")
    fused_norm = next(op for op in norm_fused if op.family == "rmsnorm_quant")
    external_out = next(op for op in norm_fused if op.op_id == "external_norm_user")
    assert [(port["op"], port["output"], port["source_op"])
            for port in input_ports(external_out)] == [
        (fused_norm.op_id, "norm", normalized.op_id)]
    residual_fused = residual_norm(norm_fused, "bf16")
    fused_residual = next(op for op in residual_fused
                          if op.family == "add_rmsnorm_quant")
    consumer_out = next(op for op in residual_fused if op.op_id == "consumer")
    assert [(port["op"], port["output"], port["source_op"])
            for port in input_ports(consumer_out)] == [
        (fused_residual.op_id, "residual", residual.op_id)]


def test_source_binding_restores_semantic_scale_and_attr_fields():
    case = _parsed("w8a8_int8")
    residual, normalized, matrix, _, _ = _graph(case)
    semantic = next(op for op in activation_quantization(
        (residual, normalized, matrix), "bf16") if op.family == "quantize")
    source = cuda_quant.make_op({"rows": 2, "width": 8, "dtype": "bf16"})
    preserved = preserve_graph(source, semantic)
    metadata = dict(preserved.metadata)
    # The selected source does not repeat these semantic fields; they must not
    # disappear when the source implementation replaces the semantic record.
    assert metadata["activation_scale_format"] == "fp32"
    assert metadata["activation_scale_shape"] == (2,)
    assert metadata["input_format"] == "bf16"
    assert metadata["output_format"] == "int8"
    assert metadata["source_binding"]["semantic_output_format"] == "int8"
    assert metadata["source_binding"]["implementation_recipe"] == cuda_quant.RECIPE
    assert metadata["dispatch"]["input_dtype"] == "bf16"
    assert metadata["dispatch"]["output_dtype"] == "int8"
    assert preserved.source_boundary == "kernel"
    assert preserved.returns == ("activation_scales",)


def test_norm_quant_exposes_rstd_only_when_consumer_slot_names_it():
    case = _parsed()
    residual, normalized, projection, _, _ = _graph(case, rows=3)
    normalized = replace(normalized, returns=("rstd",))
    consumer = vector_op(
        op_id="c", deps=(normalized.op_id,), stage=0, layer=0, requests=(),
        elements=3, ops_per_element=1, dtype="fp32", family="elementwise",
        phase="prefill", recipe="rstd_reader", shape=(3,))
    consumer = add_input_ports(consumer, (
        {"op": normalized.op_id, "output": "rstd",
         "source_op": normalized.op_id},))
    activated = activation_quantization(
        (residual, normalized, projection, consumer), "bf16")
    fused_ops = norm_quant(activated, "bf16",
                           consumers=({"recipe": "dense_gate_up"},),
                           external_output="materialize")
    fused = next(op for op in fused_ops if op.family == "rmsnorm_quant")
    values = produced_values(fused)
    assert {value["output"] for value in values} >= {"payload", "rstd"}
    assert "norm" not in {value["output"] for value in values}
    target = next(op for op in fused_ops if op.op_id == "c")
    assert [(port["op"], port["output"]) for port in input_ports(target)] == [
        (fused.op_id, "rstd")]
    operand = producer_input_values(target, {fused.op_id: values})[0]
    assert operand["output"] == "rstd"
    assert operand["read_bytes"] == 12.0


def test_norm_quant_materializes_external_norm_output_with_typed_ports():
    case = _parsed()
    ops = _graph(case)
    activated = activation_quantization(ops, "bf16")
    norm_op = next(op for op in activated if op.family == "rmsnorm")
    quant_op = next(op for op in activated if op.family == "quantize")
    fused_ops = norm_quant(
        activated, "bf16", consumers=({"recipe": "dense_gate_up"},),
        external_output="materialize")
    fused = next(op for op in fused_ops if op.family == "rmsnorm_quant")
    assert len(fused_ops) == len(activated) - 1
    assert norm_op.op_id not in {op.op_id for op in fused_ops}
    # The payload/scales are rewritten; the norm value is still materialized
    # for its external consumer.  Only the internal quant read disappears.
    assert fused.hbm_bytes == norm_op.hbm_bytes + quant_op.hbm_bytes - quant_op.mte2_bytes
    assert fused.mte3_bytes == norm_op.mte3_bytes + quant_op.mte3_bytes
    assert fused.dtype == "bf16"  # legacy input/compute declaration
    audit = audit_boundary(fused)
    assert audit["replaced_occurrences"] == (norm_op.op_id, quant_op.op_id)
    assert audit["external_outputs"] == ("norm",)
    assert set(audit["allocations"]["materialized"]) == {
        "payload", "activation_scales", "norm"}
    outputs = {value["output"]: value for value in audit["typed_outputs"]}
    assert outputs["payload"]["dtype"] == "fp8_e4m3"
    assert outputs["activation_scales"]["dtype"] == "fp8_e8m0"
    assert outputs["norm"]["dtype"] == "bf16"
    assert outputs["norm"]["materialized"] is True
    assert fused.dtype != outputs["payload"]["dtype"]
    external = next(op for op in fused_ops if op.op_id == "external_norm_user")
    assert any((port["op"], port["output"], port["source_op"]) ==
               (fused.op_id, "norm", norm_op.op_id)
               for port in input_ports(external))
    matrix = next(op for op in fused_ops if op.op_id == "projection")
    assert any(port["op"] == fused.op_id and port["output"] == "payload"
               for port in input_ports(matrix))
    assert any(port["op"] == fused.op_id and port["output"] == "activation_scales"
               for port in input_ports(matrix))
    assert dict(fused.metadata)["norm_output_materialized"] is True
    actions = {item["field"]: item["action"] for item in audit["field_dispositions"]}
    assert actions["normalization.output"] == "rewrite"
    assert actions["normalization.weight"] == "preserve"
    assert actions["quantization.group_size"] == "consume"


def test_norm_quant_does_not_cross_capture_or_stream_boundaries():
    case = _parsed()
    activated = activation_quantization(_graph(case), "bf16")
    norm_op = next(op for op in activated if op.family == "rmsnorm")
    captured_norm = replace(norm_op, metadata=norm_op.metadata + (
        ("graph_captured", True), ("graph_segment", 4)))
    ops = tuple(captured_norm if op.op_id == norm_op.op_id else op
                for op in activated)
    assert not any(op.family == "rmsnorm_quant"
                   for op in norm_quant(ops, "bf16",
                                        consumers=({"recipe": "dense_gate_up"},),
                                        external_output="materialize"))
    both = tuple(replace(op, metadata=op.metadata + (
        ("graph_captured", True), ("graph_segment", 4))) for op in ops)
    fused = [op for op in norm_quant(both, "bf16",
                                     consumers=({"recipe": "dense_gate_up"},),
                                     external_output="materialize")
             if op.family == "rmsnorm_quant"]
    assert len(fused) == 1
    assert audit_boundary(fused[0])["capture"]["graph_segment"] == 4
    other_stream = tuple(replace(op, stream="aux") if op.op_id == norm_op.op_id else op
                         for op in both)
    assert not any(op.family == "rmsnorm_quant"
                   for op in norm_quant(other_stream, "bf16",
                                        consumers=({"recipe": "dense_gate_up"},),
                                        external_output="materialize"))


def test_norm_quant_default_keeps_external_norm_producer_unfused():
    case = _parsed()
    activated = activation_quantization(_graph(case), "bf16")
    result = norm_quant(activated, "bf16", consumers=({"recipe": "dense_gate_up"},))
    assert not any(op.family == "rmsnorm_quant" for op in result)
    assert any(op.family == "rmsnorm" for op in result)
    assert any(op.family == "quantize" for op in result)


def test_residual_norm_replaces_nested_transform_and_preserves_both_outputs():
    case = _parsed()
    ops = _graph(case)
    runtime = _external_runtime(case, passes=(
        "activation_quantization", "norm_quant", "residual_norm"))
    from mindie_sim.prediction.transforms import lower_implementations
    result = lower_implementations(ops, runtime, "bf16")
    fused = next(op for op in result if op.family == "add_rmsnorm_quant")
    activated = activation_quantization(ops, "bf16")
    norm_quantized = norm_quant(
        activated, "bf16", consumers=({"recipe": "dense_gate_up"},),
        external_output="materialize")
    intermediate = next(op for op in norm_quantized if op.family == "rmsnorm_quant")
    residual = next(op for op in activated if op.recipe == "residual_add"
                    and op.op_id == "residual")
    assert fused.hbm_bytes == residual.mte2_bytes + intermediate.hbm_bytes
    assert fused.mte2_bytes == (residual.mte2_bytes + intermediate.mte2_bytes
                                 - residual.mte3_bytes)
    assert fused.mte3_bytes == residual.mte3_bytes + intermediate.mte3_bytes
    assert fused.vector_ops == residual.vector_ops + intermediate.vector_ops
    assert fused.returns == ("activation_scales", "norm", "residual")
    audit = audit_boundary(fused)
    assert audit["kind"] == "residual_norm"
    assert set(audit["replaced_occurrences"]) == {
        "residual", "norm", "projection.quantize"}
    assert set(audit["external_outputs"]) == {"residual", "norm"}
    outputs = {value["output"]: value for value in audit["typed_outputs"]}
    assert outputs["payload"]["dtype"] == "fp8_e4m3"
    assert outputs["activation_scales"]["dtype"] == "fp8_e8m0"
    assert outputs["norm"]["dtype"] == "bf16"
    assert outputs["residual"]["dtype"] == "bf16"
    external = next(op for op in result if op.op_id == "external_norm_user")
    consumer = next(op for op in result if op.op_id == "consumer")
    matrix = next(op for op in result if op.op_id == "projection")
    assert any(port["op"] == fused.op_id and port["output"] == "norm"
               for port in input_ports(external))
    assert any(port["op"] == fused.op_id and port["output"] == "residual"
               for port in input_ports(consumer))
    assert any(port["op"] == fused.op_id and port["output"] == "payload"
               for port in input_ports(matrix))
    assert any(port["op"] == fused.op_id and port["output"] == "activation_scales"
               for port in input_ports(matrix))


def test_typed_value_separates_three_valid_rows_from_four_storage_rows():
    value = vector_op(
        op_id="bucket", deps=("input",), stage=0, layer=0, requests=(),
        elements=32, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="add", shape=(4, 8),
        metadata=(("output_format", "bf16"),
                  ("graph_active_query_rows", 3),
                  ("graph_physical_query_rows", 4)))
    output = produced_values(value)[0]
    assert output["logical_shape"] == [3, 8]
    assert output["storage_shape"] == [4, 8]
    assert output["bytes"] == 3 * 8 * 2
    assert output["storage_bytes"] == 4 * 8 * 2
    assert output["layout"]["storage_shape"] == (4, 8)


def test_norm_repeated_fp32_reads_and_gamma_are_added_explicitly():
    case = _parsed("mxfp8")
    producer = vector_op(
        op_id="fp32_producer", deps=("input",), stage=0, layer=0,
        requests=("r",), elements=24, ops_per_element=1, dtype="fp32",
        family="cast", phase="prefill", recipe="cast", shape=(3, 8),
        metadata=(("output_format", "fp32"),))
    normalized = norm(
        name="norm", deps=(producer.op_id,), stage=0, layer=0,
        requests=("r",), tokens=3, width=8, model=case.model,
        mode="prefill")
    normalized = replace(
        normalized, mte2_bytes=112.0,
        metadata=normalized.metadata + (("input_repeats", {"output": 2}),))
    projection = gemm_op(
        op_id="projection", deps=(normalized.op_id,), stage=0, layer=0,
        requests=("r",), m=3, n=8, k=8, quant=quant_preset("mxfp8"),
        category="dense_mlp", phase="prefill", recipe="dense_gate_up")
    activated = activation_quantization(
        (producer, normalized, projection), "bf16")
    fused = next(op for op in norm_quant(
        activated, "bf16", consumers=({"recipe": "dense_gate_up"},))
        if op.family == "rmsnorm_quant")
    assert fused.mte2_bytes == 208
    inputs = audit_boundary(fused)["inputs"]
    assert inputs[0]["dtype"] == "fp32"
    assert inputs[0]["read_bytes"] == 192


def test_activation_quantization_reads_the_real_operand_representation():
    case = _parsed()
    producer = vector_op(
        op_id="producer", deps=("input",), stage=0, layer=0, requests=("r",),
        elements=24, ops_per_element=1, dtype="fp32", family="cast",
        phase="prefill", recipe="cast", shape=(3, 8),
        metadata=(("output_format", "fp32"),))
    matrix = gemm_op(
        op_id="projection", deps=(producer.op_id,), stage=0, layer=0,
        requests=("r",), m=3, n=8, k=8, quant=quant_preset("mxfp8"),
        category="dense_mlp", phase="prefill")
    ops = activation_quantization((producer, matrix), "bf16")
    quantized = next(op for op in ops if op.family == "quantize")
    # 3x8 FP32 source, not the region's BF16 declaration.
    assert quantized.mte2_bytes == 3 * 8 * 4
    assert quantized.dtype == "fp32"
    audit = audit_boundary(quantized)
    operand = audit["typed_inputs"][0]
    assert operand["producer"] == "producer"
    assert operand["dtype"] == "fp32"
    assert operand["logical_shape"] == [3, 8]
    assert operand["bytes"] == 96
    consumer = next(op for op in ops if op.op_id == "projection")
    assert any(port["op"] == quantized.op_id and port["output"] == "payload"
               for port in input_ports(consumer))
    assert any(port["op"] == quantized.op_id and port["output"] == "activation_scales"
               for port in input_ports(consumer))


def test_conversion_owns_single_read_and_matrix_repeats_follow_payload():
    case = _parsed("mxfp8")
    producer = vector_op(
        op_id="fp32_src", deps=("input",), stage=0, layer=0,
        requests=("r",), elements=10, ops_per_element=1, dtype="fp32",
        family="cast", phase="prefill", recipe="cast", shape=(2, 5),
        metadata=(("output_format", "fp32"),))
    matrix = gemm_op(
        op_id="matrix", deps=(producer.op_id,), stage=0, layer=0,
        requests=("r",), m=2, n=3, k=5, quant=quant_preset("w8a8_int8"),
        category="dense_mlp", phase="prefill", recipe="dense_gate_up",
        metadata=(("input_repeats", {"output": 2}),))
    out = activation_quantization((producer, matrix), "bf16")
    quantized = next(op for op in out if op.family == "quantize")
    rewritten = next(op for op in out if op.op_id == "matrix")
    assert quantized.mte2_bytes == 40
    assert input_ports(quantized)[0]["op"] == producer.op_id
    assert producer.op_id not in {port["op"] for port in input_ports(rewritten)}
    assert {port["output"] for port in input_ports(rewritten)} == {
        "payload", "activation_scales"}
    values = {quantized.op_id: produced_values(quantized)}
    operands = [value for value in producer_input_values(rewritten, values)
                if value["output"] in {"payload", "activation_scales"}]
    assert len(operands) == 2
    assert [value["output"] for value in operands] == [
        "payload", "activation_scales"]
    assert operands[0]["read_bytes"] == 20
    assert operands[1]["read_bytes"] == 8
    # The inserted conversion re-parses its original residual, not the
    # matrix's repeated payload access.
    assert producer_input_values(quantized, {
        producer.op_id: produced_values(producer)})[0]["read_bytes"] == 40


def test_quantizer_reads_named_residual_and_matrix_uses_new_ports():
    case = _parsed()
    residual, normalized, projection, external, consumer = _graph(case, rows=3)
    projection = replace(projection, deps=(residual.op_id,))
    projection = add_input_ports(projection, (
        {"op": residual.op_id, "output": "output", "source_op": residual.op_id},))
    out = activation_quantization(
        (residual, normalized, projection, external, consumer), "bf16")
    quantized = next(op for op in out if op.op_id == "projection.quantize")
    rewritten = next(op for op in out if op.op_id == "projection")
    assert input_ports(quantized)[0]["op"] == residual.op_id
    assert producer_input_values(
        quantized, {residual.op_id: produced_values(residual)})[0]["read_bytes"] == 48
    assert residual.op_id not in {port["op"] for port in input_ports(rewritten)}
    assert {port["output"] for port in input_ports(rewritten)} == {
        "payload", "activation_scales"}


def test_generic_norm_residual_rewire_keeps_unrelated_consumer_port():
    case = _parsed()
    residual, normalized, projection, external, consumer = _graph(case)
    consumer = replace(consumer, deps=(normalized.op_id, residual.op_id))
    consumer = add_input_ports(consumer, (
        {"op": normalized.op_id, "output": "norm", "source_op": normalized.op_id},
        {"op": residual.op_id, "output": "output", "source_op": residual.op_id},
    ))
    activated = activation_quantization(
        (residual, normalized, projection, external, consumer), "bf16")
    normed = norm_quant(activated, "bf16",
                        consumers=({"recipe": "dense_gate_up"},),
                        external_output="materialize")
    fused_norm = next(op for op in normed if op.family == "rmsnorm_quant")
    target = next(op for op in normed if op.op_id == "consumer")
    assert {port["op"] for port in input_ports(target)} == {
        fused_norm.op_id, residual.op_id}
    residualized = residual_norm(normed, "bf16")
    fused_residual = next(op for op in residualized
                          if op.family == "add_rmsnorm_quant")
    target = next(op for op in residualized if op.op_id == "consumer")
    assert {port["op"] for port in input_ports(target)} == {
        fused_norm.op_id, fused_residual.op_id}


def test_activation_reads_named_residual_port_not_first_payload():
    producer = vector_op(
        op_id="fused_norm", deps=("input",), stage=0, layer=0, requests=("r",),
        elements=24, ops_per_element=1, dtype="bf16",
        family="add_rmsnorm_quant", phase="prefill",
        recipe="add_rmsnorm_quantize", shape=(3, 8),
        write_bytes_override=24 + 24, returns=("activation_scales", "residual"),
        metadata=(("output_format", "int8"), ("primary_output", "payload"),
                  ("input_format", "bf16"), ("activation_scale_format", "fp32"),
                  ("activation_scale_shape", (3,)), ("activation_scale_bytes", 12)))
    matrix = gemm_op(
        op_id="projection", deps=(producer.op_id,), stage=0, layer=0,
        requests=("r",), m=3, n=8, k=8, quant=quant_preset("mxfp8"),
        category="dense_mlp", phase="prefill")
    matrix = add_input_ports(matrix, (
        {"op": producer.op_id, "output": "residual", "source_op": producer.op_id},))
    ops = activation_quantization((producer, matrix), "bf16")
    quantized = next(op for op in ops if op.family == "quantize")
    assert quantized.mte2_bytes == 3 * 8 * 2
    operand = audit_boundary(quantized)["typed_inputs"][0]
    assert operand["output"] == "residual"
    assert operand["dtype"] == "bf16"
    assert operand["logical_shape"] == [3, 8]
    assert operand["read_bytes"] == 48


def _value_fusion_group():
    """quantize(q) -> matmul(p) -> consumer(c), with c also reading q's scale."""
    key = "s0.l2.value_group"
    declaration = {
        "key": key, "role": "value_demo",
        "tensor_view": {"tokens": 2, "hidden": 5,
                        "input_dtype": "int8", "output_dtype": "bf16",
                        "scale_dtype": "fp32", "state_dtype": "fp32",
                        "index_dtype": "int64", "accumulator_dtype": "int32"},
        "implementation": {"native_kind": "ValueDemo"},
        "source": {"path": "value_demo.cc", "revision": "test"},
    }
    flags = (("graph_captured", True), ("graph_segment", 2))
    q = vector_op(
        op_id="q", deps=("input",), stage=0, layer=2, requests=(),
        elements=10, ops_per_element=3, dtype="bf16", family="quantize",
        phase="decode", recipe="dynamic_quantize", shape=(2, 5),
        read_bytes_override=32.0, write_bytes_override=20.0,
        returns=("activation_scales",),
        metadata=(("output_format", "int8"), ("primary_output", "payload"),
                  ("activation_scale_format", "fp32"),
                  ("activation_scale_shape", (2, 1)),
                  ("activation_scale_bytes", 8.0),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    p = gemm_op(
        op_id="p", deps=(q.op_id,), stage=0, layer=2, requests=(),
        m=2, n=3, k=5, quant=quant_preset("w8a8_int8"),
        category="dense_mlp", phase="decode", recipe="demo_gemm",
        metadata=(("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    c = vector_op(
        op_id="c", deps=(p.op_id, q.op_id), stage=0, layer=2, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="add", shape=(2, 3),
        metadata=(("activation_scale_producer", q.op_id),))
    c = add_input_ports(c, (
        {"op": p.op_id, "output": "output", "source_op": p.op_id},
        {"op": q.op_id, "output": "activation_scales", "source_op": q.op_id},
    ))
    return (q, p, c), declaration


def test_two_bare_primary_values_folded_into_one_fusion_keep_two_ports():
    case = _parsed()
    key = "ab.group"
    declaration = {
        "key": key, "role": "ab", "tensor_view": {"dtype": "bf16"},
        "implementation": {"native_kind": "AB"}, "source": {"path": "ab.cc"}}
    flags = (("graph_captured", True), ("graph_segment", 1))
    a = vector_op(
        op_id="a", deps=("input",), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="a", shape=(2, 3),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    b = vector_op(
        op_id="b", deps=(a.op_id,), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="b", shape=(2, 3),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    plain = vector_op(
        op_id="plain", deps=(a.op_id, b.op_id), stage=0, layer=0,
        requests=(), elements=6, ops_per_element=1, dtype="bf16",
        family="elementwise", phase="decode", recipe="add", shape=(2, 3))
    physical = lower_native_fusions(
        (a, b, plain), _fusion_runtime(case, declaration))
    fused = next(op for op in physical if op.family == "native_fusion")
    consumer = next(op for op in physical if op.op_id == "plain")
    assert consumer.deps == (fused.op_id,)
    ports = {(port["op"], port["output"]) for port in input_ports(consumer)}
    assert len([port for port in ports if port[0] == fused.op_id]) == 2
    assert {port[1] for port in ports if port[0] == fused.op_id} == {
        "output__0", "output__1"}
    operands = producer_input_values(
        consumer, {fused.op_id: produced_values(fused)})
    assert len(operands) == 2
    assert {operand["output"] for operand in operands} == {"output__0", "output__1"}
    assert sum(operand["read_bytes"] for operand in operands) == 24


def test_two_single_edge_readers_distinguish_two_primary_outputs():
    case = _parsed()
    key = "uv.group"
    declaration = {
        "key": key, "role": "uv", "tensor_view": {"dtype": "bf16"},
        "implementation": {"native_kind": "UV"}, "source": {"path": "uv.cc"}}
    flags = (("graph_captured", True), ("graph_segment", 1))
    a = vector_op(
        op_id="ua", deps=("input",), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="ua", shape=(2, 3),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    b = vector_op(
        op_id="ub", deps=(a.op_id,), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="ub", shape=(2, 3),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    u = vector_op(
        op_id="u", deps=(a.op_id,), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="u", shape=(2, 3),
        metadata=(("input_repeats", {"output": 2}),))
    v = vector_op(
        op_id="v", deps=(b.op_id,), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="v", shape=(2, 3))
    physical = lower_native_fusions(
        (a, b, u, v), _fusion_runtime(case, declaration))
    fused = next(op for op in physical if op.family == "native_fusion")
    u_out = next(op for op in physical if op.op_id == "u")
    v_out = next(op for op in physical if op.op_id == "v")
    assert u_out.deps == v_out.deps == (fused.op_id,)
    u_ports = input_ports(u_out)
    v_ports = input_ports(v_out)
    assert len(u_ports) == len(v_ports) == 1
    assert u_ports[0]["output"] != v_ports[0]["output"]
    assert {u_ports[0]["output"], v_ports[0]["output"]} == {
        "output__0", "output__1"}
    values = {fused.op_id: produced_values(fused)}
    u_value = producer_input_values(u_out, values)[0]
    v_value = producer_input_values(v_out, values)[0]
    assert u_value["read_bytes"] == 24
    assert v_value["read_bytes"] == 12


def test_native_fusion_keeps_exact_external_value_ports_and_no_internal_scale():
    case = _parsed()
    members, declaration = _value_fusion_group()
    physical = lower_native_fusions(members, _fusion_runtime(case, declaration))
    fused = next(op for op in physical if op.family == "native_fusion")
    audit = audit_boundary(fused)
    external = audit["typed"]["external_outputs"]
    assert {(value["member"], value["output"]) for value in external} == {
        ("p", "output"), ("q", "activation_scales")}
    assert fused.returns == ("activation_scales",)
    assert all(value["output"] != "payload" for value in external)
    assert not any(value["member"] == "q" and value["output"] == "payload"
                   for value in external)
    consumer = next(op for op in physical if op.op_id == "c")
    ports = input_ports(consumer)
    assert {(port["op"], port["output"]) for port in ports} == {
        (fused.op_id, "activation_scales"), (fused.op_id, "output")}


def test_alias_public_output_keeps_zero_write_and_storage():
    case = _parsed()
    key = "alias.group"
    declaration = {
        "key": key, "role": "alias", "tensor_view": {"dtype": "bf16"},
        "implementation": {"native_kind": "Alias"}, "source": {"path": "alias.cc"}}
    op = vector_op(
        op_id="alias.out", deps=("input",), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="alias", shape=(2, 3),
        metadata=(("output_format", "bf16"), ("alias_of", "base.view"),
                  ("materialized", False), ("write_bytes", 0.0),
                  ("storage", "base_pool"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)),
                  ("graph_captured", True), ("graph_segment", 1)))
    fused = lower_native_fusions((op,), _fusion_runtime(case, declaration))[0]
    value = produced_values(fused)[0]
    assert value["materialized"] is False
    assert value["alias_of"] == "base.view"
    assert value["write_bytes"] == 0.0
    assert value["storage"] == "base_pool"
    audit = audit_boundary(fused)
    assert audit["allocations"]["external_materialized"] == ()
    assert audit["allocations"]["external_aliases"] == (
        {"port": "output", "alias_of": "base.view"},)


def test_gemm_external_value_is_mn_not_mnk():
    case = _parsed()
    key = "s0.l3.gemm_group"
    declaration = {
        "key": key, "role": "gemm_demo",
        "tensor_view": {"tokens": 2, "hidden": 3,
                        "input_dtype": "int8", "output_dtype": "bf16",
                        "scale_dtype": "fp32"},
        "implementation": {"native_kind": "GemmDemo"},
        "source": {"path": "gemm_demo.cc", "revision": "test"},
    }
    projection = gemm_op(
        op_id="gemm", deps=("input",), stage=0, layer=3, requests=(),
        m=2, n=3, k=5, quant=quant_preset("w8a8_int8"),
        category="dense_mlp", phase="decode", recipe="demo_gemm",
        metadata=(("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)),
                  ("graph_captured", True), ("graph_segment", 1)))
    consumer = vector_op(
        op_id="consumer", deps=("gemm",), stage=0, layer=3, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="add", shape=(2, 3))
    physical = lower_native_fusions((projection, consumer),
                                    _fusion_runtime(case, declaration))
    fused = next(op for op in physical if op.family == "native_fusion")
    value = audit_boundary(fused)["typed"]["external_outputs"][0]
    assert value["member"] == "gemm"
    assert value["output"] == "output"
    assert value["logical_shape"] == [2, 3]
    assert value["dtype"] == "bf16"
    assert value["bytes"] == 2 * 3 * 2
    assert value["quant"] == ""
    assert value["layout"]["storage_shape"] == (2, 3)
    resolved = produced_values(fused)[0]
    assert resolved["producer"] == fused.op_id
    assert resolved["output"] == "output"
    assert resolved["dtype"] == "bf16"
    assert resolved["logical_shape"] == [2, 3]
    assert resolved["bytes"] == 2 * 3 * 2
    assert resolved["semantic_producer"] == "gemm"


def test_terminal_two_output_call_keeps_public_returns_without_consumer():
    case = _parsed()
    members, declaration = _fusion_group(0)
    q = members[0]
    physical = lower_native_fusions((q,), _fusion_runtime(case, declaration))
    fused = next(op for op in physical if op.family == "native_fusion")
    assert fused.returns == ("activation_scales",)
    values = produced_values(fused)
    assert {(value["output"], value["dtype"]) for value in values} == {
        ("payload", "int8"), ("activation_scales", "fp32")}
    audit = audit_boundary(fused)
    assert {(value["member"], value["output"])
            for value in audit["typed"]["external_outputs"]} == {
        ("s0.l0.demo.quantize", "payload"),
        ("s0.l0.demo.quantize", "activation_scales")}


def test_materialize_semantic_layout_produces_declared_tokens_topk_hidden():
    op = vector_op(
        op_id="unpack.materialize", deps=("pack", "inverse", "weights"),
        stage=0, layer=0, requests=(), elements=4 * 2 * 32,
        ops_per_element=1, dtype="bf16", family="routing",
        phase="prefill", recipe="route_unpack.materialize", shape=(4, 2, 32),
        write_bytes_override=512,
        metadata=(("output_layout", "tokens_topk_hidden"),
                  ("output_format", "bf16")))
    value = produced_values(op)[0]
    assert value["logical_shape"] == [4, 2, 32]
    assert value["storage_shape"] == [4, 2, 32]
    assert value["layout"]["shape"] == (4, 2, 32)
    assert value["layout_name"] == "tokens_topk_hidden"
    assert value["bytes"] == 512
    assert value["write_bytes"] == 512


def test_attention_output_uses_declared_q_view_not_kv_horizon():
    def attention(context):
        return vector_op(
            op_id="attention", deps=("q",), stage=0, layer=0, requests=(),
            elements=1 * context * 4 * 256, ops_per_element=1, dtype="bf16",
            family="attention", phase="decode", recipe="attention",
            shape=(1, context, 4, 256),
            metadata=(("attention_tensor_view", {
                "kind": "paged_gqa", "output_shape": (1, 4, 256),
                "query_shape": (1, 4, 256), "logical_layout": "TND",
                "output_dtype": "bf16"}),))
    first = produced_values(attention(17))[0]
    second = produced_values(attention(33))[0]
    assert first["logical_shape"] == [1, 4, 256]
    assert first["bytes"] == 2048
    assert first["bytes"] == second["bytes"]
    assert first["layout_name"] == "TND"


def test_tensor_copy_public_identity_and_missing_layout_gap():
    concat = vector_op(
        op_id="concat", deps=("a", "b"), stage=0, layer=0, requests=(),
        elements=2048, ops_per_element=1, dtype="bf16", family="tensor_copy",
        phase="prefill", recipe="concatenate", shape=(2048,),
        write_bytes_override=4096,
        metadata=(("source_outputs", ("a", 0)), ("source_layouts", ()))
    )
    value = produced_values(concat)[0]
    assert value["logical_shape"] == [2048]
    assert value["bytes"] == 4096
    assert value["layout"]["offset"] == 0
    assert value["value_id"] == "concat.output"
    missing = replace(concat, op_id="copy.unknown",
                      metadata=(("source_outputs", ("a",)),))
    assert produced_values(missing)[0]["layout_gap"] == "missing_source_layout"


def test_typed_value_preserves_existing_layout_stride_offset():
    op = vector_op(
        op_id="view", deps=("input",), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="view", shape=(2, 3))
    boundary = {"kind": "source_value_contract",
                "typed_outputs": ({
                    "producer": "view", "dtype": "bf16",
                    "layout": {"shape": (2, 3), "strides": (5, 1),
                               "offset": 1, "storage_shape": (2, 5)},
                    "logical_shape": [2, 3], "storage_shape": [2, 5],
                    "output": "output", "role": "activation",
                    "materialized": True,
                },)}
    record = {"op_id": "view", "kind": "source_value_contract",
              "layer": 0, "stage": 0, "block": None, "role": None,
              "boundary": boundary}
    op = replace(op, metadata=op.metadata + (("fused_origin_scopes", (record,)),))
    value = produced_values(op)[0]
    assert value["layout"]["shape"] == (2, 3)
    assert value["layout"]["strides"] == (5, 1)
    assert value["layout"]["offset"] == 1
    assert value["layout"]["storage_shape"] == (2, 5)


def test_replacement_connected_to_replacement_rewires_external_operand_port():
    case = _parsed()
    decl_a = {"key": "a.group", "role": "a", "tensor_view": {"dtype": "bf16"},
              "implementation": {"native_kind": "A"},
              "source": {"path": "a.cc"}}
    decl_b = {"key": "b.group", "role": "b", "tensor_view": {"dtype": "bf16"},
              "implementation": {"native_kind": "B"},
              "source": {"path": "b.cc"}}
    flags = (("graph_captured", True), ("graph_segment", 1))
    q = vector_op(
        op_id="a.q", deps=("input",), stage=0, layer=0, requests=(),
        elements=8, ops_per_element=1, dtype="bf16", family="quantize",
        phase="decode", recipe="dynamic_quantize", shape=(2, 4),
        returns=("activation_scales",),
        metadata=(("output_format", "int8"),
                  ("activation_scale_format", "fp32"),
                  ("activation_scale_shape", (2,)), ("activation_scale_bytes", 8.0),
                  ("structure_subgraphs", (decl_a,)),
                  ("structure_subgraph_memberships", ("a.group",)), *flags))
    p = vector_op(
        op_id="a.p", deps=(q.op_id,), stage=0, layer=0, requests=(),
        elements=8, ops_per_element=1, dtype="bf16", family="gemm",
        phase="decode", recipe="a_p", shape=(2, 4),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (decl_a,)),
                  ("structure_subgraph_memberships", ("a.group",)), *flags))
    b1 = vector_op(
        op_id="b.one", deps=(p.op_id,), stage=0, layer=0, requests=(),
        elements=8, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="b_one", shape=(2, 4),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (decl_b,)),
                  ("structure_subgraph_memberships", ("b.group",)),
                  ("structure_subgraph_entries", ("b.group",)), *flags))
    b1 = add_input_ports(b1, (
        {"op": p.op_id, "output": "output", "source_op": p.op_id},))
    b2 = vector_op(
        op_id="b.two", deps=(b1.op_id,), stage=0, layer=0, requests=(),
        elements=8, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="b_two", shape=(2, 4),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (decl_b,)),
                  ("structure_subgraph_memberships", ("b.group",)), *flags))
    c = vector_op(
        op_id="c", deps=(b2.op_id,), stage=0, layer=0, requests=(),
        elements=8, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="c", shape=(2, 4))
    runtime = replace(case.runtime, kernels={"lowerings": {
        "native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["a", "b"],
                        "traffic_profile": "preserve_workspaces"},
        },
    }})
    out = lower_native_fusions((q, p, b1, b2, c), runtime)
    fused = [op for op in out if op.family == "native_fusion"]
    assert len(fused) == 2
    fused_a = next(op for op in fused if op.op_id == "a.group.native")
    fused_b = next(op for op in fused if op.op_id == "b.group.native")
    assert fused_b.deps == (fused_a.op_id,)
    ports = input_ports(fused_b)
    assert len(ports) == 1
    assert (ports[0]["op"], ports[0]["output"], ports[0]["source_op"]) == (
        fused_a.op_id, "output", "a.p")


def test_explicit_total_issued_is_not_multiplied_after_port_rename():
    case = _parsed()
    key = "explicit.group"
    declaration = {
        "key": key, "role": "explicit", "tensor_view": {"dtype": "bf16"},
        "implementation": {"native_kind": "Explicit"}, "source": {"path": "e.cc"}}
    flags = (("graph_captured", True), ("graph_segment", 1))
    a = vector_op(
        op_id="ea", deps=("input",), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="ea", shape=(2, 3),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    b = vector_op(
        op_id="eb", deps=(a.op_id,), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="eb", shape=(2, 3),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    u = vector_op(
        op_id="eu", deps=(a.op_id,), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="eu", shape=(2, 3),
        metadata=(("operand_read_bytes", {"output": 32}),))
    physical = lower_native_fusions((a, b, u), _fusion_runtime(case, declaration))
    fused = next(op for op in physical if op.family == "native_fusion")
    target = next(op for op in physical if op.op_id == "eu")
    value = producer_input_values(target, {fused.op_id: produced_values(fused)})[0]
    assert value["output"] == "output__0"
    assert value["read_bytes"] == 32


def test_explicit_public_root_survives_internal_consumer():
    case = _parsed()
    key = "root.group"
    declaration = {
        "key": key, "role": "root", "tensor_view": {"dtype": "bf16"},
        "implementation": {"native_kind": "Root"}, "source": {"path": "root.cc"}}
    flags = (("graph_captured", True), ("graph_segment", 1))
    a = vector_op(
        op_id="root.a", deps=("input",), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="root_a", shape=(2, 3),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    b = vector_op(
        op_id="root.b", deps=(a.op_id,), stage=0, layer=0, requests=(),
        elements=6, ops_per_element=1, dtype="bf16", family="elementwise",
        phase="decode", recipe="root_b", shape=(2, 3),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    fused = lower_native_fusions(
        (a, b), _fusion_runtime(case, declaration),
        public_outputs=((a.op_id, "output"),))[0]
    assert "output__0" in fused.returns
    assert {value["output"] for value in produced_values(fused)} >= {"output__0"}
    ports = {value.get("port") for value in
             audit_boundary(fused)["typed"]["external_outputs"]}
    assert "output__0" in ports


def test_native_fusion_replaces_occurrences_and_keeps_typed_roles_separate():
    case = _parsed()
    members, declaration = _fusion_group(0)
    physical = lower_native_fusions(members, _fusion_runtime(case, declaration))
    fused = next(op for op in physical if op.family == "native_fusion")
    audit = audit_boundary(fused)
    assert audit["occurrences"]["replaced"] == (
        members[0].op_id, members[1].op_id)
    assert audit["occurrences"]["replacement"] == fused.op_id
    assert audit["allocations"]["external_output_ids"] == (members[1].op_id,)
    assert {value["semantic_op"] for value in audit["typed"]["external_outputs"]} == {
        members[1].op_id}
    roles = audit["typed"]["roles"]
    assert set(roles) >= {"input", "output", "scale", "state", "index", "accumulator"}
    assert {item["dtype"] for item in roles["state"]} == {"fp32"}
    assert {item["dtype"] for item in roles["index"]} == {"int64"}
    assert {item["dtype"] for item in roles["accumulator"]} == {"int32"}
    assert audit["capture"]["graph_captured"] is True
    assert audit["capture"]["graph_segment"] == 3
    assert dict(fused.metadata)["fusion_boundary"]["capture"] == audit["capture"]
    assert audit["overlap"]["declared_concurrency"] == "not_implied_by_fusion"
    assert len(dict(fused.metadata)["fusion_phases"]) == 2
    assert len(dict(fused.metadata)["fusion_semantic_ids"]) == 2
    assert fused.flops == sum(op.flops for op in members[:2])
    assert fused.hbm_bytes == sum(op.hbm_bytes for op in members[:2])
    assert physical[-1].deps == (fused.op_id,)
    assert fused.returns == ()


def _fusion_group(layer, *, captured=True):
    key = f"s0.l{layer}.demo"
    declaration = {
        "key": key, "role": "demo",
        "tensor_view": {
            "tokens": 2, "hidden": 8,
            "input_dtype": "bf16", "output_dtype": "int8",
            "scale_dtype": "fp32", "state_dtype": "fp32",
            "index_dtype": "int64", "accumulator_dtype": "int32",
        },
        "implementation": {"native_kind": "Demo"},
        "source": {"path": "demo.cc", "revision": "test"},
    }
    flags = (("graph_captured", captured), ("graph_segment", 3 if captured else None))
    quantized = vector_op(
        op_id=f"{key}.quantize", deps=("input",), stage=0, layer=layer,
        requests=(), elements=16, ops_per_element=3, dtype="bf16",
        family="quantize", phase="decode", recipe="dynamic_quantize",
        shape=(2, 8), returns=("activation_scales",),
        metadata=(
            ("output_format", "int8"), ("activation_scale_format", "fp32"),
            ("activation_scale_shape", (2, 1)), ("activation_scale_bytes", 8.0),
            ("structure_subgraphs", (declaration,)),
            ("structure_subgraph_memberships", (key,)), *flags))
    projection = vector_op(
        op_id=f"{key}.projection", deps=(quantized.op_id,), stage=0, layer=layer,
        requests=(), elements=16, ops_per_element=2, dtype="int8",
        family="gemm", phase="decode", recipe="demo_gemm", shape=(2, 8, 8),
        metadata=(("output_format", "bf16"),
                  ("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,)), *flags))
    consumer = vector_op(
        op_id=f"{key}.consumer", deps=(projection.op_id,), stage=0,
        layer=layer, requests=(), elements=16, ops_per_element=1, dtype="bf16",
        family="elementwise", phase="decode", recipe="add", shape=(16,))
    return (quantized, projection, consumer), declaration


def _fusion_runtime(case, declaration):
    return replace(case.runtime, kernels={"lowerings": {
        "native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": [declaration["role"]],
                        "traffic_profile": "preserve_workspaces"},
        },
    }})


def test_native_selection_carries_actual_program_and_recipe():
    case = _parsed()
    members, declaration = _fusion_group(0)
    fused = lower_native_fusions(members, _fusion_runtime(case, declaration))[0]
    selection = dict(fused.metadata)["fusion_boundary"]["selection"]
    assert selection["plan"]["kind"] == "declared_software_plan"
    assert selection["actual_recipe"] == fused.recipe
    assert selection["actual_phase_count"] >= 2
    assert set(selection["actual_external_outputs"]) <= {
        value["output"] for value in produced_values(fused)}


def test_fusion_audit_does_not_change_physical_signature_and_capture_is_separate():
    case = _parsed()
    first, declaration0 = _fusion_group(0, captured=True)
    second, declaration1 = _fusion_group(1, captured=True)
    runtime = _fusion_runtime(case, declaration0)
    fused0 = lower_native_fusions(first, runtime)[0]
    fused1 = lower_native_fusions(second, _fusion_runtime(case, declaration1))[0]
    assert node_physical(fused0)["physical_signature"] == (
        node_physical(fused1)["physical_signature"])
    # Capture changes neither the fused kernel count nor its physical work.
    uncaptured, declaration2 = _fusion_group(2, captured=False)
    fused2 = lower_native_fusions(uncaptured, _fusion_runtime(case, declaration2))[0]
    assert fused2.hbm_bytes == fused0.hbm_bytes
    assert fused2.flops == fused0.flops
    assert audit_boundary(fused2)["capture"]["graph_captured"] is False
    assert len(dict(fused0.metadata)["fusion_phases"]) == len(
        dict(fused2.metadata)["fusion_phases"]) == 2


def test_traffic_profile_preserves_external_value_and_reports_value_effects():
    activation = vector_op(
        op_id="act", deps=("input",), stage=0, layer=0, requests=(),
        elements=16, ops_per_element=2, dtype="bf16", family="situ",
        phase="decode", recipe="situ_activation", shape=(2, 8),
        read_bytes_override=32.0, write_bytes_override=32.0)
    quant = vector_op(
        op_id="quant", deps=("act",), stage=0, layer=0, requests=(),
        elements=16, ops_per_element=1, dtype="bf16", family="quantize",
        phase="decode", recipe="dynamic_quantize", shape=(2, 8),
        read_bytes_override=32.0, write_bytes_override=16.0,
        returns=("activation_scales",))
    declaration = {
        "key": "situ.demo", "role": "situ_quant",
        "tensor_view": {"input_dtype": "bf16", "output_dtype": "fp8_e4m3",
                        "scale_dtype": "fp8_e8m0", "axis": -1},
    }
    profile = "situ_mx_quant_arch35_2a9bd1b0"
    adjusted, ledger = apply_traffic_profile(
        (activation, quant), declaration, profile,
        external_outputs=(activation.op_id,))
    assert adjusted[0].mte3_bytes == activation.mte3_bytes
    assert adjusted[1].mte2_bytes == 0
    assert ledger["eliminated_reads"] == activation.mte3_bytes
    assert ledger["eliminated_writes"] == 0
    assert ledger["external_output_writes_preserved_phase_positions"] == [0]
    assert ledger["value_effects"][0]["action"] == "preserve"
    assert ledger["value_effects"][0]["materialized"] is True
    assert ledger["value_effects"][1]["action"] == "consume"
    assert ledger["representation_guard"]["input_dtype"] == "bf16"
    assert ledger["capture"]["implied"] is False
    assert ledger["overlap"]["implied"] is False
    # Without the external consumer the same source profile may drop the
    # producer write; the distinction is explicit rather than an export zero.
    adjusted2, ledger2 = apply_traffic_profile(
        (activation, quant), declaration, profile)
    assert adjusted2[0].mte3_bytes == 0
    assert ledger2["value_effects"][0]["action"] == "consume"


def test_model_lower_regions_uses_replay_scope_before_generic_fusion():
    case = _parsed()
    runtime = replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed",
        splitting_ops=("rmsnorm",), lowering_passes=(
            "activation_quantization", "norm_quant", "residual_norm"))
    ops = _graph(case)
    region = LoweringRegion(tuple(ops), runtime, "bf16", 2)
    lowered, _ = lower_regions((region,), graph=GraphPolicy(),
                               replay_scope="piecewise", runtime=runtime,
                               resource_fusion=False)
    assert not any(op.family in {"rmsnorm_quant", "add_rmsnorm_quant"}
                   for op in lowered)
    norm = next(op for op in lowered if op.family == "rmsnorm")
    quant = next(op for op in lowered if op.family == "quantize")
    assert dict(norm.metadata)["graph_captured"] is False
    assert dict(quant.metadata)["graph_captured"] is True
    assert dict(norm.metadata)["graph_segment"] != dict(quant.metadata)["graph_segment"]


def test_captured_norm_eager_quant_captured_matrix_keep_three_segments():
    case = _parsed()
    runtime = replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed",
        splitting_ops=("quantize",), lowering_passes=(
            "activation_quantization", "norm_quant"))
    residual, normalized, projection, external, consumer = _graph(case)
    region = LoweringRegion((residual, normalized, projection, consumer),
                            runtime, "bf16", 2)
    lowered, _ = lower_regions((region,), graph=GraphPolicy(),
                               replay_scope="piecewise", runtime=runtime,
                               resource_fusion=False)
    norm = next(op for op in lowered if op.family == "rmsnorm")
    quant = next(op for op in lowered if op.family == "quantize")
    matrix = next(op for op in lowered if op.op_id == "projection")
    assert [dict(op.metadata)["graph_captured"] for op in (norm, quant, matrix)] == [
        True, False, True]
    assert len({dict(op.metadata)["graph_segment"] for op in (norm, quant, matrix)}) == 3


def test_lower_regions_partitions_prefix_and_two_regions_once(monkeypatch):
    case = _parsed()
    key = "prefix.native"
    declaration = {
        "key": key, "role": "prefix_native",
        "tensor_view": {"tokens": 2, "hidden": 5},
        "implementation": {"native_kind": "PrefixNative"},
        "source": {"path": "prefix.cc"}}
    prefix = vector_op(
        op_id="prefix.p", deps=("input",), stage=0, layer=0,
        requests=("r",), elements=4, ops_per_element=1, dtype="bf16",
        family="elementwise", phase="prefill", recipe="prefix_p",
        shape=(2, 2))
    projection = gemm_op(
        op_id="pf.projection", deps=("input",), stage=0, layer=0,
        requests=("r",), m=2, n=3, k=5, quant=quant_preset("w8a8_int8"),
        category="dense_mlp", phase="prefill", recipe="pf_gemm",
        metadata=(("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,))))
    region1 = LoweringRegion((prefix, projection), case.runtime, "bf16", 1)
    consumer = vector_op(
        op_id="pf.consumer", deps=(projection.op_id,), stage=0, layer=0,
        requests=("r",), elements=6, ops_per_element=1, dtype="bf16",
        family="elementwise", phase="prefill", recipe="pf_add", shape=(2, 3))
    region2 = LoweringRegion((consumer,), case.runtime, "bf16", 1)
    runtime = replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed",
        lowering_passes=("activation_quantization",),
        kernels={"lowerings": {"native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["prefix_native"],
                        "traffic_profile": "preserve_workspaces"},
        }}})
    original = GraphPolicy.apply
    calls = []
    def wrapped(self, ops, scope, selected_runtime):
        calls.append(tuple(ops))
        return original(self, ops, scope, selected_runtime)
    monkeypatch.setattr(GraphPolicy, "apply", wrapped)
    lowered, _ = lower_regions(
        (region1, region2), graph=GraphPolicy(), replay_scope="piecewise",
        runtime=runtime, prefix=(), resource_fusion=False)
    assert len(calls) == 1
    assert any(op.family == "quantize" for op in lowered)
    assert any(op.op_id == "pf.projection" for op in lowered)


def test_native_admission_sees_inserted_quantizer_capture_state():
    case = _parsed()
    key = "capture.native"
    declaration = {
        "key": key, "role": "capture_demo",
        "tensor_view": {"tokens": 2, "hidden": 5},
        "implementation": {"native_kind": "CaptureDemo"},
        "source": {"path": "capture.cc"}}
    projection = gemm_op(
        op_id="capture.projection", deps=("input",), stage=0, layer=0,
        requests=("r",), m=2, n=3, k=5, quant=quant_preset("w8a8_int8"),
        category="dense_mlp", phase="prefill", recipe="capture_gemm",
        metadata=(("structure_subgraphs", (declaration,)),
                  ("structure_subgraph_memberships", (key,))))
    consumer = vector_op(
        op_id="capture.consumer", deps=(projection.op_id,), stage=0,
        layer=0, requests=("r",), elements=6, ops_per_element=1,
        dtype="bf16", family="elementwise", phase="prefill",
        recipe="capture_add", shape=(2, 3))
    runtime = replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed",
        lowering_passes=("activation_quantization",),
        kernels={"lowerings": {"native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["capture_demo"],
                        "traffic_profile": "preserve_workspaces"},
        }}})
    region = LoweringRegion((projection, consumer), runtime, "bf16", 1)
    lowered, _ = lower_regions((region,), graph=GraphPolicy(),
                               replay_scope="piecewise", runtime=runtime,
                               resource_fusion=True)
    fused = next(op for op in lowered if op.family == "native_fusion")
    assert dict(fused.metadata)["graph_captured"] is True
    assert not any(op.family in {"quantize", "gemm"} for op in lowered)
    assert produced_values(fused)[0]["logical_shape"] == [2, 3]
    assert produced_values(fused)[0]["bytes"] == 12
    assert len(dict(fused.metadata)["fusion_phases"]) >= 2


def test_same_eager_partition_fuses_without_becoming_captured():
    case = _parsed()
    runtime = replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed",
        splitting_ops=("rmsnorm", "quantize"), lowering_passes=(
            "activation_quantization", "norm_quant"))
    residual, normalized, projection, external, consumer = _graph(case)
    region = LoweringRegion((residual, normalized, projection, consumer),
                            runtime, "bf16", 2)
    lowered, _ = lower_regions((region,), graph=GraphPolicy(),
                               replay_scope="piecewise", runtime=runtime,
                               resource_fusion=False)
    fused = next(op for op in lowered if op.family == "rmsnorm_quant")
    assert dict(fused.metadata)["graph_captured"] is False
    assert not any(op.family in {"rmsnorm", "quantize"} for op in lowered)


def test_generic_fusion_honours_piecewise_split_before_capture_assignment():
    case = _parsed()
    runtime = replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed",
        splitting_ops=("rmsnorm",), lowering_passes=(
            "activation_quantization", "norm_quant", "residual_norm"))
    split = replace(case, runtime=runtime)
    ops = target_operators(split, phase="decode", tokens=1, context=0,
                           sequences=1, scope="body", apply_body_defaults=False,
                           resource_fusion=False)
    by_family = {}
    for op in ops:
        by_family.setdefault(op.family, []).append(op)
    assert "rmsnorm" in by_family
    assert "quantize" in by_family
    assert "rmsnorm_quant" not in by_family
    assert "add_rmsnorm_quant" not in by_family
    norm = by_family["rmsnorm"][0]
    quant = next(op for op in by_family["quantize"] if op.deps and op.deps[0] == norm.op_id)
    assert dict(norm.metadata).get("graph_captured") is False
    assert dict(quant.metadata).get("graph_captured") is True
    assert dict(norm.metadata)["graph_segment"] != dict(quant.metadata)["graph_segment"]
    # Removing the explicit split keeps the same-partition fusion legal.
    fused_case = replace(case, runtime=replace(runtime, splitting_ops=()))
    fused = target_operators(fused_case, phase="decode", tokens=1, context=0,
                             sequences=1, scope="body", apply_body_defaults=False,
                             resource_fusion=False)
    assert any(op.family == "rmsnorm_quant" for op in fused)
    assert not any(op.family == "rmsnorm" for op in fused)


def _repeated_transform(layer):
    case = _parsed()
    ops = tuple(replace(op, layer=layer) for op in _graph(case))
    activated = activation_quantization(ops, "bf16")
    normed = norm_quant(activated, "bf16",
                        consumers=({"recipe": "dense_gate_up"},),
                        external_output="materialize")
    return residual_norm(normed, "bf16")


def test_transform_audit_keeps_repeated_layers_physically_equal():
    first = _repeated_transform(0)
    second = _repeated_transform(1)
    left = next(op for op in first if op.family == "add_rmsnorm_quant")
    right = next(op for op in second if op.family == "add_rmsnorm_quant")
    assert node_physical(left)["physical_signature"] == node_physical(right)["physical_signature"]
    assert left.mte2_bytes == right.mte2_bytes
    assert left.mte3_bytes == right.mte3_bytes


def test_implementation_selection_reports_candidates_and_selected_plan_separately():
    case = _parsed()
    policy = {
        "cases": [
            {"domain": {"phases": ["decode"]}, "binding": "vllm_cuda_rms_norm"},
            {"domain": {"phases": ["prefill"]}, "binding": "analytical"},
        ],
        "otherwise": "error",
    }
    runtime = replace(case.runtime, kernels={"lowerings": {"rmsnorm": policy}})
    decode = SimpleNamespace(phase="decode", q_len=4, context=8)
    resolved, records = resolve(runtime, (decode,), padded_tokens=4,
                                replay_scope="none")
    record = records[0]
    assert [item["matches"] for item in record["candidates"]] == [True, False]
    assert record["selected"]["case"] == 0
    assert record["plan"]["kind"] == "declared_software_plan"
    assert record["plan"]["implementation"] == "vllm_cuda_rms_norm"
    assert resolved.kernels["lowerings"]["rmsnorm"] == "vllm_cuda_rms_norm"
    assert record["geometry"]["padded_tokens"] == 4
    prefill = SimpleNamespace(phase="prefill", q_len=4, context=0)
    resolved2, records2 = resolve(runtime, (prefill,), padded_tokens=4,
                                  replay_scope="none")
    record2 = records2[0]
    assert [item["matches"] for item in record2["candidates"]] == [False, True]
    assert record2["selected"]["case"] == 1
    assert record2["implementation"] == "analytical"
    assert record2["plan"]["kind"] == "analytical_semantic_work"
    assert "rmsnorm" not in resolved2.kernels["lowerings"]
    with pytest.raises(ModelingError, match="implementation_domain_unmatched"):
        resolve(runtime, (SimpleNamespace(phase="mixed", q_len=4, context=8),),
                padded_tokens=4, replay_scope="none")
