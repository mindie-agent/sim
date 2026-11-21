"""Typed values and causal edges through graph lowering.

[Role] Exercise conversion, fusion, replay partitioning and selection.
[Invariants] Preserve byte ledgers, public outputs and operand identity.
[Timeouts & Locks] Finite graph construction, without device or network access.
[Side-Effect Policy] Read-only fixtures and pytest-scoped patches.
"""
from dataclasses import replace
from functools import partial
from math import prod
from types import SimpleNamespace

import pytest

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.blocks import norm
from mindie_sim.prediction.graph import GraphPolicy
from mindie_sim.prediction.implementation_selection import LoweringRegion, lower_regions, resolve
from mindie_sim.prediction.implementations import (
    add_input_ports, audit_boundary, input_ports, preserve_graph, produced_values,
    producer_input_values,
)
from mindie_sim.prediction.operators.hyper_connections.native_fusion import lower_native_fusions
from mindie_sim.prediction.operators.hyper_connections.native_fusion_traffic import apply_traffic_profile
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators.vector import cuda_quant
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import (
    activation_quantization, lower_implementations, norm_quant, residual_norm,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_physical import node_physical
from _support.operator_cases import hardware_case

pytestmark = pytest.mark.tier0
CONSUMERS = ({"recipe": "dense_gate_up"},)
GENERIC_PASSES = ("activation_quantization", "norm_quant", "residual_norm")
QUANT_PORTS = {"payload", "activation_scales"}
activate = partial(activation_quantization, source_dtype="bf16")
def parsed(quant="mxfp8"):
    raw = hardware_case()
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 2e6
    case = parse_case(raw)
    return replace(case, model=replace(case.model, quant=quant_preset(quant)))

def vec(name, deps=("input",), shape=(2, 3), *, family="elementwise", recipe=None,
        phase="decode", dtype="bf16", requests=(), layer=0, metadata=None,
        elements=None, ops_per_element=1, **kwargs):
    return vector_op(
        op_id=name, deps=deps, stage=0, layer=layer, requests=requests,
        elements=prod(shape) if elements is None else elements,
        dtype=dtype, family=family, phase=phase, recipe=recipe or name, shape=shape,
        metadata=tuple((metadata or {}).items()), ops_per_element=ops_per_element, **kwargs,
    )

def mat(name, deps, shape=(2, 3, 5), *, quant="w8a8_int8", phase="prefill",
        recipe="dense_gate_up", requests=("r",), layer=0, metadata=None):
    m, n, k = shape
    return gemm_op(
        op_id=name, deps=deps, stage=0, layer=layer, requests=requests, m=m, n=n, k=k,
        quant=quant_preset(quant), category="dense_mlp", phase=phase, recipe=recipe,
        metadata=tuple((metadata or {}).items()),
    )
def graph(case, rows=2):
    residual = vec("residual", ("input", "saved"), (rows, 8), phase="prefill", requests=("r",),
                   recipe="residual_add", read_bytes_override=rows * 8 * 4,
                   write_bytes_override=rows * 8 * 2)
    normalized = norm(name="norm", deps=("residual",), stage=0, layer=0, requests=("r",),
                      tokens=rows, width=8, model=case.model, mode="prefill")
    projection = mat("projection", ("norm",), (rows, 8, 8), quant=case.model.quant.name)
    return (residual, normalized, projection,
            replace(residual, op_id="external_norm_user", deps=("norm",)),
            replace(residual, op_id="consumer", deps=("projection", "residual")))
def family(ops, name):
    return next(op for op in ops if op.family == name)
def by_id(ops, name):
    return next(op for op in ops if op.op_id == name)
def fuse_norm(ops, *, external="materialize"):
    return norm_quant(ops, "bf16", consumers=CONSUMERS, external_output=external)
def fields(record, **expected):
    assert {key: record[key] for key in expected} == expected
def attach(consumer, *operands):
    return add_input_ports(consumer, tuple(
        {"op": producer.op_id, "output": output, "source_op": producer.op_id}
        for producer, output in operands
    ))
def ports(op, *keys):
    return [tuple(port[key] for key in keys) for port in input_ports(op)]
def operand_values(reader, producer):
    return producer_input_values(reader, {producer.op_id: produced_values(producer)})
def assert_quant_ports(ops, producer):
    assert {(producer.op_id, output) for output in QUANT_PORTS} <= set(
        ports(by_id(ops, "projection"), "op", "output")
    )
def output_dtypes(audit):
    return {value["output"]: value["dtype"] for value in audit["typed_outputs"]}

def test_activation_materialization_keeps_payload_and_scale_separate():
    ops = activate(graph(parsed())[:3])
    conversion = family(ops, "quantize")
    audit = audit_boundary(conversion)
    assert audit["kind"] == "activation_materialization"
    outputs = {value["output"]: value for value in audit["typed_outputs"]}
    assert outputs["payload"]["dtype"] == "fp8_e4m3"
    fields(outputs["activation_scales"], dtype="fp8_e8m0", shape=[2, 1, 2])
    assert audit["typed_inputs"][0]["dtype"] == "bf16"
    assert {row["field"]: row["action"] for row in audit["field_dispositions"]}["activation.payload"] == "rewrite"
    assert_quant_ports(ops, conversion)

@pytest.mark.parametrize("unrelated_port", [False, True])
def test_replacements_rewire_only_the_replaced_named_ports(unrelated_port):
    residual, normalized, projection, external, consumer = graph(parsed())
    external = attach(external, (normalized, "output"))
    if unrelated_port:
        consumer = attach(replace(consumer, deps=("norm", "residual")),
                          (normalized, "norm"), (residual, "output"))
    else:
        consumer = attach(consumer, (residual, "output"))
    normed = fuse_norm(activate((residual, normalized, projection, external, consumer)))
    fused_norm = family(normed, "rmsnorm_quant")
    assert ports(by_id(normed, external.op_id), "op", "output", "source_op") == [
        (fused_norm.op_id, "norm", normalized.op_id)]
    lowered = residual_norm(normed, "bf16")
    fused_residual = family(lowered, "add_rmsnorm_quant")
    target = by_id(lowered, "consumer")
    if unrelated_port:
        assert set(ports(by_id(normed, "consumer"), "op")) == {(fused_norm.op_id,), ("residual",)}
        assert set(ports(target, "op")) == {(fused_norm.op_id,), (fused_residual.op_id,)}
    else:
        assert ports(target, "op", "output", "source_op") == [(fused_residual.op_id, "residual", "residual")]

def test_source_binding_restores_semantic_scale_and_attributes():
    semantic = family(activate(graph(parsed("w8a8_int8"))[:3]), "quantize")
    source = cuda_quant.make_op({"rows": 2, "width": 8, "dtype": "bf16"})
    preserved = preserve_graph(source, semantic)
    metadata = dict(preserved.metadata)
    fields(metadata, activation_scale_format="fp32", activation_scale_shape=(2,),
           input_format="bf16", output_format="int8")
    fields(metadata["source_binding"], semantic_output_format="int8", implementation_recipe=cuda_quant.RECIPE)
    fields(metadata["dispatch"], input_dtype="bf16", output_dtype="int8")
    assert preserved.source_boundary == "kernel"
    assert preserved.returns == ("activation_scales",)

def test_norm_quant_exposes_rstd_for_a_named_consumer():
    residual, normalized, projection, _, _ = graph(parsed(), rows=3)
    normalized = replace(normalized, returns=("rstd",))
    reader = vec("c", ("norm",), (3,), recipe="rstd_reader", phase="prefill", dtype="fp32")
    lowered = fuse_norm(activate((residual, normalized, projection, attach(reader, (normalized, "rstd")))))
    fused = family(lowered, "rmsnorm_quant")
    names = {value["output"] for value in produced_values(fused)}
    assert names >= {"payload", "rstd"} and "norm" not in names
    target = by_id(lowered, "c")
    assert ports(target, "op", "output") == [(fused.op_id, "rstd")]
    fields(operand_values(target, fused)[0], output="rstd", read_bytes=12)

def test_norm_and_residual_fusion_preserve_external_outputs_and_exact_traffic():
    case = parsed()
    ops = graph(case)
    activated = activate(ops)
    normalized, quantized = (family(activated, name) for name in ("rmsnorm", "quantize"))
    normed = fuse_norm(activated)
    fused = family(normed, "rmsnorm_quant")
    assert len(normed) == len(activated) - 1 and normalized.op_id not in {op.op_id for op in normed}
    assert fused.hbm_bytes == normalized.hbm_bytes + quantized.hbm_bytes - quantized.mte2_bytes
    assert fused.mte3_bytes == normalized.mte3_bytes + quantized.mte3_bytes
    audit = audit_boundary(fused)
    fields(audit, replaced_occurrences=(normalized.op_id, quantized.op_id), external_outputs=("norm",))
    assert set(audit["allocations"]["materialized"]) == QUANT_PORTS | {"norm"}
    fields(output_dtypes(audit), payload="fp8_e4m3", activation_scales="fp8_e8m0", norm="bf16")
    assert fused.dtype == "bf16" and fused.dtype != output_dtypes(audit)["payload"]
    assert next(v["materialized"] for v in audit["typed_outputs"] if v["output"] == "norm") is True
    assert (fused.op_id, "norm", normalized.op_id) in ports(by_id(normed, "external_norm_user"), "op", "output", "source_op")
    assert_quant_ports(normed, fused)
    assert dict(fused.metadata)["norm_output_materialized"] is True
    fields({r["field"]: r["action"] for r in audit["field_dispositions"]}, **{
        "normalization.output": "rewrite", "normalization.weight": "preserve",
        "quantization.group_size": "consume",
    })

    runtime = replace(case.runtime, lowering_passes=GENERIC_PASSES, kernels={
        "fusion_domains": {"norm_quant": {"consumers": list(CONSUMERS), "external_outputs": "materialize"}},
    })
    lowered = lower_implementations(ops, runtime, "bf16")
    residual = by_id(activated, "residual")
    combined = family(lowered, "add_rmsnorm_quant")
    assert combined.hbm_bytes == residual.mte2_bytes + fused.hbm_bytes
    assert combined.mte2_bytes == residual.mte2_bytes + fused.mte2_bytes - residual.mte3_bytes
    assert combined.mte3_bytes == residual.mte3_bytes + fused.mte3_bytes
    assert combined.vector_ops == residual.vector_ops + fused.vector_ops
    assert combined.returns == ("activation_scales", "norm", "residual")
    audit = audit_boundary(combined)
    assert audit["kind"] == "residual_norm"
    assert set(audit["replaced_occurrences"]) == {"residual", "norm", "projection.quantize"}
    assert set(audit["external_outputs"]) == {"residual", "norm"}
    fields(output_dtypes(audit), payload="fp8_e4m3", activation_scales="fp8_e8m0", norm="bf16", residual="bf16")
    for name, output in (("external_norm_user", "norm"), ("consumer", "residual")):
        assert (combined.op_id, output) in ports(by_id(lowered, name), "op", "output")
    assert_quant_ports(lowered, combined)

@pytest.mark.parametrize("partition,stream,external,expected", [
    ("norm", "default", "materialize", False),
    ("all", "default", "materialize", True),
    ("all", "aux", "materialize", False),
    ("none", "default", None, False),
])
def test_norm_quant_requires_shared_capture_stream_and_external_policy(partition, stream, external, expected):
    ops = activate(graph(parsed()))
    selected = tuple(
        replace(op, metadata=op.metadata + (("graph_captured", True), ("graph_segment", 4)))
        if partition == "all" or partition == op.op_id else op for op in ops
    )
    if stream == "aux":
        selected = tuple(replace(op, stream=stream) if op.op_id == "norm" else op for op in selected)
    result = fuse_norm(selected, external=external) if external else norm_quant(selected, "bf16", consumers=CONSUMERS)
    fused = [op for op in result if op.family == "rmsnorm_quant"]
    assert len(fused) == int(expected)
    if expected:
        assert audit_boundary(fused[0])["capture"]["graph_segment"] == 4
    if external is None:
        assert {op.family for op in result} >= {"rmsnorm", "quantize"}

def test_norm_accounts_for_repeated_fp32_reads_and_gamma():
    _, normalized, projection, _, _ = graph(parsed(), rows=3)
    source = vec("fp32_producer", shape=(3, 8), phase="prefill", dtype="fp32",
                 requests=("r",), family="cast", recipe="cast", metadata={"output_format": "fp32"})
    normalized = replace(normalized, deps=(source.op_id,), mte2_bytes=112,
                         metadata=normalized.metadata + (("input_repeats", {"output": 2}),))
    fused = family(norm_quant(activate((source, normalized, projection)), "bf16", consumers=CONSUMERS),
                   "rmsnorm_quant")
    assert fused.mte2_bytes == 2 * 3 * 8 * 4 + 8 * 2
    fields(audit_boundary(fused)["inputs"][0], dtype="fp32", read_bytes=2 * 3 * 8 * 4)

@pytest.mark.parametrize("source,output,dtype", [
    ("cast", "output", "fp32"), ("residual", "output", "bf16"), ("fused", "residual", "bf16"),
])
def test_conversion_reads_the_declared_operand(source, output, dtype):
    residual, normalized, projection, external, consumer = graph(parsed(), rows=3)
    projection = replace(projection, deps=(source,))
    if source == "cast":
        producer = vec(source, shape=(3, 8), dtype=dtype, family="cast", phase="prefill",
                       requests=("r",), metadata={"output_format": dtype})
        ops = (producer, replace(projection, recipe="matmul"))
    elif source == "fused":
        producer = vec(source, shape=(3, 8), phase="prefill", requests=("r",),
                       family="add_rmsnorm_quant", recipe="add_rmsnorm_quantize",
                       write_bytes_override=48, returns=("activation_scales", "residual"),
                       metadata={"output_format": "int8", "primary_output": "payload", "input_format": "bf16",
                                 "activation_scale_format": "fp32", "activation_scale_shape": (3,),
                                 "activation_scale_bytes": 12})
        ops = (producer, attach(replace(projection, recipe="matmul"), (producer, output)))
    else:
        producer = residual
        ops = (producer, normalized, attach(projection, (producer, output)), external, consumer)
    lowered = activate(ops)
    conversion = family(lowered, "quantize")
    payload_bytes = 3 * 8 * (4 if dtype == "fp32" else 2)
    assert conversion.mte2_bytes == payload_bytes
    operand = audit_boundary(conversion)["typed_inputs"][0]
    fields(operand, dtype=dtype, logical_shape=[3, 8])
    if source == "cast":
        assert conversion.dtype == dtype
        fields(operand, producer=source, bytes=payload_bytes)
    elif source == "fused":
        fields(operand, output=output, read_bytes=payload_bytes)
    else:
        assert ports(conversion, "op")[0] == (source,)
        assert operand_values(conversion, producer)[0]["read_bytes"] == payload_bytes
        assert {name for _, name in ports(by_id(lowered, "projection"), "op", "output")} == QUANT_PORTS
        assert (source,) not in ports(by_id(lowered, "projection"), "op")
    if source != "fused":
        assert_quant_ports(lowered, conversion)

def test_conversion_reads_once_and_repeated_matrix_reads_follow_the_payload():
    producer = vec("fp32_src", shape=(2, 5), dtype="fp32", family="cast", phase="prefill",
                   requests=("r",), metadata={"output_format": "fp32"})
    matrix = mat("matrix", (producer.op_id,), metadata={"input_repeats": {"output": 2}})
    lowered = activate((producer, matrix))
    conversion, reader = family(lowered, "quantize"), by_id(lowered, "matrix")
    assert conversion.mte2_bytes == 2 * 5 * 4 == operand_values(conversion, producer)[0]["read_bytes"]
    assert ports(conversion, "op")[0] == (producer.op_id,)
    assert (producer.op_id,) not in ports(reader, "op")
    assert {output for (output,) in ports(reader, "output")} == QUANT_PORTS
    assert [(v["output"], v["read_bytes"]) for v in operand_values(reader, conversion) if v["output"] in QUANT_PORTS] == [
        ("payload", 2 * 5 * 2), ("activation_scales", 2 * 4)]
def declare(key, role, kind, path, **view):
    return dict(key=key, role=role, tensor_view=view, implementation={"native_kind": kind},
                source={"path": path, "revision": "test"})
def member(declaration, segment=1, captured=True, **metadata):
    return dict(structure_subgraphs=(declaration,), structure_subgraph_memberships=(declaration["key"],),
                graph_captured=captured, graph_segment=segment if captured else None, **metadata)
def native_settings(*roles):
    return {"lowerings": {"native_fusion": {"implementation": "declared_native_fusion",
            "options": {"roles": list(roles), "traffic_profile": "preserve_workspaces"}}}}
def lower_native(ops, *declarations, **kwargs):
    runtime = replace(parsed().runtime, kernels=native_settings(*(row["role"] for row in declarations)))
    return lower_native_fusions(ops, runtime, **kwargs)
def scales(shape):
    return dict(output_format="int8", activation_scale_format="fp32",
                activation_scale_shape=shape, activation_scale_bytes=8.0)

@pytest.mark.parametrize("mode,role,names", [
    ("shared-reader", "ab", ("a", "b")), ("separate-readers", "uv", ("ua", "ub")),
    ("issued-total", "explicit", ("ea", "eb")), ("public-root", "root", ("root.a", "root.b")),
])
def test_fusion_preserves_primary_values_through_port_renames(mode, role, names):
    decl = declare(role + ".group", role, role.title(), role + ".cc", dtype="bf16")
    metadata = member(decl, output_format="bf16")
    a, b = vec(names[0], metadata=metadata), vec(names[1], (names[0],), metadata=metadata)
    expected_names = {"output__0", "output__1"}
    if mode == "public-root":
        fused = lower_native((a, b), decl, public_outputs=((a.op_id, "output"),))[0]
        assert "output__0" in fused.returns
        assert "output__0" in {v["output"] for v in produced_values(fused)}
        assert "output__0" in {v["port"] for v in audit_boundary(fused)["typed"]["external_outputs"]}
        return
    if mode == "shared-reader":
        fused, reader = lower_native((a, b, vec("plain", names, recipe="add")), decl)
        assert reader.deps == (fused.op_id,)
        assert {output for op, output in ports(reader, "op", "output") if op == fused.op_id} == expected_names
        operands = operand_values(reader, fused)
        assert len(operands) == 2 and {v["output"] for v in operands} == expected_names
        assert sum(v["read_bytes"] for v in operands) == 2 * 2 * 3 * 2
    elif mode == "separate-readers":
        u = vec("u", (a.op_id,), metadata={"input_repeats": {"output": 2}})
        fused, u, v = lower_native((a, b, u, vec("v", (b.op_id,))), decl)
        assert u.deps == v.deps == (fused.op_id,)
        u_ports, v_ports = ports(u, "output"), ports(v, "output")
        assert len(u_ports) == len(v_ports) == 1
        assert {u_ports[0][0], v_ports[0][0]} == expected_names
        assert [operand_values(reader, fused)[0]["read_bytes"] for reader in (u, v)] == [24, 12]
    else:
        reader = vec("eu", (a.op_id,), metadata={"operand_read_bytes": {"output": 32}})
        fused, reader = lower_native((a, b, reader), decl)
        fields(operand_values(reader, fused)[0], output="output__0", read_bytes=32)

def test_native_fusion_preserves_external_matrix_value_and_scale_only():
    decl = declare("s0.l2.value_group", "value_demo", "ValueDemo", "value_demo.cc", tokens=2, hidden=5,
                   input_dtype="int8", output_dtype="bf16", scale_dtype="fp32",
                   state_dtype="fp32", index_dtype="int64", accumulator_dtype="int32")
    q = vec("q", shape=(2, 5), layer=2, family="quantize", recipe="dynamic_quantize",
            ops_per_element=3, read_bytes_override=32, write_bytes_override=20,
            returns=("activation_scales",), metadata=member(decl, 2, primary_output="payload", **scales((2, 1))))
    p = mat("p", ("q",), phase="decode", layer=2, recipe="demo_gemm", requests=(), metadata=member(decl, 2))
    c = attach(vec("c", ("p", "q"), layer=2, recipe="add", metadata={"activation_scale_producer": "q"}),
               (p, "output"), (q, "activation_scales"))
    fused, reader = lower_native((q, p, c), decl)
    outputs = audit_boundary(fused)["typed"]["external_outputs"]
    assert {(v["member"], v["output"]) for v in outputs} == {("p", "output"), ("q", "activation_scales")}
    assert fused.returns == ("activation_scales",)
    assert set(ports(reader, "op", "output")) == {(fused.op_id, "activation_scales"), (fused.op_id, "output")}

def test_aliased_output_preserves_storage_and_has_zero_write():
    decl = declare("alias.group", "alias", "Alias", "alias.cc", dtype="bf16")
    op = vec("alias.out", recipe="alias", metadata=member(
        decl, output_format="bf16", alias_of="base.view", materialized=False, write_bytes=0, storage="base_pool"))
    fused = lower_native((op,), decl)[0]
    fields(produced_values(fused)[0], materialized=False, alias_of="base.view", write_bytes=0, storage="base_pool")
    fields(audit_boundary(fused)["allocations"], external_materialized=(),
           external_aliases=({"port": "output", "alias_of": "base.view"},))

def test_external_matrix_values_use_output_geometry():
    decl = declare("s0.l3.gemm_group", "gemm_demo", "GemmDemo", "gemm_demo.cc",
                   tokens=2, hidden=3, input_dtype="int8", output_dtype="bf16", scale_dtype="fp32")
    matrix = mat("gemm", ("input",), phase="decode", recipe="demo_gemm", requests=(), layer=3, metadata=member(decl))
    fused, _ = lower_native((matrix, vec("consumer", ("gemm",), recipe="add", layer=3)), decl)
    expected = dict(output="output", logical_shape=[2, 3], dtype="bf16", bytes=2 * 3 * 2)
    value = audit_boundary(fused)["typed"]["external_outputs"][0]
    fields(value, member="gemm", quant="", **expected)
    assert value["layout"]["storage_shape"] == (2, 3)
    fields(produced_values(fused)[0], producer=fused.op_id, semantic_producer="gemm", **expected)

@pytest.mark.parametrize("layout", ["padded", "routing", "attention", "copy", "strided"])
def test_typed_layout_preserves_logical_shape_storage_and_identity(layout):
    if layout == "padded":
        op = vec("bucket", shape=(4, 8), recipe="add", metadata={
            "output_format": "bf16", "graph_active_query_rows": 3, "graph_physical_query_rows": 4})
        value = produced_values(op)[0]
        fields(value, logical_shape=[3, 8], storage_shape=[4, 8], bytes=3 * 8 * 2, storage_bytes=4 * 8 * 2)
        assert value["layout"]["storage_shape"] == (4, 8)
    elif layout == "routing":
        shape = (4, 2, 32)
        op = vec("unpack.materialize", ("pack", "inverse", "weights"), shape, family="routing",
                 phase="prefill", recipe="route_unpack.materialize", write_bytes_override=512,
                 metadata={"output_layout": "tokens_topk_hidden", "output_format": "bf16"})
        value = produced_values(op)[0]
        fields(value, logical_shape=list(shape), storage_shape=list(shape), layout_name="tokens_topk_hidden",
               bytes=prod(shape) * 2, write_bytes=prod(shape) * 2)
        assert value["layout"]["shape"] == shape
    elif layout == "attention":
        shape = (1, 4, 256)
        view = dict(kind="paged_gqa", output_shape=shape, query_shape=shape, logical_layout="TND", output_dtype="bf16")
        first, second = [produced_values(vec("attention", ("q",), (1, context, 4, 256), family="attention",
                         metadata={"attention_tensor_view": view}))[0] for context in (17, 33)]
        fields(first, logical_shape=list(shape), bytes=prod(shape) * 2, layout_name="TND")
        assert first["bytes"] == second["bytes"]
    elif layout == "copy":
        op = vec("concat", ("a", "b"), (2048,), family="tensor_copy", recipe="concatenate",
                 phase="prefill", write_bytes_override=4096, metadata={"source_outputs": ("a", 0), "source_layouts": ()})
        value = produced_values(op)[0]
        fields(value, logical_shape=[2048], bytes=4096, value_id="concat.output")
        assert value["layout"]["offset"] == 0
        missing = replace(op, op_id="copy.unknown", metadata=(("source_outputs", ("a",)),))
        assert produced_values(missing)[0]["layout_gap"] == "missing_source_layout"
    else:
        expected = dict(shape=(2, 3), strides=(5, 1), offset=1, storage_shape=(2, 5))
        output = dict(producer="view", dtype="bf16", layout=expected, logical_shape=[2, 3], storage_shape=[2, 5],
                      output="output", role="activation", materialized=True)
        boundary = dict(kind="source_value_contract", typed_outputs=(output,))
        record = dict(op_id="view", kind="source_value_contract", layer=0, stage=0, block=None, role=None, boundary=boundary)
        fields(produced_values(vec("view", metadata={"fused_origin_scopes": (record,)}))[0]["layout"], **expected)

def test_replacement_to_replacement_rewires_the_external_operand():
    a = declare("a.group", "a", "A", "a.cc", dtype="bf16")
    b = declare("b.group", "b", "B", "b.cc", dtype="bf16")
    shape = (2, 4)
    q = vec("a.q", shape=shape, family="quantize", recipe="dynamic_quantize",
            returns=("activation_scales",), metadata=member(a, **scales((2,))))
    p = vec("a.p", ("a.q",), shape, family="gemm", recipe="a_p", metadata=member(a, output_format="bf16"))
    entry = member(b, output_format="bf16", structure_subgraph_entries=("b.group",))
    one = attach(vec("b.one", ("a.p",), shape, recipe="b_one", metadata=entry), (p, "output"))
    two = vec("b.two", ("b.one",), shape, recipe="b_two", metadata=member(b, output_format="bf16"))
    ops = lower_native((q, p, one, two, vec("c", ("b.two",), shape)), a, b)
    fused = {op.op_id: op for op in ops if op.family == "native_fusion"}
    assert set(fused) == {"a.group.native", "b.group.native"}
    target = fused["b.group.native"]
    assert target.deps == ("a.group.native",)
    assert ports(target, "op", "output", "source_op") == [("a.group.native", "output", "a.p")]

def demo(layer, captured=True):
    key = f"s0.l{layer}.demo"
    decl = declare(key, "demo", "Demo", "demo.cc", tokens=2, hidden=8, input_dtype="bf16",
                   output_dtype="int8", scale_dtype="fp32", state_dtype="fp32", index_dtype="int64",
                   accumulator_dtype="int32")
    q = vec(key + ".quantize", shape=(2, 8), layer=layer, family="quantize", recipe="dynamic_quantize",
            ops_per_element=3, returns=("activation_scales",), metadata=member(decl, 3, captured, **scales((2, 1))))
    p = vec(key + ".projection", (q.op_id,), (2, 8, 8), layer=layer, family="gemm", recipe="demo_gemm",
            dtype="int8", elements=16, ops_per_element=2, metadata=member(decl, 3, captured, output_format="bf16"))
    return (q, p, vec(key + ".consumer", (p.op_id,), (16,), layer=layer, recipe="add")), decl

def test_terminal_fusion_preserves_public_returns_without_a_consumer():
    (quant, _, _), decl = demo(0)
    fused = lower_native((quant,), decl)[0]
    assert fused.returns == ("activation_scales",)
    assert {(v["output"], v["dtype"]) for v in produced_values(fused)} == {("payload", "int8"), ("activation_scales", "fp32")}
    outputs = audit_boundary(fused)["typed"]["external_outputs"]
    assert {(v["member"], v["output"]) for v in outputs} == {(quant.op_id, "payload"), (quant.op_id, "activation_scales")}

def test_native_fusion_preserves_occurrences_roles_and_selected_plan():
    ops, decl = demo(0)
    fused, reader = lower_native(ops, decl)
    audit, metadata = audit_boundary(fused), dict(fused.metadata)
    fields(audit["occurrences"], replaced=tuple(op.op_id for op in ops[:2]), replacement=fused.op_id)
    assert audit["allocations"]["external_output_ids"] == (ops[1].op_id,)
    assert {v["semantic_op"] for v in audit["typed"]["external_outputs"]} == {ops[1].op_id}
    roles = audit["typed"]["roles"]
    assert set(roles) >= {"input", "output", "scale", "state", "index", "accumulator"}
    for role, dtype in (("state", "fp32"), ("index", "int64"), ("accumulator", "int32")):
        assert {v["dtype"] for v in roles[role]} == {dtype}
    fields(audit["capture"], graph_captured=True, graph_segment=3)
    assert metadata["fusion_boundary"]["capture"] == audit["capture"]
    assert audit["overlap"]["declared_concurrency"] == "not_implied_by_fusion"
    assert len(metadata["fusion_phases"]) == len(metadata["fusion_semantic_ids"]) == 2
    assert fused.flops == sum(op.flops for op in ops[:2])
    assert fused.hbm_bytes == sum(op.hbm_bytes for op in ops[:2])
    assert reader.deps == (fused.op_id,) and fused.returns == ()
    selection = metadata["fusion_boundary"]["selection"]
    assert selection["plan"]["kind"] == "declared_software_plan"
    assert selection["actual_recipe"] == fused.recipe and selection["actual_phase_count"] >= 2
    assert set(selection["actual_external_outputs"]) <= {v["output"] for v in produced_values(fused)}

def test_layer_identity_and_capture_preserve_physical_work():
    fused = []
    for layer, captured in ((0, True), (1, True), (2, False)):
        ops, decl = demo(layer, captured)
        fused.append(lower_native(ops, decl)[0])
    left, right, eager = fused
    assert node_physical(left)["physical_signature"] == node_physical(right)["physical_signature"]
    assert eager.hbm_bytes == left.hbm_bytes and eager.flops == left.flops
    assert audit_boundary(eager)["capture"]["graph_captured"] is False
    assert len(dict(left.metadata)["fusion_phases"]) == len(dict(eager.metadata)["fusion_phases"]) == 2
    left, right = [family(residual_norm(fuse_norm(activate(tuple(replace(op, layer=layer)
                   for op in graph(parsed())))), "bf16"), "add_rmsnorm_quant") for layer in (0, 1)]
    assert node_physical(left)["physical_signature"] == node_physical(right)["physical_signature"]
    assert left.mte2_bytes == right.mte2_bytes and left.mte3_bytes == right.mte3_bytes

@pytest.mark.parametrize("external", [True, False])
def test_traffic_profile_preserves_only_externally_visible_writes(external):
    act = vec("act", shape=(2, 8), family="situ", recipe="situ_activation",
              ops_per_element=2, read_bytes_override=32, write_bytes_override=32)
    quant = vec("quant", ("act",), (2, 8), family="quantize", recipe="dynamic_quantize",
                read_bytes_override=32, write_bytes_override=16, returns=("activation_scales",))
    decl = {"key": "situ.demo", "role": "situ_quant", "tensor_view": {
        "input_dtype": "bf16", "output_dtype": "fp8_e4m3", "scale_dtype": "fp8_e8m0", "axis": -1}}
    adjusted, ledger = apply_traffic_profile((act, quant), decl, "situ_mx_quant_arch35_2a9bd1b0",
                                             external_outputs=("act",) if external else ())
    assert adjusted[0].mte3_bytes == (act.mte3_bytes if external else 0)
    assert ledger["value_effects"][0]["action"] == ("preserve" if external else "consume")
    if external:
        assert adjusted[1].mte2_bytes == 0
        fields(ledger, eliminated_reads=act.mte3_bytes, eliminated_writes=0,
               external_output_writes_preserved_phase_positions=[0])
        assert ledger["value_effects"][0]["materialized"] is True and ledger["value_effects"][1]["action"] == "consume"
        assert ledger["representation_guard"]["input_dtype"] == "bf16"
        assert ledger["capture"]["implied"] is ledger["overlap"]["implied"] is False
def piecewise(case, splits=(), passes=GENERIC_PASSES, **kwargs):
    return replace(case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
                   piecewise_capture_keys=(8,), mixed_mode="mixed", splitting_ops=splits, lowering_passes=passes, **kwargs)
def replay(regions, runtime, resource_fusion=False):
    return lower_regions(regions, graph=GraphPolicy(), replay_scope="piecewise",
                         runtime=runtime, prefix=(), resource_fusion=resource_fusion)[0]

@pytest.mark.parametrize("splits,expected,public", [
    (("rmsnorm",), (False, True), False), (("quantize",), (True, False, True), False),
    (("rmsnorm", "quantize"), (False,), False), (("rmsnorm",), (False, True), True), ((), (True,), True),
])
def test_piecewise_replay_partitions_before_generic_fusion(splits, expected, public):
    case = parsed()
    passes = GENERIC_PASSES if public or splits == ("rmsnorm",) else GENERIC_PASSES[:2]
    runtime = piecewise(case, splits, passes)
    if public:
        ops = target_operators(replace(case, runtime=runtime), phase="decode", tokens=1, context=0, sequences=1,
                               scope="body", apply_body_defaults=False, resource_fusion=False)
    else:
        raw = graph(case)
        members = raw if splits == ("rmsnorm",) else (*raw[:3], raw[-1])
        ops = replay((LoweringRegion(members, runtime, "bf16", 2),), runtime)
    families = {op.family for op in ops}
    if len(expected) == 1:
        assert "rmsnorm_quant" in families and "rmsnorm" not in families
        if not public:
            assert "quantize" not in families
            assert dict(family(ops, "rmsnorm_quant").metadata)["graph_captured"] is expected[0]
        return
    if splits == ("rmsnorm",):
        assert not families & {"rmsnorm_quant", "add_rmsnorm_quant"}
    normalized = family(ops, "rmsnorm")
    conversion = next(op for op in ops if op.family == "quantize" and normalized.op_id in op.deps)
    selected = (normalized, conversion) if len(expected) == 2 else (normalized, conversion, by_id(ops, "projection"))
    metadata = [dict(op.metadata) for op in selected]
    assert tuple(row["graph_captured"] for row in metadata) == expected
    assert len({row["graph_segment"] for row in metadata}) == len(expected)

@pytest.mark.parametrize("resource_fusion", [False, True])
def test_replay_partitions_inserted_quantizer_before_native_admission(resource_fusion, monkeypatch):
    case = parsed()
    name = "capture" if resource_fusion else "pf"
    role = "capture_demo" if resource_fusion else "prefix_native"
    decl = declare(name + ".native", role, "CaptureDemo" if resource_fusion else "PrefixNative",
                   name + ".cc", tokens=2, hidden=5)
    metadata = {"structure_subgraphs": (decl,), "structure_subgraph_memberships": (decl["key"],)}
    matrix = mat(name + ".projection", ("input",), recipe=name + "_gemm", metadata=metadata)
    consumer = vec(name + ".consumer", (matrix.op_id,), phase="prefill", requests=("r",), recipe=name + "_add")
    runtime = piecewise(case, passes=GENERIC_PASSES[:1], kernels=native_settings(role))
    if resource_fusion:
        ops = replay((LoweringRegion((matrix, consumer), runtime, "bf16", 1),), runtime, True)
        fused = family(ops, "native_fusion")
        assert dict(fused.metadata)["graph_captured"] is True
        assert not {op.family for op in ops} & {"quantize", "gemm"}
        fields(produced_values(fused)[0], logical_shape=[2, 3], bytes=2 * 3 * 2)
        assert len(dict(fused.metadata)["fusion_phases"]) >= 2
    else:
        original = GraphPolicy.apply
        calls = []
        def record(self, ops, scope, selected_runtime):
            calls.append(tuple(ops))
            return original(self, ops, scope, selected_runtime)
        monkeypatch.setattr(GraphPolicy, "apply", record)
        prefix = vec("prefix.p", shape=(2, 2), phase="prefill", requests=("r",), recipe="prefix_p")
        regions = tuple(LoweringRegion(ops, case.runtime, "bf16", 1) for ops in ((prefix, matrix), (consumer,)))
        ops = replay(regions, runtime)
        assert len(calls) == 1
        assert family(ops, "quantize") and by_id(ops, matrix.op_id)

@pytest.mark.parametrize("phase,context,selected,matches,kind", [
    ("decode", 8, "vllm_cuda_rms_norm", [True, False], "declared_software_plan"),
    ("prefill", 0, "analytical", [False, True], "analytical_semantic_work"),
    ("mixed", 8, None, None, None),
])
def test_selection_exposes_candidates_and_chosen_plan(phase, context, selected, matches, kind):
    policy = {"cases": [
        {"domain": {"phases": [p]}, "binding": b} for p, b in
        (("decode", "vllm_cuda_rms_norm"), ("prefill", "analytical"))
    ], "otherwise": "error"}
    runtime = replace(parsed().runtime, kernels={"lowerings": {"rmsnorm": policy}})
    view = SimpleNamespace(phase=phase, q_len=4, context=context)
    if selected is None:
        with pytest.raises(ModelingError, match="implementation_domain_unmatched"):
            resolve(runtime, (view,), padded_tokens=4, replay_scope="none")
        return
    resolved, (record,) = resolve(runtime, (view,), padded_tokens=4, replay_scope="none")
    assert [item["matches"] for item in record["candidates"]] == matches
    assert record["selected"]["case"] == matches.index(True)
    assert record["plan"]["kind"] == kind
    if phase == "decode":
        assert record["plan"]["implementation"] == resolved.kernels["lowerings"]["rmsnorm"] == selected
        assert record["geometry"]["padded_tokens"] == 4
    else:
        assert record["implementation"] == selected and "rmsnorm" not in resolved.kernels["lowerings"]
