"""Declared gated activations shared by dense, shared, and routed FFNs.

[Role] Preserve SiTU's clipping and FP32 expression separately from SwiGLU.
[Invariants] Parameters come from model configuration, never profiling shapes.
  Scalar expression counts are analytical work, not native instruction counts.
[Timeouts & Locks] Pure constant-time operator construction; no I/O or locks.
[Side-Effect Policy] Returns operator metadata and appends requested groups.
"""
import math
from dataclasses import replace

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .structure_groups import declare_group


SITU_SOURCE = {
    "repository": "vllm", "base_revision": "58d3918e3ea0a544ffedadad2ba84559e9c51d8f",
    "source_state": "clean_at_revision",
    "path": "vllm/model_executor/layers/activation.py", "lines": [155, 184],
    "sha256": "157cb7610fba4237350daf7ec3b58c34c7f3fb38de8ff7b9e55584a69ca475c3",
    "symbol": "SituAndMul.forward_native",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
SITU_QUANT_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "added_in_index",
    "path": "csrc/moe/situ_mx_quant/op_host/situ_mx_quant_def.cpp",
    "lines": [20, 64],
    "sha256": "c4c9e30dd13cf759fc036b188360cb022e167ac209eeb5f86a020256e7d07367",
    "symbol": "SituMxQuant",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def gated_activation(*, plan, op_id, deps, rows, width, dtype, stage, requests,
                     phase, recipe):
    activation = plan.ffn_activation
    metadata = (("activation_kind", activation),)
    if activation == "silu":
        family, per_element = "swiglu", 4
    elif activation == "situ":
        if plan.activation_situ_beta <= 0 or (
                plan.activation_situ_linear_beta is not None
                and plan.activation_situ_linear_beta <= 0):
            raise ModelingError("situ_activation_parameters_must_be_positive")
        family = "situ"
        # Two input casts, output cast, divide/tanh/scale/sigmoid/products;
        # the optional up-branch soft clip adds divide, tanh and multiply.
        per_element = 9 + (3 if plan.activation_situ_linear_beta is not None else 0)
        metadata += (
            ("activation_beta", plan.activation_situ_beta),
            ("activation_linear_beta", plan.activation_situ_linear_beta),
            ("activation_compute_dtype", "fp32"),
            ("activation_expression", "beta*tanh(gate/beta)*sigmoid(gate)*clipped_up"),
            ("vector_count_basis", "scalar_expression_ops; transcendental_counted_once"),
            ("native_source", SITU_SOURCE),
        )
    else:
        raise ModelingError(f"unsupported_gated_ffn_activation:{activation}")
    count = rows * width
    return vector_op(
        op_id=op_id, deps=tuple(deps), stage=stage, layer=plan.index, requests=requests,
        elements=math.ceil(count), ops_per_element=per_element,
        vector_ops_override=per_element * count, dtype=dtype, family=family,
        phase=phase, recipe=recipe.replace("swiglu", family),
        shape=(math.ceil(rows), width), metadata=metadata,
        read_bytes_override=2 * count * dtype_bytes(dtype),
        write_bytes_override=count * dtype_bytes(dtype),
    )


def declare_situ_quant(ops, *, key, activation, consumer, runtime, rows, width):
    """The ordinary conversion pass supplies the output quantizer membership."""
    meta = dict(consumer.metadata)
    if (activation.family != "situ" or activation.dtype != "bf16" or not runtime
            or meta.get("activation_format") not in {"fp8_e4m3", "fp8_e5m2"}
            or meta.get("activation_scale_format") != "fp8_e8m0"
            or not meta.get("activation_scale_bytes")
            or "activation_quantization" not in runtime.lowering_passes):
        return
    for index, op in enumerate(ops):
        if op.op_id == consumer.op_id:
            ops[index] = replace(op, metadata=op.metadata + (
                ("input_conversion_subgraphs", (key,)),))
    declare_group(
        ops, key=key, role="situ_quant", family="situ",
        members=(activation.op_id,), entries=(activation.op_id,), anchor=activation.op_id,
        block="ffn", scope_role="body",
        view={"rows": rows, "width": width, "input_dtype": activation.dtype,
              "output_dtype": meta["activation_format"],
              "scale_dtype": meta["activation_scale_format"],
              "scale_shape": meta["activation_scale_shape"],
              "beta": dict(activation.metadata)["activation_beta"],
              "linear_beta": dict(activation.metadata)["activation_linear_beta"],
              "activate_left": True, "axis": -1},
        implementation={"native_kind": "SituMxQuant", "selection": None,
                        "conditions": {"scalar_activation_and_quantization_modes_not_observed": True}},
        source=SITU_QUANT_SOURCE,
        coverage={"kind": "complete_subgraph",
                  "limitations": ["native_scalar_parameters_and_quantization_layout_not_observed"]})
