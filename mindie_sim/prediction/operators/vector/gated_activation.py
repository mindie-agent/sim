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
from mindie_sim.prediction.structure_groups import declare_group


SITU_SOURCE = {
    "repository": "vllm", "base_revision": "58d3918e3ea0a544ffedadad2ba84559e9c51d8f",
    "source_state": "clean_at_revision",
    "path": "vllm/model_executor/layers/activation.py", "lines": [155, 184],
    "sha256": "157cb7610fba4237350daf7ec3b58c34c7f3fb38de8ff7b9e55584a69ca475c3",
    "symbol": "SituAndMul.forward_native",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
#: A clamped gated activation adds ``min(gate, limit)`` and the two-sided
#: ``clamp(up, -limit, limit)`` before the gate product.  ``swigluoai`` adds the
#: declared ``alpha * g`` scaling and ``(u + beta)`` offset on top of that.
SWIGLU_CLAMP_OPS_PER_ELEMENT = 3
SWIGLU_ALPHA_SCALE_OPS_PER_ELEMENT = 1
SWIGLU_BETA_OFFSET_OPS_PER_ELEMENT = 1
SWIGLU_GATE_SIGMOID_OPS_PER_ELEMENT = 4
SWIGLU_PRODUCT_OPS_PER_ELEMENT = 2
_SWIGLU_OPS_PER_ELEMENT = 4
_GEGLU_OPS_PER_ELEMENT = 8
_SITU_OPS_PER_ELEMENT = 9
_SITU_UP_CLIP_OPS_PER_ELEMENT = 3
_FP32_BYTES = 4
_SITU_EAGER_CAST_OPS = 1
_SITU_EAGER_SLICE_OPS = 1
_SITU_EAGER_DIV_OPS = 1
_SITU_EAGER_TANH_OPS = 2
_SITU_EAGER_MUL_OPS = 1
_SITU_EAGER_SIGMOID_OPS = 4
_SITU_EAGER_PAIR_TENSORS = 2
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


def _clamped_activation_ops(alpha: float, beta: float) -> int:
    """Exact scalar expression count of the declared clamped gate product."""
    return (SWIGLU_CLAMP_OPS_PER_ELEMENT
            + (SWIGLU_ALPHA_SCALE_OPS_PER_ELEMENT if alpha != 1.0 else 0)
            + SWIGLU_GATE_SIGMOID_OPS_PER_ELEMENT
            + (SWIGLU_BETA_OFFSET_OPS_PER_ELEMENT if beta else 0)
            + SWIGLU_PRODUCT_OPS_PER_ELEMENT)


def gated_activation(*, plan, op_id, deps, rows, width, dtype, stage, requests,
                     phase, recipe):
    activation = plan.ffn_activation
    metadata = (("activation_kind", activation),)
    if activation == "swigluoai" or (activation == "silu" and plan.swiglu_limit > 0):
        # Clamped gate: min(gate, limit), clamp(up, -limit, limit) and the
        # sigmoid gate.  Plain SiLU keeps alpha=1/beta=0; only ``swigluoai``
        # declares the OAI alpha/beta scalars.
        # The physical family stays the gated SwiGLU vector pipeline; the
        # recipe and metadata carry the exact OAI/clamped expression.
        family = "swiglu"
        alpha = float(plan.swiglu_alpha)
        beta = float(plan.swiglu_beta)
        limit = float(plan.swiglu_limit)
        if limit <= 0:
            raise ModelingError("clamped_ffn_activation_requires_positive_limit")
        if activation == "swigluoai" and not 0.0 < alpha:
            raise ModelingError("swigluoai_requires_positive_alpha")
        per_element = _clamped_activation_ops(alpha, beta)
        recipe = (recipe.replace("swiglu", "swigluoai")
                  if activation == "swigluoai" else recipe)
        metadata += (
            ("activation_limit", limit),
            ("activation_alpha", alpha),
            ("activation_beta", beta),
            ("activation_expression",
             "min(gate,limit)*sigmoid(alpha*min(gate,limit))"
             "*(clamp(up,-limit,limit)+beta)"),
            ("vector_count_basis",
             "scalar_expression_ops; clamp_and_sigmoid_counted_once"),
        )
    elif activation == "silu":
        family, per_element = "swiglu", _SWIGLU_OPS_PER_ELEMENT
    elif activation in {"gelu", "geglu", "gelu_tanh", "gelu_pytorch_tanh"}:
        # Tanh-approximated GeGLU: cubic + scale + add + tanh + +1 + 0.5*gate
        # + gate*up.
        family, per_element = "geglu", _GEGLU_OPS_PER_ELEMENT
    elif activation == "situ":
        if plan.activation_situ_beta <= 0 or (
                plan.activation_situ_linear_beta is not None
                and plan.activation_situ_linear_beta <= 0):
            raise ModelingError("situ_activation_parameters_must_be_positive")
        family = "situ"
        # Two input casts, output cast, divide/tanh/scale/sigmoid/products;
        # the optional up-branch soft clip adds divide, tanh and multiply.
        per_element = _SITU_OPS_PER_ELEMENT + (
            _SITU_UP_CLIP_OPS_PER_ELEMENT if plan.activation_situ_linear_beta is not None else 0)
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


def situ_eager_chain(*, plan, op_id, deps, rows, width, dtype, stage, requests, phase):
    """Unfused PyTorch eager ``SituAndMul.forward_native`` FP32 ``aclnn*`` chain.

    Matches real ``torch_npu`` kernel dispatch (14 kernels with ``linear_beta``
    up-clipping, 11 without): Cast(in->f32) -> Slice(gate) -> RealDiv -> Tanh
    -> Muls -> Slice(gate) -> Sigmoid -> Mul -> Slice(up) -> [RealDiv -> Tanh
    -> Muls] -> Mul -> Cast(f32->out).
    """
    if plan.activation_situ_beta <= 0 or (
            plan.activation_situ_linear_beta is not None
            and plan.activation_situ_linear_beta <= 0):
        raise ModelingError("situ_activation_parameters_must_be_positive")
    count = math.ceil(rows * width)
    row_dim = math.ceil(rows)
    act_bytes = dtype_bytes(dtype)
    fp32_half = count * _FP32_BYTES
    fp32_pair = _SITU_EAGER_PAIR_TENSORS * fp32_half
    act_half = count * act_bytes
    act_pair = _SITU_EAGER_PAIR_TENSORS * act_half
    base_meta = (
        ("activation_kind", "situ"),
        ("activation_beta", plan.activation_situ_beta),
        ("activation_linear_beta", plan.activation_situ_linear_beta),
        ("activation_compute_dtype", "fp32"),
        ("native_source", SITU_SOURCE),
    )

    def _emit(suffix, step_deps, *, family, recipe, elems, ops_per_elem, step_dtype,
              read_bytes, write_bytes, shape, native_kind):
        return vector_op(
            op_id=f"{op_id}.{suffix}" if suffix else op_id,
            deps=tuple(step_deps), stage=stage, layer=plan.index, requests=requests,
            elements=elems, ops_per_element=ops_per_elem,
            vector_ops_override=ops_per_elem * elems, dtype=step_dtype,
            family=family, phase=phase, recipe=recipe, shape=shape,
            read_bytes_override=read_bytes, write_bytes_override=write_bytes,
            metadata=base_meta + (("native_kind", native_kind),),
        )

    cast_in = _emit(
        "cast_in", deps, family="cast", recipe="situ_cast_fp32",
        elems=_SITU_EAGER_PAIR_TENSORS * count, ops_per_elem=_SITU_EAGER_CAST_OPS,
        step_dtype="fp32", read_bytes=act_pair, write_bytes=fp32_pair,
        shape=(row_dim, _SITU_EAGER_PAIR_TENSORS * width), native_kind="Cast")
    gate_slice = _emit(
        "gate_slice", (cast_in.op_id,), family="tensor_copy", recipe="situ_gate_slice",
        elems=count, ops_per_elem=_SITU_EAGER_SLICE_OPS, step_dtype="fp32",
        read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
        native_kind="Slice")
    gate_div = _emit(
        "gate_div", (gate_slice.op_id,), family="elementwise", recipe="situ_gate_div",
        elems=count, ops_per_elem=_SITU_EAGER_DIV_OPS, step_dtype="fp32",
        read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
        native_kind="RealDiv")
    gate_tanh = _emit(
        "gate_tanh", (gate_div.op_id,), family="elementwise", recipe="situ_gate_tanh",
        elems=count, ops_per_elem=_SITU_EAGER_TANH_OPS, step_dtype="fp32",
        read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
        native_kind="Tanh")
    gate_scale = _emit(
        "gate_scale", (gate_tanh.op_id,), family="elementwise", recipe="situ_gate_scale",
        elems=count, ops_per_elem=_SITU_EAGER_MUL_OPS, step_dtype="fp32",
        read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
        native_kind="Mul")
    sig_slice = _emit(
        "sig_slice", (cast_in.op_id,), family="tensor_copy", recipe="situ_sig_slice",
        elems=count, ops_per_elem=_SITU_EAGER_SLICE_OPS, step_dtype="fp32",
        read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
        native_kind="Slice")
    gate_sig = _emit(
        "gate_sigmoid", (sig_slice.op_id,), family="elementwise", recipe="situ_gate_sigmoid",
        elems=count, ops_per_elem=_SITU_EAGER_SIGMOID_OPS, step_dtype="fp32",
        read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
        native_kind="Sigmoid")
    gate_act = _emit(
        "gate_mul", (gate_scale.op_id, gate_sig.op_id), family="elementwise",
        recipe="situ_gate_mul", elems=count, ops_per_elem=_SITU_EAGER_MUL_OPS,
        step_dtype="fp32", read_bytes=fp32_pair, write_bytes=fp32_half,
        shape=(row_dim, width), native_kind="Mul")
    up_slice = _emit(
        "up_slice", (cast_in.op_id,), family="tensor_copy", recipe="situ_up_slice",
        elems=count, ops_per_elem=_SITU_EAGER_SLICE_OPS, step_dtype="fp32",
        read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
        native_kind="Slice")
    ops = [cast_in, gate_slice, gate_div, gate_tanh, gate_scale,
           sig_slice, gate_sig, gate_act, up_slice]
    up_ready = up_slice.op_id
    if plan.activation_situ_linear_beta is not None:
        up_div = _emit(
            "up_div", (up_ready,), family="elementwise", recipe="situ_up_div",
            elems=count, ops_per_elem=_SITU_EAGER_DIV_OPS, step_dtype="fp32",
            read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
            native_kind="RealDiv")
        up_tanh = _emit(
            "up_tanh", (up_div.op_id,), family="elementwise", recipe="situ_up_tanh",
            elems=count, ops_per_elem=_SITU_EAGER_TANH_OPS, step_dtype="fp32",
            read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
            native_kind="Tanh")
        up_scale = _emit(
            "up_scale", (up_tanh.op_id,), family="elementwise", recipe="situ_up_scale",
            elems=count, ops_per_elem=_SITU_EAGER_MUL_OPS, step_dtype="fp32",
            read_bytes=fp32_half, write_bytes=fp32_half, shape=(row_dim, width),
            native_kind="Mul")
        ops.extend((up_div, up_tanh, up_scale))
        up_ready = up_scale.op_id
    prod = _emit(
        "prod_mul", (gate_act.op_id, up_ready), family="elementwise",
        recipe="situ_prod_mul", elems=count, ops_per_elem=_SITU_EAGER_MUL_OPS,
        step_dtype="fp32", read_bytes=fp32_pair, write_bytes=fp32_half,
        shape=(row_dim, width), native_kind="Mul")
    cast_out = _emit(
        "", (prod.op_id,), family="cast", recipe="situ_cast_out",
        elems=count, ops_per_elem=_SITU_EAGER_CAST_OPS, step_dtype=dtype,
        read_bytes=fp32_half, write_bytes=act_half, shape=(row_dim, width),
        native_kind="Cast")
    ops.extend((prod, cast_out))
    return tuple(ops)


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
