"""Diffusion projection, modulation and quantization.

[Role] Build diffusion projection, modulation and quantization contracts.
[Invariants] Tensor shapes and formats determine arithmetic and materialized traffic.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from math import prod

from mindie_sim.schema.common import ModelingError, dtype_bytes

from .contracts import _as_shape, _bytes, _contract, _positive_int, _read, _role, _shape

_SWIGLU_OPS_PER_ELEMENT = 6.0
_DEFAULT_FFN_ACTIVATION_OPS_PER_ELEMENT = 5.0
_ADA_NORM_OPS_PER_ELEMENT = 5
_MXFP4_SCALE_GROUP_SIZE = 32
_MXFP4_QUANT_OPS_PER_ELEMENT = 4


def _eagle_ffn_contract(op, meta, dtype, returns):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_eagle_ffn_requires_matrix_x")
    m = prod(x_shape[:-1])
    k = x_shape[-1]
    w1 = meta.get("weight1_shape")
    w2 = meta.get("weight2_shape")
    if w1 is None or w2 is None:
        if op.flops > 0 and op.mte2_bytes > 0:
            reads = [{"name": "declared_operands", "shape": tuple(),
                      "dtype": dtype, "bytes": float(op.mte2_bytes),
                      "materialized": True, "alias": None,
                      "kind": "declared", "producer": None}]
            writes = [{"name": "y", "shape": x_shape, "dtype": dtype,
                       "bytes": float(op.mte3_bytes), "materialized": True,
                       "alias": None, "kind": "declared",
                       "source": "declared_ffn"}]
            compute = [{"name": "declared_ffn", "kind": "matrix",
                        "matrix_flops": ((dtype, float(op.flops)),),
                        "vector_ops": float(op.vector_ops),
                        "inputs": ("declared_operands",),
                        "outputs": ("declared_ffn_out",)}]
            return _contract(reads, writes, compute, attributes={},
                             basis="declared_ffn_flops_legacy",
                             limitations=("weight_shapes_not_declared",))
        raise ModelingError("mindie_sd_eagle_ffn_requires_weight1_and_weight2_shapes")
    w1 = _as_shape(w1, "weight1")
    w2 = _as_shape(w2, "weight2")
    if len(w1) != 2 or len(w2) != 2:
        raise ModelingError("mindie_sd_eagle_ffn_weights_must_be_matrix")
    if w1[1] == k:
        h_total, w1_layout = w1[0], "linear_HxK"
    elif w1[0] == k:
        h_total, w1_layout = w1[1], "canonical_KxH"
    else:
        raise ModelingError("mindie_sd_eagle_ffn_weight1_K_mismatch")
    activation = str(meta.get("activation", "silu")).lower()
    h_act = h_total
    if activation == "swiglu":
        if h_total % 2:
            raise ModelingError("mindie_sd_eagle_ffn_swiglu_requires_even_H")
        h_act = h_total // 2
    if w2[1] == h_act:
        n, w2_layout = w2[0], "linear_NxH"
    elif w2[0] == h_act:
        n, w2_layout = w2[1], "canonical_HxN"
    else:
        raise ModelingError("mindie_sd_eagle_ffn_weight2_H_mismatch")
    bias_dtype = str(meta.get("bias_dtype", dtype))
    bias1_shape = meta.get("bias1_shape")
    bias2_shape = meta.get("bias2_shape")
    if bias1_shape is not None:
        bias1_shape = _as_shape(bias1_shape, "bias1")
        if prod(bias1_shape) != h_total:
            raise ModelingError("mindie_sd_eagle_ffn_bias1_width_mismatch")
    if bias2_shape is not None:
        bias2_shape = _as_shape(bias2_shape, "bias2")
        if prod(bias2_shape) != n:
            raise ModelingError("mindie_sd_eagle_ffn_bias2_width_mismatch")
    out_shape = tuple(x_shape[:-1]) + (n,)
    hidden_bytes = float(m * h_act * dtype_bytes(dtype))
    reads = [_read("x", x_shape, dtype), _read("weight1", w1, dtype),
             _read("weight2", w2, dtype)]
    linear1_inputs = ["x", "weight1"]
    linear2_inputs = ["hidden", "weight2"]
    if bias1_shape is not None:
        reads.append(_read("bias1", bias1_shape, bias_dtype))
        linear1_inputs.append("bias1")
    if bias2_shape is not None:
        reads.append(_read("bias2", bias2_shape, bias_dtype))
        linear2_inputs.append("bias2")
    writes = [
        {"name": "hidden_store", "shape": (m, h_act), "dtype": dtype,
         "bytes": hidden_bytes, "materialized": True, "alias": None,
         "kind": "workspace", "source": "activation"},
        {"name": "y", "shape": out_shape, "dtype": dtype,
         "bytes": _bytes(out_shape, dtype), "materialized": True,
         "alias": None, "kind": "value", "source": "linear2"},
    ]
    reads.append({"name": "hidden", "shape": (m, h_act), "dtype": dtype,
                  "bytes": hidden_bytes, "materialized": True, "alias": None,
                  "kind": "workspace", "producer": "hidden_store"})
    activation_ops = (_SWIGLU_OPS_PER_ELEMENT if activation == "swiglu"
                      else _DEFAULT_FFN_ACTIVATION_OPS_PER_ELEMENT)
    compute = [
        {"name": "linear1", "kind": "matrix",
         "matrix_flops": ((dtype, 2 * m * h_total * k),),
         "inputs": tuple(linear1_inputs), "outputs": ("linear1_out",)},
        {"name": "activation", "kind": "vector",
         "vector_ops": float(m * h_act * activation_ops),
         "inputs": ("linear1_out",), "outputs": ("activation_out",)},
        {"name": "linear2", "kind": "matrix",
         "matrix_flops": ((dtype, 2 * m * n * h_act),),
         "inputs": tuple(linear2_inputs), "outputs": ("linear2_out",)},
    ]
    return _contract(reads, writes, compute, attributes={
        "activation": activation, "weight1_shape": w1, "weight2_shape": w2,
        "weight1_layout": w1_layout, "weight2_layout": w2_layout,
        "bias_dtype": bias_dtype, "bias1_shape": bias1_shape,
        "bias2_shape": bias2_shape, "hidden_shape": (m, h_act),
        "hidden_dtype": dtype, "hidden_barrier": "hidden_store",
    }, basis="two_linear_activation_hidden_gm_materialization",
        limitations=("hidden_store_is_a_real_gm_value",))


def _adalayernorm_contract(op, meta, dtype, returns):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_adalayernorm_requires_at_least_2d_x")
    rows, width = prod(x_shape[:-1]), x_shape[-1]
    scale_shape = _as_shape(meta.get("scale_shape", (width,)), "scale")
    shift_shape = _as_shape(meta.get("shift_shape", (width,)), "shift")
    batch = x_shape[0] if len(x_shape) >= 2 else 1
    def broadcast_ok(shape):
        return (prod(shape) in {width, rows * width}
                or (len(shape) == 2 and shape[0] == batch and shape[1] == width))
    if not broadcast_ok(scale_shape) or not broadcast_ok(shift_shape):
        raise ModelingError("mindie_sd_adalayernorm_modulation_width_mismatch")
    reads = [_read("x", x_shape, dtype), _read("scale", scale_shape, dtype),
             _read("shift", shift_shape, dtype)]
    weight_shape = meta.get("weight_shape")
    bias_shape = meta.get("bias_shape")
    if weight_shape is not None:
        reads.append(_read("weight", _as_shape(weight_shape, "weight"), dtype))
    if bias_shape is not None:
        reads.append(_read("bias", _as_shape(bias_shape, "bias"), dtype))
    writes = [{**_role("y", x_shape, dtype), "source": "norm"}]
    compute = [{"name": "norm", "kind": "vector",
                "vector_ops": float(rows * width * _ADA_NORM_OPS_PER_ELEMENT),
                "inputs": tuple(item["name"] for item in reads),
                "outputs": ("y_value",)}]
    if bool(meta.get("return_stats", False)):
        mean_shape = tuple(x_shape[:-1]) + (1,)
        for name in ("mean", "rstd"):
            writes.append({**_role(name, mean_shape, dtype), "source": "norm"})
    return _contract(reads, writes, compute, attributes={
        "x_shape": x_shape, "scale_shape": scale_shape, "shift_shape": shift_shape,
        "weight_shape": weight_shape, "bias_shape": bias_shape,
    }, basis="three_dimensional_adaln_modulation_and_optional_affine")


def _laurel_contract(op, meta, dtype, returns):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_laurel_requires_matrix_x")
    rows, width = prod(x_shape[:-1]), x_shape[-1]
    rank = _positive_int(meta, None, "low_rank")
    wa_shape = meta.get("weight_a_shape", (width, rank))
    wb_shape = meta.get("weight_b_shape", (rank, width))
    wa_shape = _as_shape(wa_shape, "weight_a")
    wb_shape = _as_shape(wb_shape, "weight_b")
    if len(wa_shape) != 2 or len(wb_shape) != 2 or wa_shape[-1] != wb_shape[0]:
        raise ModelingError("mindie_sd_laurel_weight_geometry_mismatch")
    reads = [_read("x", x_shape, dtype), _read("weight_a", wa_shape, dtype),
             _read("weight_b", wb_shape, dtype)]
    writes = [{**_role("y", x_shape, dtype), "source": "residual"}]
    compute = [
        {"name": "low_rank_a", "kind": "matrix",
         "matrix_flops": ((dtype, 2 * rows * wa_shape[-1] * width),),
         "inputs": ("x", "weight_a"), "outputs": ("low_rank_hidden",)},
        {"name": "low_rank_b", "kind": "matrix",
         "matrix_flops": ((dtype, 2 * rows * wb_shape[-1] * wb_shape[0]),),
         "inputs": ("low_rank_hidden", "weight_b"), "outputs": ("low_rank_out",)},
        {"name": "residual", "kind": "vector",
         "vector_ops": float(rows * width),
         "inputs": ("x", "low_rank_out"), "outputs": ("y_value",)},
    ]
    return _contract(reads, writes, compute, attributes={
        "x_shape": x_shape, "low_rank": rank,
        "weight_a_shape": wa_shape, "weight_b_shape": wb_shape,
    }, basis="two_low_rank_matrix_products_plus_residual")


def _quant_four_over_six_contract(op, meta, dtype, returns):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_quant_four_over_six_requires_matrix_x")
    rows, width = prod(x_shape[:-1]), x_shape[-1]
    groups = (width + _MXFP4_SCALE_GROUP_SIZE - 1) // _MXFP4_SCALE_GROUP_SIZE
    padded_groups = (groups + 1) // 2 * 2
    payload_width = (width + 1) // 2
    reads = [_read("x", x_shape, dtype)]
    if meta.get("scale_shape") is not None:
        reads.append(_read("scale", _as_shape(meta["scale_shape"], "scale"), "fp32"))
    payload_shape = tuple(x_shape[:-1]) + (payload_width,)
    scale_shape = tuple(x_shape[:-1]) + (padded_groups // 2, 2)
    writes = [
        {**_role("mxfp4_payload", payload_shape, "uint8"), "source": "quantize"},
        {**_role("mxfp4_scale", scale_shape, "fp8_e8m0"), "source": "quantize"},
    ]
    compute = [{"name": "quantize", "kind": "vector",
                "vector_ops": float(rows * padded_groups * _MXFP4_SCALE_GROUP_SIZE
                                    * _MXFP4_QUANT_OPS_PER_ELEMENT),
                "inputs": tuple(item["name"] for item in reads),
                "outputs": ("mxfp4_payload", "mxfp4_scale")}]
    return _contract(reads, writes, compute, attributes={
        "x_shape": x_shape, "logical_width": width,
        "num_scale_groups": groups, "padded_scale_groups": padded_groups,
        "payload_width": payload_width,
        "input_dtype": dtype, "output_payload_dtype": "fp4_e2m1",
        "payload_storage_dtype": "uint8",
        "payload_bytes": prod(payload_shape), "scale_bytes": prod(scale_shape),
        "output_scale_format": "fp8_e8m0", "scale_group_size": _MXFP4_SCALE_GROUP_SIZE,
        "scale_padding": padded_groups - groups,
    }, basis="a5_fp4_32_group_payload_and_even_padded_e8m0_scale")
