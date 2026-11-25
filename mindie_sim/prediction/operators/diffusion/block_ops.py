"""Diffusion projection, modulation and quantization.

[Role] Build diffusion projection, modulation and quantization contracts.
[Invariants] Tensor shapes and formats determine arithmetic and materialized traffic.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from dataclasses import replace
from math import prod
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from mindie_sim.prediction.values.typed import _prepend_source_value_contract, typed_value

from .contracts import _as_shape, _bytes, _contract, _positive_int, _read, _role, _shape

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="csrc/ops/eagle_ffn/op_host/eagle_ffn_def.cpp",
        source_sha256="b742ae16a54ac6c84d1452fd29fc1dc8c3fe948c295b00c6389e5ae3e7529880",
    ),
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="mindiesd/layers/eagle_ffn_linear.py",
        source_sha256="9ac63b1ed1da06c35d079a132bdd52191061e8d9863d21656883f65a7adc1f84",
    ),
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="mindiesd/layers/adalayernorm.py",
        source_sha256="14a5f04de31d179202461a8cff3d54817aafc682537b9d4c4728706a2d758c5a",
    ),
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="csrc/plugin/adalayernorm.cpp",
        source_sha256="d862775cf5193b94b411b0c1edc7ac9e85429407b7f7326a4dbd960d951ce68f",
    ),
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="csrc/plugin/quant_four_over_six_a5.cpp",
        source_sha256="2cbf50d99848a85a90a6302b92430b4eac52cc344f50c8aec609a694a980eb38",
    ),
)

_SWIGLU_OPS_PER_ELEMENT = 6.0
_DEFAULT_FFN_ACTIVATION_OPS_PER_ELEMENT = 5.0
_ADA_NORM_OPS_PER_ELEMENT = 5
_MXFP4_SCALE_GROUP_SIZE = 32
_MXFP4_QUANT_OPS_PER_ELEMENT = 4
_QUANT46_MIN_RANK = 1
_QUANT46_MAX_RANK = 7
_ADALN_TOKENWISE_RANK = 3
_A5_SOC_FAMILY = "Ascend950"


def _parse_eagle_ffn_geometry(op, meta, dtype):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_eagle_ffn_requires_matrix_x")
    m = prod(x_shape[:-1])
    k = x_shape[-1]
    w1 = meta.get("weight1_shape")
    w2 = meta.get("weight2_shape")
    if w1 is None or w2 is None:
        return None
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
    activation = str(meta.get("activation", "gelu")).lower()
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
    return {
        "x_shape": x_shape, "m": m, "k": k, "h_total": h_total, "h_act": h_act, "n": n,
        "w1": w1, "w2": w2, "w1_layout": w1_layout, "w2_layout": w2_layout,
        "activation": activation, "bias_dtype": bias_dtype,
        "bias1_shape": bias1_shape, "bias2_shape": bias2_shape,
        "out_shape": tuple(x_shape[:-1]) + (n,),
        "up_shape": tuple(x_shape[:-1]) + (h_total,),
        "act_shape": tuple(x_shape[:-1]) + (h_act,),
    }


def _eagle_ffn_unfused_ops(op: OpSpec, meta: dict, dtype: str, soc_family: str) -> tuple[OpSpec, ...]:
    """Lower A2/A3 EagleFFN into unfused linear -> activation -> linear ops."""
    geom = _parse_eagle_ffn_geometry(op, meta, dtype)
    if geom is None:
        raise ModelingError(f"mindie_sd_custom_op_soc_unsupported:eagle_ffn:{soc_family}")
    m, k, h_total, h_act, n = geom["m"], geom["k"], geom["h_total"], geom["h_act"], geom["n"]
    activation = geom["activation"]
    activation_ops = (_SWIGLU_OPS_PER_ELEMENT if activation == "swiglu"
                      else _DEFAULT_FFN_ACTIVATION_OPS_PER_ELEMENT)
    b1_bytes = _bytes(geom["bias1_shape"], geom["bias_dtype"]) if geom["bias1_shape"] is not None else 0.0
    b2_bytes = _bytes(geom["bias2_shape"], geom["bias_dtype"]) if geom["bias2_shape"] is not None else 0.0
    up_bytes = _bytes(geom["up_shape"], dtype)
    act_bytes = _bytes(geom["act_shape"], dtype)
    out_bytes = _bytes(geom["out_shape"], dtype)
    l1_read = _bytes(geom["x_shape"], dtype) + _bytes(geom["w1"], dtype) + b1_bytes
    l1_flops = float(2 * m * h_total * k)
    l2_read = act_bytes + _bytes(geom["w2"], dtype) + b2_bytes
    l2_flops = float(2 * m * n * h_act)
    act_vec = float(m * h_act * activation_ops)
    shared_meta = {
        "mindie_sd_unfused_fallback": "eagle_ffn",
        "mindie_sd_soc_family": soc_family,
        "activation": activation,
    }
    l1_id = f"{op.op_id}.linear1"
    act_id = f"{op.op_id}.activation"
    l1_shape = (m, h_total, k)
    l2_shape = (m, n, h_act)
    l1_op = replace(
        op,
        op_id=l1_id, family="gemm", recipe="gemm",
        dtype=dtype, flops=l1_flops, vector_ops=0.0,
        mte2_bytes=l1_read, mte3_bytes=up_bytes, hbm_bytes=l1_read + up_bytes,
        matrix_flops=((dtype, l1_flops),), returns=(),
        semantic_shape=l1_shape, local_shape=l1_shape, padded_shape=l1_shape, launch_shape=l1_shape,
        metadata=tuple({
            **shared_meta, "program_kind": "matmul",
            "output_shape": geom["up_shape"], "output_format": dtype,
        }.items()),
    )
    act_family = activation if activation in {"silu", "swiglu"} else "activation"
    act_op = replace(
        op,
        op_id=act_id, deps=(l1_id,), family=act_family, recipe=activation,
        dtype=dtype, flops=0.0, vector_ops=act_vec,
        mte2_bytes=up_bytes, mte3_bytes=act_bytes, hbm_bytes=up_bytes + act_bytes,
        matrix_flops=(), returns=(),
        semantic_shape=geom["act_shape"], local_shape=geom["act_shape"],
        padded_shape=geom["act_shape"], launch_shape=geom["act_shape"],
        metadata=tuple({**shared_meta, "output_format": dtype}.items()),
    )
    out_returns = tuple(op.returns) if op.returns else ("y",)
    primary_out = out_returns[0]
    l2_meta = {
        **shared_meta, "program_kind": "matmul",
        "output_shape": geom["out_shape"], "output_format": dtype,
        "primary_output": primary_out,
    }
    _prepend_source_value_contract(
        l2_meta,
        op,
        (typed_value(
            producer=op.op_id,
            dtype=dtype,
            shape=geom["out_shape"],
            output=primary_out,
            role="activation",
            source=f"{op.op_id}:linear2",
            write_bytes=out_bytes,
            contract="mindie_sd_unfused_fallback",
        ),),
    )
    l2_op = replace(
        op,
        op_id=op.op_id, deps=(act_id,), family="gemm", recipe="gemm",
        dtype=dtype, flops=l2_flops, vector_ops=0.0,
        mte2_bytes=l2_read, mte3_bytes=out_bytes, hbm_bytes=l2_read + out_bytes,
        matrix_flops=((dtype, l2_flops),), returns=out_returns,
        semantic_shape=l2_shape, local_shape=l2_shape, padded_shape=l2_shape, launch_shape=l2_shape,
        metadata=tuple(l2_meta.items()),
    )
    return (l1_op, act_op, l2_op)


def _eagle_ffn_contract(op, meta, dtype, returns):
    geom = _parse_eagle_ffn_geometry(op, meta, dtype)
    if geom is None:
        x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
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
    x_shape, m, k = geom["x_shape"], geom["m"], geom["k"]
    h_total, h_act, n = geom["h_total"], geom["h_act"], geom["n"]
    w1, w2 = geom["w1"], geom["w2"]
    w1_layout, w2_layout = geom["w1_layout"], geom["w2_layout"]
    activation, bias_dtype = geom["activation"], geom["bias_dtype"]
    bias1_shape, bias2_shape = geom["bias1_shape"], geom["bias2_shape"]
    out_shape = geom["out_shape"]
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


def _adalayernorm_modulation_ok(shape: tuple[int, ...], x_shape: tuple[int, ...]) -> bool:
    batch, width = x_shape[0], x_shape[-1]
    if shape == (width,) or shape == x_shape:
        return True
    if len(shape) == 2 and shape == (batch, width):
        return True
    if len(shape) == len(x_shape) and shape[0] == batch and shape[-1] == width:
        return all(dim in {1, x_dim} for dim, x_dim in zip(shape[1:-1], x_shape[1:-1]))
    return False


def _adalayernorm_contract(op, meta, dtype, returns, *, soc_family: str | None = None):
    x_shape = _as_shape(meta.get("x_shape", _shape(op)), "x")
    if len(x_shape) < 2:
        raise ModelingError("mindie_sd_adalayernorm_requires_at_least_2d_x")
    rows, width = prod(x_shape[:-1]), x_shape[-1]
    scale_shape = _as_shape(meta.get("scale_shape", (width,)), "scale")
    shift_shape = _as_shape(meta.get("shift_shape", (width,)), "shift")
    if (not _adalayernorm_modulation_ok(scale_shape, x_shape)
            or not _adalayernorm_modulation_ok(shift_shape, x_shape)):
        raise ModelingError("mindie_sd_adalayernorm_modulation_width_mismatch")
    tokenwise = (
        len(x_shape) == _ADALN_TOKENWISE_RANK
        and x_shape[1] > 1
        and (scale_shape == x_shape or shift_shape == x_shape)
    )
    read_scale_shape = x_shape if tokenwise else scale_shape
    read_shift_shape = x_shape if tokenwise else shift_shape
    reads = [_read("x", x_shape, dtype), _read("scale", read_scale_shape, dtype),
             _read("shift", read_shift_shape, dtype)]
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
    return_stats = bool(meta.get("return_stats", False))
    is_a5_v2 = soc_family == _A5_SOC_FAMILY
    if return_stats or is_a5_v2:
        mean_shape = tuple(x_shape[:-1]) + (1,)
        stats_kind = "value" if return_stats else "auxiliary"
        for name in ("mean", "rstd"):
            writes.append({**_role(name, mean_shape, dtype, kind=stats_kind), "source": "norm"})
    return _contract(reads, writes, compute, attributes={
        "x_shape": x_shape, "scale_shape": scale_shape, "shift_shape": shift_shape,
        "weight_shape": weight_shape, "bias_shape": bias_shape,
        "adaln_v2_stats_materialized": bool(return_stats or is_a5_v2),
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
    if not (_QUANT46_MIN_RANK <= len(x_shape) <= _QUANT46_MAX_RANK):
        raise ModelingError(f"mindie_sd_quant_four_over_six_rank_unsupported:{len(x_shape)}")
    rows = prod(x_shape[:-1]) if len(x_shape) > 1 else 1
    width = x_shape[-1]
    if width % _MXFP4_SCALE_GROUP_SIZE != 0:
        raise ModelingError(f"mindie_sd_quant_four_over_six_last_dim_unaligned:{width}")
    groups = width // _MXFP4_SCALE_GROUP_SIZE
    padded_groups = (groups + 1) // 2 * 2
    payload_width = width // 2
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
