"""xLLM fused GDN decode.

[Role] Plan the xLLM fused GDN decode occurrence.
[Invariants] Convolution and recurrent state retain their own traffic, versions and slot effects.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

from math import prod
from typing import Any

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from ..mechanism import OperatorPlan, Work
from ..program import KernelProgram, Stage

_HEAD_DIM = 128
_CONV_KERNEL_TAPS = 4
_CONV_HISTORY_TAPS = _CONV_KERNEL_TAPS - 1
_CONV_OPS_PER_ELEMENT = 9
_QKV_NORM_OPS_PER_ELEMENT = 5
_DECLARED_READOUT_OPS_PER_ELEMENT = 5
_READOUT_RMSNORM_GATE_OPS_PER_ELEMENT = 7
_STATE_INDEX_BYTES = int(dtype_bytes("int32"))
_SSM_STATE_ELEMENT_BYTES = int(dtype_bytes("fp32"))


def _convolution_traffic(batch: int, conv_dim: int, element_bytes: float) -> tuple[float, float]:
    read = (batch * conv_dim * element_bytes
            + _CONV_KERNEL_TAPS * conv_dim * element_bytes
            + _CONV_HISTORY_TAPS * batch * conv_dim * element_bytes)
    write = batch * conv_dim * element_bytes + batch * _CONV_HISTORY_TAPS * conv_dim * element_bytes
    return read, write


def _mega_gdn_config(op: OpSpec, meta: dict[str, Any]) -> dict[str, Any]:
    head_dim = int(meta.get("head_dim", _HEAD_DIM))
    if head_dim != _HEAD_DIM:
        raise ModelingError("xllm_mega_gdn_requires_source_head_dim_128")
    qkv_shape = meta.get("qkv_shape")
    z_shape = meta.get("z_shape")
    if qkv_shape is None or z_shape is None:
        raise ModelingError("xllm_mega_gdn_requires_qkv_and_z_shapes")
    qkv_shape = tuple(int(value) for value in qkv_shape)
    z_shape = tuple(int(value) for value in z_shape)
    if len(qkv_shape) != 2 or len(z_shape) != 3:
        raise ModelingError("xllm_mega_gdn_requires_2d_qkv_and_3d_BVH_Z")
    batch, conv_dim = qkv_shape
    if z_shape[0] != batch or z_shape[2] != head_dim or z_shape[1] < 1:
        raise ModelingError("xllm_mega_gdn_z_shape_disagrees_batch_or_heads")
    v_heads = z_shape[1]
    k_heads = meta.get("num_k_heads")
    if k_heads is None:
        remaining = conv_dim // head_dim - v_heads
        if remaining < 2 or remaining % 2:
            raise ModelingError("xllm_mega_gdn_cannot_resolve_key_heads")
        k_heads = remaining // 2
    k_heads, v_heads = int(k_heads), int(v_heads)
    if (k_heads < 1 or v_heads % k_heads
            or conv_dim != (2 * k_heads + v_heads) * head_dim):
        raise ModelingError("xllm_mega_gdn_convolution_geometry_mismatch")
    lengths = tuple(int(value) for value in meta.get("sequence_tokens", (1,) * batch))
    if len(lengths) != batch or any(value < 1 for value in lengths):
        raise ModelingError("xllm_mega_gdn_requires_one_positive_length_per_batch")
    conv_state_shape = _shape_meta(meta, "conv_state_shape", (batch, _CONV_HISTORY_TAPS, conv_dim))
    ssm_state_shape = _shape_meta(meta, "ssm_state_shape",
                                  (batch, v_heads, head_dim, head_dim))
    if (len(conv_state_shape) < 2 or len(ssm_state_shape) != 4
            or ssm_state_shape[1:] != (v_heads, head_dim, head_dim)):
        raise ModelingError("xllm_mega_gdn_state_geometry_mismatch")
    b_shape = _shape_meta(meta, "b_shape", (batch, v_heads))
    a_shape = _shape_meta(meta, "a_shape", (batch, v_heads))
    a_log_shape = _shape_meta(meta, "a_log_shape", (v_heads,))
    dt_bias_shape = _shape_meta(meta, "dt_bias_shape", (v_heads,))
    norm_weight_shape = _shape_meta(meta, "norm_weight_shape", (head_dim,))
    read_indices = tuple(meta.get("read_state_indices",
                                  meta.get("state_indices", (0,) * batch)))
    write_indices = tuple(meta.get("write_state_indices", read_indices))
    if (len(read_indices) != batch or len(write_indices) != batch
            or any(type(index) is not int or index < -1
                   for index in (*read_indices, *write_indices))):
        raise ModelingError("xllm_mega_gdn_state_indices_invalid")
    if prod(b_shape) != batch * v_heads or prod(a_shape) != batch * v_heads:
        raise ModelingError("xllm_mega_gdn_gate_shapes_disagree_with_batch_heads")
    if prod(a_log_shape) != v_heads or prod(dt_bias_shape) != v_heads:
        raise ModelingError("xllm_mega_gdn_a_log_dt_bias_shapes_disagree_with_heads")
    if prod(norm_weight_shape) != head_dim:
        raise ModelingError("xllm_mega_gdn_norm_weight_shape_mismatch")
    state_layout = str(meta.get("state_layout",
                                "V,K" if meta.get("fla_ssm_state_layout", True)
                                else "K,V"))
    read_versions = meta.get("read_versions")
    write_versions = meta.get("write_versions")
    for name, values in (("read_versions", read_versions),
                         ("write_versions", write_versions)):
        if values is not None and (len(values) != batch
                                   or any(type(value) is not int for value in values)):
            raise ModelingError(f"xllm_mega_gdn_{name}_invalid")
    return {
        "batch": batch, "qkv_shape": qkv_shape, "z_shape": z_shape,
        "conv_dim": conv_dim, "k_heads": k_heads, "v_heads": v_heads,
        "head_dim": head_dim, "lengths": lengths,
        "conv_state_shape": conv_state_shape,
        "ssm_state_shape": ssm_state_shape,
        "conv_weight_shape": _shape_meta(meta, "conv_weight_shape",
                                         (_CONV_KERNEL_TAPS, conv_dim)),
        "b_shape": b_shape, "a_shape": a_shape,
        "a_log_shape": a_log_shape, "dt_bias_shape": dt_bias_shape,
        "norm_weight_shape": norm_weight_shape,
        "read_state_indices": read_indices,
        "write_state_indices": write_indices,
        "read_versions": tuple(read_versions) if read_versions is not None else None,
        "write_versions": tuple(write_versions) if write_versions is not None else None,
        "state_layout": state_layout,
        "state_capacity": meta.get("state_capacity"),
    }


def _shape_meta(meta, key, default):
    value = meta.get(key, default)
    shape = tuple(int(item) for item in value)
    if not shape or any(item < 0 for item in shape):
        raise ModelingError("xllm_mega_gdn_invalid_shape:" + key)
    return shape


def _build_mega_gdn_decode(op: OpSpec, geometry):
    config = dict(dict(op.metadata).get("xllm_mega_gdn_config") or {})
    if not config:
        raise ModelingError("xllm_mega_gdn_requires_prepared_config")
    batch, dim = config["batch"], config["head_dim"]
    k_heads, v_heads, conv_dim = (config["k_heads"], config["v_heads"],
                                  config["conv_dim"])
    bf16, f32 = dtype_bytes("bf16"), dtype_bytes("fp32")
    z_bytes = batch * v_heads * dim * bf16
    conv_read, conv_write = _convolution_traffic(batch, conv_dim, bf16)
    conv_work = Work(resources={"hbm": conv_read + conv_write,
                                "mte2": conv_read, "mte3": conv_write,
                                "vector": float(batch * conv_dim * _CONV_OPS_PER_ELEMENT)})
    gate_read = (batch * conv_dim * bf16 + dim * bf16
                 + prod(config["b_shape"]) * bf16
                 + prod(config["a_shape"]) * bf16
                 + prod(config["a_log_shape"]) * f32
                 + prod(config["dt_bias_shape"]) * f32
                 + 2 * batch * _STATE_INDEX_BYTES)
    qk_norm_work = Work(resources={
        "hbm": gate_read, "mte2": gate_read,
        "vector": float(batch * (2 * k_heads + v_heads) * dim * _QKV_NORM_OPS_PER_ELEMENT
                        + batch * (v_heads + v_heads) * 2)})
    from ..operators.linear_attention.gdn_recurrent import build as build_recurrent
    recurrent_op = OpSpec(
        op_id=op.op_id + ".recurrent",
        family="recurrent_vector", phase="decode", stage=op.stage,
        layer=op.layer, requests=op.requests,
        semantic_shape=(batch, v_heads, dim, dim),
        local_shape=(batch, v_heads, dim, dim),
        padded_shape=(batch, v_heads, dim, dim),
        launch_shape=(batch, v_heads, dim, dim),
        dtype="bf16", recipe="recurrent_scan",
        metadata=tuple({
            "sequence_tokens": config["lengths"],
            "heads": v_heads, "key_heads": k_heads,
            "dk": dim, "dv": dim, "state_dtype": "fp32",
            "state_read_indices": config["read_state_indices"],
            "state_write_indices": config["write_state_indices"],
            "state_read_versions": config["read_versions"],
            "state_write_versions": config["write_versions"],
            "state_layout": config["state_layout"],
            "state_capacity": config["state_capacity"],
        }.items()),
    )
    recurrent = build_recurrent(recurrent_op, geometry)
    attention_write = batch * v_heads * dim * bf16
    # The fused source writes the final readout only once.  F-STATE's own
    # program owns the state/readout write; the post-readout vector pass below
    # consumes Z/norm on chip and does not add a second GM write.
    readout_work = Work(resources={
        "hbm": z_bytes + dim * bf16,
        "mte2": z_bytes + dim * bf16,
        "vector": float(batch * v_heads * dim * _READOUT_RMSNORM_GATE_OPS_PER_ELEMENT)})
    total = conv_work + qk_norm_work + recurrent.work + readout_work

    def program(context):
        return KernelProgram((
            Stage("conv", context.bind(conv_work)),
            Stage("qk_norm_gate", context.bind(qk_norm_work), after=("conv",)),
            Stage("recurrent_state", after=("qk_norm_gate",),
                  program=recurrent.program(context)),
            Stage("readout_rmsnorm_zgate", context.bind(readout_work),
                  after=("recurrent_state",)),
        ))

    effects = []
    for index, (read_slot, write_slot) in enumerate(zip(
            config["read_state_indices"], config["write_state_indices"])):
        read_version = (config["read_versions"][index]
                        if config["read_versions"] is not None else None)
        write_version = (config["write_versions"][index]
                         if config["write_versions"] is not None else None)
        for target, slot, version, action in (
                ("conv_state", read_slot, read_version, "read"),
                ("conv_state", write_slot, write_version, "write"),
                ("ssm_state", read_slot, read_version, "read"),
                ("ssm_state", write_slot, write_version, "write")):
            if slot is None or slot < 0:
                continue
            span = (_CONV_HISTORY_TAPS * conv_dim * bf16 if target == "conv_state"
                    else v_heads * dim * dim * _SSM_STATE_ELEMENT_BYTES)
            effects.append({
                "target": target, "slot": int(slot), "version": version,
                "action": action, "occurrence": index,
                "owner": op.op_id, "storage": target,
                "request": (op.requests[0] if op.requests else None),
                "stage": op.stage,
                "span_bytes": span,
                "span": (int(slot) * span, (int(slot) + 1) * span),
                "ready_event": f"{op.op_id}:{target}:{slot}:ready",
                "commit_event": (f"{op.op_id}:{target}:{slot}:commit"
                                 if action == "write" else None),
            })
    return OperatorPlan(total, program, metadata={
        "xllm_mega_gdn_config": config,
        "state_mechanism": "gdn_recurrent.build_program",
        "state_program": "gdn_recurrent.program",
        "gdn_state_effects": tuple(effects),
        "state_versions": {"read": config["read_versions"],
                           "write": config["write_versions"]},
        "state_traffic_bytes": {
            "conv_read": _CONV_HISTORY_TAPS * batch * conv_dim * bf16,
            "conv_write": _CONV_HISTORY_TAPS * batch * conv_dim * bf16,
            "ssm_read": batch * v_heads * dim * dim * _SSM_STATE_ELEMENT_BYTES,
            "ssm_write": batch * v_heads * dim * dim * _SSM_STATE_ELEMENT_BYTES,
        },
        "readout_order": ("conv", "qk_norm_gate", "recurrent_state",
                          "readout_rmsnorm_zgate"),
        "two_states": ("conv_state_out", "ssm_state_out"),
        "state_indices": {
            "read": config["read_state_indices"],
            "write": config["write_state_indices"],
            "layout": config["state_layout"],
            "read_versions": config["read_versions"],
            "write_versions": config["write_versions"],
        },
        "readout_onchip": True,
        "final_output_write_bytes": attention_write,
    })
