"""Shared-expert W8A8 gate/up INT32 and dequant-SwiGLU-quant caller.

[Role] Lower distinct matrix and activation kernels from the declared caller.
[Invariants] Only shared_expert + ascend_shared_gate_up_int32 selects this path.
  The INT8 matrix writes INT32 without an invented BF16 epilogue. Its consumer
  reads original row scales and stored channel scales, applies SwiGLU once,
  and produces INT8 + dynamic FP32 row scales for down without re-quantizing.
  No model-name dispatch, source substitution or measured timing multiplier.
[Timeouts & Locks] Linear graph scans and constant arithmetic; no waits or locks.
[Side-Effect Policy] Return immutable OpSpec/Work; no devices, I/O or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Literal

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef

from mindie_sim.prediction.fusion.ports import (
    control_dependency_names,
    input_port_entries,
)
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, check_live_buffers
from mindie_sim.prediction.values.typed import rebind_op_precision
from mindie_sim.prediction.runtime.l2_residency import RawDieL2Bytes, resolve_usable_l2_bytes

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="ae923905948f360f4b7ab06df52d0789a0c9316e",
        relative_path="vllm_ascend/ops/fused_moe/shared_experts.py",
        source_sha256="f1780df8ec2fc290bfeb946e5d48edcb8c1b4aee0f50c1bd9d10291def201e77",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="ae923905948f360f4b7ab06df52d0789a0c9316e",
        relative_path="csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h",
        source_sha256="2e516dc7332449057ea8379eebd61a0b6f2a2917907b7f88d07b6025d3cf72b2",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="csrc/moe/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling.cpp",
        source_sha256="318a4dc604cf10307c8d150de6fe15705014df7bc69b7d6a8389d61f6d524d73",
    ),
)

FP32_BYTES = 4
INT32_BYTES = 4
UINT8_BYTES = 1
GENERIC_EPILOGUE_SCALE_WIDTHS = frozenset({2, 2.0, FP32_BYTES, float(FP32_BYTES)})
CLAMP_OPS_PER_ELEMENT = 3
SCALE_BROADCAST_OPS_PER_ROW = 8
# Host tiler constants (vllm-ascend dequant_swiglu_quant_tiling.cpp:41-63,617-676).
TILING_BLOCK_BYTES = 32
SWIGLU_SPLIT_FACTOR = 2
BLOCK_FP32_ELEMENTS = TILING_BLOCK_BYTES // FP32_BYTES
UB_RESERVE_BYTES = 1024
DOUBLE_BUFFER_SLOTS = 2
PERFORMANCE_CORE_LIMIT = 36
MAX_DIMY_FIXED_FP32_SLOTS = 5
SWIGLU_MODE1_TMP_BYTES_PER_COL = UINT8_BYTES + INT32_BYTES


BINDING_SELECTOR = "shared_expert"
BINDING = "ascend_shared_gate_up_int32"
RECIPE = "ascend_shared_dequant_swiglu_quant"
KERNEL = "npu_dequant_swiglu_quant"
SOURCE_METADATA = "dequant_swiglu_quant"

SOURCE = {
    "caller": {
        "repository": "vllm-project/vllm-ascend",
        "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
        "path": "vllm_ascend/ops/fused_moe/shared_experts.py",
        "sha256": "f1780df8ec2fc290bfeb946e5d48edcb8c1b4aee0f50c1bd9d10291def201e77",
        "lines": "305-350",
        "call": "npu_quant_matmul(x INT8, weight INT8, weight_scale BF16, "
                "pertoken_scale=None, output_dtype=torch.int32); "
                "npu_dequant_swiglu_quant(x INT32, weight_scale FP32, "
                "activation_scale FP32, quant_mode=1) -> INT8 + FP32 row scale",
    },
    "kernel": {
        "repository": "vllm-project/vllm-ascend",
        "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
        "path": "csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h",
        "sha256": "2e516dc7332449057ea8379eebd61a0b6f2a2917907b7f88d07b6025d3cf72b2",
        "lines": "244-817",
    },
    "kernel_entry": {
        "repository": "vllm-project/vllm-ascend",
        "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
        "path": "csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.cpp",
        "sha256": "bfc9cac8008acdc9c4b9ab3933c7a275bcb362681bd9128bbe6090ceed7456d5",
        "lines": "159-180",
    },
    "fixture": {
        "path": "profiler/tests/fixtures/mixed_step_body/a3_glm53_prefill.csv",
        "line": 2,
        "capture_sha256": "ded3a03adc7a82b0801acc6a88bb85589b8f3a3f3a4a416a1322ebfb514c614f",
        "descriptor": "x[4096,6144] INT8 ND, w[16,384,16,32] INT8 NZ, "
                      "scale[512] BF16 ND -> [4096,512] INT32 ND",
        "native_events": 75,
    },
    "scope": "W8A8 dynamic-per-token SwiGLU shared expert on the A3 shapes",
    "limitations": (
        "source_coarse_logical_ledger_does_not_claim_native_pipeline_overlap",
        "native_core_partition_and_ub_tiling_not_reconstructed",
        "weight_scale_is_read_once_logically_while_native_cores_may_refetch",
        "down_matrix_BF16_epilogue_remains_an_ordinary_matrix_dequant",
        "captured_CANN_binary_identity_not_established",
    ),
}


def recipes():
    return {RECIPE: PlannedRecipe(build)}


def tensor_contract(op):
    """Active seven-input/two-output ABI, independent of timing or dispatch."""
    meta = op["metadata"]
    config = meta.get(SOURCE_METADATA, {})
    shape = tuple(op["local_shape"])
    if (op["recipe"] != RECIPE or len(shape) != 2
            or any(type(dim) is not int or dim < 1 for dim in shape)):
        return None
    rows, width = shape
    if (config.get("rows") != rows or config.get("width") != width
            or config.get("gate_width") != 2 * width
            or meta.get("input_format") != "int32"
            or meta.get("output_format") != "int8"
            or meta.get("weight_scale_format") != "fp32"
            or config.get("weight_scale_format") != "fp32"
            or meta.get("activation_scale_format") != "fp32"
            or config.get("activation_scale_format") != "fp32"
            or tuple(meta.get("activation_scale_shape", ())) != (rows,)
            or tuple(config.get("activation_scale_shape", ())) != (rows,)
            or any(config.get(key) is not False or meta.get(key) is not False
                   for key in ("bias", "group_index", "quant_scale", "quant_offset"))):
        return None
    return (
        "dequant_swiglu_quant_active", "DequantSwigluQuant",
        ((rows, 2 * width), "int32", "ND"),
        ((2 * width,), "fp32", "ND"), ((rows,), "fp32", "ND"),
        ((rows, width), "int8", "ND"), ((rows,), "fp32", "ND"),
    )


@dataclass(frozen=True)
class Config:
    rows: int
    width: int
    gate_width: int
    clamp_limit: float
    alpha: float
    beta: float
    weight_scale_bytes: int

    @property
    def columns(self):
        return self.gate_width


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ModelingError(f"shared_quant_{name}_must_be_integer_at_least_{minimum}")
    return value


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ModelingError(f"shared_quant_{name}_must_be_finite")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ModelingError(f"shared_quant_{name}_below_{minimum}")
    return value


def config_from_op(op):
    raw = dict(op.metadata).get(SOURCE_METADATA)
    if not isinstance(raw, dict):
        raise ModelingError("shared_quant_requires_explicit_dequant_configuration")
    if raw.get("activation_quant_mode") != "dynamic":
        raise ModelingError("shared_quant_requires_dynamic_output_quantization")
    if raw.get("swiglu_mode") != 1 or raw.get("activate_left") is not True:
        raise ModelingError("shared_quant_requires_source_swiglu_mode_1_activate_left")
    for key in ("bias", "group_index", "quant_scale", "quant_offset"):
        if raw.get(key) is not False:
            raise ModelingError(f"shared_quant_source_branch_requires_absent_{key}")
    rows = _integer(raw.get("rows"), "rows")
    width = _integer(raw.get("width"), "width")
    gate = _integer(raw.get("gate_width"), "gate_width")
    if gate != 2 * width:
        raise ModelingError("shared_quant_swiglu_gate_width_must_be_twice_output_width")
    if tuple(op.local_shape) != (rows, width):
        raise ModelingError("shared_quant_output_geometry_mismatch")
    scale_bytes = _integer(raw.get("weight_scale_bytes", FP32_BYTES * gate),
                           "weight_scale_bytes")
    if scale_bytes != FP32_BYTES * gate:
        raise ModelingError("shared_quant_consumer_requires_fp32_channel_scale")
    return Config(
        rows=rows, width=width, gate_width=gate,
        clamp_limit=_number(raw.get("clamp_limit", 0.0), "clamp_limit", 0.0),
        alpha=_number(raw.get("alpha", 1.0), "alpha", 0.0),
        beta=_number(raw.get("beta", 0.0), "beta"),
        weight_scale_bytes=scale_bytes,
    )


def vector_ledger(rows, width, *, clamp_limit=0.0):
    """Logical source expression counts on [rows, 2*width] -> [rows, width]."""
    rows = _integer(rows, "ledger_rows")
    width = _integer(width, "ledger_width")
    clamp = _number(clamp_limit, "ledger_clamp", 0.0)
    gate = 2 * width
    count = rows * width
    return {
        # ComputeDequant: Cast INT32->FP32, per-channel Mul, per-token Mul.
        "dequant_cast": rows * gate,
        "channel_scale_multiply": rows * gate,
        "token_scale_multiply": rows * gate,
        # SwiGluGate: gate bias add, optional 2 gate clamps + 1 act clamp,
        # -alpha multiply, Exp, +1, Div, gate multiply.
        "gate_bias_add": count,
        "clamp": CLAMP_OPS_PER_ELEMENT * count if clamp > 0 else 0,
        "silu_alpha_multiply": count,
        "silu_exp": count,
        "silu_add_one": count,
        "silu_divide": count,
        "gate_multiply": count,
        # DynamicQuant: Abs, row max reduction, max/127 factor, 8-way
        # broadcast construction, divide by scale, three casts.
        "absolute": count,
        "row_max_reduction": rows * max(0, width - 1),
        "quant_factor": rows,
        "scale_broadcast": SCALE_BROADCAST_OPS_PER_ROW * rows,
        "quant_divide": count,
        "cast_float_to_int32": count,
        "cast_int32_to_half": count,
        "cast_half_to_int8": count,
    }


def transfer_ledger(rows, width, *, weight_scale_bytes):
    read = INT32_BYTES * rows * (2 * width) + weight_scale_bytes + FP32_BYTES * rows
    write = rows * width + FP32_BYTES * rows
    return {"read_bytes": read, "write_bytes": write}


def _weight_scale_descriptor(metadata, gate_width, *, precision_abi=None):
    generic_epilogue = (
        metadata.get("matrix_output_contract") == "int32_cast_row_scale_column_scale_output_cast"
        and "weight_scale_dtype" not in metadata
    )
    aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    abi_raw = getattr(precision_abi, "weight_scale_dtype", "") or ""
    abi_fmt = aliases.get(abi_raw, abi_raw) if abi_raw else ""
    meta_raw = metadata.get("weight_scale_format") or metadata.get("weight_scale_dtype") or ""
    meta_fmt = aliases.get(meta_raw, meta_raw) if meta_raw else ""
    if abi_fmt and meta_fmt and abi_fmt != meta_fmt:
        raise ModelingError(f"shared_gate_up_weight_scale_format_conflict:{abi_fmt}:{meta_fmt}")
    fmt = abi_fmt or meta_fmt or "bf16"
    if fmt not in {"bf16", "fp16", "fp32"}:
        raise ModelingError(f"shared_gate_up_weight_scale_format_unsupported:{fmt}")
    declared = metadata.get("weight_scale_bytes")
    unit_bytes = dtype_bytes(fmt)
    abi_bytes = int(getattr(precision_abi, "weight_scale_bytes", 0) or 0)
    if abi_bytes and abi_bytes != int(unit_bytes):
        raise ModelingError("shared_gate_up_weight_scale_bytes_disagree_with_abi")
    expected = gate_width * unit_bytes
    if declared is None or (generic_epilogue and declared in (unit_bytes, float(unit_bytes))):
        declared = expected
    if not isfinite(float(declared)) or int(declared) != expected:
        raise ModelingError("shared_gate_up_weight_scale_bytes_disagree_with_abi")
    return fmt, int(declared)


def _has_scales(op):
    return "activation_scales" in op.returns


def _scale_matches(op, rows):
    meta = dict(op.metadata)
    return (meta.get("output_format", meta.get("output_dtype")) == "int8"
            and tuple(meta.get("activation_scale_shape", ())) == (rows,)
            and float(meta.get("activation_scale_bytes", 0)) == FP32_BYTES * rows)


def _scale_source(gate, by_id):
    rows = gate.local_shape[0]
    matches = [by_id[dep] for dep in gate.deps
               if dep in by_id and _has_scales(by_id[dep]) and _scale_matches(by_id[dep], rows)]
    if len(matches) != 1:
        raise ModelingError(
            "shared_quant_requires_one_original_activation_scale_producer:"
            f"{len(matches)}")
    return matches[0]


def _roles(op):
    meta = dict(op.metadata)
    return meta.get("ffn_category"), meta.get("ffn_role")


def _build_dequant_swiglu_consumer(
    gate, activation, down, scale_source, metadata, rows, width, gate_width,
    *, extra_deps=(), extra_control_deps=(),
):
    config = {
        "rows": rows, "width": width, "gate_width": gate_width,
        "activation_quant_mode": "dynamic", "swiglu_mode": 1,
        "activate_left": True, "bias": False, "group_index": False,
        "quant_scale": False, "quant_offset": False,
        "alpha": float(metadata.get("shared_alpha", 1.0)),
        "beta": float(metadata.get("shared_beta", 0.0)),
        "clamp_limit": float(metadata.get("shared_clamp_limit", 0.0)),
        "weight_scale_format": "fp32",
        "weight_scale_bytes": FP32_BYTES * gate_width,
        "activation_scale_format": "fp32",
        "activation_scale_shape": (rows,),
        "activation_scale_bytes": FP32_BYTES * rows,
    }
    ledger = vector_ledger(rows, width, clamp_limit=config["clamp_limit"])
    transfer = transfer_ledger(rows, width, weight_scale_bytes=config["weight_scale_bytes"])
    consumer_meta = {
        **dict(activation.metadata),
        "ffn_category": "shared_expert", "ffn_role": "activation",
        "input_format": "int32", "output_format": "int8",
        "activation_scale_format": "fp32", "activation_scale_shape": (rows,),
        "activation_scale_bytes": FP32_BYTES * rows,
        "weight_scale_format": "fp32", "weight_scale_bytes": FP32_BYTES * gate_width,
        "activation_quant_mode": "dynamic", "swiglu_mode": 1, "activate_left": True,
        "activation_alpha": config["alpha"], "activation_beta": config["beta"],
        "clamp_limit": config["clamp_limit"], "bias": False, "group_index": False,
        "quant_scale": False, "quant_offset": False, SOURCE_METADATA: config,
        "vector_ledger": ledger,
        "source": {**SOURCE, "graph_producers": {
            "accumulator": gate.op_id, "input_row_scales": scale_source.op_id,
            "down_consumer": down.op_id}},
        "semantic_recipe": activation.recipe, "semantic_family": activation.family,
    }
    if extra_control_deps:
        existing_controls = tuple(consumer_meta.get("control_deps", ()) or ())
        consumer_meta["control_deps"] = tuple(dict.fromkeys((*existing_controls, *extra_control_deps)))
    consumer_deps = tuple(dict.fromkeys((
        gate.op_id, scale_source.op_id, *activation.deps, *extra_deps,
    )))
    consumer = vector_op(
        op_id=activation.op_id, deps=consumer_deps,
        stage=activation.stage, layer=activation.layer,
        requests=activation.requests, elements=rows * width, ops_per_element=0,
        dtype="int8", family="dequant_swiglu_quant", phase=activation.phase,
        recipe=RECIPE, shape=(rows, width),
        read_bytes_override=transfer["read_bytes"], write_bytes_override=transfer["write_bytes"],
        vector_ops_override=sum(ledger.values()), returns=("activation_scales",),
        metadata=tuple(consumer_meta.items()),
        notes=(
            "source_boundary=kernel:npu_dequant_swiglu_quant",
            "one_source_kernel_per_shared_expert_caller",
            *SOURCE["limitations"],
        ),
    )
    consumer = replace(consumer, source_boundary="kernel", layout="contiguous", stream=activation.stream)
    if dict(consumer.metadata).get("activation_scale_shape") != (rows,):
        raise ModelingError("shared_quant_scale_contract_lost")
    return consumer


def _rewrite(ops, gate):
    """[Cohesive-Pipeline] Single-pass graph rewrite fusing shared MX-quantized matrix producer/consumer chains."""
    by_id = {op.op_id: op for op in ops}
    users = {op.op_id: [] for op in ops}
    data_users = {op.op_id: [] for op in ops}
    for op in ops:
        controls = control_dependency_names(op, by_id)
        seen_data: set[str] = set()
        for dep in op.deps:
            users.setdefault(dep, []).append(op.op_id)
            if dep not in controls:
                data_users.setdefault(dep, []).append(op.op_id)
                seen_data.add(dep)
        for dep, _port in input_port_entries(op.metadata):
            if dep not in seen_data:
                data_users.setdefault(dep, []).append(op.op_id)
                seen_data.add(dep)
            if op.op_id not in users.setdefault(dep, []):
                users[dep].append(op.op_id)
    metadata = dict(gate.metadata)
    rows, gate_width, hidden = gate.local_shape
    if gate_width % 2:
        raise ModelingError("shared_quant_gate_width_must_be_even")
    width = gate_width // 2
    scale_source = _scale_source(gate, by_id)
    activations = [op for op in ops
                   if _roles(op) == ("shared_expert", "activation")
                   and gate.op_id in op.deps]
    if len(activations) != 1:
        raise ModelingError(
            f"shared_quant_requires_one_activation_consumer:{len(activations)}")
    activation = activations[0]
    if set(data_users.get(gate.op_id, ())) != {activation.op_id}:
        raise ModelingError("shared_quant_gate_has_other_consumers")
    if activation.family == "situ":
        raise ModelingError("shared_quant_situ_unsupported")
    if activation.family != "swiglu":
        raise ModelingError(f"shared_quant_activation_unsupported:{activation.family}")
    if tuple(activation.local_shape) != (rows, width):
        raise ModelingError("shared_quant_activation_geometry_mismatch")
    def consumes_activation(op):
        if op.deps == (activation.op_id,):
            return True
        return (len(op.deps) == 1 and op.deps[0] in by_id
                and by_id[op.deps[0]].family == "quantize"
                and activation.op_id in by_id[op.deps[0]].deps)

    downs = [op for op in ops
             if _roles(op) == ("shared_expert", "down")
             and op.family == "down_projection" and consumes_activation(op)]
    if len(downs) != 1:
        raise ModelingError(f"shared_quant_requires_one_down_consumer:{len(downs)}")
    down = downs[0]
    if down.local_shape[-1] != width:
        raise ModelingError("shared_quant_down_width_mismatch")
    down_metadata = dict(down.metadata)
    if (down_metadata.get("activation_format") != "int8"
            or tuple(down_metadata.get("activation_scale_shape", ())) != (rows,)
            or float(down_metadata.get("activation_scale_bytes", 0)) != FP32_BYTES * rows):
        raise ModelingError("shared_quant_down_requires_consumer_int8_row_scale")

    quant = None
    if down.deps == (activation.op_id,):
        pass
    elif len(down.deps) == 1 and down.deps[0] in by_id:
        candidate = by_id[down.deps[0]]
        if candidate.family == "quantize" and activation.op_id in candidate.deps:
            quant = candidate
        else:
            raise ModelingError("shared_quant_down_must_consume_activation_or_its_quantize")
    else:
        raise ModelingError("shared_quant_down_dependency_ambiguous")
    if quant is not None:
        if users.get(activation.op_id) != [quant.op_id]:
            raise ModelingError("shared_quant_activation_has_other_consumers")
        if users.get(quant.op_id) != [down.op_id]:
            raise ModelingError("shared_quant_down_quantize_has_other_consumers")
    elif users.get(activation.op_id) != [down.op_id]:
        raise ModelingError("shared_quant_activation_has_other_consumers")

    weight_scale_format, weight_scale_bytes = _weight_scale_descriptor(
        metadata, gate_width, precision_abi=gate.precision_abi)
    weight_payload = int(metadata.get("weight_payload_bytes", gate_width * hidden))
    if weight_payload != gate_width * hidden:
        raise ModelingError("shared_gate_up_weight_payload_must_be_int8_matrix")
    dequant_scale_format = metadata.get("dequant_weight_scale_format", "fp32")
    if dequant_scale_format not in {"fp32", "float32"}:
        raise ModelingError("shared_quant_consumer_requires_fp32_weight_scale")
    gate_read = rows * hidden + weight_payload + weight_scale_bytes
    gate_write = INT32_BYTES * rows * gate_width
    gate_updates = {
        **metadata,
        "output_format": "int32",
        "matrix_output_contract": "int32_quant_matmul_without_epilogue",
        "column_metadata_policy": "actual_weight_scale_only",
        "weight_scale_format": weight_scale_format,
        "weight_scale_bytes": weight_scale_bytes,
        "weight_payload_bytes": weight_payload,
        "packed_weight_bytes": weight_payload + weight_scale_bytes,
        "activation_scale_in_mte2": False,
        "activation_scale_read_by_producer": False,
        "dequant_weight_scale_format": "fp32",
        "dequant_weight_scale_bytes": FP32_BYTES * gate_width,
        "shared_gate_up_int32": True,
        "shared_quant_source": SOURCE,
        "shared_quant_limitations": SOURCE["limitations"],
    }
    gate_updates.pop("matrix_epilogue", None)
    # The original input quantizer still produces these scales for the
    # activation kernel. They are not an input slot of the INT32 matrix call.
    for key in ("activation_scale_shape", "activation_scale_format",
                "activation_scale_bytes"):
        gate_updates.pop(key, None)
    new_gate = replace(
        gate, hbm_bytes=gate_read + gate_write, mte2_bytes=gate_read,
        mte3_bytes=gate_write, vector_ops=0,
        metadata=tuple(gate_updates.items()))
    if gate.precision_abi is not None:
        new_gate_abi = replace(
            gate.precision_abi,
            output_dtype="int32",
            weight_scale_dtype=weight_scale_format,
            weight_scale_bytes=int(dtype_bytes(weight_scale_format)),
            act_group_size=0,
            act_scale_dtype="",
            act_scale_pack="",
            activation_quant_mode="none",
        )
        new_gate = rebind_op_precision(new_gate, new_gate_abi, metadata=gate_updates)

    quant_extra_deps = (
        tuple(dep for dep in quant.deps if dep != activation.op_id)
        if quant is not None else ()
    )
    act_data_deps = (
        {gate.op_id, scale_source.op_id}
        | (set(activation.deps) - set(control_dependency_names(activation, by_id)))
        | {dep for dep, _port in input_port_entries(activation.metadata)}
    )
    quant_controls = control_dependency_names(quant, by_id) if quant is not None else frozenset()
    quant_extra_controls = tuple(
        dep for dep in quant_extra_deps
        if dep in quant_controls and dep not in act_data_deps
    )
    consumer = _build_dequant_swiglu_consumer(
        gate, activation, down, scale_source, metadata, rows, width, gate_width,
        extra_deps=quant_extra_deps,
        extra_control_deps=quant_extra_controls,
    )

    down_meta = {
        **dict(down.metadata),
        "payload_producer": consumer.op_id,
        "shared_quant_dequant_chain": True,
    }
    new_down = replace(down, deps=(consumer.op_id,),
                       metadata=tuple(down_meta.items()))

    result = []
    for op in ops:
        if op.op_id == gate.op_id:
            result.append(new_gate)
        elif op.op_id == activation.op_id:
            if quant is None:
                result.append(consumer)
        elif quant is not None and op.op_id == quant.op_id:
            result.append(consumer)
        elif op.op_id == down.op_id:
            result.append(new_down)
        else:
            if quant is not None and quant.op_id in op.deps:
                raise ModelingError("shared_quant_down_quantize_has_consumer")
            result.append(op)
    return tuple(result)


def lower_shared_experts(ops, bindings):
    """Apply only the explicitly selected structural shared-expert branch."""
    from mindie_sim.prediction.implementations import select
    selected = select(bindings, BINDING_SELECTOR)
    if selected is None:
        return tuple(ops)
    current = tuple(ops)
    gates = [op for op in current if _roles(op) == ("shared_expert", "gate_up")
             and dict(op.metadata).get("program_kind") == "matmul"
             and not dict(op.metadata).get("shared_gate_up_int32")]
    for gate in gates:
        metadata = dict(gate.metadata)
        # W4 branches are a different source mechanism.  They remain
        # unselected rather than being approximated by this W8A8 ledger.
        if metadata.get("weight_format") not in {"int8", "w8a8"}:
            continue
        if metadata.get("activation_format") != "int8":
            raise ModelingError("shared_gate_up_int32_requires_int8_activations")
        current = _rewrite(current, next(
            op for op in current if op.op_id == gate.op_id))
    return current


def _tiling_plan(config: Config, geometry) -> tuple[int, int, int]:
    """Mirror DequantSwigluQuantDskTiling alignment, UB ceiling, row block & usedCoreNum."""
    if config.width % TILING_BLOCK_BYTES != 0:
        raise ModelingError(f"shared_quant_width_must_be_multiple_of_{TILING_BLOCK_BYTES}:{config.width}")
    ub = geometry.buffers.get("ub")
    if ub is None or ub <= 0:
        raise ModelingError("shared_quant_requires_ub_capacity")
    ub = int(ub)
    fixed_max_y = (
        UB_RESERVE_BYTES + TILING_BLOCK_BYTES
        + DOUBLE_BUFFER_SLOTS * TILING_BLOCK_BYTES + FP32_BYTES
        + BLOCK_FP32_ELEMENTS * BLOCK_FP32_ELEMENTS * FP32_BYTES
    )
    per_col_max_y = (
        MAX_DIMY_FIXED_FP32_SLOTS * FP32_BYTES
        + DOUBLE_BUFFER_SLOTS * SWIGLU_SPLIT_FACTOR * FP32_BYTES
        + UINT8_BYTES + SWIGLU_MODE1_TMP_BYTES_PER_COL
    )
    required_single_row_ub = fixed_max_y + per_col_max_y * config.width
    max_out_dimy = (
        ((ub - fixed_max_y) // per_col_max_y) // TILING_BLOCK_BYTES * TILING_BLOCK_BYTES
        if ub > fixed_max_y else 0
    )
    if config.width > max_out_dimy:
        check_live_buffers(
            (LiveBufferSpec(
                "dequant_swiglu_quant_ub", "ub", required_single_row_ub, 0, 1,
                owner_kind="aiv", alignment_bytes=1,
            ),),
            geometry,
            family=RECIPE,
        )
        raise ModelingError(f"shared_quant_width_exceeds_ub_limit:{config.width}:{max_out_dimy}")
    fixed_ub = (
        UB_RESERVE_BYTES + TILING_BLOCK_BYTES
        + (1 + SWIGLU_SPLIT_FACTOR) * config.width * FP32_BYTES
    )
    per_row_ub = (
        DOUBLE_BUFFER_SLOTS * (config.width * SWIGLU_SPLIT_FACTOR + BLOCK_FP32_ELEMENTS) * FP32_BYTES
        + config.width * UINT8_BYTES + FP32_BYTES
        + config.width * SWIGLU_SPLIT_FACTOR * FP32_BYTES
        + config.width * SWIGLU_MODE1_TMP_BYTES_PER_COL
        + BLOCK_FP32_ELEMENTS * FP32_BYTES
    )
    ub_factor_dimx = min(config.rows, max(1, (ub - fixed_ub) // per_row_ub))
    ub_used = fixed_ub + ub_factor_dimx * per_row_ub
    while ub_factor_dimx > 1 and ((ub_used + TILING_BLOCK_BYTES - 1) // TILING_BLOCK_BYTES) * TILING_BLOCK_BYTES > ub:
        ub_factor_dimx -= 1
        ub_used = fixed_ub + ub_factor_dimx * per_row_ub
    check_live_buffers(
        (LiveBufferSpec(
            "dequant_swiglu_quant_ub", "ub", ub_used, 0, 1,
            owner_kind="aiv", alignment_bytes=TILING_BLOCK_BYTES,
        ),),
        geometry,
        family=RECIPE,
    )
    blocks = (config.rows + ub_factor_dimx - 1) // ub_factor_dimx
    aiv_count = int(geometry.resource_units.get("vector", geometry.compute_units))
    used_cores = max(1, min(blocks, PERFORMANCE_CORE_LIMIT, aiv_count))
    return ub_factor_dimx, ub_used, used_cores


def build(op, geometry):
    config = config_from_op(op)
    ub_factor_dimx, ub_used, used_cores = _tiling_plan(config, geometry)
    ledger = vector_ledger(
        config.rows, config.width, clamp_limit=config.clamp_limit)
    transfer = transfer_ledger(
        config.rows, config.width, weight_scale_bytes=config.weight_scale_bytes)
    row_read = INT32_BYTES * config.rows * config.gate_width + FP32_BYTES * config.rows
    write = transfer["write_bytes"]
    read_pipe = read_role(geometry.resource_units, aiv=0)
    scale_bytes = config.weight_scale_bytes
    # Every active AIV loads the full channel-scale vector into its private UB;
    # subsequent cores hit shared L2 when the die has an L2 pipe with capacity.
    hierarchy = getattr(getattr(geometry, "physical_mechanisms", None), "memory_hierarchy", None)
    l2_ceiling = (
        int(hierarchy.effective_l2_bytes)
        if hierarchy is not None else int(geometry.raw_l2_bytes)
    )
    usable_l2 = (
        int(resolve_usable_l2_bytes(
            getattr(geometry, "admission_memory_view", None),
            int(getattr(geometry, "rank", None) or 0),
            fallback_raw_l2=RawDieL2Bytes(max(0, l2_ceiling)),
        ))
        if l2_ceiling > 0 else 0
    )
    scale_l2_hit = (
        used_cores > 1
        and "l2" in geometry.resource_units
        and usable_l2 >= scale_bytes
    )
    total_read_pipe = row_read + used_cores * scale_bytes
    scale_hbm = scale_bytes if scale_l2_hit else used_cores * scale_bytes
    scale_l2 = (used_cores - 1) * scale_bytes if scale_l2_hit else 0
    work_resources = {
        "hbm": row_read + scale_hbm + write,
        read_pipe: total_read_pipe,
        "mte3": write,
        "vector": float(sum(ledger.values())),
    }
    if scale_l2:
        work_resources["l2"] = scale_l2
    work = Work(resources=work_resources)
    base_rows, extra_rows = divmod(config.rows, used_cores)
    core_row_counts = tuple(base_rows + (1 if idx < extra_rows else 0) for idx in range(used_cores))

    def core_program(context, core_id, core_rows):
        aiv_units = context.resource_units.get("vector", context.core_count)
        read_place = context.owned(core_id, aiv_units, (read_pipe,))
        vec_place = context.owned(core_id, aiv_units, ("vector",))
        store_place = context.owned(core_id, aiv_units, ("mte3",))
        allocs = ()
        if "ub" in context.pool_names:
            ub_units = context.pool_units.get("ub", aiv_units)
            allocs = (context.allocate(
                f"dequant_swiglu_quant_ub{core_id}", "ub", ub_used,
                unit_group(core_id, aiv_units, ub_units)),)
        use_l2_scale = (
            scale_l2_hit and core_id > 0
            and context.resource_names.get("l2") is not None
        )
        scale_tier = "l2" if use_l2_scale else "hbm"
        n_tiles = (core_rows + ub_factor_dimx - 1) // ub_factor_dimx
        if n_tiles == 1:
            tile_read = INT32_BYTES * core_rows * config.gate_width + FP32_BYTES * core_rows
            load_res = {read_pipe: tile_read + scale_bytes, "hbm": tile_read}
            load_res[scale_tier] = load_res.get(scale_tier, 0) + scale_bytes
            tile_vec = float(sum(vector_ledger(core_rows, config.width, clamp_limit=config.clamp_limit).values()))
            tile_write = core_rows * config.width + FP32_BYTES * core_rows
            return KernelProgram((
                Stage("load", context.bind(Work(resources=load_res)),
                      acquire=allocs, placement=read_place),
                Stage("dequant_swiglu_quant", context.bind(Work(resources={"vector": tile_vec})),
                      after=("load",), placement=vec_place),
                Stage("store", context.bind(Work(resources={"hbm": tile_write, "mte3": tile_write})),
                      after=("dequant_swiglu_quant",), free=allocs, placement=store_place),
            ))

        def tile(tile_index):
            rows_in_tile = min(ub_factor_dimx, core_rows - tile_index * ub_factor_dimx)
            tile_read = INT32_BYTES * rows_in_tile * config.gate_width + FP32_BYTES * rows_in_tile
            tile_vec = float(sum(vector_ledger(rows_in_tile, config.width, clamp_limit=config.clamp_limit).values()))
            tile_write = rows_in_tile * config.width + FP32_BYTES * rows_in_tile
            return (
                Stage("load", context.bind(Work(resources={"hbm": tile_read, read_pipe: tile_read})),
                      placement=read_place),
                Stage("dequant_swiglu_quant", context.bind(Work(resources={"vector": tile_vec})),
                      after=("load",), placement=vec_place),
                Stage("store", context.bind(Work(resources={"hbm": tile_write, "mte3": tile_write})),
                      after=("dequant_swiglu_quant",), placement=store_place),
            )

        scale_work = context.bind(Work(resources={scale_tier: scale_bytes, read_pipe: scale_bytes}))
        return KernelProgram((
            Stage("scale_load", scale_work, acquire=allocs, placement=read_place),
            Stage("row_tiles", after=("scale_load",), program=PipelineProgram(n_tiles, 1, tile)),
            Stage("ub_release", after=("row_tiles",), free=allocs),
        ))

    def program(context):
        if len(core_row_counts) == 1:
            return core_program(context, 0, core_row_counts[0])
        return ParallelProgram(tuple(
            core_program(context, core_id, core_rows)
            for core_id, core_rows in enumerate(core_row_counts)
        ))

    return OperatorPlan(work, program, {
        "ub_factor_dimx": ub_factor_dimx,
        "ub_bytes_per_aiv": ub_used,
        "used_cores": used_cores,
    })
