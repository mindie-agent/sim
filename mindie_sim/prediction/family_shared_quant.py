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

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes

from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, Stage


BINDING_SELECTOR = "shared_expert"
BINDING = "ascend_shared_gate_up_int32"
RECIPE = "ascend_shared_dequant_swiglu_quant"
KERNEL = "npu_dequant_swiglu_quant"
SOURCE_METADATA = "dequant_swiglu_quant"

SOURCE = {
    "caller": {
        "repository": "vllm-ascend-v41-private",
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
        "repository": "vllm-ascend-v41-private",
        "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
        "path": "csrc/moe/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant.h",
        "sha256": "2e516dc7332449057ea8379eebd61a0b6f2a2917907b7f88d07b6025d3cf72b2",
        "lines": "244-817",
    },
    "kernel_entry": {
        "repository": "vllm-ascend-v41-private",
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
    scale_bytes = _integer(raw.get("weight_scale_bytes", 4 * gate),
                           "weight_scale_bytes")
    if scale_bytes != 4 * gate:
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
        "clamp": 3 * count if clamp > 0 else 0,
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
        "scale_broadcast": 8 * rows,
        "quant_divide": count,
        "cast_float_to_int32": count,
        "cast_int32_to_half": count,
        "cast_half_to_int8": count,
    }


def transfer_ledger(rows, width, *, weight_scale_bytes):
    read = 4 * rows * (2 * width) + weight_scale_bytes + 4 * rows
    write = rows * width + 4 * rows
    return {"read_bytes": read, "write_bytes": write}


def _weight_scale_descriptor(metadata, gate_width):
    generic_epilogue = (
        metadata.get("matrix_output_contract") == "int32_cast_row_scale_column_scale_output_cast"
        and "weight_scale_dtype" not in metadata
    )
    fmt = (
        "bf16"
        if generic_epilogue and metadata.get("weight_scale_bytes") in {2, 2.0, 4, 4.0}
        else metadata.get("weight_scale_format", metadata.get("weight_scale_dtype", "bf16"))
    )
    aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    fmt = aliases.get(fmt, fmt)
    if fmt not in {"bf16", "fp16", "fp32"}:
        raise ModelingError(f"shared_gate_up_weight_scale_format_unsupported:{fmt}")
    declared = metadata.get("weight_scale_bytes")
    expected = gate_width * dtype_bytes(fmt)
    if declared is None or (generic_epilogue and declared in {2, 2.0, 4, 4.0}):
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
            and float(meta.get("activation_scale_bytes", 0)) == 4 * rows)


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


def _rewrite(ops, gate):
    by_id = {op.op_id: op for op in ops}
    users = {op.op_id: [] for op in ops}
    for op in ops:
        for dep in op.deps:
            users.setdefault(dep, []).append(op.op_id)
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
            or float(down_metadata.get("activation_scale_bytes", 0)) != 4 * rows):
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
        metadata, gate_width)
    weight_payload = int(metadata.get("weight_payload_bytes", gate_width * hidden))
    if weight_payload != gate_width * hidden:
        raise ModelingError("shared_gate_up_weight_payload_must_be_int8_matrix")
    dequant_scale_format = metadata.get("dequant_weight_scale_format", "fp32")
    if dequant_scale_format not in {"fp32", "float32"}:
        raise ModelingError("shared_quant_consumer_requires_fp32_weight_scale")
    gate_read = rows * hidden + weight_payload + weight_scale_bytes
    gate_write = 4 * rows * gate_width
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
        "dequant_weight_scale_bytes": 4 * gate_width,
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

    config = {
        "rows": rows, "width": width, "gate_width": gate_width,
        "activation_quant_mode": "dynamic", "swiglu_mode": 1,
        "activate_left": True, "bias": False, "group_index": False,
        "quant_scale": False, "quant_offset": False,
        "alpha": float(metadata.get("shared_alpha", 1.0)),
        "beta": float(metadata.get("shared_beta", 0.0)),
        "clamp_limit": float(metadata.get("shared_clamp_limit", 0.0)),
        "weight_scale_format": "fp32",
        "weight_scale_bytes": 4 * gate_width,
        "activation_scale_format": "fp32",
        "activation_scale_shape": (rows,),
        "activation_scale_bytes": 4 * rows,
    }
    ledger = vector_ledger(rows, width, clamp_limit=config["clamp_limit"])
    transfer = transfer_ledger(
        rows, width, weight_scale_bytes=config["weight_scale_bytes"])
    consumer_meta = {
        **dict(activation.metadata),
        "ffn_category": "shared_expert",
        "ffn_role": "activation",
        "input_format": "int32",
        "output_format": "int8",
        "activation_scale_format": "fp32",
        "activation_scale_shape": (rows,),
        "activation_scale_bytes": 4 * rows,
        "weight_scale_format": "fp32",
        "weight_scale_bytes": 4 * gate_width,
        "activation_quant_mode": "dynamic",
        "swiglu_mode": 1,
        "activate_left": True,
        "activation_alpha": config["alpha"],
        "activation_beta": config["beta"],
        "clamp_limit": config["clamp_limit"],
        "bias": False,
        "group_index": False,
        "quant_scale": False,
        "quant_offset": False,
        SOURCE_METADATA: config,
        "vector_ledger": ledger,
        "source": {**SOURCE, "graph_producers": {
            "accumulator": gate.op_id, "input_row_scales": scale_source.op_id,
            "down_consumer": down.op_id}},
        "semantic_recipe": activation.recipe,
        "semantic_family": activation.family,
    }
    consumer = vector_op(
        op_id=activation.op_id, deps=(gate.op_id, scale_source.op_id),
        stage=activation.stage, layer=activation.layer,
        requests=activation.requests, elements=rows * width, ops_per_element=0,
        dtype="int8", family="dequant_swiglu_quant", phase=activation.phase,
        recipe=RECIPE, shape=(rows, width),
        read_bytes_override=transfer["read_bytes"],
        write_bytes_override=transfer["write_bytes"],
        vector_ops_override=sum(ledger.values()),
        returns=("activation_scales",), metadata=tuple(consumer_meta.items()),
        notes=(
            "source_boundary=kernel:npu_dequant_swiglu_quant",
            "one_source_kernel_per_shared_expert_caller",
            *SOURCE["limitations"],
        ),
    )
    consumer = replace(consumer, source_boundary="kernel", layout="contiguous",
                       stream=activation.stream)
    if dict(consumer.metadata).get("activation_scale_shape") != (rows,):
        raise ModelingError("shared_quant_scale_contract_lost")

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
            result.append(consumer)
        elif quant is not None and op.op_id == quant.op_id:
            continue
        elif op.op_id == down.op_id:
            result.append(new_down)
        else:
            if quant is not None and quant.op_id in op.deps:
                raise ModelingError("shared_quant_down_quantize_has_consumer")
            result.append(op)
    return tuple(result)


def lower_shared_experts(ops, bindings):
    """Apply only the explicitly selected structural shared-expert branch."""
    from .implementations import select
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


def build(op, geometry):
    del geometry
    config = config_from_op(op)
    ledger = vector_ledger(
        config.rows, config.width, clamp_limit=config.clamp_limit)
    transfer = transfer_ledger(
        config.rows, config.width, weight_scale_bytes=config.weight_scale_bytes)
    read, write = transfer["read_bytes"], transfer["write_bytes"]
    work = Work(resources={
        "hbm": read + write, "mte2": read, "mte3": write,
        "vector": float(sum(ledger.values())),
    })

    def program(context):
        load = context.bind(Work(resources={"hbm": read, "mte2": read}))
        compute = context.bind(Work(resources={"vector": float(sum(ledger.values()))}))
        store = context.bind(Work(resources={"hbm": write, "mte3": write}))
        return KernelProgram((
            Stage("load", load),
            Stage("dequant_swiglu_quant", compute, after=("load",)),
            Stage("store", store, after=("dequant_swiglu_quant",)),
        ))

    return OperatorPlan(work, program)
