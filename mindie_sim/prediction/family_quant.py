"""Per-row symmetric dynamic INT8 conversion on semantic model graphs.

[Role] Connect explicit activation producers to the existing CUDA quant plan.
[Invariants] The ordinary model binding requires the BF16 activation chain.
  One integer row owns one FP32 scale; block or fractional routing
  moments cannot be reinterpreted as this kernel's per-token contract.
  The source boundary survives subsequent generic norm/quant fusion passes.
[Timeouts & Locks] Pure bounded metadata work; no subprocesses or locks.
[Side-Effect Policy] Returns operator records without measurements or devices.
"""
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .implementations import preserve_graph
from .mechanism import PlannedRecipe
from .operators import cuda_quant


def recipes():
    return {cuda_quant.RECIPE: PlannedRecipe(cuda_quant.build)}


def lower_op(op, *, input_byte_offset=0, output_byte_offset=0, zero_absmax=False):
    # Quantized model matrices currently return BF16. Other model activation
    # types need explicit producer/cast contracts before this graph can bind.
    if op.dtype != "bf16":
        raise ModelingError("cuda_quant_model_binding_requires_bf16_producer_chain")
    metadata = dict(op.metadata)
    rows, width = op.local_shape
    if (op.layout != "contiguous" or metadata.get("output_format") != "int8"
            or metadata.get("activation_group_size") != width
            or metadata.get("activation_scale_format") != "fp32"
            or tuple(metadata.get("activation_scale_shape", ())) != (rows,)
            or metadata.get("activation_scale_bytes") != rows * 4
            or op.mte2_bytes != rows * width * dtype_bytes(op.dtype)
            or op.mte3_bytes != rows * width + rows * 4):
        raise ModelingError("cuda_quant_requires_integer_per_row_int8_fp32_scale_contract")
    source = cuda_quant.make_op({
        "rows": rows, "width": width, "dtype": op.dtype,
        "input_byte_offset": input_byte_offset, "output_byte_offset": output_byte_offset,
        "zero_absmax": zero_absmax,
    })
    if source.returns != op.returns:
        raise ModelingError("cuda_quant_output_contract_mismatch")
    return preserve_graph(source, op)
