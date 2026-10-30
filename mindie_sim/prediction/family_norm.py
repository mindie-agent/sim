"""Declared CUDA normalization implementations on semantic model graphs.

[Role] Bind ordinary and residual RMSNorm to their existing source planner.
[Invariants] Reduction width, weights, dtype and residual outputs are semantic
  inputs. A source selection cannot consume quantization or erase extra outputs.
[Timeouts & Locks] Pure per-operator metadata work; no locks or subprocesses.
[Side-Effect Policy] Returns records; no device initialization or observations.
"""
import math

from mindie_sim.schema.common import ModelingError
from .implementations import preserve_graph
from .mechanism import PlannedRecipe
from .operators import cuda_norm


_KERNELS = {
    "rmsnorm": cuda_norm.RMS_NORM,
    "add_rmsnorm": cuda_norm.FUSED_ADD_RMS_NORM,
}


def recipes():
    return {f"cuda_norm.{kernel}": PlannedRecipe(cuda_norm.build)
            for kernel in _KERNELS.values()}


def lower_op(op, *, aligned=True, batch_invariant=False):
    metadata = dict(op.metadata)
    if "reduction_width" not in metadata or "norm_has_weight" not in metadata:
        raise ModelingError(f"normalization_semantic_contract_required:{op.recipe}")
    width = int(metadata["reduction_width"])
    elements = math.prod(op.local_shape)
    if width < 1 or elements % width or op.layout != "contiguous":
        raise ModelingError(f"normalization_requires_contiguous_complete_rows:{op.recipe}")
    if metadata.get("activation_scale_bytes") or metadata.get("output_format", op.dtype) != op.dtype:
        raise ModelingError(f"normalization_quantization_requires_own_implementation:{op.recipe}")
    source = cuda_norm.make_op({
        "rows": elements // width, "width": width, "dtype": op.dtype,
        "kernel": _KERNELS[op.family], "has_weight": metadata["norm_has_weight"],
        "aligned": aligned, "batch_invariant": batch_invariant,
    })
    if source.returns != op.returns:
        raise ModelingError(f"normalization_output_contract_mismatch:{op.recipe}")
    return preserve_graph(source, op)
