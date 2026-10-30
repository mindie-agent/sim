"""Explicit native KDA decode selection and semantic boundary replacement.

[Role] Register one recurrent source kernel for an ordinary frontend binding.
[Invariants] Caller supplies raw inputs and removes fused norm/gate members;
  this adapter never infers fused dependencies from already-normalized scans.
[Timeouts & Locks] Pure metadata operations, no I/O or locks.
[Side-Effect Policy] Returns immutable operators and local recipe mappings.
"""
from mindie_sim.schema.common import ModelingError
from .mechanism import PlannedRecipe
from .operators import kda_recurrent

IMPLEMENTATION = kda_recurrent.IMPLEMENTATION
SOURCE = kda_recurrent.SOURCE
CAPABILITIES = frozenset({"decode", "in_kernel_gates", "decode_recurrent_source"})


def recipes():
    return {kda_recurrent.RECIPE: PlannedRecipe(kda_recurrent.build)}


def validate(*, key_heads, value_heads, key_width, value_width, model):
    if (key_width != 128 or value_width not in (128, 256)
            or not 1 <= key_heads <= value_heads <= 256 or value_heads % key_heads
            or model.activation_dtype != "bf16"):
        raise ModelingError("recurrent_kda_source_head_geometry_or_precision_unsupported")


def bind_decode_scans(scans, *, lengths, heads, key_heads, dk, dv, dtype,
                      state_dtype, requests, deps, **flags):
    scans = tuple(scans)
    if not scans:
        raise ModelingError("recurrent_kda_decode_scans_missing")
    first = scans[0]
    return kda_recurrent.make_op({
        "lengths": tuple(lengths), "heads": heads, "key_heads": key_heads,
        "dk": dk, "dv": dv, "dtype": dtype, "state_dtype": state_dtype, **flags,
    }, op_id=first.op_id, deps=deps, requests=requests, stage=first.stage,
        layer=first.layer, phase=first.phase, stream=first.stream, notes=first.notes)
