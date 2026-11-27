"""Explicit native KDA decode selection and semantic boundary replacement.

[Role] Register one recurrent source kernel for an ordinary frontend binding.
[Invariants] Caller supplies raw inputs and removes fused norm/gate members;
  this adapter never infers fused dependencies from already-normalized scans.
[Timeouts & Locks] Pure metadata operations, no I/O or locks.
[Side-Effect Policy] Returns immutable operators and local recipe mappings.
"""
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from .kda_source import view_state_declaration
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators.kda import kda_recurrent

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
IMPLEMENTATION = kda_recurrent.IMPLEMENTATION
SOURCE = kda_recurrent.SOURCE
SOURCES: tuple[SourceRef, ...] = kda_recurrent.SOURCES
CAPABILITIES = frozenset({"decode", "in_kernel_gates", "decode_recurrent_source"})
_SOURCE_KEY_WIDTH = 128
_SOURCE_VALUE_WIDTHS = (128, 256)
_SOURCE_MAX_VALUE_HEADS = 256


def recipes():
    return {kda_recurrent.RECIPE: PlannedRecipe(kda_recurrent.build)}


def validate(*, key_heads, value_heads, key_width, value_width, model):
    if (key_width != _SOURCE_KEY_WIDTH or value_width not in _SOURCE_VALUE_WIDTHS
            or not 1 <= key_heads <= value_heads <= _SOURCE_MAX_VALUE_HEADS
            or value_heads % key_heads
            or model.activation_dtype != "bf16"):
        raise ModelingError("recurrent_kda_source_head_geometry_or_precision_unsupported")


def bind_decode_scans(scans, *, lengths, heads, key_heads, dk, dv, dtype,
                      state_dtype, requests, deps, views=None,
                      state_values=None, state_slots=None, state_ready=(),
                      state_grants=(), state_decisions=(),
                      state_generation=None, **flags):
    scans = tuple(scans)
    if not scans:
        raise ModelingError("recurrent_kda_decode_scans_missing")
    if views is not None:
        contexts, acceptance = view_state_declaration(views)
        if flags.get("state_contexts") is None:
            flags["state_contexts"] = contexts
        if flags.get("state_acceptance") is None:
            flags["state_acceptance"] = acceptance
    first = scans[0]
    return kda_recurrent.make_op({
        "lengths": tuple(lengths), "heads": heads, "key_heads": key_heads,
        "dk": dk, "dv": dv, "dtype": dtype, "state_dtype": state_dtype, **flags,
    }, op_id=first.op_id, deps=deps, requests=requests, stage=first.stage,
        layer=first.layer, phase=first.phase, stream=first.stream,
        notes=first.notes, state_values=state_values, state_slots=state_slots,
        state_ready=state_ready, state_grants=state_grants,
        state_decisions=state_decisions, state_generation=state_generation)
