"""Runtime verify-window speculative state and compact-record residency.

[Role]
  One pure geometry for every associative linear-attention family: the active
  verify window decides how many recurrent state slots and compact trajectory
  records are resident.  Capacity, execution metadata and commit accounting
  consume this plan instead of repeating a one-query assumption.

[Invariants]
  * ``q=1`` is identical under both RecoverSSM settings.
  * ``q>1`` without replay keeps ``q*S`` checkpoint slots; with replay it keeps
    ``S`` plus ``q*R`` compact records.  The base state is one of the ``q``
    slots, never an additional copy.
  * Convolution history stays per sequence; the default rollback buffer is
    ``(kernel_history + q - 1) * local_conv_width * conv_element_bytes`` and
    ``q=1`` reproduces the historical size exactly.
  * A declared source state contract may provide separate snapshot/transient
    allocations; they are reported, not silently compressed.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  Read-only immutable records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes


ZERO = 0
ONE = 1
TWO = 2
CORRECTION_BYTES = 4
DEFAULT_RECORD_KG_DTYPE = "bf16"
PHASE_DECODE = "decode"
MODE_SINGLE = "single_query"
MODE_CHECKPOINT = "checkpoint_slots"
MODE_REPLAY = "replay_records"
LINEAR_ATTENTION_KINDS = frozenset({
    "kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet",
})


@dataclass(frozen=True)
class SpeculativeStatePlan:
    """Resident state/record bytes for one layer's active verify window."""

    query_tokens: int
    state_slots: int
    recurrent_bytes: float
    conv_bytes: float
    replay_bytes: float
    transient_bytes: float
    mode: str

    @property
    def total_bytes(self) -> float:
        return float(self.recurrent_bytes + self.conv_bytes
                     + self.replay_bytes + self.transient_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_tokens": int(self.query_tokens),
            "state_slots": int(self.state_slots),
            "recurrent_bytes": float(self.recurrent_bytes),
            "conv_bytes": float(self.conv_bytes),
            "replay_bytes": float(self.replay_bytes),
            "transient_bytes": float(self.transient_bytes),
            "total_bytes": self.total_bytes,
            "mode": self.mode,
        }


def _local_heads(layer: Any, parallel: Any) -> int:
    heads = int(getattr(layer, "linear_value_heads", ZERO)
                or getattr(layer, "linear_heads", ZERO) or ONE)
    tp = max(ONE, int(parallel.tp))
    if heads % tp != ZERO:
        raise ModelingError(f"linear_state_heads_not_divisible_by_tp:{heads}%{tp}")
    return max(ONE, heads // tp)


def _head_geometry(layer: Any) -> tuple[int, int]:
    dk = int(getattr(layer, "linear_key_head_dim", ZERO)
             or getattr(layer, "linear_head_dim", ZERO) or ONE)
    dv = int(getattr(layer, "linear_head_dim", ZERO) or ONE)
    return max(ONE, dk), max(ONE, dv)


def record_head_bytes(dk: int, dv: int, kg_dtype: str = DEFAULT_RECORD_KG_DTYPE) -> int:
    """Per-token per-head compact record: FP32 correction + two k/g arrays."""
    return int(dv) * CORRECTION_BYTES + TWO * int(dk) * int(dtype_bytes(kg_dtype))


def _conv_geometry(layer: Any, parallel: Any) -> tuple[int, int, str]:
    elements = float(getattr(layer, "conv_state_elements", ZERO) or ZERO)
    elements /= max(ONE, int(parallel.tp))
    kernel = max(ONE, int(getattr(layer, "conv_kernel", ZERO) or ONE))
    width = elements / float(kernel) if kernel else elements
    dtype = str(getattr(layer, "conv_state_dtype", "") or "bf16")
    return int(round(width)), kernel, dtype


def speculative_state_plan(
    model: Any, layer: Any, parallel: Any, *, query_tokens: int,
    recover_ssm: bool, capacity_padded: bool, state_contract: Mapping[str, Any] | None = None,
) -> SpeculativeStatePlan:
    """Resolve one layer's active verify-window state/record residency."""
    del model, capacity_padded
    kind = str(getattr(layer, "attention", ""))
    if kind not in LINEAR_ATTENTION_KINDS:
        raise ModelingError(f"speculative_state_requires_linear_attention:{kind}")
    q = int(query_tokens)
    if q < ONE:
        raise ModelingError(f"speculative_state_query_tokens_invalid:{query_tokens}")
    contract = dict(state_contract or {})
    heads = _local_heads(layer, parallel)
    dk, dv = _head_geometry(layer)
    state_dtype = str(getattr(layer, "recurrent_state_dtype", "") or "fp32")
    kg_dtype = str(contract.get("record_kg_dtype", DEFAULT_RECORD_KG_DTYPE))
    recurrent_elements = float(getattr(layer, "recurrent_state_elements", ZERO) or ZERO)
    recurrent_elements /= max(ONE, int(parallel.tp))
    state_bytes = recurrent_elements * float(dtype_bytes(state_dtype))
    conv_width, kernel, conv_dtype = _conv_geometry(layer, parallel)
    conv_element = float(dtype_bytes(conv_dtype))
    history = int(contract.get("kernel_history", kernel))
    history = max(ONE, history)
    conv_snapshots = int(contract.get("conv_snapshot_slots", ZERO) or ZERO)
    if conv_snapshots > ZERO:
        conv_bytes = float(conv_snapshots) * conv_width * history * conv_element
    else:
        conv_bytes = float(history + q - ONE) * conv_width * conv_element
    transient = float(contract.get("transient_bytes", ZERO) or ZERO)
    if q <= ONE:
        return SpeculativeStatePlan(
            q, ONE, state_bytes, conv_bytes, ZERO, transient, MODE_SINGLE)
    if not recover_ssm:
        slots = int(contract.get("base_state_slots", q) or q)
        if slots < q:
            raise ModelingError(f"speculative_state_slots_below_verify:{slots}<{q}")
        return SpeculativeStatePlan(
            q, slots, state_bytes * float(slots), conv_bytes, ZERO,
            transient, MODE_CHECKPOINT)
    record = record_head_bytes(dk, dv, kg_dtype) * float(heads)
    override = contract.get("record_bytes_per_token")
    if override is not None:
        record = float(override)
    if record < ZERO:
        raise ModelingError("speculative_state_record_bytes_invalid")
    return SpeculativeStatePlan(
        q, ONE, state_bytes, conv_bytes, record * float(q),
        transient, MODE_REPLAY)


def resolve_speculative_query_tokens(
    runtime: Any, *, view: Any = None, tokens_per_seq: int | None = None,
    phase: str = PHASE_DECODE,
) -> int:
    """Resolve the actual verify width from the scheduled view/declaration."""
    if phase != PHASE_DECODE:
        return ONE
    if view is not None:
        width = int(getattr(view, "q_len", ZERO) or ZERO)
        if width < ONE:
            raise ModelingError(f"speculative_state_view_width_invalid:{width}")
        return width
    if tokens_per_seq is not None:
        width = int(tokens_per_seq)
        if width < ONE:
            raise ModelingError(
                f"speculative_state_tokens_per_seq_invalid:{tokens_per_seq}")
        return width
    dspark = getattr(runtime, "dspark", None)
    draft_len = int(getattr(dspark, "draft_len", ZERO) or ZERO)
    if draft_len > ZERO:
        return draft_len
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    proposals = int(metadata.get("draft_proposal_tokens", ZERO) or ZERO)
    if int(getattr(runtime, "mtp_acceptance_length", ONE) or ONE) <= ONE and proposals <= ZERO:
        return ONE
    if str(metadata.get("draft_mode", "mtp")) == "dspark":
        return max(ONE, proposals)
    return max(ONE, proposals + ONE)
