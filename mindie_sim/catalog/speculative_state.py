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

from mindie_sim.catalog.cache_pools import _ceil_div, exact_element_bytes
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


KDA_PARALLEL_AXIS = "tp_x_pcp"


def _kda_shards(layer: Any, parallel: Any, runtime: Any) -> int:
    """Unplugged W5a socket — duplicate KDA TP x PCP calculator deleted."""
    del layer, parallel, runtime
    raise NotImplementedError(
        "LEGACY_PURGED: delegate KDA TPxPCP sharding to single canonical helper "
        "in catalog/context_parallel.py"
    )


def _local_heads(layer: Any, parallel: Any, shards: int = 1) -> int:
    heads = int(getattr(layer, "linear_value_heads", 0)
                or getattr(layer, "linear_heads", 0) or 1)
    tp = max(1, int(parallel.tp)) * max(1, int(shards))
    if heads % tp != 0:
        raise ModelingError(f"linear_state_heads_not_divisible_by_tp:{heads}%{tp}")
    return max(1, heads // tp)


def _head_geometry(layer: Any) -> tuple[int, int]:
    dk = int(getattr(layer, "linear_key_head_dim", ZERO)
             or getattr(layer, "linear_head_dim", ZERO) or ONE)
    dv = int(getattr(layer, "linear_head_dim", ZERO) or ONE)
    return max(ONE, dk), max(ONE, dv)


def record_head_bytes(dk: int, dv: int, kg_dtype: str = DEFAULT_RECORD_KG_DTYPE) -> int:
    """Per-token per-head compact record: FP32 correction + two k/g arrays."""
    return int(dv) * CORRECTION_BYTES + 2 * exact_element_bytes(int(dk), kg_dtype)


def _conv_slice_width(layer: Any, total_shards: int, local_conv: int, kernel: int) -> int:
    """Exact per-token convolution channel width on this shard."""
    del local_conv, kernel
    conv_width = int(getattr(layer, "conv_width", 0) or 0)
    if conv_width > 0:
        return _ceil_div(conv_width, total_shards)
    raise NotImplementedError(
        "LEGACY_PURGED: require explicit conv_width > 0 on every linear-attention LayerPlan "
        "(legacy synthetic conv_width==0 fallback deleted)"
    )


def speculative_state_plan(
    model: Any, layer: Any, parallel: Any, *, query_tokens: int,
    recover_ssm: bool, capacity_padded: bool, state_contract: Mapping[str, Any] | None = None,
    runtime: Any = None,
) -> SpeculativeStatePlan:
    """Resolve one layer's active verify-window state/record residency."""
    del model, capacity_padded
    kind = str(getattr(layer, "attention", ""))
    if kind not in LINEAR_ATTENTION_KINDS:
        raise ModelingError(f"speculative_state_requires_linear_attention:{kind}")
    q = int(query_tokens)
    if q < 1:
        raise ModelingError(f"speculative_state_query_tokens_invalid:{query_tokens}")
    contract = dict(state_contract or {})
    shards = _kda_shards(layer, parallel, runtime)
    heads = _local_heads(layer, parallel, shards)
    dk, dv = _head_geometry(layer)
    state_dtype = str(getattr(layer, "recurrent_state_dtype", "") or "").strip()
    if not state_dtype:
        raise NotImplementedError(
            "LEGACY_PURGED: require explicit recurrent_state_dtype on linear-attention LayerPlan"
        )
    kg_dtype = str(contract.get("record_kg_dtype", DEFAULT_RECORD_KG_DTYPE))
    total_shards = max(1, int(parallel.tp)) * shards
    recurrent_elements = _ceil_div(
        int(getattr(layer, "recurrent_state_elements", 0) or 0), total_shards,
    )
    state_bytes = float(exact_element_bytes(recurrent_elements, state_dtype))
    conv_raw = int(getattr(layer, "conv_state_elements", 0) or 0)
    local_conv = _ceil_div(conv_raw, total_shards)
    kernel = max(1, int(getattr(layer, "conv_kernel", 0) or 1))
    conv_dtype = str(getattr(layer, "conv_state_dtype", "") or "").strip()
    if not conv_dtype:
        raise NotImplementedError(
            "LEGACY_PURGED: require explicit conv_state_dtype on linear-attention LayerPlan"
        )
    slice_width = _conv_slice_width(layer, total_shards, local_conv, kernel)
    default_history = max(1, kernel - 1) if (
        int(getattr(layer, "conv_width", 0) or 0) > 0
        or (local_conv > 0 and local_conv % max(1, kernel - 1) == 0)
    ) else kernel
    history = max(1, int(contract.get("kernel_history", default_history)))
    conv_snapshots = int(contract.get("conv_snapshot_slots", 0) or 0)
    base_conv_elements = local_conv if history == default_history else slice_width * history
    if conv_snapshots > 0:
        conv_bytes = float(exact_element_bytes(conv_snapshots * base_conv_elements, conv_dtype))
    else:
        rollback_elements = (q - 1) * slice_width if q > 1 else 0
        conv_bytes = float(exact_element_bytes(base_conv_elements + rollback_elements, conv_dtype))
    transient = float(contract.get("transient_bytes", 0) or 0)
    if q <= 1:
        return SpeculativeStatePlan(
            q, 1, state_bytes, conv_bytes, 0, transient, MODE_SINGLE)
    if not recover_ssm:
        slots = int(contract.get("base_state_slots", q) or q)
        if slots < q:
            raise ModelingError(f"speculative_state_slots_below_verify:{slots}<{q}")
        return SpeculativeStatePlan(
            q, slots, state_bytes * float(slots), conv_bytes, 0,
            transient, MODE_CHECKPOINT)
    if kind == "gdn":
        raise NotImplementedError(
            "LEGACY_PURGED: define gdn_record_head_bytes in L1 catalog/speculative_state.py "
            "(L1->L3 upward import from prediction.operators deleted)"
        )
    else:
        record = record_head_bytes(dk, dv, kg_dtype) * float(heads)
    override = contract.get("record_bytes_per_token")
    if override is not None:
        record = float(override)
    if record < 0:
        raise ModelingError("speculative_state_record_bytes_invalid")
    return SpeculativeStatePlan(
        q, 1, state_bytes, conv_bytes, record * float(q),
        transient, MODE_REPLAY)


def runtime_recover_ssm(runtime: Any) -> bool:
    """Whether the resolved runtime keeps compact RecoverSSM records."""
    if runtime is None:
        return False
    if bool(getattr(runtime, "recover_ssm", False)):
        return True
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    return metadata.get("recover_ssm") is True


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
    if dspark is not None:
        # DSpark verifies the confidence-truncated proposal prefix plus the bonus token.
        return ONE + int(dspark.step_spec().draft_len)
    raise NotImplementedError(
        "LEGACY_PURGED: read verify width directly from SpeculativeExecutionContract / "
        "DraftStepSchedule (ad-hoc metadata draft_proposal_tokens/mtp_acceptance_length purged)"
    )
