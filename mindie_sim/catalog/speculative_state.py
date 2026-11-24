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
KDA_TP_ONLY_AXIS = "tp"
NON_NEURAL_SPEC_METHODS = frozenset({"none", "ngram", "ngram_gpu"})


def kda_state_shards(layer: Any, parallel: Any, runtime: Any = None) -> int:
    """Canonical extra KDA state shard multiplier (PCP unless ``kda_parallel_axis == 'tp'``)."""
    kind = str(getattr(layer, "attention", "kda") or "kda") if layer is not None else "kda"
    if kind != "kda":
        return ONE
    meta = (
        dict(runtime) if isinstance(runtime, Mapping)
        else dict(getattr(runtime, "metadata", ()) or ())
    )
    axis = str(meta.get("kda_parallel_axis") or "").strip().lower()
    pcp = max(ONE, int(getattr(parallel, "pcp", ONE) or ONE))
    return ONE if axis == KDA_TP_ONLY_AXIS else pcp


def kda_total_state_shards(layer: Any, parallel: Any, runtime: Any = None) -> int:
    """Canonical total KDA state shards ``tp * kda_state_shards(...)``."""
    tp = max(ONE, int(getattr(parallel, "tp", ONE) or ONE))
    return tp * kda_state_shards(layer, parallel, runtime)


def _kda_shards(layer: Any, parallel: Any, runtime: Any) -> int:
    return kda_state_shards(layer, parallel, runtime)


def gdn_record_head_bytes(dk: int, dv: int) -> int:
    """Per-token per-head GDN compact replay record: FP32 ``(k + v + beta) = (dk + dv + 1) * 4 B``."""
    return (int(dk) + int(dv) + ONE) * CORRECTION_BYTES


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
    raise ModelingError("linear_attention_conv_width_missing")


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
        raise ModelingError("linear_attention_recurrent_state_dtype_missing")
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
        raise ModelingError("linear_attention_conv_state_dtype_missing")
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
        record = float(gdn_record_head_bytes(dk, dv) * heads)
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


def _declared_speculative_policy(runtime: Any) -> tuple[str, int]:
    """Extract canonical ``(spec_method, num_speculative_tokens)`` from runtime contracts."""
    if runtime is None:
        return "", 0
    contract = getattr(runtime, "speculative_contract", None)
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if contract is None:
        raw_contract = meta.get("speculative_contract")
        if raw_contract is not None:
            contract = raw_contract
    if contract is not None:
        method = str(
            getattr(contract, "spec_method", None)
            or (contract.get("spec_method") if isinstance(contract, Mapping) else "")
            or ""
        ).strip().lower()
        k_val = int(
            getattr(contract, "num_speculative_tokens", None)
            if not isinstance(contract, Mapping)
            else contract.get("num_speculative_tokens", 0) or 0
        )
        return method, max(0, k_val)
    spine = meta.get("single_spine")
    if isinstance(spine, Mapping):
        scen = spine.get("scenario_contract")
        if isinstance(scen, Mapping):
            pol = scen.get("speculative_policy")
            if isinstance(pol, Mapping):
                method = str(pol.get("spec_method") or "").strip().lower()
                k_val = int(pol.get("num_speculative_tokens", 0) or 0)
                if method or k_val > 0:
                    return method, max(0, k_val)
    method = str(meta.get("spec_method") or meta.get("draft_mode") or "").strip().lower()
    raw_k = meta.get("num_speculative_tokens")
    if raw_k is None:
        raw_k = meta.get("draft_proposal_tokens")
    k_val = int(raw_k or 0)
    return method, max(0, k_val)


def resolve_speculative_proposals(runtime: Any, *, phase: str = PHASE_DECODE) -> int:
    """Resolve declared integer speculative proposal count ``K >= 0`` from runtime contracts."""
    if phase != PHASE_DECODE or runtime is None:
        return 0
    dspark = getattr(runtime, "dspark", None)
    if dspark is not None:
        return max(0, int(dspark.step_spec().draft_len))
    method, k_val = _declared_speculative_policy(runtime)
    if method == "none":
        return 0
    return k_val


def resolve_speculative_query_tokens(
    runtime: Any, *, view: Any = None, tokens_per_seq: int | None = None,
    phase: str = PHASE_DECODE,
) -> int:
    """Resolve the actual verify width from the scheduled view or declared contract."""
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
    return ONE + resolve_speculative_proposals(runtime, phase=phase)
