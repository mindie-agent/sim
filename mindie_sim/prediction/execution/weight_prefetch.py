"""Pass 2.5 static weight prefetch bounded by a shared on-chip pool.

[Role] Identify static projection weights with no RAW dependency on preceding
  activations and decide how many *local rank* bytes may move into a shared
  prefetch pool before the consumer's start.
[Invariants] The pool is shared by every live prefetched tile; a decision sees
  the bytes already reserved/resident and never the declared capacity alone.
  The local weight shard is identical in the baseline, the traffic ledger, the
  prefetch target and the remaining foreground read, so
  ``P + (W_rank - P) == W_rank`` exactly. Only bytes the Engine actually
  completed before the consumer start are credited; a saturated window gets 0.
[Timeouts & Locks] O(number of operator inputs); no I/O, clocks or locks.
[Side-Effect Policy] Pure decisions and Stage records; no Engine submission.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from mindie_sim.prediction.program import Stage

from mindie_sim.prediction.fusion.graph import Constraint, TypedGraph, local_nbytes

_PREFETCH_BANDWIDTH_SAFETY = 0.9
_PREFETCH_ALIGNMENT_BYTES = 256
_PREFETCH_MIN_WEIGHT_FRACTION = 0.1
PREFETCH_TASK_PRIORITY = 10


@dataclass(frozen=True)
class PrefetchPhysics:
    """Declared per-rank hardware capability, never a vendor/model lookup."""
    hbm_bytes_per_ms: float
    onchip_bytes_per_ms: float
    capacity_bytes: int
    tensor_parallel: int = 1
    safety: float = _PREFETCH_BANDWIDTH_SAFETY
    alignment: int = _PREFETCH_ALIGNMENT_BYTES
    min_fraction: float = _PREFETCH_MIN_WEIGHT_FRACTION
    priority: int = PREFETCH_TASK_PRIORITY

    def __post_init__(self):
        if self.hbm_bytes_per_ms <= 0 or self.onchip_bytes_per_ms <= 0:
            raise ValueError("prefetch_bandwidth_must_be_positive")
        if self.capacity_bytes < 0 or self.tensor_parallel < 1:
            raise ValueError("prefetch_capacity_or_tp_invalid")
        if not 0 < self.safety <= 1 or self.alignment < 1 or not 0 <= self.min_fraction < 1:
            raise ValueError("prefetch_policy_invalid")
        if self.priority < 0:
            raise ValueError("prefetch_priority_invalid")

    @property
    def speedup_ratio(self) -> float:
        return self.hbm_bytes_per_ms / self.onchip_bytes_per_ms


@dataclass(frozen=True)
class PrefetchDecision:
    op_id: str
    weight_ids: tuple[str, ...]
    weight_bytes: int
    target_bytes: int
    bytes_prefetched: int
    hbm_read_bytes: int
    onchip_read_bytes: int
    window_ms: float
    spare_bytes_per_ms: float
    estimated_gain_ms: float
    allowed: bool
    reason: str
    constraints: tuple[Constraint, ...]
    target_ms: float = 0.0
    priority: int = PREFETCH_TASK_PRIORITY
    shard_basis: str = "declared_local"
    hbm_rate_ms: float = 0.0
    onchip_rate_ms: float = 0.0
    alignment: int = _PREFETCH_ALIGNMENT_BYTES

    @property
    def prefetch_fraction(self) -> float:
        return self.bytes_prefetched / self.weight_bytes if self.weight_bytes else 0.0

    @property
    def conserved(self) -> bool:
        return self.bytes_prefetched + self.hbm_read_bytes == self.weight_bytes

    def stage(self, entry_name: str, *, name: str | None = None) -> Stage | None:
        if not self.allowed:
            return None
        return Stage(name or f"prefetch_W_{self.op_id}",
                     {"hbm": float(self.bytes_prefetched)},
                     after=(entry_name,), priority=self.priority,
                     caps={"hbm": float(self.spare_bytes_per_ms)})

    def settled(self, completed_bytes: int) -> "PrefetchDecision":
        """Credit only Engine-completed bytes before the consumer start."""
        done = max(0, min(int(completed_bytes), int(self.target_bytes)))
        done = (done // max(1, self.alignment)) * max(1, self.alignment)
        if done <= 0:
            return replace(self, bytes_prefetched=0, hbm_read_bytes=self.weight_bytes,
                           onchip_read_bytes=0, estimated_gain_ms=0.0, allowed=False,
                           reason="prefetch_did_not_complete_before_consumer")
        gain = done / self.hbm_rate_ms - done / self.onchip_rate_ms
        reason = ("prefetch_full_weight" if done >= self.weight_bytes
                  else "prefetch_partial_completion")
        return replace(self, bytes_prefetched=done,
                       hbm_read_bytes=self.weight_bytes - done,
                       onchip_read_bytes=done,
                       estimated_gain_ms=max(0.0, gain), allowed=True, reason=reason)

    def record(self) -> dict:
        return {"op_id": self.op_id, "weight_ids": list(self.weight_ids),
                "weight_bytes": self.weight_bytes,
                "target_bytes": self.target_bytes,
                "bytes_prefetched": self.bytes_prefetched,
                "hbm_read_bytes": self.hbm_read_bytes,
                "onchip_read_bytes": self.onchip_read_bytes,
                "window_ms": self.window_ms,
                "spare_bytes_per_ms": self.spare_bytes_per_ms,
                "estimated_gain_ms": self.estimated_gain_ms,
                "prefetch_fraction": self.prefetch_fraction,
                "priority": self.priority, "shard_basis": self.shard_basis,
                "allowed": self.allowed, "reason": self.reason,
                "hbm_conserved": self.conserved,
                "constraints": [c.record() for c in self.constraints]}


def static_weight_values(graph: TypedGraph, op_id: str) -> tuple[str, ...]:
    """Weight inputs of ``op_id`` with no producer and no state version."""
    op = graph.ops[op_id]
    found = []
    for value_id in op.inputs:
        value = graph.values[value_id]
        if value.role == "weight" and value.producer is None and value.state_version is None:
            found.append(value_id)
    return tuple(dict.fromkeys(found))


def raw_independent(graph: TypedGraph, op_id: str, weight_ids: tuple[str, ...]) -> bool:
    """True only when no graph path produces the static weight."""
    produced = {v for member in graph.ops.values() for v in member.outputs}
    return bool(weight_ids) and not (set(weight_ids) & produced) and all(
        graph.values[v].producer is None for v in weight_ids)


def weight_shard(graph: TypedGraph, op_id: str, *,
                 tensor_parallel: int = 1) -> tuple[tuple[str, ...], int, str]:
    """Local weight ids/bytes and the sharding provenance actually used."""
    ids = static_weight_values(graph, op_id)
    if not ids:
        return (), 0, "no_static_weight"
    local = sum(local_nbytes(graph.values[v], tensor_parallel) for v in ids)
    declared_local = all(graph.values[v].sharded for v in ids)
    basis = "declared_local" if declared_local else (
        "local_shard" if tensor_parallel == 1 else "tp_divided_once")
    return ids, int(local), basis


def _reject(op_id: str, weight_ids: tuple[str, ...], weight_bytes: int, window_ms: float,
            spare: float, reason: str, *constraints: Constraint,
            target: int = 0, priority: int = PREFETCH_TASK_PRIORITY,
            basis: str = "declared_local") -> PrefetchDecision:
    return PrefetchDecision(op_id, weight_ids, weight_bytes, target, 0, weight_bytes, 0,
                            window_ms, spare, 0.0, False, reason, tuple(constraints),
                            0.0, priority, basis)


def plan_prefetch(graph: TypedGraph, op_id: str, *, window_ms: float,
                  foreground_hbm_bytes: float, physics: PrefetchPhysics,
                  capacity_bytes: int | None = None,
                  tensor_parallel: int | None = None,
                  reserved_bytes: int = 0,
                  resident_bytes: int = 0) -> PrefetchDecision:
    """Decide the capacity target; the caller must settle actual completion."""
    if op_id not in graph.ops:
        raise ValueError(f"prefetch_operator_unknown:{op_id}")
    tp = int(tensor_parallel or physics.tensor_parallel)
    if tp < 1:
        raise ValueError("prefetch_tensor_parallel_invalid")
    weight_ids, weight_bytes, basis = weight_shard(graph, op_id, tensor_parallel=tp)
    if not weight_ids:
        return _reject(op_id, (), 0, window_ms, 0.0, "no_static_weight",
                       Constraint("static_weight", "violated",
                                  "operator has no static weight"), basis=basis)
    if not raw_independent(graph, op_id, weight_ids):
        return _reject(op_id, weight_ids, weight_bytes, window_ms, 0.0,
                       "weight_has_raw_dependency",
                       Constraint("raw_independence", "violated",
                                  "a graph producer owns the weight"), basis=basis)
    capacity = int(physics.capacity_bytes if capacity_bytes is None else capacity_bytes)
    reserved = max(0, int(reserved_bytes))
    resident = max(0, int(resident_bytes))
    free = max(0, capacity - reserved - resident)
    constraints = [
        Constraint("raw_independence", "satisfied", "weight has no producing operator"),
        Constraint("tensor_parallel_shard", "satisfied",
                   f"local W_rank={weight_bytes}B ({basis}, TP{tp})"),
    ]
    if window_ms <= 0 or not math.isfinite(window_ms):
        return _reject(op_id, weight_ids, weight_bytes, window_ms, 0.0,
                       "prefetch_window_empty", *constraints, basis=basis)
    if capacity <= 0:
        return _reject(op_id, weight_ids, weight_bytes, window_ms, 0.0,
                       "prefetch_capacity_zero",
                       *constraints,
                       Constraint("prefetch_capacity", "violated", "C=0"), basis=basis)
    if free <= 0:
        return _reject(op_id, weight_ids, weight_bytes, window_ms, 0.0,
                       "prefetch_pool_saturated",
                       *constraints,
                       Constraint("shared_prefetch_pool", "violated",
                                  f"{resident}+{reserved} live of {capacity}B"),
                       basis=basis)
    spare = physics.hbm_bytes_per_ms - foreground_hbm_bytes / window_ms
    if spare <= 0:
        return _reject(op_id, weight_ids, weight_bytes, window_ms, max(0.0, spare),
                       "hbm_saturated_no_overlap_gain", *constraints,
                       Constraint("spare_hbm_bandwidth", "violated",
                                  "foreground already saturates HBM"), basis=basis)
    budget = spare * window_ms * physics.safety
    target = min(float(weight_bytes), float(free), budget)
    target = int(max(0, min(target, float(weight_bytes))))
    target = (target // physics.alignment) * physics.alignment
    if target <= 0:
        return _reject(op_id, weight_ids, weight_bytes, window_ms, spare,
                       "prefetch_capacity_zero", *constraints,
                       Constraint("prefetch_capacity", "violated",
                                  "aligned target is empty"), basis=basis)
    fraction = target / weight_bytes if weight_bytes else 0.0
    gain = target / physics.hbm_bytes_per_ms - target / physics.onchip_bytes_per_ms
    if gain <= 0:
        return _reject(op_id, weight_ids, weight_bytes, window_ms, spare,
                       "onchip_path_not_faster", *constraints,
                       Constraint("onchip_path", "violated",
                                  "on-chip service is not faster than HBM"), basis=basis)
    if fraction < physics.min_fraction:
        reason = ("prefetch_capacity_zero" if capacity <= 0
                  else "prefetch_weight_exceeds_capacity")
        return _reject(op_id, weight_ids, weight_bytes, window_ms, spare, reason,
                       *constraints,
                       Constraint("prefetch_capacity", "violated",
                                  f"{target}B of {weight_bytes}B fits"), basis=basis)
    constraints.extend([
        Constraint("shared_prefetch_pool", "satisfied",
                   f"{target}B <= C_free={free}B (C={capacity}, "
                   f"resident={resident}, reserved={reserved})"),
        Constraint("spare_hbm_bandwidth", "satisfied",
                   f"{spare:.3f} B/ms spare over {window_ms:.6f} ms"),
        Constraint("hbm_conservation", "satisfied",
                   "prefetched bytes equal the foreground read removed"),
    ])
    decision = PrefetchDecision(
        op_id, weight_ids, weight_bytes, target, target, weight_bytes - target,
        target, window_ms, spare, gain, True,
        "prefetch_full_weight" if target >= weight_bytes else "prefetch_partial_weight",
        tuple(constraints), 0.0, physics.priority, basis,
        physics.hbm_bytes_per_ms, physics.onchip_bytes_per_ms, physics.alignment)
    return replace(decision, target_ms=target / max(1e-12, physics.hbm_bytes_per_ms))


def apply_to_work(work: tuple[tuple[str, float], ...],
                  decision: PrefetchDecision) -> tuple[tuple[str, float], ...]:
    """Move only the credited bytes from foreground HBM to the on-chip path."""
    if not decision.allowed:
        return work
    values = dict(work)
    hbm = values.get("hbm", 0.0)
    if decision.bytes_prefetched > hbm + 1e-9:
        raise ValueError("prefetch_exceeds_foreground_hbm_work")
    values["hbm"] = max(0.0, hbm - decision.bytes_prefetched)
    values["mte1"] = values.get("mte1", 0.0) + decision.bytes_prefetched
    if not decision.conserved:
        raise ValueError("prefetch_hbm_conservation_broken")
    return tuple(sorted(values.items()))


def prefetch_weight_entries(graph: TypedGraph, op_id: str, *,
                            tensor_parallel: int = 1) -> tuple[dict, ...]:
    """Typed local weight ledger for the optimizer's kernel report."""
    return tuple({"value_id": v,
                  "bytes": local_nbytes(graph.values[v], tensor_parallel),
                  "raw_bytes": graph.values[v].nbytes,
                  "sharded": graph.values[v].sharded,
                  "dtype": graph.values[v].dtype,
                  "shape": list(graph.values[v].shape)}
                 for v in static_weight_values(graph, op_id))


def residency_peak(intervals: tuple[tuple[float, float, int], ...]) -> tuple[int, tuple]:
    """Peak simultaneous resident bytes over ``(start, end, bytes)`` intervals."""
    events: list[tuple[float, int, int, str]] = []
    for start, end, amount in intervals:
        if amount <= 0:
            continue
        events.append((float(start), 0, amount, "start"))
        events.append((float(max(start, end)), 1, amount, "end"))
    live = peak = 0
    for _time, kind, amount, _label in sorted(events):
        live += amount if kind == 0 else -amount
        peak = max(peak, live)
    return peak, tuple(sorted((float(t), int(a)) for t, k, a, _ in events))
