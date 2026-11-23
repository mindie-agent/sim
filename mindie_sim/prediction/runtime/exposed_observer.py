"""Live-scope four-clock observer for the causal Engine.

[Role]
  Maintain the four ordered clocks for submitted/resolved events, serial or
  collective edges, fluid service intervals, periodic skips and terminal joins
  of one execution scope, then reduce them with the pure exposed-time helper.

[Invariants]
  * Event clocks join componentwise over real predecessors; internal waits add
    no separate duration and a resource wait names its actual blocker.
  * Periodic skip advances all four clocks by the proved cycle correspondence
    and delta; a changed attribution structure invalidates the skip.
  * Reference-counted side tables release entries once no live successor or
    scope needs them; no all-token timeline is retained.
  * Attribution is observation only: no service, rate or Engine state changes.

[Timeouts & Locks]
  Single-threaded callbacks; no locks, I/O or background work.

[Side-Effect Policy]
  Mutates only this observer's bounded scope tables.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import heapq
import re
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from ..efficiency.exposed_breakdown import (
    ExposedClocks, advance_clocks, cube_clocks,
    finish_breakdown, join_clocks, memory_clocks, roofline_clocks,
    unavailable_breakdown, vector_clocks, communication_clocks,
)
from ..efficiency.exposed_program import (
    ROLE_COMMUNICATION, ROLE_CONTROL, ROLE_COUPLED, ROLE_CUBE, ROLE_MEMORY,
    ROLE_VECTOR, StageDescriptor, declarations_from_program,
)
from .exposed_intervals import IntervalClocks, bind_interval, interval_clocks


ZERO = 0.0
ONE = 1
FIRST = 0
INT_ZERO = 0
MAX_EVENTS = 32768
MAX_SCOPES = 64
SCOPE_DEVICE_STEP = "device_step"
STREAM0_INDEX = 0
COMM_STREAM_INDEX = 2
SECONDARY_MASKER_SHARE = 0.35
MAX_MASKED_LABEL_LEN = 96
NS_PER_MS = 1_000_000
NS_PER_US = 1000.0
_COLLECTIVE_TASK_IDX_RE = re.compile(r"^c:(\d+):")


class ServiceGrantObserver:
    """Lightweight read-only ``WaveRecorder`` observer capturing per-op Engine grants."""

    __slots__ = ("current_node_idx", "task_to_node", "admit_ms_by_node", "_grant_acc", "_slot_gen")

    def __init__(self) -> None:
        self.current_node_idx: int | None = None
        self.task_to_node: dict[int, int] = {}
        self.admit_ms_by_node: dict[int, float] = {}
        self._grant_acc: dict[tuple[float, float, str, int], float] = {}
        self._slot_gen: dict[tuple[str, int, str], int] = {}

    @property
    def grants(self) -> list[tuple[float, float, str, int, float]]:
        return [(t0, t1, rname, idx, deliv) for (t0, t1, rname, idx), deliv in self._grant_acc.items()]

    def to_observed_grants(
        self,
        node_ids: Sequence[str],
        *,
        step_start_ns: int = 0,
    ) -> tuple[ObservedServiceGrant, ...]:
        """Convert accumulated Engine grants to integer-ns ``ObservedServiceGrant`` records."""
        out: list[ObservedServiceGrant] = []
        n_nodes = len(node_ids)
        for (t0_ms, t1_ms, rname, idx), deliv in self._grant_acc.items():
            if deliv <= 0.0 or t1_ms <= t0_ms or idx < 0 or idx >= n_nodes:
                continue
            s_ns = max(0, int(round(t0_ms * float(NS_PER_MS))) - step_start_ns)
            e_ns = max(s_ns, int(round(t1_ms * float(NS_PER_MS))) - step_start_ns)
            if e_ns > s_ns:
                out.append(ObservedServiceGrant(node_ids[idx], rname, s_ns, e_ns, float(deliv)))
        return tuple(out)

    def register_pipe(self, _name: str) -> int:
        return 0

    def on_submit(self, task_id: int, name: str = "", *_args: Any, **_kwargs: Any) -> None:
        idx = self.current_node_idx
        if idx is None and name:
            match = _COLLECTIVE_TASK_IDX_RE.match(str(name))
            if match is not None:
                idx = int(match.group(1))
        if idx is not None:
            self.task_to_node[int(task_id)] = idx

    def on_admit(self, task_id: int, now_ms: float, *_args: Any, **_kwargs: Any) -> None:
        idx = self.task_to_node.get(int(task_id))
        if idx is not None:
            prev = self.admit_ms_by_node.get(idx)
            if prev is None or now_ms < prev:
                self.admit_ms_by_node[idx] = float(now_ms)

    def on_service(
        self, task_id: int, physical_row: str, _owner: int,
        t0_ms: float, t1_ms: float, delivered: float, **_kwargs: Any,
    ) -> None:
        if delivered <= 0.0 or t1_ms <= t0_ms:
            return
        idx = self.task_to_node.get(int(task_id))
        if idx is not None:
            key = (float(t0_ms), float(t1_ms), str(physical_row), idx)
            self._grant_acc[key] = self._grant_acc.get(key, 0.0) + float(delivered)

    def allocate_slot_generation(self, pool: str, unit: int, name: str) -> tuple[int, int]:
        key = (str(pool), int(unit), str(name))
        gen = self._slot_gen.get(key, 0) + 1
        self._slot_gen[key] = gen
        return 0, gen

    def release_slot_generation(self, pool: str, unit: int, name: str) -> tuple[int, int]:
        return 0, self._slot_gen.get((str(pool), int(unit), str(name)), 0)

    def on_allocation(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def extract_cycle_pattern(self, *_args: Any, **_kwargs: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        return (), ()

    def on_period(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass(frozen=True)
class ObservedInterval:
    """One half-open execution interval ``[start_ns, end_ns)`` on a device stream."""

    occurrence_id: str
    stream: str
    stream_index: int
    start_ns: int
    end_ns: int
    ordinal: int = INT_ZERO
    label: str = ""
    replica: int = INT_ZERO


@dataclass(frozen=True)
class ObservedServiceGrant:
    """Delivered physical resource service slice for contention attribution."""

    occurrence_id: str
    resource: str
    start_ns: int
    end_ns: int
    delivered: float


@dataclass(frozen=True)
class StreamExposureResult:
    """Exact integer-nanosecond exposed attribution with strict conservation."""

    start_ns: int
    terminal_ns: int
    exposed_ns_by_op: dict[str, int]
    idle_or_wait_ns: int
    masked_by: dict[str, str]
    conservation_delta_ns: int


def _is_stream0(interval: ObservedInterval) -> bool:
    return int(interval.stream_index) == STREAM0_INDEX or str(interval.stream or "main") == "main"


def _trace_stream0_witness_set(
    intervals: Sequence[ObservedInterval],
    causal_preds: Mapping[str, Sequence[str]] | None,
) -> frozenset[str]:
    active = [iv for iv in intervals if iv.end_ns > iv.start_ns]
    if not active:
        return frozenset()
    by_id = {iv.occurrence_id: iv for iv in active}
    cur = max(active, key=lambda iv: (iv.end_ns, int(_is_stream0(iv)), -iv.ordinal))
    witness: set[str] = set()
    preds_map = causal_preds or {}
    while cur.occurrence_id not in witness:
        witness.add(cur.occurrence_id)
        candidates = [
            by_id[pid] for pid in preds_map.get(cur.occurrence_id, ())
            if pid in by_id and by_id[pid].end_ns <= cur.start_ns
        ]
        if not candidates:
            break
        cur = max(candidates, key=lambda iv: (iv.end_ns, int(_is_stream0(iv)), -iv.ordinal))
    return frozenset(witness)


def _pick_stream0_witness(
    active: Sequence[ObservedInterval],
    witness_set: frozenset[str],
) -> ObservedInterval:
    main_active = [iv for iv in active if _is_stream0(iv)]
    pool = main_active if main_active else list(active)
    in_witness = [iv for iv in pool if iv.occurrence_id in witness_set]
    candidates = in_witness if in_witness else pool
    return min(
        candidates,
        key=lambda iv: (
            0 if _is_stream0(iv) else (1 if int(iv.stream_index) == COMM_STREAM_INDEX else 2),
            -iv.end_ns,
            -(iv.end_ns - iv.start_ns),
            iv.ordinal,
            iv.occurrence_id,
        ),
    )


def _allocate_integer_shares(dt: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Divide ``dt`` integer nanoseconds proportionally with zero rounding drift."""
    total = sum(v for v in weights.values() if v > ZERO)
    if dt <= INT_ZERO or total <= ZERO:
        return {}
    exact = [(oid, dt * (val / total)) for oid, val in weights.items() if val > ZERO]
    floors = {oid: int(share) for oid, share in exact}
    remainder = dt - sum(floors.values())
    if remainder > INT_ZERO:
        order = sorted(exact, key=lambda item: (-(item[1] - int(item[1])), item[0]))
        for idx in range(remainder):
            oid = order[idx % len(order)][0]
            floors[oid] += ONE
    return floors


def _attribute_live_slice(
    dt: int,
    live: Sequence[ObservedInterval],
    live_grants: Sequence[ObservedServiceGrant],
    witness_set: frozenset[str],
    exposed: dict[str, int],
    mask_overlap: dict[str, dict[str, int]],
) -> None:
    """Attribute one concurrent slice ``dt`` ns across active intervals and grants."""
    witness = _pick_stream0_witness(live, witness_set)
    live_ids = {iv.occurrence_id for iv in live}
    grants_by_res: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for gr in live_grants:
        if gr.occurrence_id in live_ids:
            span = float(gr.end_ns - gr.start_ns)
            if span > 0.0:
                grants_by_res[gr.resource][gr.occurrence_id] += gr.delivered * (float(dt) / span)
    contested = [
        res for res, claimants in grants_by_res.items()
        if witness.occurrence_id in claimants and len(claimants) >= 2
    ]
    if contested:
        combined: dict[str, float] = defaultdict(float)
        for res in contested:
            claimants = grants_by_res[res]
            res_tot = sum(claimants.values())
            if res_tot > 0.0:
                for oid, amt in claimants.items():
                    combined[oid] += amt / res_tot
        shares = _allocate_integer_shares(dt, combined)
        for oid, ns in shares.items():
            exposed[oid] += ns
        for iv in live:
            if iv.occurrence_id not in shares and iv.stream_index != witness.stream_index:
                mask_overlap[iv.occurrence_id][witness.label] += dt
        return
    exposed[witness.occurrence_id] += dt
    for iv in live:
        if iv.occurrence_id != witness.occurrence_id and iv.stream_index != witness.stream_index:
            mask_overlap[iv.occurrence_id][witness.label] += dt


def attribute_stream0_intervals(
    intervals: Sequence[ObservedInterval],
    *,
    start_ns: int,
    terminal_ns: int,
    grants: Sequence[ObservedServiceGrant] = (),
    causal_preds: Mapping[str, Sequence[str]] | None = None,
) -> StreamExposureResult:
    """Assign every half-open ``[t0, t1)`` slice once via O((N+G) log(N+G)) sweep-line.

    * Stream 0 (``main``) is preferred whenever active; aux side-streams or
      collective/comm tails are exposed only when Stream 0 is idle (or while
      physically contesting a shared resource with Stream 0).
    * Contested resources divide slice duration by delivered service grants
      using exact largest-remainder integer-nanosecond conservation.
    * Uncovered gaps are assigned to ``idle_or_wait_ns`` so
      ``sum(exposed_ns_by_op.values()) + idle_or_wait_ns == terminal_ns - start_ns``.
    """
    start_i = int(start_ns)
    term_i = max(start_i, int(terminal_ns))
    exposed: dict[str, int] = {iv.occurrence_id: 0 for iv in intervals}
    mask_overlap: dict[str, dict[str, int]] = {iv.occurrence_id: defaultdict(int) for iv in intervals}
    if term_i <= start_i:
        return StreamExposureResult(start_i, term_i, exposed, 0, {iv.occurrence_id: "" for iv in intervals}, 0)

    clamped = [
        ObservedInterval(
            iv.occurrence_id, iv.stream, iv.stream_index,
            max(start_i, min(term_i, int(iv.start_ns))),
            max(start_i, min(term_i, int(iv.end_ns))),
            iv.ordinal, iv.label or iv.occurrence_id, iv.replica,
        )
        for iv in intervals
    ]
    active_ivs = sorted(
        (iv for iv in clamped if iv.end_ns > iv.start_ns),
        key=lambda iv: (iv.start_ns, iv.end_ns, iv.ordinal, iv.occurrence_id),
    )
    witness_set = _trace_stream0_witness_set(active_ivs, causal_preds)
    cuts_set: set[int] = {start_i, term_i}
    for iv in active_ivs:
        cuts_set.add(iv.start_ns)
        cuts_set.add(iv.end_ns)
    clamped_grants: list[ObservedServiceGrant] = []
    for gr in grants:
        lo = max(start_i, min(term_i, int(gr.start_ns)))
        hi = max(start_i, min(term_i, int(gr.end_ns)))
        if hi > lo and gr.delivered > 0.0:
            clamped_grants.append(ObservedServiceGrant(gr.occurrence_id, gr.resource, lo, hi, float(gr.delivered)))
            cuts_set.add(lo)
            cuts_set.add(hi)
    clamped_grants.sort(key=lambda gr: (gr.start_ns, gr.end_ns, gr.occurrence_id, gr.resource))

    cuts = sorted(cuts_set)
    idle_wait_ns = 0
    iv_ptr = gr_ptr = 0
    n_ivs, n_grants = len(active_ivs), len(clamped_grants)
    iv_heap: list[tuple[int, int]] = []
    gr_heap: list[tuple[int, int]] = []
    live_ivs: dict[int, ObservedInterval] = {}
    live_grants: dict[int, ObservedServiceGrant] = {}

    for lo, hi in zip(cuts, cuts[1:]):
        dt = hi - lo
        if dt <= 0:
            continue
        while iv_ptr < n_ivs and active_ivs[iv_ptr].start_ns <= lo:
            iv = active_ivs[iv_ptr]
            if iv.end_ns > lo:
                live_ivs[iv_ptr] = iv
                heapq.heappush(iv_heap, (iv.end_ns, iv_ptr))
            iv_ptr += 1
        while iv_heap and iv_heap[0][0] <= lo:
            _, expired_idx = heapq.heappop(iv_heap)
            live_ivs.pop(expired_idx, None)
        while gr_ptr < n_grants and clamped_grants[gr_ptr].start_ns <= lo:
            gr = clamped_grants[gr_ptr]
            if gr.end_ns > lo:
                live_grants[gr_ptr] = gr
                heapq.heappush(gr_heap, (gr.end_ns, gr_ptr))
            gr_ptr += 1
        while gr_heap and gr_heap[0][0] <= lo:
            _, expired_g = heapq.heappop(gr_heap)
            live_grants.pop(expired_g, None)

        if not live_ivs:
            idle_wait_ns += dt
            continue
        if len(live_ivs) == 1:
            only_iv = next(iter(live_ivs.values()))
            exposed[only_iv.occurrence_id] += dt
            continue
        _attribute_live_slice(
            dt,
            tuple(live_ivs.values()),
            tuple(live_grants.values()),
            witness_set,
            exposed,
            mask_overlap,
        )

    masked_by: dict[str, str] = {}
    for iv in clamped:
        dur = max(0, iv.end_ns - iv.start_ns)
        ov = mask_overlap[iv.occurrence_id]
        if dur - exposed[iv.occurrence_id] <= 0 or not ov:
            masked_by[iv.occurrence_id] = ""
            continue
        ordered = sorted(ov.items(), key=lambda item: (-item[1], item[0]))
        top_lbl, top_ov = ordered[0]
        if len(ordered) > 1 and ordered[1][1] > top_ov * SECONDARY_MASKER_SHARE:
            masked_by[iv.occurrence_id] = f"{top_lbl} + {ordered[1][0]}"[:MAX_MASKED_LABEL_LEN]
        else:
            masked_by[iv.occurrence_id] = top_lbl[:MAX_MASKED_LABEL_LEN]

    total_attributed = sum(exposed.values()) + idle_wait_ns
    delta_ns = total_attributed - (term_i - start_i)
    return StreamExposureResult(
        start_ns=start_i,
        terminal_ns=term_i,
        exposed_ns_by_op=exposed,
        idle_or_wait_ns=idle_wait_ns,
        masked_by=masked_by,
        conservation_delta_ns=delta_ns,
    )


def _short_exposed_label(node: Any) -> str:
    fam = str(getattr(node, "family", "") or "").lower()
    rec = str(getattr(node, "recipe", "") or "").lower()
    if fam in {"grouped_moe", "mc2", "fused_moe_mc2", "moe"} or "megamoe" in rec or "dispatch_ffn_combine" in rec:
        return "MoE Routed GMM / MegaMoE"
    if fam in {"allgather", "all_gather"}:
        return "Collective AllGather"
    if fam in {"indexer", "index_pool"}:
        return "Sparse Indexer"
    role = str(getattr(node, "logical_role", "") or getattr(node, "ascend_kernel", "") or getattr(node, "op_id", ""))
    return role[:MAX_MASKED_LABEL_LEN]


def attribute_engine_exposure(
    nodes: Sequence[Any],
    intervals_us: Sequence[tuple[float, float]],
    observer: ServiceGrantObserver | None,
    causal_preds: Mapping[int, tuple[int, ...]] | None,
    step_span_us: float,
    *,
    step_start_ns: int = INT_ZERO,
) -> tuple[list[float], list[str]]:
    """Attribute Engine wall time additively via ``attribute_stream0_intervals``."""
    n = len(nodes)
    if n == INT_ZERO or step_span_us <= ZERO:
        return [ZERO] * n, [""] * n
    term_ns = max(INT_ZERO, int(round(float(step_span_us) * NS_PER_US)))
    if term_ns <= INT_ZERO:
        return [ZERO] * n, [""] * n
    node_ids = tuple(str(i) for i in range(n))
    obs_intervals: list[ObservedInterval] = []
    for idx, node in enumerate(nodes):
        s_us, e_us = intervals_us[idx]
        s_ns = max(INT_ZERO, min(term_ns, int(round(float(s_us) * NS_PER_US))))
        e_ns = max(s_ns, min(term_ns, int(round(float(e_us) * NS_PER_US))))
        occ = getattr(node, "occurrence", None)
        obs_intervals.append(ObservedInterval(
            occurrence_id=node_ids[idx],
            stream=str(getattr(node, "stream", "main") or "main"),
            stream_index=int(getattr(node, "stream_index", INT_ZERO) or INT_ZERO),
            start_ns=s_ns,
            end_ns=e_ns,
            ordinal=int(getattr(node, "ordinal", idx) or idx),
            label=_short_exposed_label(node),
            replica=int(getattr(occ, "replica", INT_ZERO) or INT_ZERO),
        ))
    obs_grants = (
        observer.to_observed_grants(node_ids, step_start_ns=step_start_ns)
        if observer is not None else ()
    )
    preds_str = (
        {
            node_ids[k]: tuple(node_ids[p] for p in v if INT_ZERO <= p < n)
            for k, v in causal_preds.items()
            if INT_ZERO <= k < n
        }
        if causal_preds else None
    )
    res = attribute_stream0_intervals(
        obs_intervals, start_ns=INT_ZERO, terminal_ns=term_ns,
        grants=obs_grants, causal_preds=preds_str,
    )
    exposed_us = [float(res.exposed_ns_by_op.get(node_ids[i], INT_ZERO)) / NS_PER_US for i in range(n)]
    masked = [res.masked_by.get(node_ids[i], "") for i in range(n)]
    return exposed_us, masked


@dataclass
class _Scope:
    start_ms: float
    inputs: tuple[ExposedClocks, ...] = ()
    events: dict[Any, ExposedClocks] = field(default_factory=dict)
    event_time: dict[Any, float] = field(default_factory=dict)
    metadata: dict[Any, dict[str, Any]] = field(default_factory=dict)
    releases: dict[Any, tuple[ExposedClocks, ...]] = field(default_factory=dict)
    refcounts: Counter = field(default_factory=Counter)
    service: ExposedClocks = field(default_factory=lambda: ExposedClocks(ZERO, ZERO, ZERO, ZERO))
    control_ms: float = ZERO
    opaque_control_ms: float = ZERO
    terminals: list[Any] = field(default_factory=list)
    periodic_delta: ExposedClocks | None = None
    periodic_repeats: int = INT_ZERO


def _clock_for_declaration(declaration: Any) -> tuple[ExposedClocks, float, float]:
    if isinstance(declaration, StageDescriptor):
        control, opaque = declaration.control_ms, declaration.opaque_control_ms
    elif isinstance(declaration, Mapping):
        serial_after = tuple(declaration.get("serial_after", ()) or ())
        del serial_after
        role = str(declaration.get("role", ROLE_CONTROL))
        cube = float(declaration.get("cube_ms", ZERO) or ZERO)
        vector = float(declaration.get("vector_ms", ZERO) or ZERO)
        memory = float(declaration.get("memory_ms", ZERO) or ZERO)
        communication = float(declaration.get("communication_ms", ZERO) or ZERO)
        control = float(declaration.get("control_ms", ZERO) or ZERO)
        opaque = float(declaration.get("opaque_control_ms", ZERO) or ZERO)
        if role == ROLE_COUPLED:
            clocks = roofline_clocks(cube_ms=cube, vector_ms=vector,
                                     memory_ms=memory,
                                     communication_ms=communication)
        elif role == ROLE_CUBE:
            clocks = cube_clocks(cube)
        elif role == ROLE_VECTOR:
            clocks = vector_clocks(vector)
        elif role == ROLE_MEMORY:
            clocks = memory_clocks(memory)
        elif role == ROLE_COMMUNICATION:
            clocks = communication_clocks(communication)
        else:
            clocks = vector_clocks(vector)
        clocks, control, opaque = bind_interval(
            interval=IntervalClocks(clocks), control_ms=control,
            opaque_control_ms=opaque)
        return clocks, control, opaque
    else:
        raise ModelingError("exposed_observer_declaration_unsupported")
    role = declaration.role
    if role == ROLE_COUPLED:
        clocks = roofline_clocks(
            cube_ms=declaration.cube_ms, vector_ms=declaration.vector_ms,
            memory_ms=declaration.memory_ms,
            communication_ms=declaration.communication_ms)
    elif role == ROLE_CUBE:
        clocks = cube_clocks(declaration.cube_ms)
    elif role == ROLE_VECTOR:
        clocks = vector_clocks(declaration.vector_ms)
    elif role == ROLE_MEMORY:
        clocks = memory_clocks(declaration.memory_ms)
    elif role == ROLE_COMMUNICATION:
        clocks = communication_clocks(declaration.communication_ms)
    else:
        clocks = vector_clocks(declaration.vector_ms)
    clocks, control, opaque = bind_interval(
        interval=IntervalClocks(clocks), control_ms=control,
        opaque_control_ms=opaque)
    return clocks, control, opaque


def _edge_keys(after: Any) -> tuple[Any, ...]:
    if after is None:
        return ()
    if isinstance(after, (str, bytes)):
        return (after,)
    if isinstance(after, Mapping):
        return tuple(after.keys())
    try:
        return tuple(after)
    except TypeError:
        return (after,)


class ExposedObserver:
    """Bounded side-table observer with Stream 0 attribution and replica filtering."""

    def __init__(
        self,
        *,
        max_events: int = MAX_EVENTS,
        observed_replicas: Sequence[int] | None = None,
    ) -> None:
        self.max_events = max(ONE, int(max_events))
        self.observed_replicas: frozenset[int] | None = (
            frozenset(int(r) for r in observed_replicas)
            if observed_replicas is not None else None
        )
        self._scopes: dict[Any, _Scope] = {}
        self._active: Any = None
        self._intervals: list[ObservedInterval] = []
        self._grants: list[ObservedServiceGrant] = []
        self._peer_ready_ms: dict[Any, float] = {}
        self._peer_counts: Counter = Counter()
        self.filtered_occurrences: int = INT_ZERO

    def should_retain_replica(self, replica: int) -> bool:
        """True if full per-occurrence records are retained for ``replica``."""
        return self.observed_replicas is None or int(replica) in self.observed_replicas

    def record_peer_readiness(self, join_key: Any, replica: int, ready_ms: float) -> None:
        """Retain only lightweight collective peer readiness for non-observed replicas."""
        del replica
        prev = self._peer_ready_ms.get(join_key)
        val = float(ready_ms)
        if prev is None or val > prev:
            self._peer_ready_ms[join_key] = val
        self._peer_counts[join_key] += ONE

    def collective_peer_ready_ms(self, join_key: Any) -> float | None:
        return self._peer_ready_ms.get(join_key)

    def collective_peer_count(self, join_key: Any) -> int:
        return int(self._peer_counts.get(join_key, INT_ZERO))

    def record_interval(self, interval: ObservedInterval) -> bool:
        """Record an occurrence interval only for observed replicas."""
        if not self.should_retain_replica(interval.replica):
            self.filtered_occurrences += ONE
            return False
        self._intervals.append(interval)
        return True

    def record_service_grant(self, grant: ObservedServiceGrant, *, replica: int = INT_ZERO) -> bool:
        if not self.should_retain_replica(replica):
            return False
        self._grants.append(grant)
        return True

    def finish_stream0_exposure(
        self,
        *,
        start_ns: int,
        terminal_ns: int,
        causal_preds: Mapping[str, Sequence[str]] | None = None,
    ) -> StreamExposureResult:
        """Compute exact conserved Stream 0 exposed attribution over recorded intervals."""
        return attribute_stream0_intervals(
            self._intervals,
            start_ns=start_ns,
            terminal_ns=terminal_ns,
            grants=self._grants,
            causal_preds=causal_preds,
        )

    def begin_scope(self, scope_id: Any, start_ms: float, inputs: Any = ()) -> None:
        clocks = ()
        seeded: dict[Any, ExposedClocks] = {}
        if isinstance(inputs, Mapping):
            seeded = {key: value for key, value in inputs.items()
                      if isinstance(value, ExposedClocks)}
            clocks = tuple(seeded.values())
        elif isinstance(inputs, (list, tuple)):
            clocks = tuple(value for value in inputs
                           if isinstance(value, ExposedClocks))
        elif isinstance(inputs, ExposedClocks):
            clocks = (inputs,)
        self._scopes[scope_id] = _Scope(float(start_ms), clocks, events=seeded)
        self._active = scope_id
        if len(self._scopes) > MAX_SCOPES:
            oldest = next(iter(self._scopes))
            if oldest != scope_id:
                self._scopes.pop(oldest, None)

    def _scope(self, scope_id: Any | None = None) -> _Scope:
        key = self._active if scope_id is None else scope_id
        if key not in self._scopes:
            raise ModelingError(f"exposed_observer_scope_unknown:{key}")
        return self._scopes[key]

    def on_submit(self, event: Any, *, after: Any = (), release: Any = (),
                 declaration: Any = None) -> None:
        scope = self._scope()
        predecessors = [scope.events[key] for key in _edge_keys(after)
                        if key in scope.events]
        for key in _edge_keys(release):
            predecessors.extend(scope.releases.get(key, ()))
        duration, control, opaque = _clock_for_declaration(declaration)
        scope.events[event] = advance_clocks(predecessors, duration)
        scope.control_ms = max(scope.control_ms, control)
        scope.opaque_control_ms = max(scope.opaque_control_ms, opaque)
        for key in _edge_keys(after):
            if key in scope.events:
                scope.refcounts[key] += ONE
        if len(scope.events) > self.max_events:
            oldest = next(iter(scope.events))
            scope.events.pop(oldest, None)
            scope.event_time.pop(oldest, None)

    def on_service_interval(
        self, *, start_ms: float, end_ms: float, services: Mapping[str, float],
        offered_capacities: Mapping[str, float], blockers: Any = (),
    ) -> IntervalClocks:
        if end_ms < start_ms:
            raise ModelingError("exposed_observer_interval_regressed")
        interval = interval_clocks(
            delivered_work=services, offered_capacities=offered_capacities,
            blockers=blockers)
        scope = self._scope()
        elapsed = float(end_ms) - float(start_ms)
        clocks = interval.clocks
        if clocks.all_ms > ZERO:
            scale = elapsed / clocks.all_ms
            clocks = ExposedClocks(
                clocks.all_ms * scale, clocks.without_communication_ms * scale,
                clocks.compute_ms * scale, clocks.cube_ms * scale)
        scope.service = ExposedClocks(
            scope.service.all_ms + clocks.all_ms,
            scope.service.without_communication_ms + clocks.without_communication_ms,
            scope.service.compute_ms + clocks.compute_ms,
            scope.service.cube_ms + clocks.cube_ms)
        return interval

    def on_resolve(self, event: Any, *, after: Any = (), moment_ms: float) -> ExposedClocks:
        scope = self._scope()
        predecessors = [scope.events[key] for key in _edge_keys(after)
                        if key in scope.events]
        base = scope.events.get(event)
        clocks = join_clocks([*predecessors, *([base] if base else [])])
        scope.events[event] = clocks
        scope.event_time[event] = float(moment_ms)
        for key in _edge_keys(after):
            scope.refcounts[key] -= ONE
            if scope.refcounts[key] <= INT_ZERO:
                scope.refcounts.pop(key, None)
        return clocks

    def release(self, event: Any) -> None:
        scope = self._scope()
        clock = scope.events.get(event)
        if clock is not None:
            scope.releases[event] = (clock,)

    def on_periodic_skip(self, *, correspondence: Any, repeat: int,
                         clock_delta: Any) -> None:
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < ONE:
            raise ModelingError("exposed_observer_repeat_invalid")
        scope = self._scope()
        if isinstance(clock_delta, ExposedClocks):
            delta = clock_delta
        elif isinstance(clock_delta, Mapping):
            delta = ExposedClocks(
                float(clock_delta.get("all_ms", ZERO)),
                float(clock_delta.get("without_communication_ms", ZERO)),
                float(clock_delta.get("compute_ms", ZERO)),
                float(clock_delta.get("cube_ms", ZERO)))
        else:
            delta = ExposedClocks(float(clock_delta), float(clock_delta),
                                  float(clock_delta), float(clock_delta))
        scaled = ExposedClocks(
            delta.all_ms * repeat, delta.without_communication_ms * repeat,
            delta.compute_ms * repeat, delta.cube_ms * repeat)
        scope.periodic_delta = scaled
        scope.periodic_repeats = repeat
        pairs = correspondence.items() if isinstance(correspondence, Mapping) else correspondence
        for source, target in pairs:
            clock = scope.events.get(source)
            if clock is not None:
                scope.events[target] = ExposedClocks(
                    clock.all_ms + scaled.all_ms,
                    clock.without_communication_ms + scaled.without_communication_ms,
                    clock.compute_ms + scaled.compute_ms,
                    clock.cube_ms + scaled.cube_ms)

    def finish_scope(self, scope_id: Any, terminal: Any) -> dict[str, Any]:
        scope = self._scope(scope_id)
        terminals = _edge_keys(terminal)
        terminal_clocks = [scope.events[key] for key in terminals
                           if key in scope.events]
        if terminal_clocks:
            clocks = join_clocks(terminal_clocks)
        elif scope.service.all_ms > ZERO:
            clocks = scope.service
        else:
            return unavailable_breakdown(
                "exposed_observer_no_terminal_clock", boundary=SCOPE_DEVICE_STEP)
        duration = clocks.all_ms
        result = finish_breakdown(
            clocks, duration_ms=duration, boundary=SCOPE_DEVICE_STEP,
            control_ms=scope.control_ms,
            opaque_control_ms=scope.opaque_control_ms)
        result["exposed_breakdown_basis"]["periodic_skips"] = int(
            scope.periodic_repeats)
        return result

    def scope_events(self, scope_id: Any) -> dict[Any, ExposedClocks]:
        return dict(self._scope(scope_id).events)

    def program_scope(
        self, scope_id: Any, start_ms: float, descriptors: Sequence[Any],
        *, inputs: Any = (),
    ) -> Any:
        """Drive a serial program's declared descriptors for a compact scope."""
        self.begin_scope(scope_id, start_ms, inputs)
        previous: tuple[Any, ...] = ()
        terminal: Any = None
        for index, descriptor in enumerate(descriptors):
            event = f"{scope_id}:stage{index}"
            self.on_submit(event, after=previous, declaration=descriptor)
            previous = (event,)
            terminal = event
        return terminal
