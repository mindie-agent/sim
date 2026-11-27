"""Python reference and C11 ctypes parity harness for prepared-row epoch solving.

[Role] Provide test-only ``PreparedTermsRunner`` and Python prepared-row
  weighted max-min epoch helpers (``PreparedCapacityRow``, ``PreparedTerm``,
  ``PreparedDelivery``, ``PreparedProgress``, ``allocate_prepared_epoch``,
  ``integrate_prepared_epoch``) verifying bit-exact / bounded-ULP parity with
  ``_engine_accel_terms.h`` outside production ``mindie_sim/`` modules.
[Invariants] Prepared-row conservation holds: every flow consumes at most
  ``z_left`` and delivers ``sum(dz) * work_per_z`` to declared counters with
  residual snapping within ``RESIDUAL_SNAP_ULPS`` imported from
  ``mindie_sim.prediction.runtime.prepared_terms``.
[Timeouts & Locks] Bounded progressive-filling loops (``guard <= n + 1``) and
  caller-bounded C-ABI calls; no threads or network I/O.
[Side-Effect Policy] Mutates only caller-owned C11 contexts and explicit
  ``z_left`` / ``counters`` lists passed to ``integrate_prepared_epoch``.
"""
from __future__ import annotations

from collections.abc import Sequence
import ctypes
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.prediction.runtime.accel_terms import (
    MsimAccess,
    MsimCapacityRow,
    MsimDelivery,
    MsimDep,
    MsimHbmBus,
    MsimProgress,
    MsimSlotGen,
    MsimTerm,
    bind_terms_abi,
    verify_terms_abi,
)
from mindie_sim.prediction.runtime.memory_physics import (
    HBM_DIR_READ,
    HBM_DIR_WRITE,
    HbmBusSpec,
)
from mindie_sim.prediction.runtime.prepared_terms import RESIDUAL_SNAP_ULPS
from mindie_sim.schema.common import ModelingError

POLICY_FAIR = 0
POLICY_FIFO = 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1


@dataclass(slots=True)
class PreparedCapacityRow:
    """One normalized capacity row matching C11 ``msim_capacity_row``."""

    row_id: int
    scope: int = 0
    unit: int = 0
    policy: int = POLICY_FAIR
    capacity: float = 1.0
    fifo_owner_seq: int | None = None


@dataclass(frozen=True, slots=True)
class PreparedTerm:
    """One nonnegative ``ms_per_z`` coefficient linking a flow to a capacity row."""

    row: int
    flow: int
    ms_per_z: float


@dataclass(frozen=True, slots=True)
class PreparedDelivery:
    """One counter delivery coefficient ``work_per_z`` for a flow."""

    counter: int
    flow: int
    work_per_z: float


@dataclass(slots=True)
class PreparedProgress:
    """Mutable progress cursor for one flow matching C11 ``msim_progress``."""

    sequence: int
    task: int
    priority: int
    placement_id: int
    member_map: int
    z_left: float
    weight: float
    max_dz_ms: float = math.inf


def allocate_prepared_epoch(
    rows: Sequence[PreparedCapacityRow],
    progs: Sequence[PreparedProgress],
    terms: Sequence[PreparedTerm],
) -> list[float]:
    """Reference prepared-row weighted max-min matching C11 ``msim_allocate_epoch``."""
    n_rows = len(rows)
    n_progs = len(progs)
    dz = [0.0] * n_progs
    if n_rows == 0 or n_progs == 0:
        return dz

    coalesced: dict[tuple[int, int], float] = {}
    for term in terms:
        if term.ms_per_z > 0.0:
            key = (term.flow, term.row)
            coalesced[key] = coalesced.get(key, 0.0) + float(term.ms_per_z)

    order = sorted(
        range(n_progs),
        key=lambda f: (progs[f].priority, progs[f].sequence, progs[f].placement_id, progs[f].member_map, f),
    )
    flow_terms: list[list[tuple[int, float]]] = [[] for _ in range(n_progs)]
    for (flow, row), coeff in sorted(coalesced.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        flow_terms[flow].append((row, coeff))

    fifo_winner: dict[int, int] = {}
    for row_idx, row_obj in enumerate(rows):
        if row_obj.policy != POLICY_FIFO or row_obj.fifo_owner_seq is None:
            continue
        incumbent_seq = row_obj.fifo_owner_seq
        for f in order:
            if progs[f].sequence == incumbent_seq and progs[f].z_left > 0.0 and any(
                r == row_idx and coeff > 0.0 for r, coeff in flow_terms[f]
            ):
                fifo_winner[row_idx] = f
                break
        if row_idx not in fifo_winner:
            row_obj.fifo_owner_seq = None

    rem_cap = [float(r.capacity) for r in rows]
    by_prio: dict[int, list[int]] = {}
    for f in order:
        by_prio.setdefault(progs[f].priority, []).append(f)

    for prio in sorted(by_prio):
        free: list[int] = []
        for f in by_prio[prio]:
            p = progs[f]
            if p.z_left <= 0.0 or p.weight <= 0.0 or p.max_dz_ms <= 0.0:
                continue
            blocked = False
            for row, coeff in flow_terms[f]:
                if coeff > 0.0:
                    if rows[row].policy == POLICY_FIFO and row in fifo_winner and fifo_winner[row] != f:
                        blocked = True
                        break
                    if rem_cap[row] <= 0.0:
                        blocked = True
                        break
            if not blocked:
                for row, coeff in flow_terms[f]:
                    if coeff > 0.0 and rows[row].policy == POLICY_FIFO and row not in fifo_winner:
                        fifo_winner[row] = f
                free.append(f)
        guard = 0
        while free and guard <= n_progs + 1:
            guard += 1
            slope = [0.0] * n_rows
            for f in free:
                w_f = progs[f].weight
                for row, coeff in flow_terms[f]:
                    slope[row] += coeff * w_f
            delta = math.inf
            bounds = [math.inf] * n_rows
            for j in range(n_rows):
                if slope[j] > 0.0:
                    b = rem_cap[j] / slope[j]
                    bounds[j] = b
                    if b < delta:
                        delta = b
            for f in free:
                rem_v = (progs[f].max_dz_ms - dz[f]) / progs[f].weight
                if rem_v < delta:
                    delta = rem_v
            if not math.isfinite(delta) or delta < 0.0:
                break
            for f in free:
                dz[f] += delta * progs[f].weight
            saturated: set[int] = set()
            for j in range(n_rows):
                if slope[j] > 0.0:
                    nv = rem_cap[j] - delta * slope[j]
                    rem_cap[j] = nv if nv > 0.0 else 0.0
                    if bounds[j] <= delta or rem_cap[j] <= 0.0:
                        saturated.add(j)
            next_free: list[int] = []
            for f in free:
                if dz[f] >= progs[f].max_dz_ms:
                    continue
                if any(row in saturated for row, _coeff in flow_terms[f]):
                    continue
                next_free.append(f)
            free = next_free
    for row_idx, winner_f in fifo_winner.items():
        if rows[row_idx].fifo_owner_seq is None and dz[winner_f] > 0.0:
            rows[row_idx].fifo_owner_seq = progs[winner_f].sequence
    return dz


def integrate_prepared_epoch(
    progs: Sequence[PreparedProgress],
    deliveries: Sequence[PreparedDelivery],
    dz_per_ms: Sequence[float],
    dt_ms: float,
    counters: list[float],
) -> list[float]:
    """Integrate one epoch ``dt_ms`` matching C11 ``msim_integrate_epoch``."""
    if not math.isfinite(dt_ms) or dt_ms < 0.0:
        raise ModelingError(f"prepared_epoch_dt_invalid:{dt_ms!r}")
    dz_steps = [0.0] * len(progs)
    for f, p in enumerate(progs):
        rate = float(dz_per_ms[f])
        if p.z_left <= 0.0 or rate <= 0.0:
            continue
        step = dt_ms * rate
        if step >= p.z_left:
            used = p.z_left
            p.z_left = 0.0
        else:
            residual = p.z_left - step
            if residual <= RESIDUAL_SNAP_ULPS * math.ulp(p.z_left):
                used = p.z_left
                p.z_left = 0.0
            else:
                used = step
                p.z_left = residual
        dz_steps[f] = used
    for d in deliveries:
        step = dz_steps[d.flow]
        if step > 0.0:
            counters[d.counter] += step * d.work_per_z
    return dz_steps


class PreparedTermsRunner:
    """Direct C11 wrapper for ``msim_hbm_terms`` and prepared-row epoch solving."""

    def __init__(self, library: Any = None) -> None:
        if library is None:
            from mindie_sim.prediction.runtime import accel_build, accel_loader
            status = accel_build.build_accel()
            if not status.get("ok"):
                raise ModelingError(f"accel_build_unavailable:{status.get('reason')}")
            library = accel_loader.load_library(status)
        self.library = bind_terms_abi(library)
        self.ctx = self.library.msim_ctx_create()
        if not self.ctx:
            raise ModelingError("msim_ctx_create_failed")
        self._nprogs = 0
        self._ncounters = 0

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.library.msim_ctx_free(self.ctx)
            self.ctx = None

    def __enter__(self) -> PreparedTermsRunner:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def hbm_terms(self, bus: HbmBusSpec, direction: str, dram_bytes: float) -> tuple[float, float]:
        """Call C11 ``msim_hbm_terms`` and return ``(ms_r, ms_w)``."""
        c_bus = MsimHbmBus(
            die=int(bus.die), row_r=0, row_w=1, ctr_r=0, ctr_w=1,
            bw_r_bytes_per_ms=float(bus.read_bytes_per_ms),
            bw_w_bytes_per_ms=float(bus.write_bytes_per_ms),
            alpha_rw=float(bus.alpha_rw),
        )
        dir_code = HBM_DIR_READ if direction == "read" else HBM_DIR_WRITE
        out_r = ctypes.c_double(0.0)
        out_w = ctypes.c_double(0.0)
        rc = int(self.library.msim_hbm_terms(
            ctypes.byref(c_bus), dir_code, float(dram_bytes),
            ctypes.byref(out_r), ctypes.byref(out_w),
        ))
        if rc != 0:
            raise ModelingError(f"msim_hbm_terms_failed:{rc}")
        return float(out_r.value), float(out_w.value)

    def validate_access(self, acc: MsimAccess) -> int:
        """Call C11 ``msim_validate_access``."""
        return int(self.library.msim_validate_access(ctypes.byref(acc)))

    def config_rows(self, rows: Sequence[PreparedCapacityRow]) -> None:
        """Upload normalized capacity rows into the C11 context."""
        n = len(rows)
        arr = (MsimCapacityRow * max(1, n))()
        for i, row in enumerate(rows):
            arr[i].row_id = int(row.row_id)
            arr[i].scope = int(row.scope)
            arr[i].unit = int(row.unit)
            arr[i].policy = int(row.policy)
            if row.fifo_owner_seq is None:
                arr[i].has_fifo_owner = 0
                arr[i].fifo_owner_seq = 0
            else:
                owner_seq = int(row.fifo_owner_seq)
                if owner_seq < 0 or owner_seq > _UINT64_MAX:
                    raise ModelingError(f"prepared_fifo_owner_out_of_range:{owner_seq}")
                arr[i].has_fifo_owner = 1
                arr[i].fifo_owner_seq = owner_seq
            arr[i].capacity = float(row.capacity)
        rc = int(self.library.msim_config_rows(self.ctx, arr if n else None, n))
        if rc != 0:
            raise ModelingError(f"msim_config_rows_failed:{rc}")
        self._nprogs = 0
        self._ncounters = 0

    def terms_abi_probe(self) -> tuple[int, ...]:
        """Return the C11 16-cell struct size/offset probe vector."""
        return verify_terms_abi(self.library)

    def roundtrip_ring(self, ring: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Pack, validate via C11 ``msim_validate_ring``, and decode ring records."""
        from mindie_sim.prediction.runtime.ring_buffer import DepRecord, SlotGenRecord
        slot_recs, dep_recs = ring.to_c_records()
        ns, nd = len(slot_recs), len(dep_recs)
        s_arr = (MsimSlotGen * max(1, ns))()
        for i, s in enumerate(slot_recs):
            s_arr[i] = MsimSlotGen(
                occurrence=int(s.occurrence), generation=int(s.generation),
                allocation_id=int(s.allocation_id), die=int(s.die),
                owner=int(s.owner), pool_id=int(s.pool_id), slot_id=int(s.slot_id),
                producer_node=int(s.producer_node), publish_node=int(s.publish_node),
                last_use_node=int(s.last_use_node), consumer_count=int(s.consumer_count),
            )
        d_arr = (MsimDep * max(1, nd))()
        for i, d in enumerate(dep_recs):
            d_arr[i] = MsimDep(
                src_node=int(d.src_node), dst_node=int(d.dst_node),
                kind=int(d.kind), flags=int(d.flags), pad=0,
            )
        nnodes = len(ring._nodes)
        rc = int(self.library.msim_validate_ring(
            s_arr if ns else None, ns, d_arr if nd else None, nd, nnodes,
        ))
        if rc != 0:
            raise ModelingError(f"msim_validate_ring_failed:{rc}")
        dec_slots = tuple(
            SlotGenRecord(
                occurrence=int(s_arr[i].occurrence), generation=int(s_arr[i].generation),
                allocation_id=int(s_arr[i].allocation_id), die=int(s_arr[i].die),
                owner=int(s_arr[i].owner), pool_id=int(s_arr[i].pool_id),
                slot_id=int(s_arr[i].slot_id), producer_node=int(s_arr[i].producer_node),
                publish_node=int(s_arr[i].publish_node),
                last_use_node=int(s_arr[i].last_use_node),
                consumer_count=int(s_arr[i].consumer_count),
            )
            for i in range(ns)
        )
        dec_deps = tuple(
            DepRecord(
                src_node=int(d_arr[i].src_node), dst_node=int(d_arr[i].dst_node),
                kind=int(d_arr[i].kind), flags=int(d_arr[i].flags),
            )
            for i in range(nd)
        )
        return dec_slots, dec_deps

    def prepare_terms(
        self,
        progs: Sequence[PreparedProgress],
        terms: Sequence[PreparedTerm],
        deliveries: Sequence[PreparedDelivery],
        ncounters: int,
    ) -> None:
        """Upload flows, coalesced row terms, and counter deliveries into C11."""
        np, nt, nd = len(progs), len(terms), len(deliveries)
        p_arr = (MsimProgress * max(1, np))()
        for i, p in enumerate(progs):
            seq = int(p.sequence)
            prio = int(p.priority)
            if seq < 0 or seq > _UINT64_MAX:
                raise ModelingError(f"prepared_sequence_out_of_range:{seq}")
            if prio < _INT64_MIN or prio > _INT64_MAX:
                raise ModelingError(f"prepared_priority_out_of_range:{prio}")
            p_arr[i] = MsimProgress(
                sequence=seq, priority=prio, task=int(p.task),
                placement_id=int(p.placement_id), member_map=int(p.member_map),
                pad=0, z_left=float(p.z_left),
                weight=float(p.weight), max_dz_ms=float(p.max_dz_ms),
            )
        t_arr = (MsimTerm * max(1, nt))()
        for i, t in enumerate(terms):
            t_arr[i] = MsimTerm(row=int(t.row), flow=int(t.flow), ms_per_z=float(t.ms_per_z))
        d_arr = (MsimDelivery * max(1, nd))()
        for i, d in enumerate(deliveries):
            d_arr[i] = MsimDelivery(
                counter=int(d.counter), flow=int(d.flow), work_per_z=float(d.work_per_z)
            )
        rc = int(self.library.msim_prepare_terms(
            self.ctx,
            p_arr if np else None, np,
            t_arr if nt else None, nt,
            d_arr if nd else None, nd,
            int(ncounters),
        ))
        if rc != 0:
            raise ModelingError(f"msim_prepare_terms_failed:{rc}")
        self._nprogs = np
        self._ncounters = int(ncounters)

    def allocate_epoch(self) -> list[float]:
        """Run C11 ``msim_allocate_epoch`` and return per-flow ``dz/dt`` rates."""
        out = (ctypes.c_double * max(1, self._nprogs))()
        rc = int(self.library.msim_allocate_epoch(
            self.ctx, out if self._nprogs else None, self._nprogs
        ))
        if rc != 0:
            raise ModelingError(f"msim_allocate_epoch_failed:{rc}")
        return [float(out[i]) for i in range(self._nprogs)]

    def integrate_epoch(self, dz_per_ms: Sequence[float], dt_ms: float) -> list[float]:
        """Run C11 ``msim_integrate_epoch`` and return per-flow consumed ``dz``."""
        in_arr = (ctypes.c_double * max(1, self._nprogs))(*[float(v) for v in dz_per_ms])
        out_arr = (ctypes.c_double * max(1, self._nprogs))()
        rc = int(self.library.msim_integrate_epoch(
            self.ctx, in_arr if self._nprogs else None, self._nprogs,
            float(dt_ms), out_arr if self._nprogs else None,
        ))
        if rc != 0:
            raise ModelingError(f"msim_integrate_epoch_failed:{rc}")
        return [float(out_arr[i]) for i in range(self._nprogs)]

    def read_state(self) -> tuple[list[float], list[float]]:
        """Read back ``(z_left, counters)`` from the C11 context."""
        z_arr = (ctypes.c_double * max(1, self._nprogs))()
        c_arr = (ctypes.c_double * max(1, self._ncounters))()
        rc = int(self.library.msim_read_term_state(
            self.ctx,
            z_arr if self._nprogs else None, self._nprogs,
            c_arr if self._ncounters else None, self._ncounters,
        ))
        if rc != 0:
            raise ModelingError(f"msim_read_term_state_failed:{rc}")
        return (
            [float(z_arr[i]) for i in range(self._nprogs)],
            [float(c_arr[i]) for i in range(self._ncounters)],
        )


__all__ = [
    "POLICY_FAIR",
    "POLICY_FIFO",
    "PreparedCapacityRow",
    "PreparedDelivery",
    "PreparedProgress",
    "PreparedTerm",
    "PreparedTermsRunner",
    "allocate_prepared_epoch",
    "integrate_prepared_epoch",
]
