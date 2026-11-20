"""C11 bindings for prepared-row terms, 2-row HBM bus, and WaveRecorder obs buffer.

[Role] Expose typed ``ctypes`` bindings for ``_engine_accel_terms.h``
  (``msim_hbm_bus``, ``msim_access``, ``msim_term``, ``msim_delivery``,
  ``msim_progress``, ``msim_capacity_row``, ``msim_obs``, ``msim_slot_gen``,
  ``msim_dep``), a ``PreparedTermsRunner`` for direct Python==C11 epoch parity,
  and the zero-overhead ``ObsBridge`` connecting native C11 runs to
  ``WaveRecorder`` plus asymmetric-cohort hand-back detection.
[Invariants] Struct layouts match ``_engine_accel_terms.h`` and
  ``_engine_accel_core.h`` exactly. Caller-owned ``msim_obs`` arrays are never
  retained past detach, and ``has_asymmetric_cohort`` only triggers hand-back
  when a live peer actually breaks a cohort's per-member placement symmetry.
[Timeouts & Locks] Pure C-ABI calls with caller-bounded buffers; no threads or I/O.
[Side-Effect Policy] Mutates only caller-owned C11 contexts and attached
  ``WaveRecorder`` instances.
"""
from __future__ import annotations

from collections.abc import Sequence
import ctypes
from typing import Any
import weakref

from mindie_sim.schema.common import ModelingError
from .memory_physics import HBM_DIR_READ, HBM_DIR_WRITE, HbmBusSpec
from .prepared_terms import (
    PreparedCapacityRow,
    PreparedDelivery,
    PreparedProgress,
    PreparedTerm,
)

DEFAULT_OBS_BUFFER_CAP = 4096
NS_PER_MS = 1_000_000.0
MAX_WIDENED_LINK_BITS = 64
_PAD_BYTES_CAPACITY_ROW = 7
_OBS_BUFFERS: weakref.WeakKeyDictionary[Any, tuple[Any, Any, int]] = (
    weakref.WeakKeyDictionary()
)


class MsimSlice(ctypes.Structure):
    _fields_ = [("offset", ctypes.c_uint32), ("count", ctypes.c_uint32)]


class MsimHbmBus(ctypes.Structure):
    _fields_ = [
        ("die", ctypes.c_uint32),
        ("row_r", ctypes.c_uint32),
        ("row_w", ctypes.c_uint32),
        ("ctr_r", ctypes.c_uint32),
        ("ctr_w", ctypes.c_uint32),
        ("bw_r_bytes_per_ms", ctypes.c_double),
        ("bw_w_bytes_per_ms", ctypes.c_double),
        ("alpha_rw", ctypes.c_double),
    ]


class MsimAccess(ctypes.Structure):
    _fields_ = [
        ("logical_bytes", ctypes.c_uint64),
        ("aligned_bytes", ctypes.c_uint64),
        ("hit_bytes", ctypes.c_uint64),
        ("miss_bytes", ctypes.c_uint64),
        ("sector_bytes", ctypes.c_uint64),
        ("dram_service_bytes", ctypes.c_double),
        ("copy_pipe_row", ctypes.c_uint32),
        ("l2_row", ctypes.c_uint32),
        ("die", ctypes.c_uint16),
        ("direction", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
    ]


class MsimTerm(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.c_uint32),
        ("flow", ctypes.c_uint32),
        ("ms_per_z", ctypes.c_double),
    ]


class MsimDelivery(ctypes.Structure):
    _fields_ = [
        ("counter", ctypes.c_uint32),
        ("flow", ctypes.c_uint32),
        ("work_per_z", ctypes.c_double),
    ]


class MsimProgress(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("task", ctypes.c_uint32),
        ("placement_id", ctypes.c_uint32),
        ("member_map", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
        ("z_left", ctypes.c_double),
        ("weight", ctypes.c_double),
        ("max_dz_ms", ctypes.c_double),
    ]


class MsimCapacityRow(ctypes.Structure):
    _fields_ = [
        ("row_id", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("unit", ctypes.c_uint32),
        ("policy", ctypes.c_uint8),
        ("pad", ctypes.c_uint8 * _PAD_BYTES_CAPACITY_ROW),
        ("capacity", ctypes.c_double),
    ]


class MsimObs(ctypes.Structure):
    _fields_ = [
        ("t_start_ns", ctypes.c_uint64),
        ("t_end_ns", ctypes.c_uint64),
        ("task_id", ctypes.c_uint64),
        ("pipe_id", ctypes.c_uint32),
        ("owner_id", ctypes.c_uint32),
        ("kind", ctypes.c_uint16),
        ("reason_code", ctypes.c_uint16),
        ("work_done", ctypes.c_double),
    ]


class MsimSlotGen(ctypes.Structure):
    _fields_ = [
        ("ring_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("gen", ctypes.c_uint64),
        ("producer_task_id", ctypes.c_uint64),
        ("publish_event_id", ctypes.c_uint64),
        ("last_use_event_id", ctypes.c_uint64),
        ("consumer_count", ctypes.c_uint32),
        ("reader_done_count", ctypes.c_uint32),
    ]


class MsimDep(ctypes.Structure):
    _fields_ = [
        ("src_event_id", ctypes.c_uint64),
        ("dst_task_id", ctypes.c_uint64),
        ("kind", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("pad", ctypes.c_uint32),
    ]


def bind_terms_abi(library: Any) -> Any:
    """Bind ``_engine_accel_terms.h`` signatures on a loaded ctypes library."""
    if getattr(library, "_msim_terms_bound", False):
        return library
    c_void = ctypes.c_void_p
    c_i32 = ctypes.c_int32
    c_u32 = ctypes.c_uint32
    c_u8 = ctypes.c_uint8
    c_dbl = ctypes.c_double

    library.msim_ctx_create.argtypes = []
    library.msim_ctx_create.restype = c_void
    library.msim_ctx_free.argtypes = [c_void]
    library.msim_ctx_free.restype = None

    library.msim_hbm_terms.argtypes = [
        ctypes.POINTER(MsimHbmBus), c_u8, c_dbl,
        ctypes.POINTER(c_dbl), ctypes.POINTER(c_dbl),
    ]
    library.msim_hbm_terms.restype = c_i32

    library.msim_validate_access.argtypes = [ctypes.POINTER(MsimAccess)]
    library.msim_validate_access.restype = c_i32

    library.msim_set_obs_buffer.argtypes = [c_void, ctypes.POINTER(MsimObs), c_u32]
    library.msim_set_obs_buffer.restype = c_i32

    library.msim_drain_obs.argtypes = [c_void, ctypes.POINTER(c_u32)]
    library.msim_drain_obs.restype = c_u32

    library.msim_config_rows.argtypes = [c_void, ctypes.POINTER(MsimCapacityRow), c_u32]
    library.msim_config_rows.restype = c_i32

    library.msim_prepare_terms.argtypes = [
        c_void,
        ctypes.POINTER(MsimProgress), c_u32,
        ctypes.POINTER(MsimTerm), c_u32,
        ctypes.POINTER(MsimDelivery), c_u32,
        c_u32,
    ]
    library.msim_prepare_terms.restype = c_i32

    library.msim_allocate_epoch.argtypes = [c_void, ctypes.POINTER(c_dbl), c_u32]
    library.msim_allocate_epoch.restype = c_i32

    library.msim_integrate_epoch.argtypes = [
        c_void, ctypes.POINTER(c_dbl), c_u32, c_dbl, ctypes.POINTER(c_dbl),
    ]
    library.msim_integrate_epoch.restype = c_i32

    library.msim_read_term_state.argtypes = [
        c_void, ctypes.POINTER(c_dbl), c_u32, ctypes.POINTER(c_dbl), c_u32,
    ]
    library.msim_read_term_state.restype = c_i32

    library._msim_terms_bound = True
    return library


def has_asymmetric_cohort(engine: Any) -> bool:
    """True when any live task has cohort_members broken by another live task."""
    tasks = engine._tasks
    if len(tasks) < 2:
        return False
    for seq, task in tasks.items():
        if task.cohort_size <= 1 or not task.cohort_members or not task.remaining:
            continue
        members_dict = dict(task.cohort_members)
        for peer_seq, peer in tasks.items():
            if peer_seq == seq or not peer.placement or not peer.remaining:
                continue
            for rname, members in members_dict.items():
                if rname not in task.remaining or peer.remaining.get(rname, 0.0) <= 0.0:
                    continue
                peer_units = peer.placement.get(rname)
                if peer_units is None:
                    continue
                peer_set = set(peer_units)
                if len({len(peer_set.intersection(m)) for m in members}) > 1:
                    return True
    return False


def widened_mask_for_task(task: Any) -> int | None:
    """Encode ``task.cohort_widened`` as a 64-bit link mask, or ``None`` if >64 links."""
    widened = getattr(task, "cohort_widened", ())
    if not widened:
        return 0
    if task.cohort_size <= 1:
        return None
    wset = set(widened)
    mask = 0
    for idx, name in enumerate(task.remaining):
        if name in wset:
            if idx >= MAX_WIDENED_LINK_BITS:
                return None
            mask |= 1 << idx
    return mask


def prepare_obs_bridge(backend: Any, engine: Any) -> None:
    """Attach or detach the C11 ``msim_obs`` buffer to match ``engine._wave_recorder``."""
    arena = backend.arena
    if arena is None:
        return
    lib = bind_terms_abi(arena.library)
    rec = getattr(engine, "_wave_recorder", None)
    entry = _OBS_BUFFERS.get(backend)
    if rec is None:
        if entry is not None:
            lib.msim_set_obs_buffer(arena.ctx, None, 0)
            _OBS_BUFFERS.pop(backend, None)
        return
    if entry is None or entry[0] != arena.ctx:
        buf = (MsimObs * DEFAULT_OBS_BUFFER_CAP)()
        rc = int(lib.msim_set_obs_buffer(arena.ctx, buf, DEFAULT_OBS_BUFFER_CAP))
        if rc != 0:
            raise ModelingError(f"obs_buffer_attach_failed:{rc}")
        _OBS_BUFFERS[backend] = (arena.ctx, buf, DEFAULT_OBS_BUFFER_CAP)


def drain_obs_bridge(backend: Any, engine: Any) -> None:
    """Drain C11 ``msim_obs`` service records into ``engine._wave_recorder``."""
    rec = getattr(engine, "_wave_recorder", None)
    arena = backend.arena
    entry = _OBS_BUFFERS.get(backend)
    if rec is None or arena is None or entry is None or entry[0] != arena.ctx:
        return
    _, buf, cap = entry
    lib = bind_terms_abi(arena.library)
    overflow = ctypes.c_uint32(0)
    count = int(lib.msim_drain_obs(arena.ctx, ctypes.byref(overflow)))
    res_names = backend.res_names
    for i in range(count):
        item = buf[i]
        pipe_idx = int(item.pipe_id)
        if not (0 <= pipe_idx < len(res_names)):
            continue
        rname = res_names[pipe_idx]
        seq = int(item.task_id)
        rec.on_service(
            seq,
            rname,
            int(item.owner_id),
            float(item.t_start_ns) / NS_PER_MS,
            float(item.t_end_ns) / NS_PER_MS,
            float(item.work_done),
        )
    if overflow.value > 0:
        new_cap = cap * 2
        new_buf = (MsimObs * new_cap)()
        rc = int(lib.msim_set_obs_buffer(arena.ctx, new_buf, new_cap))
        if rc != 0:
            raise ModelingError(f"obs_buffer_grow_failed:{rc}")
        _OBS_BUFFERS[backend] = (arena.ctx, new_buf, new_cap)


def record_native_starts(backend: Any, engine: Any, nstart: int) -> None:
    """Mirror native C11 start records into ``engine._wave_recorder.on_admit``."""
    rec = getattr(engine, "_wave_recorder", None)
    if rec is None or nstart <= 0 or backend.arena is None:
        return
    tasks = engine._tasks
    for seq, moment in backend.arena.starts(nstart):
        task = tasks.get(seq)
        if task is not None:
            owners = tuple(
                sorted({u for units in (task.placement or {}).values() for u in units})
            ) or (0,)
            rec.on_admit(seq, moment, owners=owners)


class PreparedTermsRunner:
    """Direct C11 wrapper for ``msim_hbm_terms`` and prepared-row epoch solving."""

    def __init__(self, library: Any = None) -> None:
        if library is None:
            from . import accel_build, accel_loader
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
            arr[i].capacity = float(row.capacity)
        rc = int(self.library.msim_config_rows(self.ctx, arr if n else None, n))
        if rc != 0:
            raise ModelingError(f"msim_config_rows_failed:{rc}")

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
            p_arr[i] = MsimProgress(
                sequence=int(p.sequence), task=int(p.task),
                placement_id=int(p.placement_id), member_map=int(p.member_map),
                priority=int(p.priority), z_left=float(p.z_left),
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
