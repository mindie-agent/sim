"""C11 bindings for prepared-row terms, 2-row HBM bus, and WaveRecorder obs buffer.

[Role] Expose typed ``ctypes`` bindings for ``_engine_accel_terms.h``
  (``msim_hbm_bus``, ``msim_access``, ``msim_term``, ``msim_delivery``,
  ``msim_progress``, ``msim_capacity_row``, ``msim_obs``, ``msim_slot_gen``,
  ``msim_dep``) and the zero-overhead ``ObsBridge`` connecting native C11 runs
  to ``WaveRecorder`` plus asymmetric-cohort hand-back detection.
[Invariants] Struct layouts match ``_engine_accel_terms.h`` and
  ``_engine_accel_core.h`` exactly. Caller-owned ``msim_obs`` arrays are never
  retained past detach, and ``has_asymmetric_cohort`` only triggers hand-back
  when a live peer actually breaks a cohort's per-member placement symmetry.
[Timeouts & Locks] Pure C-ABI calls with caller-bounded buffers; no threads or I/O.
[Side-Effect Policy] Mutates only caller-owned C11 contexts and attached
  ``WaveRecorder`` instances.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import math
from typing import Any
import weakref

from mindie_sim.schema.common import ModelingError

DEFAULT_OBS_BUFFER_CAP = 4096
NS_PER_MS = 1_000_000.0
TERMS_ABI_VERSION = 4
TERMS_ABI_PROBE_CELLS = 16
_PAD_BYTES_CAPACITY_ROW = 2
_OBS_BUFFERS: weakref.WeakKeyDictionary[Any, tuple[Any, Any, int]] = (
    weakref.WeakKeyDictionary()
)


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
        ("priority", ctypes.c_int64),
        ("task", ctypes.c_uint32),
        ("placement_id", ctypes.c_uint32),
        ("member_map", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
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
        ("has_fifo_owner", ctypes.c_uint8),
        ("pad", ctypes.c_uint8 * _PAD_BYTES_CAPACITY_ROW),
        ("capacity", ctypes.c_double),
        ("fifo_owner_seq", ctypes.c_uint64),
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
        ("occurrence", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("allocation_id", ctypes.c_uint64),
        ("die", ctypes.c_uint32),
        ("owner", ctypes.c_uint32),
        ("pool_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("producer_node", ctypes.c_uint32),
        ("publish_node", ctypes.c_uint32),
        ("last_use_node", ctypes.c_uint32),
        ("consumer_count", ctypes.c_uint32),
    ]


class MsimDep(ctypes.Structure):
    _fields_ = [
        ("src_node", ctypes.c_uint32),
        ("dst_node", ctypes.c_uint32),
        ("kind", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("pad", ctypes.c_uint32),
    ]


def expected_terms_abi_probe() -> tuple[int, ...]:
    """Return expected 16-cell ``msim_terms_abi_probe`` size/offset vector."""
    return (
        TERMS_ABI_VERSION,
        ctypes.sizeof(MsimHbmBus),
        int(MsimHbmBus.alpha_rw.offset),
        ctypes.sizeof(MsimAccess),
        int(MsimAccess.dram_service_bytes.offset),
        ctypes.sizeof(MsimTerm),
        ctypes.sizeof(MsimDelivery),
        ctypes.sizeof(MsimProgress),
        int(MsimProgress.max_dz_ms.offset),
        ctypes.sizeof(MsimCapacityRow),
        int(MsimCapacityRow.capacity.offset),
        ctypes.sizeof(MsimObs),
        int(MsimObs.work_done.offset),
        ctypes.sizeof(MsimSlotGen),
        int(MsimSlotGen.consumer_count.offset),
        ctypes.sizeof(MsimDep),
    )


def verify_terms_abi(library: Any) -> tuple[int, ...]:
    """Verify terms ABI version (>2) and all 16 struct sizes/offsets against C11."""
    ver = int(library.msim_terms_abi_version())
    if ver != TERMS_ABI_VERSION:
        raise ModelingError(f"terms_abi_version_mismatch:{ver}!={TERMS_ABI_VERSION}")
    buf = (ctypes.c_uint64 * TERMS_ABI_PROBE_CELLS)()
    rc = int(library.msim_terms_abi_probe(buf, TERMS_ABI_PROBE_CELLS))
    if rc != 0:
        raise ModelingError(f"terms_abi_probe_failed:{rc}")
    got = tuple(int(buf[i]) for i in range(TERMS_ABI_PROBE_CELLS))
    exp = expected_terms_abi_probe()
    if got != exp:
        raise ModelingError(f"terms_abi_layout_mismatch:{got}!={exp}")
    return got


def bind_terms_abi(library: Any) -> Any:
    """Bind ``_engine_accel_terms.h`` signatures and verify layout on a CDLL."""
    if getattr(library, "_msim_terms_bound", False):
        return library
    c_void = ctypes.c_void_p
    c_i32 = ctypes.c_int32
    c_u32 = ctypes.c_uint32
    c_u64 = ctypes.c_uint64
    c_u8 = ctypes.c_uint8
    c_dbl = ctypes.c_double

    library.msim_ctx_create.argtypes = []
    library.msim_ctx_create.restype = c_void
    library.msim_ctx_free.argtypes = [c_void]
    library.msim_ctx_free.restype = None

    library.msim_terms_abi_version.argtypes = []
    library.msim_terms_abi_version.restype = c_u32
    library.msim_terms_abi_probe.argtypes = [ctypes.POINTER(c_u64), ctypes.c_size_t]
    library.msim_terms_abi_probe.restype = c_i32
    library.msim_validate_ring.argtypes = [
        ctypes.POINTER(MsimSlotGen), c_u32,
        ctypes.POINTER(MsimDep), c_u32, c_u32,
    ]
    library.msim_validate_ring.restype = c_i32

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

    verify_terms_abi(library)
    library._msim_terms_bound = True
    return library


def coupled_placement_proportional(engine: Any, task: Any) -> bool:
    """True when multi-unit resources on a coupled task have proportional widths."""
    if not (task.coupled and task.placement and len(task.remaining) > 1):
        return True
    partial_live = getattr(engine, "_partial_placed_live", None)
    first_u = first_tot = 0
    for name in task.remaining:
        res_obj = engine.resources[name]
        if res_obj.units <= 1:
            continue
        units = task.placement.get(name)
        if units is None and (not partial_live or name not in partial_live):
            continue
        u_len = res_obj.units if units is None else len(units)
        if first_tot == 0:
            first_u, first_tot = u_len, res_obj.units
        elif u_len * first_tot != first_u * res_obj.units:
            return False
    return True


def _peer_breaks_cohort(task: Any, peer: Any) -> bool:
    if not peer.remaining:
        return False
    if not task.coupled and task.cohort_size > 1:
        t_place = task.placement or {}
        p_place = peer.placement or {}
        for rname in task.remaining:
            if peer.remaining.get(rname, 0.0) <= 0.0:
                continue
            t_units = t_place.get(rname)
            p_units = p_place.get(rname)
            if t_units is None or p_units is None or not set(p_units).isdisjoint(t_units):
                return True
    if not task.cohort_members or not peer.placement:
        return False
    for rname, members in task.cohort_members:
        if rname not in task.remaining or peer.remaining.get(rname, 0.0) <= 0.0:
            continue
        peer_units = peer.placement.get(rname)
        if peer_units is None:
            continue
        peer_set = set(peer_units)
        if len({len(peer_set.intersection(m)) for m in members}) > 1:
            return True
    return False


def _has_other_unit_overlap(
    t_units: Any,
    owner_task: Any,
    peer_map: Mapping[Any, tuple[int, Any]],
) -> bool:
    t_set = None if t_units is None else set(t_units)
    for p_units, (cnt, single) in peer_map.items():
        if cnt == 1 and single is owner_task:
            continue
        if t_set is None or p_units is None or not t_set.isdisjoint(p_units):
            return True
    return False


def _has_other_member_asymmetry(
    members: tuple[frozenset[int], ...],
    owner_task: Any,
    peer_map: Mapping[Any, tuple[int, Any]],
) -> bool:
    for p_units, (cnt, single) in peer_map.items():
        if p_units is None or (cnt == 1 and single is owner_task):
            continue
        p_set = set(p_units)
        if len({len(p_set.intersection(m)) for m in members}) > 1:
            return True
    return False


def _record_sig_count(bucket: dict[Any, tuple[int, Any]], sig: Any, task: Any) -> None:
    prev = bucket.get(sig)
    bucket[sig] = (1, task) if prev is None else (prev[0] + 1, None)


def _build_cohort_peer_index(tasks: Mapping[int, Any], cohort_seqs: Any = None):
    """Build deduplicated resource-indexed cohort and peer placement tables."""
    if cohort_seqs is None:
        cohorts = [t for t in tasks.values() if t.remaining and t.cohort_size > 1]
    else:
        cohorts = [
            t for seq in cohort_seqs
            if (t := tasks.get(seq)) is not None and t.remaining and t.cohort_size > 1
        ]
    if not cohorts:
        return None
    cohort_res: set[str] = set()
    uncoupled_by_res: dict[str, dict[Any, tuple[int, Any]]] = {}
    members_by_res: dict[str, dict[Any, tuple[int, Any]]] = {}
    for c in cohorts:
        cohort_res.update(c.remaining)
        if not c.coupled:
            c_place = c.placement or {}
            for rname in c.remaining:
                _record_sig_count(uncoupled_by_res.setdefault(rname, {}), c_place.get(rname), c)
        if c.cohort_members:
            for rname, members in c.cohort_members:
                if rname in c.remaining:
                    _record_sig_count(members_by_res.setdefault(rname, {}), members, c)
    peer_units_by_res: dict[str, dict[Any, tuple[int, Any]]] = {}
    for task in tasks.values():
        rem = task.remaining
        if not rem:
            continue
        p_place = task.placement or {}
        for rname, amt in rem.items():
            if amt > 0.0 and rname in cohort_res:
                _record_sig_count(peer_units_by_res.setdefault(rname, {}), p_place.get(rname), task)
    return cohort_res, uncoupled_by_res, members_by_res, peer_units_by_res


def _peer_breaks_indexed_cohort(
    peer: Any,
    uncoupled_by_res: Mapping[str, Mapping[Any, tuple[int, Any]]],
    members_by_res: Mapping[str, Mapping[Any, tuple[int, Any]]],
) -> bool:
    rem = peer.remaining
    if not rem:
        return False
    p_place = peer.placement or {}
    for rname, amt in rem.items():
        if amt <= 0.0:
            continue
        p_units = p_place.get(rname)
        u_map = uncoupled_by_res.get(rname)
        if u_map and _has_other_unit_overlap(p_units, peer, u_map):
            return True
        m_map = members_by_res.get(rname)
        if m_map and p_units is not None:
            p_set = set(p_units)
            for members, (cnt, single) in m_map.items():
                if cnt == 1 and single is peer:
                    continue
                if len({len(p_set.intersection(m)) for m in members}) > 1:
                    return True
    return False


def _cohort_needs_split_indexed(
    cohort: Any,
    peer_units_by_res: Mapping[str, Mapping[Any, tuple[int, Any]]],
    fifo_names: Any,
) -> bool:
    if fifo_names and any(r in fifo_names for r in cohort.remaining):
        return True
    if not cohort.cohort_members and cohort.coupled:
        return False
    if not cohort.coupled and cohort.cohort_size > 1:
        c_place = cohort.placement or {}
        for rname in cohort.remaining:
            p_map = peer_units_by_res.get(rname)
            if p_map and _has_other_unit_overlap(c_place.get(rname), cohort, p_map):
                return True
    if cohort.cohort_members:
        for rname, members in cohort.cohort_members:
            if rname not in cohort.remaining:
                continue
            p_map = peer_units_by_res.get(rname)
            if p_map and _has_other_member_asymmetry(members, cohort, p_map):
                return True
    return False


def has_asymmetric_cohort(engine: Any, cohort_seqs: Any = None) -> bool:
    """True when an active/ready cohort touches FIFO or has symmetry broken now."""
    now = engine.now
    tasks = engine._tasks
    if cohort_seqs is None:
        active_cohorts = [
            t for t in tasks.values()
            if t.cohort_size > 1 and t.remaining
            and (t.start is not None or (t.waiting == 0 and t.release <= now))
        ]
    else:
        active_cohorts = [
            t for seq in cohort_seqs
            if (t := tasks.get(seq)) is not None and t.remaining
            and (t.start is not None or (t.waiting == 0 and t.release <= now))
        ]
    if not active_cohorts:
        return False
    fifo_names = engine.fifo_resource_names() if engine._fifo_live else ()
    breakable: list[Any] = []
    cohort_res: set[str] = set()
    for task in active_cohorts:
        if fifo_names and any(r in fifo_names for r in task.remaining):
            return True
        if task.cohort_members or not task.coupled:
            breakable.append(task)
            cohort_res.update(task.remaining)
    if not breakable:
        return False
    for peer in tasks.values():
        rem = peer.remaining
        if not rem or (peer.start is None and (peer.waiting != 0 or peer.release > now)):
            continue
        if cohort_res.isdisjoint(rem):
            continue
        for task in breakable:
            if peer is not task and _peer_breaks_cohort(task, peer):
                return True
    return False


def cohort_break_horizon(engine: Any, horizon: float, cohort_seqs: Any = None) -> float:
    """Clamp epoch horizon to the earliest future peer or cohort release breaking symmetry."""
    now = engine.now
    if getattr(engine, "_max_release", math.inf) <= now:
        return horizon
    tasks = engine._tasks
    index = _build_cohort_peer_index(tasks, cohort_seqs)
    if index is None:
        return horizon
    cohort_res, uncoupled_by_res, members_by_res, peer_units_by_res = index
    fifo_names = engine.fifo_resource_names() if engine._fifo_live else ()
    eff = horizon
    for fut in tasks.values():
        rem = fut.remaining
        if not rem or fut.start is not None or fut.release <= now or fut.release >= eff:
            continue
        if fut.cohort_size <= 1 and cohort_res.isdisjoint(rem):
            continue
        if _peer_breaks_indexed_cohort(fut, uncoupled_by_res, members_by_res) or (
            fut.cohort_size > 1 and _cohort_needs_split_indexed(fut, peer_units_by_res, fifo_names)
        ):
            eff = fut.release
    return eff


def stop_slots_for_epoch(
    engine: Any,
    until: Sequence[Any],
    cohort_seqs: Any = None,
    backend: Any = None,
) -> list[int]:
    """Collect ``until`` slots plus predecessors waking an asymmetric peer or cohort."""
    slots = engine._native_slots
    out = [slots[e.sequence] for e in until if not e.done and e.sequence in slots]
    cache_key = None
    if backend is not None:
        cache_key = (engine._sequence, backend.epoch_tasks, id(backend.arena), bool(engine._fifo_live))
        if backend._cohort_stop_key == cache_key:
            return out
    tasks = engine._tasks
    index = _build_cohort_peer_index(tasks, cohort_seqs)
    if index is None:
        if backend is not None:
            backend._cohort_stop_key = cache_key
        return out
    cohort_res, uncoupled_by_res, members_by_res, peer_units_by_res = index
    fifo_names = engine.fifo_resource_names() if engine._fifo_live else ()
    seen = set(out)
    for seq, slot in slots.items():
        if slot in seen:
            continue
        pred = tasks.get(seq)
        if pred is None or not pred.event._waiters:
            continue
        for wseq in pred.event._waiters:
            w = tasks.get(wseq)
            if w is None or not w.remaining:
                continue
            if w.cohort_size <= 1 and cohort_res.isdisjoint(w.remaining):
                continue
            if _peer_breaks_indexed_cohort(w, uncoupled_by_res, members_by_res) or (
                w.cohort_size > 1 and _cohort_needs_split_indexed(w, peer_units_by_res, fifo_names)
            ):
                out.append(slot)
                seen.add(slot)
                break
    if backend is not None:
        backend._cohort_stop_key = cache_key
    return out


def prepare_obs_bridge(backend: Any, engine: Any) -> None:
    """Attach or detach the C11 ``msim_obs`` buffer to match ``engine._wave_recorder``."""
    arena = backend.arena
    if arena is None:
        return
    rec = getattr(engine, "_wave_recorder", None)
    entry = _OBS_BUFFERS.get(backend)
    if rec is None:
        if entry is not None:
            bind_terms_abi(arena.library).msim_set_obs_buffer(arena.ctx, None, 0)
            _OBS_BUFFERS.pop(backend, None)
        return
    lib = bind_terms_abi(arena.library)
    if entry is None or entry[0] != arena.ctx:
        buf = (MsimObs * DEFAULT_OBS_BUFFER_CAP)()
        rc = int(lib.msim_set_obs_buffer(arena.ctx, buf, DEFAULT_OBS_BUFFER_CAP))
        if rc != 0:
            raise ModelingError(f"obs_buffer_attach_failed:{rc}")
        _OBS_BUFFERS[backend] = (arena.ctx, buf, DEFAULT_OBS_BUFFER_CAP)


def drain_obs_bridge(backend: Any, engine: Any) -> bool:
    """Drain C11 ``msim_obs`` service records into ``engine._wave_recorder``."""
    rec = getattr(engine, "_wave_recorder", None)
    arena = backend.arena
    entry = _OBS_BUFFERS.get(backend)
    if rec is None or arena is None or entry is None or entry[0] != arena.ctx:
        return False
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
        new_cap = max(cap * 2, int(overflow.value))
        new_buf = (MsimObs * new_cap)()
        rc = int(lib.msim_set_obs_buffer(arena.ctx, new_buf, new_cap))
        if rc != 0:
            raise ModelingError(f"obs_buffer_grow_failed:{rc}")
        _OBS_BUFFERS[backend] = (arena.ctx, new_buf, new_cap)
    return bool(count > 0 or overflow.value > 0)


def sync_native_hbm_accounting(backend: Any, engine: Any) -> None:
    """Credit incremental scalar HBM service from live native directional slots."""
    arena = backend.arena
    if arena is None or not engine._hbm_acct_live or not engine._native_slots:
        return
    from ._engine_hbm_bus import _sync_task_hbm_accounting
    tasks = engine._tasks
    export_fn = arena.library.msim_ctx_export
    ctx = arena.ctx
    for seq, slot in engine._native_slots.items():
        task = tasks.get(seq)
        if task is None or not task.hbm_accounting:
            continue
        rem_names = tuple(task.remaining)
        n = len(rem_names)
        if n == 0:
            _sync_task_hbm_accounting(engine, task, {})
            continue
        buf = (ctypes.c_double * (2 * n))()
        got = int(export_fn(ctx, slot, buf, 2 * n))
        if got != n:
            continue
        live_rem = {rem_names[i]: float(buf[i]) for i in range(n)}
        _sync_task_hbm_accounting(engine, task, live_rem)


def sync_native_deliveries(backend: Any, engine: Any) -> None:
    """Flush partial delivery counters for live native tasks at epoch/horizon boundaries."""
    arena = backend.arena
    if arena is None or not engine._delivery_live or not engine._native_slots:
        return
    from ._engine_hbm_bus import _hbm_delivery_fraction
    tasks = engine._tasks
    deliv = engine.delivered
    export_fn = arena.library.msim_ctx_export
    ctx = arena.ctx
    for seq, slot in engine._native_slots.items():
        task = tasks.get(seq)
        if task is None or not task.deliveries:
            continue
        ref_r = next(iter(task.demand), None)
        if ref_r is None:
            continue
        rem_names = tuple(task.remaining)
        n = len(rem_names)
        if task.hbm_accounting:
            if n == 0:
                live_rem: dict[str, float] = {}
            else:
                buf = (ctypes.c_double * (2 * n))()
                got = int(export_fn(ctx, slot, buf, 2 * n))
                if got != n:
                    continue
                live_rem = {rem_names[i]: float(buf[i]) for i in range(n)}
            hbm_frac = _hbm_delivery_fraction(task, live_rem)
            if hbm_frac is not None:
                frac = hbm_frac
            elif n == 0 or ref_r not in task.remaining:
                frac = 1.0
            else:
                idx = rem_names.index(ref_r)
                svc = task.service.get(ref_r, 0.0) + float(buf[n + idx])
                ref_d = task.demand[ref_r]
                frac = min(1.0, svc / ref_d) if ref_d > 0.0 else 1.0
        elif n == 0 or ref_r not in task.remaining:
            frac = 1.0
        else:
            buf = (ctypes.c_double * (2 * n))()
            got = int(export_fn(ctx, slot, buf, 2 * n))
            if got != n:
                continue
            idx = rem_names.index(ref_r)
            svc = task.service.get(ref_r, 0.0) + float(buf[n + idx])
            ref_d = task.demand[ref_r]
            frac = min(1.0, svc / ref_d) if ref_d > 0.0 else 1.0
        acc = task.delivered_acc
        totals: dict[str, float] = {}
        for ctr_id, total_amt in task.deliveries:
            totals[ctr_id] = totals.get(ctr_id, 0.0) + total_amt
        for ctr_id, total_amt in totals.items():
            prev = acc.get(ctr_id, 0.0) if acc is not None else 0.0
            target = total_amt * frac
            delta = target - prev
            if delta > 0.0:
                deliv[ctr_id] = deliv.get(ctr_id, 0.0) + delta
                if acc is not None:
                    acc[ctr_id] = target


def record_native_starts(backend: Any, engine: Any, nstart: int, ncomp: int = 0) -> None:
    """Mirror native C11 start records into ``engine._wave_recorder.on_admit``."""
    rec = getattr(engine, "_wave_recorder", None)
    if rec is None or nstart <= 0 or backend.arena is None:
        return
    tasks = engine._tasks
    comp_times: dict[int, float] | None = None
    for seq, moment in backend.arena.starts(nstart):
        task = tasks.get(seq)
        if task is None:
            continue
        owners = tuple(
            sorted({u for units in (task.placement or {}).values() for u in units})
        ) or (0,)
        gen = int(task.metadata["generation"]) if task.metadata and "generation" in task.metadata else 0
        blockers = ()
        p_pipe = next(iter(task.demand), "")
        if p_pipe:
            if task.pending_preds is not None and len(task.pending_preds) >= 2:
                if comp_times is None:
                    comp_times = (
                        {cseq: cmom for cseq, cmom, _ in backend.arena.completions(ncomp)}
                        if ncomp > 0 else {}
                    )
                ordered = sorted(
                    (p.time if p.time is not None else comp_times.get(p.sequence, moment), i, p)
                    for i, p in enumerate(task.pending_preds)
                )
                prev_t, _prev_i, _prev_p = ordered[len(ordered) - 2]
                _last_t, _last_i, last_p = ordered[-1]
                sole_since = task.release if task.release > prev_t else prev_t
                blockers = ((p_pipe, last_p.name, sole_since, "publish"),)
            elif task.blocker_event:
                r_ms = task.ready_at if task.ready_at is not None else moment
                blockers = ((p_pipe, task.blocker_event, r_ms, task.blocker_reason or "publish"),)
        rec.on_admit(seq, moment, owners=owners, generation=gen, blockers=blockers)
