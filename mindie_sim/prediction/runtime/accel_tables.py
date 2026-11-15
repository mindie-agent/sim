"""ctypes ABI layer for the client-compiled C11 engine kernel (engine-arena/2).

[Role] Mirror the ``_engine_accel_core.h`` wire layout exactly, own one
  ``msim_ctx`` per Engine, batch pure-DAG task submission and translate epoch
  results into the canonical Engine.
[Invariants] One ABI version.  Only u32/u64/i32/binary64 cells and documented
  offsets cross the ABI; no Python object or movable pointer is handed to C.
  Packing preserves the canonical order (resource declaration order, ascending
  task sequence, resource order inside a task's remaining map).  Eligibility is
  conservative and *decided before* the call: an unsupported shape never enters
  C, and a non-zero C status is a fatal ``ExecutionError`` (never a replay).
[Timeouts & Locks] No locks, threads, sleeps or I/O.  Calls are bounded by the
  caller's step budget; buffers are fixed and validated by the kernel.
[Side-Effect Policy] Writes only this arena's ctypes buffers and the Engine
  cells the reference integration block would have written.
"""
from __future__ import annotations

import ctypes
import math
import struct
import weakref

from .engine_types import _EMPTY_FROZENSET as _ENGINE_EMPTY_FROZENSET

ABI_VERSION = 2
HDR_CELLS = 16
ST_CELLS = 8
IO_CELLS = 8
TASK_STRIDE = 12
LINK_STRIDE = 6
COMP_STRIDE = 3
SVC_STRIDE = 3
MAX_T = 65536
MAX_R = 32
MAX_U = 64
MAX_L = 262144
MAX_F = 65536
MAX_COEF = 262144
MAX_MROW = 65536
MAX_SLOT = 8192
MAX_SENT = 2097152
MAX_UENT = 2097152
MAX_OWN = 32768
INT64_MAX = (1 << 63) - 1
INT32_MAX = (1 << 31) - 1
UINT64_MAX = (1 << 64) - 1


def wide_integer(task) -> bool:
    """True when a legal Python integer has no exact C ABI cell.

    Non-exact integer subclasses are conservatively wide (and declined) before
    any comparison can dispatch user ``__lt__``/``__eq__`` code.
    """
    priority, cohort = task.priority, task.cohort_size
    sequence = task.event.sequence
    if (type(priority) is not int or type(cohort) is not int
            or type(sequence) is not int):
        return True
    return (priority < 0 or priority > INT64_MAX
            or cohort < 1 or cohort > UINT64_MAX - 1
            or sequence < 0 or sequence > UINT64_MAX)


#: Exact non-weakrefable CPython built-ins checked by type identity (`tp_weaklistoffset == 0`).
_NONE_TYPE = type(None)
_ROOT_DEPTH = 0
_INERT_MAX_DEPTH = 8


def _is_inert_builtin(obj, depth: int = _ROOT_DEPTH) -> bool:
    """Exact immutable built-in (identity-only types).

    The depth budget is checked first: an exhausted traversal is unproven and
    fails closed even for a scalar leaf.  Only exact ``tuple`` recurses; an
    exact ``frozenset`` is weakrefable and is never certified inert.
    """
    if depth >= _INERT_MAX_DEPTH:
        return False
    kind = type(obj)
    if (kind is _NONE_TYPE or kind is bool or kind is int or kind is float
            or kind is str or kind is bytes):
        return True
    if kind is tuple:
        return all(_is_inert_builtin(item, depth + 1) for item in obj)
    return False


def _inert_mapping(mapping) -> bool:
    """One exact private ``dict`` whose keys and values are inert built-ins."""
    if type(mapping) is not dict:
        return False
    return all(_is_inert_builtin(key) and _is_inert_builtin(value)
               for key, value in mapping.items())


def _eligible_metadata(metadata) -> bool:
    """True when native metadata is ``None`` or an all-inert exact dict."""
    return metadata is None or _inert_mapping(metadata)


def _inert_scalar_mapping(mapping) -> bool:
    """Exact private ``str`` -> exact ``int``/``float`` dictionary."""
    if type(mapping) is not dict:
        return False
    for key, value in mapping.items():
        if type(key) is not str:
            return False
        if type(value) is not int and type(value) is not float:
            return False
    return True


def _exact_number_or_none(value) -> bool:
    """No numeric subclass may own a finalizer behind a task cell."""
    return value is None or type(value) is int or type(value) is float


def _inert_pool_set(pools, engine=None) -> bool:
    """Only an Engine-retained exact ``frozenset[str]`` is inert.

    ``Engine.submit`` keeps the caller's exact frozenset identity for
    ``memory_pools=payload`` (``engine.py:630``), so a caller-owned set is
    weakrefable and can gain ``weakref.finalize``.  Only the Engine's own
    ``_memory_pools_frozen`` snapshot, or the module empty sentinel with no
    attached weakref, is provable; without an Engine, just that sentinel.
    """
    if type(pools) is not frozenset:
        return False
    if pools is not _ENGINE_EMPTY_FROZENSET and not (
            engine is not None and pools is engine._memory_pools_frozen):
        return False
    return (weakref.getweakrefcount(pools) == 0
            and all(type(pool) is str for pool in pools))


def _inert_str_tuple(value) -> bool:
    """An engine-owned tuple of exact allocation names."""
    return type(value) is tuple and all(type(item) is str for item in value)


def _event_is_stable(event) -> bool:
    """The owned Event cells carry no user subclass or pending callback."""
    if type(event.name) is not str or type(event.sequence) is not int:
        return False
    if event.time is not None and type(event.time) is not float:
        return False
    if event.started is not None and type(event.started) is not float:
        return False
    if event._callbacks is not None or type(event._waiters) is not list:
        return False
    return all(type(key) is int for key in event._waiters)


def _contains_mutable_container(obj, depth: int = _ROOT_DEPTH) -> bool:
    """A caller-aliased container can gain a finalizer after admission.

    Reaching the depth budget is unproven, so it fails closed.  Nine
    exact tuple wrappers around a mutable leaf therefore stay in Python.
    """
    kind = type(obj)
    if kind is list or kind is dict or kind is set or kind is bytearray:
        return True
    if depth >= _INERT_MAX_DEPTH:
        return True
    if kind is tuple or kind is frozenset:
        return any(_contains_mutable_container(item, depth + 1) for item in obj)
    return False


def metadata_is_stable(metadata) -> bool:
    """The private top-level copy has no caller-aliased mutable value.

    ``Engine.submit`` copies the top-level mapping, so its keys are stable.  A
    nested ``list``/``dict``/``set``/``bytearray`` is still the caller's object
    and may acquire a weakrefable value between two ``advance`` calls; an
    exhausted traversal budget is unproven.  Both stay in Python.
    ``_stable_task_payloads`` enforces stricter recursive inertness on metadata.
    """
    if type(metadata) is not dict:
        return False
    for key, value in metadata.items():
        if not _is_inert_builtin(key):
            return False
        if _contains_mutable_container(value):
            return False
    return True


def _stable_task_payloads(task, engine=None) -> bool:
    """Audit *every* retained ``_Task`` payload for exact built-in types.

    A custom subclass in any field can own a finalizer and is retained by a
    Python dict/list or the stage cache at a different instant than the native
    row.  Every test is an identity check, so no user ``__eq__``/``__lt__``
    runs.  Admits metadata only when it is ``None`` or one exact private
    dict of recursively inert built-ins, and admits pools only through
    ``_inert_pool_set`` (Engine-retained sentinels).  Declining keeps every
    release instant exactly the Python reference's.
    """
    if not _event_is_stable(task.event) or not _eligible_metadata(task.metadata):
        return False
    if task.owner is not None and type(task.owner) is not str:
        return False
    if type(task.priority) is not int or type(task.cohort_size) is not int:
        return False
    if type(task.waiting) is not int or task.waiting < 0:
        return False
    if not _inert_scalar_mapping(task.remaining):
        return False
    if not _inert_scalar_mapping(task.demand):
        return False
    if not _inert_scalar_mapping(task.caps):
        return False
    if not _inert_scalar_mapping(task.service):
        return False
    if not _exact_number_or_none(task.release):
        return False
    if not _exact_number_or_none(task.latency):
        return False
    if type(task.coupled) is not bool:
        return False
    if task.start is not None and type(task.start) is not float:
        return False
    if type(task.memory_granted) is not bool:
        return False
    # A ledger is an arbitrary user object; declining keeps its reference
    # lifetime exactly the Python ``_Task`` lifetime.
    if task.memory_ledger is not None:
        return False
    if type(task.memory_replica) is not int or type(task.memory_stage) is not int:
        return False
    if not _inert_pool_set(task.memory_pools, engine):
        return False
    if task.memory_bindings is not None:
        return False
    if not _inert_str_tuple(task.memory_grant_names):
        return False
    if type(task.acquire) is not tuple or type(task.free) is not tuple:
        return False
    if type(task.allocation_demands) is not tuple:
        return False
    if type(task.placement) is not dict:
        return False
    for name, units in task.placement.items():
        if type(name) is not str or type(units) is not tuple:
            return False
        if any(type(unit) is not int for unit in units):
            return False
    return True


def callback_capable(task, engine=None) -> bool:
    """True when releasing this row may run user-controlled finalization."""
    return not (
        _event_is_stable(task.event)
        and _eligible_metadata(task.metadata)
        and _is_inert_builtin(task.owner)
        and _is_inert_builtin(task.priority)
        and _is_inert_builtin(task.cohort_size)
        and _is_inert_builtin(task.waiting)
        and _inert_mapping(task.remaining)
        and _inert_mapping(task.demand)
        and _inert_mapping(task.caps)
        and _inert_mapping(task.service)
        and _exact_number_or_none(task.release)
        and _exact_number_or_none(task.latency)
        and type(task.coupled) is bool
        and _is_inert_builtin(task.start)
        and _is_inert_builtin(task.memory_granted)
        and _is_inert_builtin(task.memory_ledger)
        and _is_inert_builtin(task.memory_replica)
        and _is_inert_builtin(task.memory_stage)
        and _inert_pool_set(task.memory_pools, engine)
        and task.memory_bindings is None
        and _inert_str_tuple(task.memory_grant_names)
        and _inert_mapping(task.placement))


def is_inert_trace_entry(record) -> bool:
    """True when one bounded-trace row retains no release-sensitive object."""
    if type(record) is tuple and len(record) == 9:
        return (
            _is_inert_builtin(record[0]) and _is_inert_builtin(record[1])
            and _is_inert_builtin(record[2]) and _is_inert_builtin(record[3])
            and _inert_mapping(record[4]) and _inert_mapping(record[5])
            and _eligible_metadata(record[6]) and _inert_mapping(record[7])
            and _is_inert_builtin(record[8]))
    return (
        _is_inert_builtin(record.get("task"))
        and _is_inert_builtin(record.get("owner"))
        and _is_inert_builtin(record.get("start_ms"))
        and _is_inert_builtin(record.get("end_ms"))
        and _inert_mapping(record.get("demand"))
        and _inert_mapping(record.get("serviced"))
        and _eligible_metadata(record.get("metadata"))
        and _inert_mapping(record.get("placement"))
        and _is_inert_builtin(record.get("coupled")))


def pure_task(engine, task) -> bool:
    """Kernel-eligible callback-free numeric row (no scheduler hooks)."""
    if engine._periodic_runs:
        return False
    if (task.start_callback is not None or task.callback is not None
            or task.acquire or task.free or task.allocation_demands):
        return False
    if not _stable_task_payloads(task, engine):
        return False
    if task.metadata is not None and "occurrence" in task.metadata:
        return False
    try:
        # An exact Python int beyond the binary64 range is legal in the
        # reference clock but has no native ABI cell: decline, never fault.
        if not math.isfinite(task.release):
            return False
        if not math.isfinite(task.latency) or task.latency < 0.0:
            return False
    except (OverflowError, TypeError):
        return False
    if wide_integer(task):
        return False
    for name in task.remaining:
        if name not in engine.resources:
            return False
        work = task.remaining[name]
        if not math.isfinite(work) or work <= 0.0:
            return False
        cap = task.caps.get(name, math.inf)
        if math.isnan(cap) or cap <= 0.0:
            return False
        units = task.placement.get(name) if task.placement else None
        if units is not None:
            if len(units) < 1 or len(units) > MAX_U:
                return False
            res_obj = engine.resources[name]
            limit = res_obj.units
            if res_obj.policy == "fifo" and len(units) < limit:
                return False
            for unit in units:
                if type(unit) is not int or unit < 0 or unit >= limit:
                    return False
    if task.coupled and len(task.remaining) > 1 and task.placement:
        u0 = tot0 = 0
        for name in task.remaining:
            res_obj = engine.resources[name]
            units = task.placement.get(name)
            ulen = res_obj.units if units is None else len(units)
            if tot0 == 0:
                u0, tot0 = ulen, res_obj.units
            elif ulen * tot0 != u0 * res_obj.units:
                return False
    return True

#: Boundary kinds returned by ``msim_ctx_run``.
B_NONE, B_CALLBACK, B_HORIZON, B_LATENCY, B_BUDGET, B_IDLE, B_RELEASE, B_STOP = range(8)
B_OUTPUT = 8

E_RANGE, E_STALL, E_VALUE, E_ARITH, E_FULL, E_STATE, E_ABI = range(1, 8)

#: ``st[0]`` status bits: failure (1) and accepted prefix (2).
ST_FAILED, ST_PREFIX = 1, 2

F_COUPLED, F_LATENCY, F_CALLBACK, F_STOP, F_OCCURRENCE = 1, 2, 4, 8, 16
F_EXT_START = 32


def needs_external_start(task) -> bool:
    return bool(task.acquire or task.free or task.allocation_demands)


def task_flags(task, engine=None) -> int:
    """Kernel flags for one mirrored task (Python keeps every hook)."""
    flags = 0
    if task.coupled:
        flags |= F_COUPLED
    if task.latency > 0.0:
        flags |= F_LATENCY
    if task.acquire or task.free or task.allocation_demands:
        flags |= F_EXT_START | F_CALLBACK
    if (task.callback is not None or "occurrence" in task.metadata
            or callback_capable(task, engine)):
        # A finalizer-capable row ends *time* integration at its own
        # completion instant, so any observer registered by that release is
        # causal before later numeric work is irrevocably accepted.
        flags |= F_CALLBACK
    return flags


def resource_table(engine):
    """Declared rates/units/policies in deterministic resource order."""
    rates, units, policy = [], [], []
    for name in engine.resources:
        resource = engine.resources[name]
        rates.append(float(resource.rate))
        units.append(int(resource.units))
        policy.append(1 if resource.policy == "fifo" else 0)
    return rates, units, policy

#: Reference ``engine-arena/2`` golden probe cells (scalar sizes + binary64).
PROBE_GOLDEN = (
    2, 8, 4, 8, ctypes.sizeof(ctypes.c_void_p), 16, 12, 6, 3, 3,
    struct.unpack("<Q", struct.pack("<d", math.ulp(1.0)))[0],
    struct.unpack("<Q", struct.pack("<d", 5e-324))[0],
    struct.unpack("<Q", struct.pack("<d", math.nextafter(1.0, 2.0) - 1.0))[0],
    struct.unpack("<Q", struct.pack("<d", 1.0 / 3.0))[0],
    1, MAX_R,
)

U32_T = ctypes.c_uint32
U64_T = ctypes.c_uint64
I32_T = ctypes.c_int32
F64_T = ctypes.c_double


class KernelFault(RuntimeError):
    """A C-side contract/arithmetic failure: fatal, never replayed in Python."""

    def __init__(self, code: int, boundary: str, detail: str = "",
                 prefix=None, has_prefix: bool = False) -> None:
        super().__init__(f"accel_kernel_error:{boundary}:{code}")
        self.code = int(code)
        self.boundary = boundary
        self.detail = detail
        #: ``(now, boundary, steps, ncomp, nstart)`` accepted by the
        #: kernel before a failure, or None when no prefix exists.  The
        #: ``has_prefix`` bit is the kernel's own trustworthy indicator.
        self.prefix = prefix
        self.has_prefix = bool(has_prefix)


def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def from_bits(value: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(value) & 0xFFFFFFFFFFFFFFFF))[0]


def install_signatures(library: ctypes.CDLL) -> None:
    """Declare every ABI symbol once for this process-local shared library."""
    P = ctypes.POINTER
    library.msim_ctx_link.restype = ctypes.c_int32
    library.msim_ctx_link.argtypes = [
        ctypes.c_void_p, P(ctypes.c_int32), P(ctypes.c_int32),
        P(ctypes.c_uint64), ctypes.c_size_t, P(ctypes.c_int32), ctypes.c_size_t]
    library.msim_abi_version.restype = ctypes.c_uint32
    library.msim_abi_version.argtypes = ()
    library.msim_abi_probe.restype = ctypes.c_int32
    library.msim_abi_probe.argtypes = [P(ctypes.c_uint64), ctypes.c_size_t]
    library.msim_ctx_create.restype = ctypes.c_void_p
    library.msim_ctx_create.argtypes = ()
    library.msim_ctx_free.restype = None
    library.msim_ctx_free.argtypes = [ctypes.c_void_p]
    library.msim_ctx_reset.restype = ctypes.c_int32
    library.msim_ctx_reset.argtypes = [ctypes.c_void_p]
    library.msim_ctx_config.restype = ctypes.c_int32
    library.msim_ctx_config.argtypes = [
        ctypes.c_void_p, P(ctypes.c_double), P(ctypes.c_int32),
        P(ctypes.c_int32), ctypes.c_size_t, P(ctypes.c_int32), ctypes.c_size_t]
    library.msim_ctx_seed.restype = ctypes.c_int32
    library.msim_ctx_seed.argtypes = [
        ctypes.c_void_p, P(ctypes.c_double), P(ctypes.c_double), ctypes.c_size_t]
    library.msim_ctx_counters.restype = ctypes.c_int32
    library.msim_ctx_counters.argtypes = [
        ctypes.c_void_p, P(ctypes.c_double), P(ctypes.c_double), ctypes.c_size_t]
    library.msim_ctx_submit.restype = ctypes.c_int32
    library.msim_ctx_submit.argtypes = [
        ctypes.c_void_p, P(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_size_t,
        P(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_size_t,
        P(ctypes.c_uint32), ctypes.c_size_t, P(ctypes.c_int32), P(ctypes.c_int32),
        ctypes.c_size_t, P(ctypes.c_int32), ctypes.c_size_t,
        P(ctypes.c_int32), ctypes.c_size_t]
    for name in ("msim_ctx_deliver", "msim_ctx_drop", "msim_ctx_mark_external"):
        fn = getattr(library, name)
        fn.restype = ctypes.c_int32
        fn.argtypes = [ctypes.c_void_p, P(ctypes.c_int32), ctypes.c_size_t,
                       P(ctypes.c_int32), ctypes.c_size_t]
    library.msim_ctx_activate.restype = ctypes.c_int32
    library.msim_ctx_activate.argtypes = [
        ctypes.c_void_p, P(ctypes.c_int32), ctypes.c_size_t, ctypes.c_double,
        P(ctypes.c_int32), ctypes.c_size_t]
    library.msim_ctx_export.restype = ctypes.c_int32
    library.msim_ctx_export.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, P(ctypes.c_double), ctypes.c_size_t]
    library.msim_ctx_run.restype = ctypes.c_int32
    library.msim_ctx_run.argtypes = [
        ctypes.c_void_p, P(ctypes.c_uint64), ctypes.c_size_t,
        P(ctypes.c_int32), ctypes.c_size_t, P(ctypes.c_double), ctypes.c_size_t,
        P(ctypes.c_uint64), ctypes.c_size_t, P(ctypes.c_double), ctypes.c_size_t,
        P(ctypes.c_uint64), ctypes.c_size_t]


def _grow(array: ctypes.Array, needed: int, elem):
    if needed <= len(array):
        return array
    return (elem * max(needed, len(array) * 2 + 64))()


def _free_ctx_box(library: ctypes.CDLL, box) -> None:
    """Finalizer body: free at most once through the shared mutable box.

    The box holds only the ``c_void_p`` address, never the Arena, backend or
    Engine, so the same primitive serves Engine finalizers cycle-free.
    """
    ctx = box[0]
    if not ctx:
        return
    box[0] = None
    library.msim_ctx_free(ctx)


class Arena:
    """One Engine's context handle plus the bounded wire buffers."""

    __slots__ = (
        "library", "ctx", "ctx_box", "rows", "links", "place", "hdr", "st",
        "io", "comp", "svc", "started", "out_slots", "res_names", "closed",
        "nres", "last_held", "_finalizer", "__weakref__",
    )

    def __init__(self, library: ctypes.CDLL) -> None:
        self.library = library
        handle = library.msim_ctx_create()
        if not handle:
            raise MemoryError("accel_ctx_create_failed")
        self.ctx_box = [ctypes.c_void_p(handle)]
        self.ctx = self.ctx_box[0]
        self._finalizer = weakref.finalize(
            self, _free_ctx_box, library, self.ctx_box)
        self.rows = (U64_T * (TASK_STRIDE * 256))()
        self.links = (U64_T * (LINK_STRIDE * 1024))()
        self.place = (U32_T * 1024)()
        self.hdr = (U64_T * HDR_CELLS)()
        self.st = (I32_T * ST_CELLS)()
        self.io = (F64_T * IO_CELLS)()
        self.comp = (U64_T * (COMP_STRIDE * 4096))()
        self.svc = (F64_T * (SVC_STRIDE * 32768))()
        self.started = (U64_T * (2 * 4096))()
        self.out_slots = (I32_T * 256)()
        self.res_names: tuple[str, ...] = ()
        self.closed = False
        self.nres = 0
        self.last_held = False

    # -- lifecycle -----------------------------------------------------
    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        self.ctx_box[0] = None
        self.ctx = ctypes.c_void_p(None)
        self.closed = True

    def reset(self) -> None:
        """Drop every kernel row (a fresh attachment owns no stale task)."""
        if self.closed or not self.ctx:
            return
        if self.library.msim_ctx_reset(self.ctx) != 0:
            raise KernelFault(E_STATE, "reset")

    def configure(self, rates, units, policy) -> None:
        n = len(rates)
        rr = (F64_T * n)(*[float(x) for x in rates])
        uu = (I32_T * n)(*[int(x) for x in units])
        pp = (I32_T * n)(*[int(x) for x in policy])
        rc = self.library.msim_ctx_config(self.ctx, rr, uu, pp, n, self.st, ST_CELLS)
        if rc != 0:
            raise KernelFault(int(self.st[1]), "configure", f"nres={n}")
        self.nres = n

    def seed(self, busy, serv) -> None:
        n = self.nres
        b = (F64_T * n)(*[float(x) for x in busy])
        s = (F64_T * n)(*[float(x) for x in serv])
        if self.library.msim_ctx_seed(self.ctx, b, s, n) != 0:
            raise KernelFault(E_VALUE, "seed")

    def counters(self):
        n = self.nres
        b = (F64_T * n)()
        s = (F64_T * n)()
        if self.library.msim_ctx_counters(self.ctx, b, s, n) != 0:
            raise KernelFault(E_VALUE, "counters")
        return list(b), list(s)

    # -- submission ----------------------------------------------------
    def submit(self, tasks, deps=()) -> list[int]:
        """``tasks`` is a list of ``(seq, flags, prio, cohort, waiting, start,
        latency, release, [(res, remaining, cap, placement)])`` tuples."""
        nrows = len(tasks)
        if not nrows:
            return []
        nlinks = sum(len(row[8]) for row in tasks)
        nplace = sum(len(link[3]) for row in tasks for link in row[8])
        self.rows = _grow(self.rows, nrows * TASK_STRIDE, U64_T)
        self.links = _grow(self.links, nlinks * LINK_STRIDE, U64_T)
        self.place = _grow(self.place, nplace + 1, U32_T)
        self.out_slots = _grow(self.out_slots, nrows, I32_T)
        place = 0
        li = 0
        for i, row in enumerate(tasks):
            seq, flags, prio, cohort, waiting, start, latency, release, links = row
            # Representable bounds are enforced *before* ctypes packing so a
            # wide Python integer can never wrap into a valid-looking cell.
            if not isinstance(prio, int) or isinstance(prio, bool) or prio < 0 \
                    or prio > 0x7FFFFFFFFFFFFFFF:
                raise KernelFault(E_RANGE, "submit", "priority_out_of_range")
            if not isinstance(cohort, int) or isinstance(cohort, bool) \
                    or cohort < 1 or cohort > 0xFFFFFFFFFFFFFFFE:
                raise KernelFault(E_RANGE, "submit", "cohort_out_of_range")
            if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0 \
                    or seq > 0xFFFFFFFFFFFFFFFF:
                raise KernelFault(E_RANGE, "submit", "sequence_out_of_range")
            base = i * TASK_STRIDE
            self.rows[base + 0] = seq
            self.rows[base + 1] = flags
            self.rows[base + 2] = prio
            self.rows[base + 3] = cohort
            self.rows[base + 4] = waiting
            self.rows[base + 5] = 0
            self.rows[base + 6] = bits(start)
            self.rows[base + 7] = bits(latency)
            self.rows[base + 8] = bits(release)
            self.rows[base + 9] = len(links)
            self.rows[base + 10] = li
            self.rows[base + 11] = 0
            for res, remaining, cap, units in links:
                off = li * LINK_STRIDE
                self.links[off + 0] = i
                self.links[off + 1] = res
                self.links[off + 2] = bits(remaining)
                self.links[off + 3] = bits(cap)
                self.links[off + 4] = place
                self.links[off + 5] = len(units)
                for u in units:
                    self.place[place] = u
                    place += 1
                li += 1
        rc = self.library.msim_ctx_submit(
            self.ctx, self.rows, nrows, TASK_STRIDE, self.links, nlinks,
            LINK_STRIDE, self.place, place, None, None, 0,
            self.out_slots, nrows, self.st, ST_CELLS)
        if rc != 0:
            raise KernelFault(int(self.st[1]), "submit", f"nrows={nrows}")
        return [int(self.out_slots[i]) for i in range(nrows)]

    def link(self, edges) -> None:
        """``edges`` is a list of ``(pred_slot, succ_slot, succ_seq)`` triples."""
        if not edges:
            return
        n = len(edges)
        pred = (I32_T * n)(*[int(e[0]) for e in edges])
        succ = (I32_T * n)(*[int(e[1]) for e in edges])
        seqs = (ctypes.c_uint64 * n)(*[int(e[2]) for e in edges])
        rc = self.library.msim_ctx_link(self.ctx, pred, succ, seqs, n,
                                        self.st, ST_CELLS)
        if rc != 0:
            raise KernelFault(int(self.st[1]), "link")

    def call(self, name: str, slots) -> None:
        if not slots:
            return
        buf = (I32_T * len(slots))(*[int(x) for x in slots])
        rc = getattr(self.library, name)(self.ctx, buf, len(slots), self.st, ST_CELLS)
        if rc != 0:
            raise KernelFault(int(self.st[1]), name)

    # -- epoch ---------------------------------------------------------
    def run(self, now: float, horizon: float, next_release: float, budget: int):
        self.hdr[0] = ABI_VERSION
        self.hdr[1] = 0
        self.hdr[2] = int(budget)
        self.io[0] = now
        self.io[1] = horizon
        self.io[2] = 0.0
        self.io[3] = next_release
        rc = self.library.msim_ctx_run(
            self.ctx, self.hdr, HDR_CELLS, self.st, ST_CELLS, self.io, IO_CELLS,
            self.comp, COMP_STRIDE * 4096, self.svc, SVC_STRIDE * 32768,
            self.started, 2 * 4096)
        if rc != 0:
            code = int(self.st[1])
            # the kernel publishes an accepted prefix (clock, counts and
            # records) only when status bit ``ST_PREFIX`` is set.  A preflight
            # or step-0 failure must not fabricate one from stale buffers.
            has_prefix = bool(int(self.st[0]) & ST_PREFIX)
            prefix = None
            if has_prefix:
                prefix = (
                    float(self.io[0]), int(self.st[2]), int(self.st[3]),
                    int(self.st[4]), int(self.st[6]),
                )
            raise KernelFault(code, "run", prefix=prefix,
                              has_prefix=has_prefix)
        # hdr[3] reports that a callback-capable completion ended this
        # call before the awakened ready rows were admitted.
        self.last_held = bool(int(self.hdr[3]))
        return (
            float(self.io[0]), int(self.st[2]), int(self.st[3]),
            int(self.st[4]), int(self.st[6]),
        )

    def completions(self, count: int):
        out = []
        for i in range(count):
            base = i * COMP_STRIDE
            out.append((
                int(self.comp[base + 0]),
                from_bits(int(self.comp[base + 1])),
                int(self.comp[base + 2]),
            ))
        return out

    def services(self, start: int, count: int):
        out = []
        for i in range(count):
            base = (start + i) * SVC_STRIDE
            out.append((
                int(self.svc[base + 0]),
                float(self.svc[base + 1]),
            ))
        return out

    def starts(self, count: int):
        return [
            (int(self.started[i * 2]), from_bits(int(self.started[i * 2 + 1])))
            for i in range(count)
        ]
