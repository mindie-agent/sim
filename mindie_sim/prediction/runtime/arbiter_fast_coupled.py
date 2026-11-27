"""Single-tier fast per-unit progressive filling for coupled/placed tasks.

[Role] Solve the common single-priority, non-FIFO case of ``_tiered_coupled``
  without per-step dictionary/set allocations by folding private physical units
  into per-flow ``alpha`` upper bounds and running progressive filling only over
  genuinely shared physical units.
[Invariants] Bit-identical ``demand`` and ``weight`` scaling to ``_tier_flows``,
  exact ``math.fsum`` per-unit summation, 16-ULP unit saturation, 8-ULP cap
  retirement, and an independent final ``unit_service_overallocated`` guard on
  every private and shared unit. Returns ``None`` whenever multiple priority
  tiers or FIFO resources are present so ``_tiered_coupled`` handles them.
[Timeouts & Locks] Bounded progressive filling; no threads, sleeps, locks or I/O.
[Side-Effect Policy] Pure arithmetic over caller-supplied task and resource views.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import math
import os
import threading

_HALF_STEP = 0.5
_CAP_ULPS = 8.0
_SAT_ULPS = 16.0
_NEXTAFTER_STEPS = 64
_MAX_ATTEMPTS = 256
_UNIT_RANGE_CACHE: dict[int, tuple[int, ...]] = {}

_FAULT_OK = 0
_FAULT_SCALE_UNDERFLOW = 1
_FAULT_OVERALLOCATED = 2
_FAULT_CELLS = 3
_FAULT_CODE_IDX = 0
_FAULT_RES_IDX = 1
_FAULT_UNIT_IDX = 2
_INITIAL_SCRATCH_CAP = 64
_SCRATCH_GROWTH_FACTOR = 2
_MIN_UNIT_COUNT = 1
_MAX_NATIVE_COHORT = 2_147_483_647
_INT32_BYTES = ctypes.sizeof(ctypes.c_int32)
_PLACE_PACK_CACHE_MAXSIZE = 1024
_PYTHON_MODES = frozenset({"python", "py", "off", "0"})
_UNSET = object()

_NATIVE_FN_CACHE: dict[tuple, object] = {}
_NATIVE_FN_LOCK = threading.Lock()
_THREAD_SCRATCH = threading.local()


@lru_cache(maxsize=_PLACE_PACK_CACHE_MAXSIZE)
def _pack_place_units(decl: tuple[int, ...]) -> tuple[ctypes.Array, int, int]:
    """Pack a repeated placement tuple into a contiguous ``c_int32`` buffer."""
    count = len(decl)
    arr = (ctypes.c_int32 * count)(*(int(u) for u in decl))
    return arr, count, count * _INT32_BYTES


class _CoupledScratch:
    """Reusable growable ctypes buffers avoiding per-call heap churn."""

    __slots__ = (
        "res_cap", "res_rate", "res_units",
        "task_cap", "task_coupled", "task_cohort", "task_link_off",
        "link_cap", "link_res", "link_rem", "link_cap_arr",
        "link_widened", "link_unplaced", "link_place_off", "out_rate",
        "place_cap", "place_units", "out_fault",
    )

    def __init__(self) -> None:
        cap = _INITIAL_SCRATCH_CAP
        self.res_cap = cap
        self.res_rate = (ctypes.c_double * cap)()
        self.res_units = (ctypes.c_int32 * cap)()
        self.task_cap = cap
        self.task_coupled = (ctypes.c_uint8 * cap)()
        self.task_cohort = (ctypes.c_int32 * cap)()
        self.task_link_off = (ctypes.c_int32 * (cap + _MIN_UNIT_COUNT))()
        self.link_cap = cap
        self.link_res = (ctypes.c_int32 * cap)()
        self.link_rem = (ctypes.c_double * cap)()
        self.link_cap_arr = (ctypes.c_double * cap)()
        self.link_widened = (ctypes.c_uint8 * cap)()
        self.link_unplaced = (ctypes.c_uint8 * cap)()
        self.link_place_off = (ctypes.c_int32 * (cap + _MIN_UNIT_COUNT))()
        self.out_rate = (ctypes.c_double * cap)()
        self.place_cap = cap
        self.place_units = (ctypes.c_int32 * cap)()
        self.out_fault = (ctypes.c_int32 * _FAULT_CELLS)()

    def ensure_res(self, need: int) -> None:
        if need <= self.res_cap:
            return
        cap = max(need, self.res_cap * _SCRATCH_GROWTH_FACTOR)
        self.res_cap = cap
        self.res_rate, self.res_units = (ctypes.c_double * cap)(), (ctypes.c_int32 * cap)()

    def ensure_tasks(self, need: int) -> None:
        if need <= self.task_cap:
            return
        cap = max(need, self.task_cap * _SCRATCH_GROWTH_FACTOR)
        self.task_cap = cap
        self.task_coupled, self.task_cohort = (ctypes.c_uint8 * cap)(), (ctypes.c_int32 * cap)()
        self.task_link_off = (ctypes.c_int32 * (cap + _MIN_UNIT_COUNT))()

    def ensure_links(self, need: int) -> None:
        if need <= self.link_cap:
            return
        cap = max(need, self.link_cap * _SCRATCH_GROWTH_FACTOR)
        self.link_cap = cap
        self.link_res, self.link_rem = (ctypes.c_int32 * cap)(), (ctypes.c_double * cap)()
        self.link_cap_arr, self.out_rate = (ctypes.c_double * cap)(), (ctypes.c_double * cap)()
        self.link_widened, self.link_unplaced = (ctypes.c_uint8 * cap)(), (ctypes.c_uint8 * cap)()
        self.link_place_off = (ctypes.c_int32 * (cap + _MIN_UNIT_COUNT))()

    def ensure_place(self, need: int) -> None:
        if need <= self.place_cap:
            return
        cap = max(need, self.place_cap * _SCRATCH_GROWTH_FACTOR)
        self.place_cap, self.place_units = cap, (ctypes.c_int32 * cap)()


def _get_scratch() -> _CoupledScratch:
    scratch = getattr(_THREAD_SCRATCH, "buf", None)
    if scratch is None:
        scratch = _CoupledScratch()
        _THREAD_SCRATCH.buf = scratch
    return scratch


def _get_native_coupled_fn():
    """Return bound C11 ``msim_fast_single_tier_coupled`` or ``None``."""
    env = os.environ
    mode_raw = env.get("MINDIE_SIM_ACCEL")
    if mode_raw is None:
        mode_raw = env.get("MINDIE_SIM_ENGINE_BACKEND")
    if mode_raw is not None and mode_raw.strip().lower() in _PYTHON_MODES:
        return None
    cache_base = env.get("MINDIE_CACHE_HOME")
    if cache_base:
        base_tag = ("cache", cache_base)
    else:
        data_base = env.get("MINDIE_DATA_HOME")
        if data_base:
            base_tag = ("data", data_base)
        else:
            xdg_base = env.get("XDG_CACHE_HOME")
            if xdg_base:
                base_tag = ("xdg", xdg_base)
            else:
                base_tag = ("home", env.get("HOME"), env.get("USERPROFILE"))
    env_key = (mode_raw, base_tag, env.get("MINDIE_SIM_CC"))
    cached = _NATIVE_FN_CACHE.get(env_key, _UNSET)
    if cached is not _UNSET:
        return cached
    from . import accel_loader, accel_sync

    if accel_sync.resolve_mode(None) == "python":
        return None
    with _NATIVE_FN_LOCK:
        if env_key in _NATIVE_FN_CACHE:
            return _NATIVE_FN_CACHE[env_key]
    fn = None
    try:
        _status, lib = accel_loader.load_accel_cdll()
        candidate = getattr(lib, "msim_fast_single_tier_coupled", None)
        if candidate is not None:
            fn = candidate
    except TimeoutError:
        raise
    except (accel_loader.AccelUnavailable, OSError, RuntimeError):
        fn = None
    with _NATIVE_FN_LOCK:
        _NATIVE_FN_CACHE[env_key] = fn
    return fn


def reset_native_coupled_cache() -> None:
    """Clear process-local native coupled-arbiter handle memo (tests only)."""
    with _NATIVE_FN_LOCK:
        _NATIVE_FN_CACHE.clear()


def _collect_single_tier(tasks, active) -> list[int] | None:
    """Return sorted active task keys if single-priority, else ``None``."""
    first_prio = None
    tier: list[int] = []
    for key in active:
        task = tasks[key]
        if not task.remaining:
            continue
        if not task.coupled and getattr(task, "cohort_size", 1) > 1:
            return None
        prio = task.priority
        if first_prio is None:
            first_prio = prio
        elif prio != first_prio:
            return None
        tier.append(key)
    if len(tier) > _MIN_UNIT_COUNT:
        tier.sort()
    return tier


def fast_single_tier_coupled_c(
    resources, tasks, active, *, native_fn=None,
) -> dict[str, dict[int, float]] | None:
    """Native C11 fast path for single-priority non-FIFO coupled tiers."""
    tier = _collect_single_tier(tasks, active)
    if tier is None:
        return None
    if not tier:
        return {}
    fn = _get_native_coupled_fn() if native_fn is None else native_fn
    if fn is None:
        return fast_single_tier_coupled_py(resources, tasks, active)

    res_ids: dict[str, int] = {}
    res_names: list[str] = []
    n_links = 0
    n_place = 0
    for key in tier:
        task = tasks[key]
        pl = task.placement
        for name, amount in task.remaining.items():
            if amount <= 0.0:
                continue
            if name not in res_ids:
                res = resources[name]
                if res.policy == "fifo":
                    return None
                res_ids[name] = len(res_names)
                res_names.append(name)
            n_links += _MIN_UNIT_COUNT
            if pl:
                decl = pl.get(name)
                if decl is not None:
                    n_place += len(decl)
    if not n_links:
        return {}

    buf = _get_scratch()
    n_res = len(res_names)
    n_tasks = len(tier)
    buf.ensure_res(n_res)
    buf.ensure_tasks(n_tasks)
    buf.ensure_links(n_links)
    buf.ensure_place(max(_MIN_UNIT_COUNT, n_place))
    place_base = ctypes.addressof(buf.place_units)
    memmove = ctypes.memmove

    for ri, name in enumerate(res_names):
        res = resources[name]
        buf.res_rate[ri] = float(res.rate)
        buf.res_units[ri] = max(_MIN_UNIT_COUNT, int(res.units))

    link_res_names: list[str] = []
    link_task_keys: list[int] = []
    li = 0
    pi = 0
    inf = math.inf
    for ti, key in enumerate(tier):
        task = tasks[key]
        buf.task_coupled[ti] = _MIN_UNIT_COUNT if task.coupled else _FAULT_OK
        cohort = int(task.cohort_size or _MIN_UNIT_COUNT)
        if cohort > _MAX_NATIVE_COHORT:
            return fast_single_tier_coupled_py(resources, tasks, active)
        buf.task_cohort[ti] = cohort if cohort >= _MIN_UNIT_COUNT else _MIN_UNIT_COUNT
        buf.task_link_off[ti] = li
        pl = task.placement
        caps_map = task.caps
        widened = getattr(task, "cohort_widened", ())
        for name, amount in task.remaining.items():
            if amount <= 0.0:
                continue
            buf.link_res[li] = res_ids[name]
            buf.link_rem[li] = float(amount)
            cap_val = caps_map.get(name) if caps_map else None
            buf.link_cap_arr[li] = inf if cap_val is None else float(cap_val)
            buf.link_widened[li] = _MIN_UNIT_COUNT if (widened and name in widened) else _FAULT_OK
            decl = pl.get(name) if pl else None
            buf.link_place_off[li] = pi
            if decl is None:
                buf.link_unplaced[li] = _MIN_UNIT_COUNT
            else:
                buf.link_unplaced[li] = _FAULT_OK
                if type(decl) is tuple and decl:
                    packed_arr, packed_cnt, packed_bytes = _pack_place_units(decl)
                    memmove(place_base + pi * _INT32_BYTES, packed_arr, packed_bytes)
                    pi += packed_cnt
                else:
                    for unit in decl:
                        buf.place_units[pi] = int(unit)
                        pi += _MIN_UNIT_COUNT
            link_res_names.append(name)
            link_task_keys.append(key)
            li += _MIN_UNIT_COUNT
    buf.task_link_off[n_tasks] = li
    buf.link_place_off[li] = pi

    rc = int(fn(
        n_res, buf.res_rate, buf.res_units,
        n_tasks, buf.task_coupled, buf.task_cohort, buf.task_link_off,
        li, buf.link_res, buf.link_rem, buf.link_cap_arr,
        buf.link_widened, buf.link_unplaced, buf.link_place_off, buf.place_units,
        buf.out_rate, buf.out_fault,
    ))
    fault_code = int(buf.out_fault[_FAULT_CODE_IDX])
    if rc != _FAULT_OK:
        if fault_code == _FAULT_SCALE_UNDERFLOW:
            raise ZeroDivisionError("coupled_scale_underflow")
        if fault_code == _FAULT_OVERALLOCATED:
            r_idx = int(buf.out_fault[_FAULT_RES_IDX])
            u_idx = int(buf.out_fault[_FAULT_UNIT_IDX])
            r_name = res_names[r_idx] if 0 <= r_idx < n_res else str(r_idx)
            raise AssertionError(f"unit_service_overallocated:{r_name}:{u_idx}")
        return fast_single_tier_coupled_py(resources, tasks, active)

    result: dict[str, dict[int, float]] = {}
    out_rate = buf.out_rate
    for idx in range(li):
        rate = float(out_rate[idx])
        if rate <= 0.0:
            continue
        rname = link_res_names[idx]
        tkey = link_task_keys[idx]
        target = result.get(rname)
        if target is None:
            result[rname] = {tkey: rate}
        else:
            target[tkey] = target.get(tkey, 0.0) + rate
    return result


def fast_single_tier_coupled(
    resources, tasks, active, *, allow_native: bool = True,
) -> dict[str, dict[int, float]] | None:
    """Fast path for single-priority non-FIFO cohorts; ``None`` delegates out."""
    if allow_native:
        fn = _get_native_coupled_fn()
        if fn is not None:
            return fast_single_tier_coupled_c(resources, tasks, active, native_fn=fn)
    return fast_single_tier_coupled_py(resources, tasks, active)


def _push_slot_item(sbox, is_full: bool, is_unplaced: bool, units, item) -> None:
    """Append ``(flow_idx, weight, span_id)`` into resource full/unit buckets."""
    if is_unplaced:
        sbox[2] = True
    if is_full:
        fslot = sbox[0]
        if fslot is None:
            sbox[0] = item
        elif type(fslot) is tuple:
            sbox[0] = [fslot, item]
        else:
            fslot.append(item)
        return
    umap = sbox[1]
    if umap is None:
        umap = sbox[1] = {}
    for u in units:
        cur = umap.get(u)
        if cur is None:
            umap[u] = item
        elif type(cur) is tuple:
            umap[u] = [cur, item]
        else:
            cur.append(item)


def _collect_coupled_flow(resources, task, key, res_info, flow_keys, flow_limits, flow_rates, span_meta):
    res_get = res_info.get
    range_get = _UNIT_RANGE_CACHE.get
    ulp = math.ulp
    remaining = task.remaining
    placement = task.placement
    caps_map = task.caps
    inf = math.inf
    isfinite = math.isfinite
    cohort = task.cohort_size or 1
    if cohort < 1:
        cohort = 1
    widened_set = getattr(task, "cohort_widened", ())
    spans: list[tuple[str, float, tuple[int, ...], int, float, list, bool, bool]] = []
    blocked = False
    scale = 0.0
    for name, amount in remaining.items():
        if amount <= 0.0:
            continue
        info = res_get(name)
        if info is None:
            res = resources[name]
            if res.policy == "fifo":
                return False
            cnt = max(1, int(res.units))
            u_cap = float(res.rate) / cnt
            sat_tol = _SAT_ULPS * ulp(u_cap)
            bind_thresh = u_cap - _SAT_ULPS * 2.0 * ulp(u_cap)
            full_u = range_get(cnt)
            if full_u is None:
                full_u = _UNIT_RANGE_CACHE[cnt] = tuple(range(cnt))
            info = res_info[name] = (u_cap, cnt, sat_tol, bind_thresh, [None, None, False], full_u)
        u_cap, cnt, _sat_tol, _bind_thresh, sbox, full_u = info
        if u_cap <= 0.0:
            blocked = True
            break
        decl = placement.get(name) if placement else None
        if decl is None:
            units, n_u, is_full, is_unplaced = full_u, cnt, True, cnt > 1
        else:
            units = (decl if (decl and 0 <= decl[0] and decl[-1] < cnt
                              and (len(decl) <= 2 or all(0 <= u < cnt for u in decl)))
                     else tuple(u for u in decl if 0 <= u < cnt))
            n_u = len(units)
            if not n_u:
                blocked = True
                break
            is_full = units == full_u
            is_unplaced = False
        amt_f = float(amount)
        denom = u_cap * n_u
        if cohort > 1 and name not in widened_set:
            denom *= cohort
        if denom <= 0.0:
            scale = inf
        elif scale != inf:
            val = amt_f / denom
            if val > scale:
                scale = val
        spans.append((name, amt_f, units, n_u, u_cap, sbox, is_full, is_unplaced))
    if blocked or not spans:
        return True
    if scale == 0.0:
        raise ZeroDivisionError("coupled_scale_underflow")
    if not isfinite(scale) or scale < 0.0:
        return True
    limit = 1.0
    if caps_map:
        for name, amt_f, _u, _nu, _uc, _sb, _if, _iu in spans:
            cap = caps_map.get(name)
            if cap is not None:
                demand = amt_f / scale
                c_lim = (max(0.0, float(cap)) / demand) if demand > 0.0 else 0.0
                if c_lim < limit:
                    limit = c_lim
        if limit <= 0.0:
            return True
    f_idx = len(flow_keys)
    flow_keys.append(key)
    flow_limits.append(limit)
    flow_rates.append([(name, amt_f / scale) for name, amt_f, _u, _nu, _uc, _sb, _if, _iu in spans])
    s_flow, s_dem, s_rem, s_unp, s_w, s_ba = span_meta
    for name, amt_f, units, n_u, _u_cap, sbox, is_full, is_unplaced in spans:
        demand = amt_f / scale
        weight = demand / n_u
        s_id = len(s_flow)
        s_flow.append(f_idx)
        s_dem.append(demand)
        s_rem.append(n_u)
        s_unp.append(is_unplaced)
        s_w.append(weight)
        s_ba.append(0.0)
        _push_slot_item(sbox, is_full, is_unplaced, units, (f_idx, weight, s_id))
    return True


def _collect_independent_flows(resources, task, key, res_info, flow_keys, flow_limits, flow_rates, span_meta):
    res_get = res_info.get
    range_get = _UNIT_RANGE_CACHE.get
    ulp = math.ulp
    remaining = task.remaining
    placement = task.placement
    caps_map = task.caps
    s_flow, s_dem, s_rem, s_unp, s_w, s_ba = span_meta
    for name, amount in remaining.items():
        if amount <= 0.0:
            continue
        info = res_get(name)
        if info is None:
            res = resources[name]
            if res.policy == "fifo":
                return False
            cnt = max(1, int(res.units))
            u_cap = float(res.rate) / cnt
            sat_tol = _SAT_ULPS * ulp(u_cap)
            bind_thresh = u_cap - _SAT_ULPS * 2.0 * ulp(u_cap)
            full_u = range_get(cnt)
            if full_u is None:
                full_u = _UNIT_RANGE_CACHE[cnt] = tuple(range(cnt))
            info = res_info[name] = (u_cap, cnt, sat_tol, bind_thresh, [None, None, False], full_u)
        u_cap, cnt, _sat_tol, _bind_thresh, sbox, full_u = info
        if u_cap <= 0.0:
            continue
        decl = placement.get(name) if placement else None
        if decl is None:
            units, n_u, is_full, is_unplaced = full_u, cnt, True, cnt > 1
        else:
            units = (decl if (decl and 0 <= decl[0] and decl[-1] < cnt
                              and (len(decl) <= 2 or all(0 <= u < cnt for u in decl)))
                     else tuple(u for u in decl if 0 <= u < cnt))
            n_u = len(units)
            if not n_u:
                continue
            is_full = units == full_u
            is_unplaced = False
        demand = u_cap * n_u
        if demand <= 0.0:
            continue
        limit = 1.0
        if caps_map:
            cap = caps_map.get(name)
            if cap is not None:
                c_lim = max(0.0, float(cap)) / demand
                if c_lim < limit:
                    limit = c_lim
            if limit <= 0.0:
                continue
        f_idx = len(flow_keys)
        flow_keys.append(key)
        flow_limits.append(limit)
        flow_rates.append(((name, demand),))
        weight = demand / n_u
        s_id = len(s_flow)
        s_flow.append(f_idx)
        s_dem.append(demand)
        s_rem.append(n_u)
        s_unp.append(is_unplaced)
        s_w.append(weight)
        s_ba.append(0.0)
        _push_slot_item(sbox, is_full, is_unplaced, units, (f_idx, weight, s_id))
    return True


def _fold_private_slot(pc_append, flow_limits, name: str, u: int, u_cap: float, item) -> None:
    f_idx, w, _sid = item
    pc_append((name, u, u_cap, f_idx, w))
    if w > 0.0:
        b = u_cap / w
        if b < flow_limits[f_idx]:
            flow_limits[f_idx] = b
        while flow_limits[f_idx] > 0.0 and flow_limits[f_idx] * w > u_cap:
            flow_limits[f_idx] = math.nextafter(flow_limits[f_idx], 0.0)


def _partition_capacity_slots(res_info, flow_limits, s_unp):
    shared_units = []
    su_append = shared_units.append
    private_checks: list[tuple[str, int, float, int, float]] = []
    pc_append = private_checks.append
    for name, (u_cap, cnt, sat_tol, bind_thresh, sbox, full_u) in res_info.items():
        fslot, umap, has_unplaced = sbox
        if umap is None:
            if type(fslot) is tuple:
                s_unp[fslot[2]] = False
                _fold_private_slot(pc_append, flow_limits, name, 0, u_cap, fslot)
            elif fslot is not None:
                # All ``cnt`` units carry identical contenders and saturate
                # simultaneously; collapsed spans cannot redistribute.
                for _f_idx, _w, s_id in fslot:
                    s_unp[s_id] = False
                su_append((name, 0, u_cap, sat_tol, bind_thresh, fslot, False))
            continue
        if fslot is None:
            for u in (sorted(umap) if len(umap) > 1 else umap):
                cur = umap[u]
                if type(cur) is tuple:
                    _fold_private_slot(pc_append, flow_limits, name, u, u_cap, cur)
                else:
                    su_append((name, u, u_cap, sat_tol, bind_thresh, cur, False))
            continue
        f_is_tuple = type(fslot) is tuple
        f_part = (fslot,) if f_is_tuple else fslot
        if has_unplaced and cnt > 1:
            for u in range(cnt):
                cur = umap.get(u)
                if cur is None:
                    terms = list(f_part)
                else:
                    u_part = (cur,) if type(cur) is tuple else cur
                    terms = sorted([*f_part, *u_part])
                su_append((name, u, u_cap, sat_tol, bind_thresh, terms, True))
            continue
        if len(umap) < cnt:
            u_rep = next(u for u in full_u if u not in umap)
            if f_is_tuple:
                _fold_private_slot(pc_append, flow_limits, name, u_rep, u_cap, fslot)
            else:
                su_append((name, u_rep, u_cap, sat_tol, bind_thresh, fslot, False))
        for u in (sorted(umap) if len(umap) > 1 else umap):
            cur = umap[u]
            if f_is_tuple and type(cur) is tuple:
                terms = [fslot, cur] if fslot[0] <= cur[0] else [cur, fslot]
            else:
                u_part = (cur,) if type(cur) is tuple else cur
                terms = sorted([*f_part, *u_part])
            su_append((name, u, u_cap, sat_tol, bind_thresh, terms, False))
    return private_checks, shared_units


def _binding_capacity_slots(shared_units, flow_limits):
    fsum = math.fsum
    binding_units = []
    shared_to_bind: list[int] = [-1] * len(shared_units)
    seen_binding: dict[tuple[float, tuple[tuple[int, float], ...]], int] = {}
    for idx, (_name, _u, cap, sat_tol, bind_thresh, terms, force_bind) in enumerate(shared_units):
        if force_bind:
            shared_to_bind[idx] = len(binding_units)
            binding_units.append((cap, sat_tol, terms, True))
            continue
        n_t = len(terms)
        max_tot = (
            terms[0][1] * flow_limits[terms[0][0]] if n_t == 1
            else (terms[0][1] * flow_limits[terms[0][0]] + terms[1][1] * flow_limits[terms[1][0]]) if n_t == 2
            else fsum([flow_limits[f_idx] * w for f_idx, w, _sid in terms])
        )
        if max_tot > bind_thresh:
            key_u = (cap, tuple((f_idx, w) for f_idx, w, _sid in terms))
            hit = seen_binding.get(key_u)
            if hit is None:
                hit = len(binding_units)
                seen_binding[key_u] = hit
                binding_units.append((cap, sat_tol, terms, False))
            shared_to_bind[idx] = hit
    return binding_units, shared_to_bind


def _isolated_slot_delta(cap, sat_tol, terms, flow_limits):
    """Solve a slot whose flows have no other binding capacity slots."""
    nextafter = math.nextafter
    fsum = math.fsum
    ulp = math.ulp
    n_t = len(terms)
    slope = (
        terms[0][1] if n_t == 1
        else (terms[0][1] + terms[1][1]) if n_t == 2
        else fsum([w for _f, w, _sid in terms])
    )
    denom = slope if n_t == 1 else nextafter(slope, math.inf)
    delta = cap / denom
    if not all(delta <= flow_limits[f_idx] for f_idx, _w, _sid in terms):
        return None
    for attempt in range(_MAX_ATTEMPTS):
        if delta <= 0.0:
            break
        tot = (
            (delta * terms[0][1]) if n_t == 1
            else (delta * terms[0][1] + delta * terms[1][1]) if n_t == 2
            else fsum([delta * w for _f, w, _sid in terms])
        )
        if tot <= cap:
            res_v = cap - tot
            tol = (sat_tol if 0.0 <= res_v <= cap
                   else _SAT_ULPS * ulp(max(abs(cap), abs(res_v))))
            if res_v <= tol:
                return delta
            break
        delta = nextafter(delta, 0.0) if attempt < _NEXTAFTER_STEPS else delta * _HALF_STEP
    return None


def _solve_binding_slots(binding_units, flow_limits, span_meta):
    n_flows = len(flow_limits)
    s_unp = span_meta[3]
    alphas = [0.0] * n_flows
    deg = [0] * n_flows
    for _cap, _sat_tol, terms, _fb in binding_units:
        for f_idx, _w, _sid in terms:
            deg[f_idx] += 1
    active_flags = [False] * n_flows
    for f_idx in range(n_flows):
        lim = flow_limits[f_idx]
        if deg[f_idx] == 0:
            alphas[f_idx] = lim
        elif lim > 0.0:
            active_flags[f_idx] = True
    rem_units = []
    bind_to_rem: list[int] = [-1] * len(binding_units)
    for b_idx, (cap, sat_tol, terms, force_bind) in enumerate(binding_units):
        if not force_bind and all(
            deg[f_idx] == 1 and active_flags[f_idx] and not s_unp[s_id]
            for f_idx, _w, s_id in terms
        ):
            delta = _isolated_slot_delta(cap, sat_tol, terms, flow_limits)
            if delta is not None:
                for f_idx, _w, _sid in terms:
                    alphas[f_idx] = delta
                    active_flags[f_idx] = False
                continue
        bind_to_rem[b_idx] = len(rem_units)
        rem_units.append((cap, sat_tol, terms))
    committed = (
        _fill_shared_slots(rem_units, flow_limits, alphas, active_flags, span_meta)
        if rem_units else []
    )
    return alphas, bind_to_rem, committed


def _slot_term_val(f_idx: int, s_id: int, alpha_next: float, base_c: float, s_w, s_ba) -> float:
    ba = s_ba[s_id]
    w = s_w[s_id]
    return (base_c + (alpha_next - ba) * w) if ba else (alpha_next * w)


def _fit_shared_step(delta, act_indices, open_shared, rem_units, base_comm, committed,
                     alphas, flow_limits, active_flags, s_w, s_ba):
    nextafter = math.nextafter
    fsum = math.fsum
    blocking: set[int] = set()
    for attempt in range(_MAX_ATTEMPTS):
        if delta <= 0.0:
            return 0.0, [], blocking
        c_sub: list[tuple[int, float]] = []
        failed: set[int] = set()
        cap_fail = any(alphas[f_idx] + delta > flow_limits[f_idx] for f_idx in act_indices)
        if not cap_fail:
            for s_idx in open_shared:
                cap, _sat_tol, terms = rem_units[s_idx]
                comm_s = committed[s_idx]
                bcomm_s = base_comm[s_idx]
                vals = [
                    (_slot_term_val(f_idx, s_id, alphas[f_idx] + delta, bcomm_s[j], s_w, s_ba)
                     if active_flags[f_idx] else comm_s[j])
                    for j, (f_idx, _w, s_id) in enumerate(terms)
                ]
                tot = vals[0] if len(vals) == 1 else (vals[0] + vals[1] if len(vals) == 2 else fsum(vals))
                if tot > cap:
                    failed.add(s_idx)
                else:
                    c_sub.append((s_idx, tot))
        if not cap_fail and not failed:
            return delta, c_sub, blocking
        if failed:
            blocking = failed
        delta = nextafter(delta, 0.0) if attempt < _NEXTAFTER_STEPS else delta * _HALF_STEP
    return 0.0, [], blocking


def _retire_shared_units(open_shared, rem_units, totals, blocking,
                         alphas, flow_limits, active_flags,
                         base_comm, committed, span_meta):
    """Freeze pinned flows and redistribute unplaced spans on saturated slots."""
    ulp = math.ulp
    s_flow, s_dem, s_rem, s_unp, s_w, s_ba = span_meta
    next_open: list[int] = []
    redist: set[int] = set()
    froze_any = False
    for s_idx in open_shared:
        cap, sat_tol, terms = rem_units[s_idx]
        res_v = cap - totals[s_idx]
        tol = sat_tol if 0.0 <= res_v <= cap else _SAT_ULPS * ulp(max(abs(cap), abs(res_v)))
        has_uncapped = any(
            active_flags[f_idx]
            and alphas[f_idx] < flow_limits[f_idx] - _CAP_ULPS * ulp(
                max(abs(flow_limits[f_idx]), abs(alphas[f_idx])))
            for f_idx, _w, _sid in terms
        )
        unit_bound = res_v <= 0.0 or ((s_idx in blocking or has_uncapped) and res_v <= tol)
        if not unit_bound:
            next_open.append(s_idx)
            continue
        for f_idx, _w, s_id in terms:
            if not active_flags[f_idx]:
                continue
            froze_any = True
            if not s_unp[s_id]:
                active_flags[f_idx] = False
                continue
            s_rem[s_id] -= 1
            if s_rem[s_id] <= 0:
                active_flags[f_idx] = False
            else:
                redist.add(s_id)
    if redist:
        for os_idx in next_open:
            comm_s = committed[os_idx]
            bcomm_s = base_comm[os_idx]
            for j, (_f_idx, _w, s_id) in enumerate(rem_units[os_idx][2]):
                if s_id in redist:
                    bcomm_s[j] = comm_s[j]
        for s_id in redist:
            f_idx = s_flow[s_id]
            if active_flags[f_idx]:
                s_ba[s_id] = alphas[f_idx]
                s_w[s_id] = s_dem[s_id] / s_rem[s_id]
    return next_open, froze_any


def _fill_shared_slots(rem_units, flow_limits, alphas, active_flags, span_meta):
    nextafter = math.nextafter
    fsum = math.fsum
    ulp = math.ulp
    inf = math.inf
    isfinite = math.isfinite
    s_w, s_ba = span_meta[4], span_meta[5]
    n_rem = len(rem_units)
    act_indices = [i for i, ok in enumerate(active_flags) if ok]
    base_comm = [[0.0] * len(rem_units[s_idx][2]) for s_idx in range(n_rem)]
    committed = [[0.0] * len(rem_units[s_idx][2]) for s_idx in range(n_rem)]
    totals = [0.0] * n_rem
    open_shared = list(range(n_rem))
    while act_indices and open_shared:
        bound_limit = inf
        still_open: list[int] = []
        for s_idx in open_shared:
            cap, _sat_tol, terms = rem_units[s_idx]
            weights = [s_w[s_id] for f_idx, _w, s_id in terms if active_flags[f_idx]]
            if not weights:
                continue
            still_open.append(s_idx)
            cnt_w = len(weights)
            slope = weights[0] if cnt_w == 1 else (weights[0] + weights[1] if cnt_w == 2 else fsum(weights))
            if slope > 0.0:
                denom = slope if cnt_w == 1 else nextafter(slope, inf)
                b = (cap - totals[s_idx]) / denom
                if b < bound_limit:
                    bound_limit = b
        open_shared = still_open
        if not open_shared:
            for f_idx in act_indices:
                alphas[f_idx] = flow_limits[f_idx]
            break
        headroom = min(flow_limits[f_idx] - alphas[f_idx] for f_idx in act_indices)
        raw_delta = bound_limit if bound_limit < headroom else headroom
        if not isfinite(raw_delta) or raw_delta < 0.0:
            break
        delta, cand_sub, blocking = _fit_shared_step(
            raw_delta, act_indices, open_shared, rem_units, base_comm, committed,
            alphas, flow_limits, active_flags, s_w, s_ba,
        )
        advanced = False
        if delta > 0.0:
            for f_idx in act_indices:
                lim_f = flow_limits[f_idx]
                raw_f = alphas[f_idx] + delta
                val = lim_f if raw_f > lim_f else raw_f
                if val != alphas[f_idx]:
                    advanced = True
                alphas[f_idx] = val
            for s_idx in open_shared:
                comm_s = committed[s_idx]
                bcomm_s = base_comm[s_idx]
                for j, (f_idx, _w, s_id) in enumerate(rem_units[s_idx][2]):
                    if active_flags[f_idx]:
                        comm_s[j] = _slot_term_val(f_idx, s_id, alphas[f_idx], bcomm_s[j], s_w, s_ba)
            for s_idx, tot in cand_sub:
                totals[s_idx] = tot
        open_shared, froze_any = _retire_shared_units(
            open_shared, rem_units, totals, blocking,
            alphas, flow_limits, active_flags, base_comm, committed, span_meta,
        )
        no_step = not advanced
        next_act: list[int] = []
        for f_idx in act_indices:
            if not active_flags[f_idx]:
                continue
            lim = flow_limits[f_idx]
            al = alphas[f_idx]
            cap_tol = _CAP_ULPS * ulp(max(abs(lim), abs(al)))
            if al >= lim - cap_tol or (no_step and (lim - al) <= headroom + cap_tol):
                active_flags[f_idx] = False
                froze_any = True
            else:
                next_act.append(f_idx)
        if not froze_any:
            break
        act_indices = next_act
    return committed


def fast_single_tier_coupled_py(resources, tasks, active) -> dict[str, dict[int, float]] | None:
    """Pure-Python reference for single-priority non-FIFO coupled tiers."""
    tier = _collect_single_tier(tasks, active)
    if tier is None:
        return None
    if not tier:
        return {}

    res_info: dict[str, tuple[float, int, float, float, list, tuple[int, ...]]] = {}
    flow_keys: list[int] = []
    flow_limits: list[float] = []
    flow_rates: list[tuple[tuple[str, float], ...]] = []
    span_meta = ([], [], [], [], [], [])
    for key in tier:
        task = tasks[key]
        collect = _collect_coupled_flow if task.coupled else _collect_independent_flows
        if not collect(resources, task, key, res_info, flow_keys, flow_limits, flow_rates, span_meta):
            return None

    if not flow_keys:
        return {}

    private_checks, shared_units = _partition_capacity_slots(res_info, flow_limits, span_meta[3])
    binding_units, shared_to_bind = _binding_capacity_slots(shared_units, flow_limits)
    if binding_units:
        alphas, bind_to_rem, committed = _solve_binding_slots(binding_units, flow_limits, span_meta)
    else:
        alphas, bind_to_rem, committed = flow_limits, [], []

    fsum = math.fsum
    for name, u, cap, f_idx, w in private_checks:
        if alphas[f_idx] * w > cap:
            raise AssertionError(f"unit_service_overallocated:{name}:{u}")
    for s_idx, (name, u, cap, _sat_tol, _bind_thresh, terms, fb) in enumerate(shared_units):
        b_idx = shared_to_bind[s_idx]
        r_idx = bind_to_rem[b_idx] if b_idx >= 0 else -1
        if fb and r_idx >= 0:
            vals = committed[r_idx]
            n_t = len(vals)
            tot_u = vals[0] if n_t == 1 else (vals[0] + vals[1] if n_t == 2 else fsum(vals))
        else:
            n_t = len(terms)
            tot_u = (
                alphas[terms[0][0]] * terms[0][1] if n_t == 1
                else (alphas[terms[0][0]] * terms[0][1] + alphas[terms[1][0]] * terms[1][1]) if n_t == 2
                else fsum([alphas[f_idx] * w for f_idx, w, _sid in terms])
            )
        if tot_u > cap:
            raise AssertionError(f"unit_service_overallocated:{name}:{u}")

    result: dict[str, dict[int, float]] = {}
    for f_idx, alpha in enumerate(alphas):
        if alpha <= 0.0:
            continue
        key = flow_keys[f_idx]
        for name, demand in flow_rates[f_idx]:
            rate = alpha * demand
            if rate <= 0.0:
                continue
            target = result.get(name)
            if target is None:
                result[name] = {key: rate}
            else:
                target[key] = target.get(key, 0.0) + rate
    return result
