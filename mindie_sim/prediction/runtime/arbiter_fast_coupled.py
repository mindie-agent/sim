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
import math
import threading

_MIN_RATE_PER_MS = 1e-12
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

_NATIVE_FN_CACHE: dict[str, object] = {}
_NATIVE_FN_LOCK = threading.Lock()
_THREAD_SCRATCH = threading.local()


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
    from . import accel_build, accel_loader, accel_sync

    if accel_sync.resolve_mode(None) == "python":
        return None
    cache_key = str(accel_build.cache_root())
    with _NATIVE_FN_LOCK:
        if cache_key in _NATIVE_FN_CACHE:
            return _NATIVE_FN_CACHE[cache_key]
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
        _NATIVE_FN_CACHE[cache_key] = fn
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


def fast_single_tier_coupled_py(resources, tasks, active) -> dict[str, dict[int, float]] | None:
    """Pure-Python reference for single-priority non-FIFO coupled tiers."""
    ni, zi, oi, ti = -1, 0, 1, 2
    zf, of, tf = 0.0, 1.0, 2.0

    first_prio = None
    tier: list[int] = []
    tier_append = tier.append
    for key in active:
        task = tasks[key]
        if not task.remaining:
            continue
        prio = task.priority
        if first_prio is None:
            first_prio = prio
        elif prio != first_prio:
            return None
        tier_append(key)
    if not tier:
        return {}
    if len(tier) > oi:
        tier.sort()

    res_info: dict[str, tuple[float, int, float, float, list, tuple[int, ...]]] = {}
    res_get = res_info.get
    range_get = _UNIT_RANGE_CACHE.get
    flow_keys: list[int] = []
    flow_limits: list[float] = []
    flow_rates: list[tuple[tuple[str, float], ...]] = []
    shared_units: list[tuple[str, int, float, float, float, list[tuple[int, float]]]] = []
    fk_append = flow_keys.append
    fl_append = flow_limits.append
    fr_append = flow_rates.append
    su_append = shared_units.append

    nextafter = math.nextafter
    fsum = math.fsum
    ulp = math.ulp
    inf = math.inf
    isfinite = math.isfinite

    for key in tier:
        task = tasks[key]
        remaining = task.remaining
        placement = task.placement
        caps_map = task.caps
        if task.coupled:
            cohort = task.cohort_size or oi
            if cohort < oi:
                cohort = oi
            widened_set = getattr(task, "cohort_widened", ())
            spans: list[tuple[str, float, tuple[int, ...], int, float, list, bool]] = []
            spans_append = spans.append
            blocked = False
            scale = zf
            for name, amount in remaining.items():
                if amount <= zf:
                    continue
                info = res_get(name)
                if info is None:
                    res = resources[name]
                    if res.policy == "fifo":
                        return None
                    cnt = max(oi, int(res.units))
                    u_cap = float(res.rate) / cnt
                    cap_scale = of if u_cap < of else u_cap
                    sat_tol = max(_MIN_RATE_PER_MS, _SAT_ULPS * ulp(cap_scale))
                    bind_thresh = u_cap - max(_MIN_RATE_PER_MS, _SAT_ULPS * tf * ulp(cap_scale))
                    full_u = range_get(cnt)
                    if full_u is None:
                        full_u = _UNIT_RANGE_CACHE[cnt] = tuple(range(cnt))
                    info = res_info[name] = (u_cap, cnt, sat_tol, bind_thresh, [None, None], full_u)
                u_cap, cnt, _sat_tol, _bind_thresh, sbox, full_u = info
                if u_cap <= _MIN_RATE_PER_MS:
                    blocked = True
                    break
                decl = placement.get(name) if placement else None
                if decl is None:
                    units = full_u
                    n_u = cnt
                    is_full = True
                else:
                    units = (decl if (decl and zi <= decl[zi] and decl[ni] < cnt
                                      and (len(decl) <= ti or all(zi <= u < cnt for u in decl)))
                             else tuple(u for u in decl if zi <= u < cnt))
                    n_u = len(units)
                    if not n_u:
                        blocked = True
                        break
                    is_full = units == full_u
                amt_f = float(amount)
                denom = u_cap * n_u
                if cohort > oi and name not in widened_set:
                    denom *= cohort
                if denom <= zf:
                    scale = inf
                elif scale != inf:
                    val = amt_f / denom
                    if val > scale:
                        scale = val
                spans_append((name, amt_f, units, n_u, u_cap, sbox, is_full))
            if blocked or not spans:
                continue
            if scale == zf:
                raise ZeroDivisionError("coupled_scale_underflow")
            if not isfinite(scale) or scale < zf:
                continue
            limit = of
            if caps_map:
                for name, amt_f, _units, _n_u, _u_cap, _sbox, _is_full in spans:
                    cap = caps_map.get(name)
                    if cap is not None:
                        demand = amt_f / scale
                        c_lim = (max(zf, float(cap)) / demand) if demand > zf else zf
                        if c_lim < limit:
                            limit = c_lim
                if limit <= zf:
                    continue
            f_idx = len(flow_keys)
            fk_append(key)
            fl_append(limit)
            fr_append([(name, amt_f / scale)
                       for name, amt_f, _u, _nu, _uc, _sb, _if in spans])
            for name, amt_f, units, n_u, _u_cap, sbox, is_full in spans:
                weight = (amt_f / scale) / n_u
                item = (f_idx, weight)
                if is_full:
                    fslot = sbox[zi]
                    if fslot is None:
                        sbox[zi] = item
                    elif type(fslot) is tuple:
                        sbox[zi] = [fslot, item]
                    else:
                        fslot.append(item)
                else:
                    umap = sbox[oi]
                    if umap is None:
                        umap = sbox[oi] = {}
                    for u in units:
                        cur = umap.get(u)
                        if cur is None:
                            umap[u] = item
                        elif type(cur) is tuple:
                            umap[u] = [cur, item]
                        else:
                            cur.append(item)
            continue

        for name, amount in remaining.items():
            if amount <= zf:
                continue
            info = res_get(name)
            if info is None:
                res = resources[name]
                if res.policy == "fifo":
                    return None
                cnt = max(oi, int(res.units))
                u_cap = float(res.rate) / cnt
                cap_scale = of if u_cap < of else u_cap
                sat_tol = max(_MIN_RATE_PER_MS, _SAT_ULPS * ulp(cap_scale))
                bind_thresh = u_cap - max(_MIN_RATE_PER_MS, _SAT_ULPS * tf * ulp(cap_scale))
                full_u = range_get(cnt)
                if full_u is None:
                    full_u = _UNIT_RANGE_CACHE[cnt] = tuple(range(cnt))
                info = res_info[name] = (u_cap, cnt, sat_tol, bind_thresh, [None, None], full_u)
            u_cap, cnt, _sat_tol, _bind_thresh, sbox, full_u = info
            if u_cap <= _MIN_RATE_PER_MS:
                continue
            decl = placement.get(name) if placement else None
            if decl is None:
                units = full_u
                n_u = cnt
                is_full = True
            else:
                units = (decl if (decl and zi <= decl[zi] and decl[ni] < cnt
                                  and (len(decl) <= ti or all(zi <= u < cnt for u in decl)))
                         else tuple(u for u in decl if zi <= u < cnt))
                n_u = len(units)
                if not n_u:
                    continue
                is_full = units == full_u
            demand = u_cap * n_u
            if demand <= zf:
                continue
            limit = of
            if caps_map:
                cap = caps_map.get(name)
                if cap is not None:
                    c_lim = max(zf, float(cap)) / demand
                    if c_lim < limit:
                        limit = c_lim
                if limit <= zf:
                    continue
            f_idx = len(flow_keys)
            fk_append(key)
            fl_append(limit)
            fr_append(((name, demand),))
            weight = demand / n_u
            item = (f_idx, weight)
            if is_full:
                fslot = sbox[zi]
                if fslot is None:
                    sbox[zi] = item
                elif type(fslot) is tuple:
                    sbox[zi] = [fslot, item]
                else:
                    fslot.append(item)
            else:
                umap = sbox[oi]
                if umap is None:
                    umap = sbox[oi] = {}
                for u in units:
                    cur = umap.get(u)
                    if cur is None:
                        umap[u] = item
                    elif type(cur) is tuple:
                        umap[u] = [cur, item]
                    else:
                        cur.append(item)

    n_flows = len(flow_keys)
    if not n_flows:
        return {}

    private_checks: list[tuple[str, int, float, int, float]] = []
    pc_append = private_checks.append
    for name, (u_cap, cnt, sat_tol, bind_thresh, sbox, full_u) in res_info.items():
        fslot = sbox[zi]
        umap = sbox[oi]
        if umap is None:
            if type(fslot) is tuple:
                f_idx, w = fslot
                pc_append((name, zi, u_cap, f_idx, w))
                if w > zf:
                    b = u_cap / w
                    if b < flow_limits[f_idx]:
                        flow_limits[f_idx] = b
                    while flow_limits[f_idx] > zf and flow_limits[f_idx] * w > u_cap:
                        flow_limits[f_idx] = nextafter(flow_limits[f_idx], zf)
            elif fslot is not None:
                su_append((name, zi, u_cap, sat_tol, bind_thresh, fslot))
            continue
        if fslot is None:
            for u in (sorted(umap) if len(umap) > oi else umap):
                cur = umap[u]
                if type(cur) is tuple:
                    f_idx, w = cur
                    pc_append((name, u, u_cap, f_idx, w))
                    if w > zf:
                        b = u_cap / w
                        if b < flow_limits[f_idx]:
                            flow_limits[f_idx] = b
                        while flow_limits[f_idx] > zf and flow_limits[f_idx] * w > u_cap:
                            flow_limits[f_idx] = nextafter(flow_limits[f_idx], zf)
                else:
                    su_append((name, u, u_cap, sat_tol, bind_thresh, cur))
            continue
        if len(umap) < cnt:
            u_rep = next(u for u in full_u if u not in umap)
            if type(fslot) is tuple:
                f_idx, w = fslot
                pc_append((name, u_rep, u_cap, f_idx, w))
                if w > zf:
                    b = u_cap / w
                    if b < flow_limits[f_idx]:
                        flow_limits[f_idx] = b
                    while flow_limits[f_idx] > zf and flow_limits[f_idx] * w > u_cap:
                        flow_limits[f_idx] = nextafter(flow_limits[f_idx], zf)
            else:
                su_append((name, u_rep, u_cap, sat_tol, bind_thresh, fslot))
        f_is_tuple = type(fslot) is tuple
        for u in (sorted(umap) if len(umap) > oi else umap):
            cur = umap[u]
            if f_is_tuple and type(cur) is tuple:
                terms = [fslot, cur] if fslot[zi] <= cur[zi] else [cur, fslot]
            else:
                f_part = (fslot,) if f_is_tuple else fslot
                u_part = (cur,) if type(cur) is tuple else cur
                terms = sorted([*f_part, *u_part])
            su_append((name, u, u_cap, sat_tol, bind_thresh, terms))

    n_shared = len(shared_units)
    if n_shared == zi:
        alphas = flow_limits
    else:
        binding_units: list[tuple[float, float, list[tuple[int, float]]]] = []
        seen_binding: set[tuple[float, tuple[tuple[int, float], ...]]] = set()
        for _name, _u, cap, sat_tol, bind_thresh, terms in shared_units:
            n_t = len(terms)
            max_tot = (
                terms[zi][oi] * flow_limits[terms[zi][zi]] if n_t == oi
                else (terms[zi][oi] * flow_limits[terms[zi][zi]] + terms[oi][oi] * flow_limits[terms[oi][zi]]) if n_t == ti
                else fsum([flow_limits[f_idx] * w for f_idx, w in terms])
            )
            if max_tot > bind_thresh:
                key_u = (cap, tuple(terms))
                if key_u not in seen_binding:
                    seen_binding.add(key_u)
                    binding_units.append((cap, sat_tol, terms))
        if not binding_units:
            alphas = flow_limits
        else:
            alphas = [zf] * n_flows
            deg = [zi] * n_flows
            for _cap, _sat_tol, terms in binding_units:
                for f_idx, _w in terms:
                    deg[f_idx] += oi
            active_flags = [False] * n_flows
            for f_idx in range(n_flows):
                lim = flow_limits[f_idx]
                if deg[f_idx] == zi:
                    alphas[f_idx] = lim
                elif lim > zf:
                    active_flags[f_idx] = True
            rem_units: list[tuple[float, float, list[tuple[int, float]]]] = []
            for cap, sat_tol, terms in binding_units:
                if all(deg[f_idx] == oi and active_flags[f_idx] for f_idx, _w in terms):
                    n_t = len(terms)
                    slope = (
                        terms[zi][oi] if n_t == oi
                        else (terms[zi][oi] + terms[oi][oi]) if n_t == ti
                        else fsum([w for _f, w in terms])
                    )
                    denom = slope if n_t == oi else nextafter(slope, inf)
                    b = cap / denom
                    if all(b <= flow_limits[f_idx] for f_idx, _w in terms):
                        delta = b
                        solved = False
                        for attempt in range(_MAX_ATTEMPTS):
                            if delta <= zf:
                                break
                            tot = (
                                (delta * terms[zi][oi]) if n_t == oi
                                else (delta * terms[zi][oi] + delta * terms[oi][oi]) if n_t == ti
                                else fsum([delta * w for _f, w in terms])
                            )
                            if tot <= cap:
                                res_v = cap - tot
                                tol = (sat_tol if zf <= res_v <= cap
                                       else max(_MIN_RATE_PER_MS, _SAT_ULPS * ulp(max(of, abs(cap), abs(res_v)))))
                                if res_v <= tol:
                                    for f_idx, _w in terms:
                                        alphas[f_idx] = delta
                                        active_flags[f_idx] = False
                                    solved = True
                                break
                            delta = nextafter(delta, zf) if attempt < _NEXTAFTER_STEPS else delta * _HALF_STEP
                        if solved:
                            continue
                rem_units.append((cap, sat_tol, terms))
            if rem_units:
                n_rem = len(rem_units)
                act_indices = [i for i, ok in enumerate(active_flags) if ok]
                committed = [[zf] * len(rem_units[s_idx][ti]) for s_idx in range(n_rem)]
                totals = [zf] * n_rem
                open_shared = list(range(n_rem))
                while act_indices and open_shared:
                    bound_limit = inf
                    still_open: list[int] = []
                    for s_idx in open_shared:
                        cap, _sat_tol, terms = rem_units[s_idx]
                        if len(terms) == ti:
                            (f0, w0), (f1, w1) = terms
                            a0, a1 = active_flags[f0], active_flags[f1]
                            if not a0 and not a1:
                                continue
                            still_open.append(s_idx)
                            if a0 and a1:
                                slope = w0 + w1
                                denom = nextafter(slope, inf) if slope > zf else zf
                            else:
                                slope = w0 if a0 else w1
                                denom = slope
                            if denom > zf:
                                b = (cap - totals[s_idx]) / denom
                                if b < bound_limit:
                                    bound_limit = b
                        else:
                            weights = [w for f_idx, w in terms if active_flags[f_idx]]
                            if weights:
                                still_open.append(s_idx)
                                cnt_w = len(weights)
                                slope = weights[zi] if cnt_w == oi else fsum(weights)
                                if slope > zf:
                                    denom = slope if cnt_w == oi else nextafter(slope, inf)
                                    b = (cap - totals[s_idx]) / denom
                                    if b < bound_limit:
                                        bound_limit = b
                    open_shared = still_open
                    if not open_shared:
                        for f_idx in act_indices:
                            alphas[f_idx] = flow_limits[f_idx]
                        break
                    headroom = inf
                    for f_idx in act_indices:
                        hr = flow_limits[f_idx] - alphas[f_idx]
                        if hr < headroom:
                            headroom = hr
                    delta = bound_limit if bound_limit < headroom else headroom
                    if not isfinite(delta) or delta < zf:
                        break
                    cand_sub: list[tuple[int, float]] = []
                    for attempt in range(_MAX_ATTEMPTS):
                        if delta <= zf:
                            cand_sub = []
                            break
                        c_sub: list[tuple[int, float]] = []
                        unsafe = any(alphas[f_idx] + delta > flow_limits[f_idx] for f_idx in act_indices)
                        if not unsafe:
                            for s_idx in open_shared:
                                cap, _sat_tol, terms = rem_units[s_idx]
                                comm_s = committed[s_idx]
                                if len(terms) == ti:
                                    (f0, w0), (f1, w1) = terms
                                    v0 = (alphas[f0] + delta) * w0 if active_flags[f0] else comm_s[zi]
                                    v1 = (alphas[f1] + delta) * w1 if active_flags[f1] else comm_s[oi]
                                    tot = v0 + v1
                                else:
                                    vals = [((alphas[f_idx] + delta) * w if active_flags[f_idx] else comm_s[j])
                                            for j, (f_idx, w) in enumerate(terms)]
                                    tot = vals[zi] if len(vals) == oi else fsum(vals)
                                if tot > cap:
                                    unsafe = True
                                    break
                                c_sub.append((s_idx, tot))
                        if not unsafe:
                            cand_sub = c_sub
                            break
                        delta = nextafter(delta, zf) if attempt < _NEXTAFTER_STEPS else delta * _HALF_STEP
                    else:
                        delta = zf
                        cand_sub = []
                    advanced = False
                    if delta > zf:
                        for f_idx in act_indices:
                            lim_f = flow_limits[f_idx]
                            raw_f = alphas[f_idx] + delta
                            val = lim_f if raw_f > lim_f else raw_f
                            if val != alphas[f_idx]:
                                advanced = True
                            alphas[f_idx] = val
                        for s_idx in open_shared:
                            comm_s = committed[s_idx]
                            for j, (f_idx, w) in enumerate(rem_units[s_idx][ti]):
                                if active_flags[f_idx]:
                                    comm_s[j] = alphas[f_idx] * w
                        for s_idx, tot in cand_sub:
                            totals[s_idx] = tot
                    next_open: list[int] = []
                    froze_any = False
                    for s_idx in open_shared:
                        cap, sat_tol, terms = rem_units[s_idx]
                        res_v = cap - totals[s_idx]
                        tol = (sat_tol if zf <= res_v <= cap
                               else max(_MIN_RATE_PER_MS, _SAT_ULPS * ulp(max(of, abs(cap), abs(res_v)))))
                        if res_v <= tol:
                            for f_idx, _w in terms:
                                if active_flags[f_idx]:
                                    active_flags[f_idx] = False
                                    froze_any = True
                        else:
                            next_open.append(s_idx)
                    open_shared = next_open
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

    for name, u, cap, f_idx, w in private_checks:
        if alphas[f_idx] * w > cap:
            raise AssertionError(f"unit_service_overallocated:{name}:{u}")
    for name, u, cap, _sat_tol, _bind_thresh, terms in shared_units:
        n_t = len(terms)
        tot_u = (
            alphas[terms[zi][zi]] * terms[zi][oi] if n_t == oi
            else (alphas[terms[zi][zi]] * terms[zi][oi] + alphas[terms[oi][zi]] * terms[oi][oi]) if n_t == ti
            else fsum([alphas[f_idx] * w for f_idx, w in terms])
        )
        if tot_u > cap:
            raise AssertionError(f"unit_service_overallocated:{name}:{u}")

    result: dict[str, dict[int, float]] = {}
    for f_idx, alpha in enumerate(alphas):
        if alpha <= zf:
            continue
        key = flow_keys[f_idx]
        for name, demand in flow_rates[f_idx]:
            target = result.get(name)
            if target is None:
                result[name] = {key: alpha * demand}
            else:
                target[key] = target.get(key, zf) + alpha * demand
    return result
