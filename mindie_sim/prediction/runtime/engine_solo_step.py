"""Single-active-task fast-path step integration for the causal Engine.

[Role] Integrate one eligible active consumer (uncoupled single/multi-resource
  or all-fair coupled) without allocating intermediate rate dictionaries.
[Invariants] Zero allocated service rates raise ``service_rate_underflow``
  before any state mutation; falls back cleanly (returns ``None``) when FIFO
  or multi-tier coupled constraints require the general arbiter.
[Timeouts & Locks] O(task resources) arithmetic; no locks, sleeps or I/O.
[Side-Effect Policy] Mutates live task remaining/service maps and Engine counters.
"""
from __future__ import annotations

import heapq
import math

from ..runtime.coupled import _coupled_demand_and_cap_limit
from ._engine_hbm_bus import _hbm_delivery_fraction, _sync_task_hbm_accounting
from .engine_types import ExecutionError, _Task
from .prepared_terms import RESIDUAL_SNAP_ULPS


def _emit_wave_service(
    wave_rec, key: int, rname: str, placement,
    t0_ms: float, t1_ms: float, used: float,
) -> None:
    """Publish service across explicitly placed unit owners with exact work conservation."""
    pl_u = placement.get(rname) if placement else None
    if not pl_u:
        wave_rec.on_service(key, rname, 0, t0_ms, t1_ms, used)
        return
    n_u = len(pl_u)
    if n_u == 1:
        wave_rec.on_service(key, rname, pl_u[0], t0_ms, t1_ms, used)
        return
    prev = 0.0
    denom = float(n_u)
    for idx, owner_u in enumerate(pl_u, 1):
        cur = used if idx == n_u else used * (float(idx) / denom)
        share = cur - prev
        prev = cur
        wave_rec.on_service(key, rname, owner_u, t0_ms, t1_ms, share)


class _EngineSoloStepMixin:
    def _step_solo_fast(
        self, key: int, task: _Task, now: float, horizon: float,
        max_steps: int | None,
    ) -> float | None:
        rem = task.remaining
        releases = self._releases
        latency_active = self._latency_active
        tasks = self._tasks
        _isfinite = math.isfinite
        _nextafter = math.nextafter
        _inf = math.inf
        stop_dt = horizon - now
        if releases:
            rel_dt = releases[0][0] - now
            if rel_dt < stop_dt:
                stop_dt = rel_dt
        if latency_active:
            if len(latency_active) == 1:
                ltask = tasks[next(iter(latency_active))]
                anchor = ltask.first_service if ltask.latency_from_first_service else ltask.start
                if anchor is not None and anchor + ltask.latency > now:
                    lat_dt = anchor + ltask.latency - now
                    if lat_dt < stop_dt:
                        stop_dt = lat_dt
            else:
                for lkey in latency_active:
                    ltask = tasks[lkey]
                    anchor = ltask.first_service if ltask.latency_from_first_service else ltask.start
                    if anchor is not None and anchor + ltask.latency > now:
                        lat_dt = anchor + ltask.latency - now
                        if lat_dt < stop_dt:
                            stop_dt = lat_dt
        if not rem:
            if not _isfinite(stop_dt):
                names = ",".join(t.event.name for t in tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + stop_dt
            if stop_dt > 0 and next_time == now:
                next_time = _nextafter(now, _inf)
            return next_time
        if task.coupled:
            return self._step_solo_coupled(key, task, now, stop_dt)
        prep = task.prepared
        s_rates = prep.solo_rates if prep is not None else None
        if len(rem) == 1:
            return self._step_solo_resource(key, task, now, stop_dt, s_rates)
        return self._step_solo_uncoupled(key, task, now, stop_dt, max_steps, s_rates)

    def _step_solo_resource(self, key, task, now, stop_dt, s_rates):
        rem = task.remaining
        tasks = self._tasks
        resources = self.resources
        busy_ms = self.busy_ms
        serviced = self.serviced
        completable = self._completable
        completable_seen = self._completable_seen
        _isfinite = math.isfinite
        _ulp = math.ulp
        _nextafter = math.nextafter
        _heappush = heapq.heappush
        _inf = math.inf
        rname, remaining = next(iter(rem.items()))
        if s_rates is not None:
            rate = s_rates[rname]
        else:
            res = resources[rname]
            if res.policy == "fifo":
                rate = res.rate
            elif task.placement and rname in task.placement:
                rate = res.rate / res.units * len(task.placement[rname])
            else:
                rate = res.rate
            if task.caps and rname in task.caps:
                cap = task.caps[rname]
                if cap < rate:
                    rate = cap
        if rate == 0.0:
            raise ExecutionError(f"service_rate_underflow:{rname}")
        work_dt = remaining / rate
        dt = work_dt if work_dt < stop_dt else stop_dt
        if not _isfinite(dt):
            names = ",".join(t.event.name for t in tasks.values())
            raise ExecutionError(f"execution_stalled:{names[:300]}")
        next_time = now + dt
        if dt > 0 and next_time == now:
            next_time = _nextafter(now, _inf)
        busy_ms[rname] += dt
        scaled = dt * rate
        if scaled >= remaining:
            used, residual = remaining, 0.0
        else:
            residual = remaining - scaled
            if residual <= RESIDUAL_SNAP_ULPS * _ulp(remaining):
                used, residual = remaining, 0.0
            else:
                used = scaled
        service = task.service
        service[rname] = service.get(rname, 0.0) + used
        serviced[rname] += used
        wave_rec = self._wave_recorder
        if wave_rec is not None and dt > 0.0:
            _emit_wave_service(wave_rec, key, rname, task.placement, now, next_time, used)
        if residual:
            rem[rname] = residual
        else:
            del rem[rname]
            if task.latency <= 0.0:
                completable_seen.add(key)
                _heappush(completable, key)
        return next_time

    def _step_solo_uncoupled(self, key, task, now, stop_dt, max_steps, s_rates):
        rem = task.remaining
        tasks = self._tasks
        resources = self.resources
        busy_ms = self.busy_ms
        serviced = self.serviced
        completable = self._completable
        completable_seen = self._completable_seen
        _isfinite = math.isfinite
        _ulp = math.ulp
        _nextafter = math.nextafter
        _heappush = heapq.heappush
        _inf = math.inf
        entries = []
        min_wdt = stop_dt
        max_wdt = 0.0
        t_place = task.placement
        t_caps = task.caps
        for rname, remaining in rem.items():
            if s_rates is not None:
                rate = s_rates[rname]
            else:
                res = resources[rname]
                if res.policy == "fifo":
                    rate = res.rate
                elif t_place and rname in t_place:
                    rate = res.rate / res.units * len(t_place[rname])
                else:
                    rate = res.rate
                if t_caps and rname in t_caps:
                    cap = t_caps[rname]
                    if cap < rate:
                        rate = cap
            if rate == 0.0:
                raise ExecutionError(f"service_rate_underflow:{rname}")
            wdt = remaining / rate
            if wdt < min_wdt:
                min_wdt = wdt
            if wdt > max_wdt:
                max_wdt = wdt
            entries.append((rname, remaining, rate, wdt))
        if max_steps is None and _isfinite(max_wdt) and max_wdt <= stop_dt:
            next_time = now + max_wdt
            if max_wdt > 0 and next_time == now:
                next_time = _nextafter(now, _inf)
            service = task.service
            wave_rec = self._wave_recorder
            for rname, remaining, _rate, wdt in entries:
                busy_ms[rname] += wdt
                service[rname] = service.get(rname, 0.0) + remaining
                serviced[rname] += remaining
                if wave_rec is not None and wdt > 0.0:
                    _emit_wave_service(wave_rec, key, rname, task.placement, now, now + wdt, remaining)
            rem.clear()
            if task.latency <= 0.0:
                completable_seen.add(key)
                _heappush(completable, key)
            return next_time
        dt = min_wdt
        if not _isfinite(dt):
            names = ",".join(t.event.name for t in tasks.values())
            raise ExecutionError(f"execution_stalled:{names[:300]}")
        next_time = now + dt
        if dt > 0 and next_time == now:
            next_time = _nextafter(now, _inf)
        service = task.service
        wave_rec = self._wave_recorder
        for rname, remaining, rate, _wdt in entries:
            busy_ms[rname] += dt
            scaled = dt * rate
            if scaled >= remaining:
                used, residual = remaining, 0.0
            else:
                residual = remaining - scaled
                if residual <= RESIDUAL_SNAP_ULPS * _ulp(remaining):
                    used, residual = remaining, 0.0
                else:
                    used = scaled
            service[rname] = service.get(rname, 0.0) + used
            serviced[rname] += used
            if wave_rec is not None and dt > 0.0:
                _emit_wave_service(wave_rec, key, rname, task.placement, now, next_time, used)
            if residual:
                rem[rname] = residual
            else:
                del rem[rname]
        if not rem and task.latency <= 0.0:
            completable_seen.add(key)
            _heappush(completable, key)
        return next_time

    def _step_solo_coupled(self, key, task, now, stop_dt):
        rem = task.remaining
        tasks = self._tasks
        resources = self.resources
        busy_ms = self.busy_ms
        serviced = self.serviced
        completable = self._completable
        completable_seen = self._completable_seen
        _isfinite = math.isfinite
        _ulp = math.ulp
        _nextafter = math.nextafter
        _heappush = heapq.heappush
        _inf = math.inf
        prep = task.prepared
        c_rates = (prep.coupled_solo_rates
                   if (prep is not None
                       and not task.service
                       and prep.coupled_solo_rates is not None
                       and len(rem) == len(prep.coupled_solo_rates))
                   else None)
        if c_rates is not None:
            dt = stop_dt
            for rname, rate in c_rates:
                if rate == 0.0:
                    raise ExecutionError(f"service_rate_underflow:{rname}")
                wdt = rem[rname] / rate
                if wdt < dt:
                    dt = wdt
        elif all(resources[rname].policy != "fifo" for rname in rem):
            c_demand, c_limit = _coupled_demand_and_cap_limit(resources, task)
            t_place = task.placement
            for rname, amount in c_demand.items():
                if amount:
                    res = resources[rname]
                    u_cnt = (len(t_place[rname])
                             if t_place and rname in t_place else res.units)
                    per_unit = amount if u_cnt == 1 else amount / u_cnt
                    if per_unit:
                        cap_lim = (res.rate / res.units) / per_unit
                        if cap_lim < c_limit:
                            c_limit = cap_lim
            if 0.0 < c_limit < _inf:
                dt = stop_dt
                c_rates_list = []
                for rname, amount in c_demand.items():
                    rate = amount * c_limit
                    if rate == 0.0:
                        raise ExecutionError(f"service_rate_underflow:{rname}")
                    wdt = rem[rname] / rate
                    if wdt < dt:
                        dt = wdt
                    c_rates_list.append((rname, rate))
                c_rates = c_rates_list
        if c_rates is None:
            return None
        if not _isfinite(dt):
            names = ",".join(t.event.name for t in tasks.values())
            raise ExecutionError(f"execution_stalled:{names[:300]}")
        next_time = now + dt
        if dt > 0 and next_time == now:
            next_time = _nextafter(now, _inf)
        service = task.service
        wave_rec = self._wave_recorder
        for rname, rate in c_rates:
            busy_ms[rname] += dt
            remaining = rem[rname]
            scaled = dt * rate
            if scaled >= remaining:
                used, residual = remaining, 0.0
            else:
                residual = remaining - scaled
                if residual <= RESIDUAL_SNAP_ULPS * _ulp(remaining):
                    used, residual = remaining, 0.0
                else:
                    used = scaled
            service[rname] = service.get(rname, 0.0) + used
            serviced[rname] += used
            if wave_rec is not None and dt > 0.0:
                _emit_wave_service(wave_rec, key, rname, task.placement, now, next_time, used)
            if residual:
                rem[rname] = residual
            else:
                del rem[rname]
        if not rem and task.latency <= 0.0:
            completable_seen.add(key)
            _heappush(completable, key)
        return next_time

    def _update_task_deliveries(self, task: _Task) -> None:
        if not task.deliveries:
            return
        hbm_frac = _hbm_delivery_fraction(task) if task.hbm_accounting else None
        if hbm_frac is not None:
            frac = hbm_frac
        else:
            ref_r = next(iter(task.demand), None)
            if ref_r is None:
                return
            ref_d = task.demand[ref_r]
            frac = min(1.0, task.service.get(ref_r, 0.0) / ref_d) if ref_d > 0.0 else 1.0
        deliv = self.delivered
        acc = task.delivered_acc
        for ctr_id, total_amt in task.deliveries:
            prev = acc.get(ctr_id, 0.0) if acc is not None else 0.0
            target = total_amt * frac
            delta = target - prev
            if delta > 0.0:
                deliv[ctr_id] = deliv.get(ctr_id, 0.0) + delta
                if acc is not None:
                    acc[ctr_id] = target

    def _emit_fast_step_observers(
        self, wave_rec, watched, now: float, next_time: float,
    ) -> None:
        for k, t, prev_svc in watched:
            if wave_rec is not None and next_time > now:
                for rn, post in t.service.items():
                    delta = post - prev_svc.get(rn, 0.0)
                    if delta > 0.0:
                        _emit_wave_service(wave_rec, k, rn, t.placement, now, next_time, delta)
            if t.hbm_accounting:
                _sync_task_hbm_accounting(self, t)
            if t.deliveries:
                self._update_task_deliveries(t)
