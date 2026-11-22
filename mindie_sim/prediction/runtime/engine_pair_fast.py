"""Closed-form two-task and independent-third-task step integration.

[Role] Integrate uncached two-task (plus optional disjoint third-task) active
  cohorts directly from ``_coupled_pair_fast`` tuples without dict churn.
[Invariants] Zero allocated service rates raise ``service_rate_underflow``
  before any progress mutation; 4-ULP residual snap matches ``service_step``.
[Timeouts & Locks] O(active links) arithmetic; no locks, sleeps or I/O.
[Side-Effect Policy] Mutates live task remaining/service maps and Engine counters.
"""
from __future__ import annotations

import heapq
import math

from .engine_types import ExecutionError, _Task

_PAIR_MODE_COUPLED_RATES = 3
_PAIR_MODE_DUAL_SINGLE = 4
_PAIR_MODE_COUPLED_SCALED = 5


class _EnginePairFastMixin:
    def _step_pair_fast(
        self, cp_fast, now: float, dt: float,
        ks: int = 0, ts: _Task | None = None, rs: str = "", rate_s: float = 0.0,
    ) -> float:
        busy_ms = self.busy_ms
        serviced = self.serviced
        completable = self._completable
        completable_seen = self._completable_seen
        _isfinite = math.isfinite
        _ulp = math.ulp
        _nextafter = math.nextafter
        _heappush = heapq.heappush
        _inf = math.inf
        if ts is not None:
            if (rate_s) == 0.0:
                raise ExecutionError(f"service_rate_underflow:{rs}")
            wdt_s = ts.remaining[rs] / rate_s
            if wdt_s < dt:
                dt = wdt_s
        mode = cp_fast[0]
        if mode == _PAIR_MODE_COUPLED_RATES:
            _, kc, tc, c_rates, ku, tu, name_u, rate_u = cp_fast
            rem_c = tc.remaining
            rem_u = tu.remaining
            for rname, rate_c in c_rates:
                if (rate_c) == 0.0:
                    raise ExecutionError(f"service_rate_underflow:{rname}")
                wdt = rem_c[rname] / rate_c
                if wdt < dt:
                    dt = wdt
            if rate_u:
                wdt_u = rem_u[name_u] / rate_u
                if wdt_u < dt:
                    dt = wdt_u
            if not math.isfinite(dt):
                names = ",".join(t.event.name for t in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = math.nextafter(now, math.inf)
            svc_c = tc.service
            for rname, rate_c in c_rates:
                busy_ms[rname] += dt
                remaining = rem_c[rname]
                scaled = dt * rate_c
                if scaled >= remaining:
                    used = remaining
                    residual = 0.0
                else:
                    residual = remaining - scaled
                    if residual <= 4 * math.ulp(remaining):
                        used, residual = remaining, 0.0
                    else:
                        used = scaled
                svc_c[rname] = svc_c.get(rname, 0.0) + used
                serviced[rname] += used
                if residual:
                    rem_c[rname] = residual
                else:
                    del rem_c[rname]
            if not rem_c and tc.latency <= 0.0:
                completable_seen.add(kc)
                heapq.heappush(completable, kc)
            if rate_u:
                svc_u = tu.service
                busy_ms[name_u] += dt
                remaining = rem_u[name_u]
                scaled = dt * rate_u
                if scaled >= remaining:
                    used = remaining
                    residual = 0.0
                else:
                    residual = remaining - scaled
                    if residual <= 4 * math.ulp(remaining):
                        used, residual = remaining, 0.0
                    else:
                        used = scaled
                svc_u[name_u] = svc_u.get(name_u, 0.0) + used
                serviced[name_u] += used
                if residual:
                    rem_u[name_u] = residual
                else:
                    del rem_u[name_u]
                if not rem_u and tu.latency <= 0.0:
                    completable_seen.add(ku)
                    heapq.heappush(completable, ku)
        elif mode == _PAIR_MODE_DUAL_SINGLE:
            _, k0, t0, n0, rate0, k1, t1, n1, rate1 = cp_fast
            rem0 = t0.remaining
            rem1 = t1.remaining
            if (rate0) == 0.0:
                raise ExecutionError(f"service_rate_underflow:{n0}")
            wdt0 = rem0[n0] / rate0
            if wdt0 < dt:
                dt = wdt0
            if (rate1) == 0.0:
                raise ExecutionError(f"service_rate_underflow:{n1}")
            wdt1 = rem1[n1] / rate1
            if wdt1 < dt:
                dt = wdt1
            if not math.isfinite(dt):
                names = ",".join(t.event.name for t in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = math.nextafter(now, math.inf)
            busy_ms[n0] += dt
            rem_v0 = rem0[n0]
            scaled0 = dt * rate0
            if scaled0 >= rem_v0:
                used0, res0 = rem_v0, 0.0
            else:
                res0 = rem_v0 - scaled0
                if res0 <= 4 * math.ulp(rem_v0):
                    used0, res0 = rem_v0, 0.0
                else:
                    used0 = scaled0
            svc0 = t0.service
            svc0[n0] = svc0.get(n0, 0.0) + used0
            serviced[n0] += used0
            if res0:
                rem0[n0] = res0
            else:
                del rem0[n0]
                if t0.latency <= 0.0:
                    completable_seen.add(k0)
                    heapq.heappush(completable, k0)
            busy_ms[n1] += dt
            rem_v1 = rem1[n1]
            scaled1 = dt * rate1
            if scaled1 >= rem_v1:
                used1, res1 = rem_v1, 0.0
            else:
                res1 = rem_v1 - scaled1
                if res1 <= 4 * math.ulp(rem_v1):
                    used1, res1 = rem_v1, 0.0
                else:
                    used1 = scaled1
            svc1 = t1.service
            svc1[n1] = svc1.get(n1, 0.0) + used1
            serviced[n1] += used1
            if res1:
                rem1[n1] = res1
            else:
                del rem1[n1]
                if t1.latency <= 0.0:
                    completable_seen.add(k1)
                    heapq.heappush(completable, k1)
        elif mode == _PAIR_MODE_COUPLED_SCALED:
            _, kc, tc, c_items, fc, ku, tu, name_u, rate_u = cp_fast
            rem_c = tc.remaining
            rem_u = tu.remaining
            if fc:
                for rname, amt in c_items:
                    if (amt * fc) == 0.0:
                        raise ExecutionError(f"service_rate_underflow:{rname}")
                    wdt = rem_c[rname] / (amt * fc)
                    if wdt < dt:
                        dt = wdt
            if rate_u:
                wdt_u = rem_u[name_u] / rate_u
                if wdt_u < dt:
                    dt = wdt_u
            if not math.isfinite(dt):
                names = ",".join(t.event.name for t in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = math.nextafter(now, math.inf)
            if fc:
                svc_c = tc.service
                for rname, amt in c_items:
                    busy_ms[rname] += dt
                    remaining = rem_c[rname]
                    scaled = dt * (amt * fc)
                    if scaled >= remaining:
                        used = remaining
                        residual = 0.0
                    else:
                        residual = remaining - scaled
                        if residual <= 4 * math.ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    if rname in svc_c:
                        svc_c[rname] += used
                    else:
                        svc_c[rname] = used
                    serviced[rname] += used
                    if residual:
                        rem_c[rname] = residual
                    else:
                        del rem_c[rname]
                if not rem_c and tc.latency <= 0.0:
                    completable_seen.add(kc)
                    heapq.heappush(completable, kc)
            if rate_u:
                svc_u = tu.service
                if not fc:
                    busy_ms[name_u] += dt
                remaining = rem_u[name_u]
                scaled = dt * rate_u
                if scaled >= remaining:
                    used = remaining
                    residual = 0.0
                else:
                    residual = remaining - scaled
                    if residual <= 4 * math.ulp(remaining):
                        used, residual = remaining, 0.0
                    else:
                        used = scaled
                if name_u in svc_u:
                    svc_u[name_u] += used
                else:
                    svc_u[name_u] = used
                serviced[name_u] += used
                if residual:
                    rem_u[name_u] = residual
                else:
                    del rem_u[name_u]
                    if tu.latency <= 0.0:
                        completable_seen.add(ku)
                        heapq.heappush(completable, ku)
        elif mode == 1:
            _, ck0, ct0, dem0, f0, ck1, ct1, dem1, f1 = cp_fast
            rem0 = ct0.remaining
            rem1 = ct1.remaining
            if f0:
                for rname, amt in dem0.items():
                    if (amt * f0) == 0.0:
                        raise ExecutionError(f"service_rate_underflow:{rname}")
                    wdt = rem0[rname] / (amt * f0)
                    if wdt < dt:
                        dt = wdt
            if f1:
                for rname, amt in dem1.items():
                    if (amt * f1) == 0.0:
                        raise ExecutionError(f"service_rate_underflow:{rname}")
                    wdt = rem1[rname] / (amt * f1)
                    if wdt < dt:
                        dt = wdt
            if not math.isfinite(dt):
                names = ",".join(t.event.name for t in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = math.nextafter(now, math.inf)
            if f0:
                svc0 = ct0.service
                for rname, amt in dem0.items():
                    busy_ms[rname] += dt
                    remaining = rem0[rname]
                    scaled = dt * (amt * f0)
                    if scaled >= remaining:
                        used = remaining
                        residual = 0.0
                    else:
                        residual = remaining - scaled
                        if residual <= 4 * math.ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    if rname in svc0:
                        svc0[rname] += used
                    else:
                        svc0[rname] = used
                    serviced[rname] += used
                    if residual:
                        rem0[rname] = residual
                    else:
                        del rem0[rname]
                if not rem0 and ct0.latency <= 0.0:
                    completable_seen.add(ck0)
                    heapq.heappush(completable, ck0)
            if f1:
                svc1 = ct1.service
                for rname, amt in dem1.items():
                    if not f0 or rname not in dem0:
                        busy_ms[rname] += dt
                    remaining = rem1[rname]
                    scaled = dt * (amt * f1)
                    if scaled >= remaining:
                        used = remaining
                        residual = 0.0
                    else:
                        residual = remaining - scaled
                        if residual <= 4 * math.ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    if rname in svc1:
                        svc1[rname] += used
                    else:
                        svc1[rname] = used
                    serviced[rname] += used
                    if residual:
                        rem1[rname] = residual
                    else:
                        del rem1[rname]
                if not rem1 and ct1.latency <= 0.0:
                    completable_seen.add(ck1)
                    heapq.heappush(completable, ck1)
        elif mode == 0:
            _, kc, tc, dem_c, fc, ku, tu, indep_u = cp_fast
            rem_c = tc.remaining
            rem_u = tu.remaining
            # A coupled resource that finished on an earlier step is removed
            # from ``rem_c`` but is still declared by ``coupled_demand``.  The
            # cached declaration is only safe while every declared key is still
            # live; otherwise fall back to a live-only projection so the
            # per-resource amounts match the rescaled ``dem_c`` plan.
            c_info = (tc.prepared.coupled_res_info
                      if tc.prepared is not None else None)
            c_iter = (c_info[1]
                      if (c_info is not None
                          and len(rem_c) == len(dem_c) == len(c_info[1]))
                      else tuple((rname, amt) for rname, amt in dem_c.items()
                                 if rname in rem_c and amt > 0.0))
            if fc:
                for rname, amt in c_iter:
                    if (amt * fc) == 0.0:
                        raise ExecutionError(f"service_rate_underflow:{rname}")
                    wdt = rem_c[rname] / (amt * fc)
                    if wdt < dt:
                        dt = wdt
            for rname, rate_u in indep_u:
                remaining = rem_u.get(rname)
                if remaining is None:
                    continue
                if (rate_u) == 0.0:
                    raise ExecutionError(f"service_rate_underflow:{rname}")
                wdt = remaining / rate_u
                if wdt < dt:
                    dt = wdt
            if not math.isfinite(dt):
                names = ",".join(t.event.name for t in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = math.nextafter(now, math.inf)
            if fc:
                svc_c = tc.service
                for rname, amt in c_iter:
                    busy_ms[rname] += dt
                    remaining = rem_c[rname]
                    scaled = dt * (amt * fc)
                    if scaled >= remaining:
                        used = remaining
                        residual = 0.0
                    else:
                        residual = remaining - scaled
                        if residual <= 4 * math.ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    if rname in svc_c:
                        svc_c[rname] += used
                    else:
                        svc_c[rname] = used
                    serviced[rname] += used
                    if residual:
                        rem_c[rname] = residual
                    else:
                        del rem_c[rname]
                if not rem_c and tc.latency <= 0.0:
                    completable_seen.add(kc)
                    heapq.heappush(completable, kc)
            if indep_u:
                svc_u = tu.service
                for rname, rate_u in indep_u:
                    remaining = rem_u.get(rname)
                    if remaining is None:
                        continue
                    if not fc or rname not in dem_c:
                        busy_ms[rname] += dt
                    scaled = dt * rate_u
                    if scaled >= remaining:
                        used = remaining
                        residual = 0.0
                    else:
                        residual = remaining - scaled
                        if residual <= 4 * math.ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    if rname in svc_u:
                        svc_u[rname] += used
                    else:
                        svc_u[rname] = used
                    serviced[rname] += used
                    if residual:
                        rem_u[rname] = residual
                    else:
                        del rem_u[rname]
                if not rem_u and tu.latency <= 0.0:
                    completable_seen.add(ku)
                    heapq.heappush(completable, ku)
        else:
            _, ck0, ct0, rates0, ck1, ct1, rates1 = cp_fast
            rem0 = ct0.remaining
            rem1 = ct1.remaining
            for rname, rate0 in rates0:
                if (rate0) == 0.0:
                    raise ExecutionError(f"service_rate_underflow:{rname}")
                wdt = rem0[rname] / rate0
                if wdt < dt:
                    dt = wdt
            for rname, rate1, _ in rates1:
                if (rate1) == 0.0:
                    raise ExecutionError(f"service_rate_underflow:{rname}")
                wdt = rem1[rname] / rate1
                if wdt < dt:
                    dt = wdt
            if not math.isfinite(dt):
                names = ",".join(t.event.name for t in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = math.nextafter(now, math.inf)
            if rates0:
                svc0 = ct0.service
                for rname, rate0 in rates0:
                    busy_ms[rname] += dt
                    remaining = rem0[rname]
                    scaled = dt * rate0
                    if scaled >= remaining:
                        used = remaining
                        residual = 0.0
                    else:
                        residual = remaining - scaled
                        if residual <= 4 * math.ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    svc0[rname] = svc0.get(rname, 0.0) + used
                    serviced[rname] += used
                    if residual:
                        rem0[rname] = residual
                    else:
                        del rem0[rname]
                if not rem0 and ct0.latency <= 0.0:
                    completable_seen.add(ck0)
                    heapq.heappush(completable, ck0)
            if rates1:
                svc1 = ct1.service
                for rname, rate1, add_busy in rates1:
                    if add_busy:
                        busy_ms[rname] += dt
                    remaining = rem1[rname]
                    scaled = dt * rate1
                    if scaled >= remaining:
                        used = remaining
                        residual = 0.0
                    else:
                        residual = remaining - scaled
                        if residual <= 4 * math.ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    svc1[rname] = svc1.get(rname, 0.0) + used
                    serviced[rname] += used
                    if residual:
                        rem1[rname] = residual
                    else:
                        del rem1[rname]
                if not rem1 and ct1.latency <= 0.0:
                    completable_seen.add(ck1)
                    heapq.heappush(completable, ck1)
        if ts is not None:
            rem_s = ts.remaining
            remaining_s = rem_s[rs]
            busy_ms[rs] += dt
            scaled_s = dt * rate_s
            if scaled_s >= remaining_s:
                used_s = remaining_s
                residual_s = 0.0
            else:
                residual_s = remaining_s - scaled_s
                if residual_s <= 4 * math.ulp(remaining_s):
                    used_s, residual_s = remaining_s, 0.0
                else:
                    used_s = scaled_s
            svc_s = ts.service
            svc_s[rs] = svc_s.get(rs, 0.0) + used_s
            serviced[rs] += used_s
            if residual_s:
                rem_s[rs] = residual_s
            else:
                del rem_s[rs]
                if ts.latency <= 0.0:
                    completable_seen.add(ks)
                    heapq.heappush(completable, ks)
        return next_time
