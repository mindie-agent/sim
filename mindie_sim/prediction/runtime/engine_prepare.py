"""Task preparation, static stage caching and public submission entrypoints.

[Role] Validate task work, latency, supply caps, placement and buffer demands
  against an Engine's declared resources and pools, caching immutable Stage
  templates across repeated tiles.
[Invariants] Every invalid resource, unit, rate, priority, cohort or pool
  demand raises ``ExecutionError`` before any task enters the frontier.
[Timeouts & Locks] Bounded structural/weak-identity caches; no locks or I/O.
[Side-Effect Policy] Populates per-Engine stage preparation caches only.
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
import weakref

from ._engine_hbm_bus import (
    _classify_hbm_split,
    _discover_hbm_bus_aliases,
    _expand_hbm_bus_work,
    _hbm_bus_alias_terms,
    _is_pure_coupled_hbm_work,
    _refresh_prepared_for_hbm_bus,
    _retrofit_hbm_bus_tasks,
)
from .engine_types import (
    Allocation, Event, ExecutionError, _EMPTY_CAPS, _EMPTY_FROZENSET,
    _EMPTY_PLACEMENT, _EMPTY_TASK, _PreparedTask, _compute_pool_unit_masks,
    _units_to_mask,
)


_STAGE_PREPARATION_CACHE_LIMIT = 32768
_VALIDATED_PLACEMENT_CACHE_LIMIT = 8192


def _normalize_cohort_symmetry(
    name: str,
    placement: dict[str, tuple[int, ...]],
    cohort_size: int,
    cohort_widened: Sequence[str] | None,
    cohort_members: Mapping[str, Sequence[Sequence[int]]] | Sequence | None,
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]]:
    """Validate explicit cohort widening and per-member unit tuples."""
    if not cohort_widened and not cohort_members:
        return (), ()
    if cohort_size <= 1:
        raise ExecutionError(f"cohort_widened_requires_cohort:{name}")
    widened_set: set[str] = set()
    members_map: dict[str, tuple[tuple[int, ...], ...]] = {}
    if cohort_members:
        items = (
            cohort_members.items()
            if isinstance(cohort_members, Mapping)
            else cohort_members
        )
        for key, raw_rows in items:
            if type(key) is not str or key not in placement:
                raise ExecutionError(f"cohort_members_resource_missing:{name}:{key}")
            rows = tuple(tuple(int(u) for u in row) for row in raw_rows)
            if len(rows) != cohort_size or any(not row for row in rows):
                raise ExecutionError(f"cohort_members_count_invalid:{name}:{key}")
            members_map[key] = rows
            if len(set(rows)) > 1:
                if len({len(r) for r in rows}) != 1:
                    raise ExecutionError(f"cohort_members_asymmetric:{name}:{key}")
                flat = tuple(u for row in rows for u in row)
                if len(set(flat)) != len(flat) or set(flat) != set(placement[key]):
                    raise ExecutionError(f"cohort_members_not_disjoint_union:{name}:{key}")
                widened_set.add(key)
    if cohort_widened:
        for key in cohort_widened:
            if type(key) is not str or key not in placement:
                raise ExecutionError(f"cohort_widened_resource_missing:{name}:{key}")
            units = placement[key]
            if len(units) < cohort_size or len(units) % cohort_size != 0:
                raise ExecutionError(f"cohort_widened_width_invalid:{name}:{key}")
            if key in members_map and key not in widened_set:
                raise ExecutionError(f"cohort_widened_conflicts_with_shared_members:{name}:{key}")
            widened_set.add(key)
            if key not in members_map:
                stride = len(units) // cohort_size
                members_map[key] = tuple(
                    units[i * stride:(i + 1) * stride] for i in range(cohort_size)
                )
    widened_tuple = tuple(sorted(widened_set))
    members_tuple = tuple(sorted(members_map.items()))
    return widened_tuple, members_tuple


class _EnginePrepareMixin:
    def submit(
        self, name: str, work: Mapping[str, float] | None = None, *,
        after: Sequence[Event] = (), release: float | None = None,
        latency: float = 0.0, caps: Mapping[str, float] | None = None,
        acquire: Sequence[Allocation] = (), free: Sequence[Allocation] = (),
        owner: str = "", metadata: Mapping | None = None,
        on_complete: Callable[[Event], None] | None = None,
        on_start: Callable[[Event], None] | None = None,
        placement: Mapping[str, Sequence[int]] | None = None,
        coupled: bool = False, priority: int = 0, cohort_size: int = 1,
        cohort_widened: Sequence[str] | None = None,
        cohort_members: Mapping[str, Sequence[Sequence[int]]] | None = None,
        memory=None, memory_pools: Sequence[str] | None = None,
        memory_bindings: Mapping[str, Mapping] | None = None,
        native_hook: bool = False,
        deliveries: Sequence[tuple[str, float]] = (),
        latency_from_first_service: bool = False,
    ) -> Event:
        if (work is None and latency == 0.0 and caps is None and not acquire
                and not free and placement is None and not coupled
                and cohort_size == 1 and not cohort_widened and not cohort_members
                and not latency_from_first_service):
            prepared = _EMPTY_TASK
        else:
            prepared = self._prepare_uncoupled_cached(
                name, work, latency=latency, caps=caps, acquire=acquire,
                free=free, placement=placement, coupled=coupled,
                priority=priority, cohort_size=cohort_size,
                cohort_widened=cohort_widened, cohort_members=cohort_members,
                latency_from_first_service=latency_from_first_service,
            )
        return self._submit_prepared(
            prepared, name, after=after, release=release, owner=owner,
            metadata=metadata, on_complete=on_complete, on_start=on_start,
            memory=memory, memory_pools=memory_pools,
            memory_bindings=memory_bindings, native_hook=native_hook,
            deliveries=deliveries,
        )

    def _uncoupled_fast_key(self, work, placement):
        """Return a hashable cache key for exact-typed uncoupled work/placement, or None."""
        wcls = work.__class__
        if wcls is not dict and wcls is not MappingProxyType:
            return None
        w_items = []
        for k, v in work.items():
            vt = type(v)
            if type(k) is not str or (vt is not float and vt is not int):
                return None
            if v:
                w_items.append((k, float(v)))
        if not placement:
            return ("_u", tuple(w_items), ())
        pcls = placement.__class__
        if pcls is not dict and pcls is not MappingProxyType:
            return None
        p_items = []
        for k, units in placement.items():
            if type(k) is not str or units.__class__ is not tuple:
                return None
            for u in units:
                if type(u) is not int:
                    return None
            p_items.append((k, units))
        return ("_u", tuple(w_items), tuple(p_items))

    def _prepare_uncoupled_cached(
        self, name, work=None, *, latency=0.0, caps=None, acquire=(), free=(),
        placement=None, coupled=False, priority=0, cohort_size=1,
        cohort_widened=None, cohort_members=None,
        latency_from_first_service=False,
    ) -> _PreparedTask:
        if (work is not None and not caps and not acquire and not free
                and not coupled and latency == 0.0 and not priority
                and cohort_size == 1 and not cohort_widened and not cohort_members
                and not latency_from_first_service and not self._hbm_bus_aliases
                and type(latency) in (float, int)
                and type(priority) is int and type(cohort_size) is int):
            ukey = self._uncoupled_fast_key(work, placement)
            if ukey is not None:
                cache = self._stage_prepared
                hit = cache.get(ukey)
                if hit is not None:
                    return hit
                prepared = (
                    self._prepare_plain_work(name, work, ())
                    if not placement
                    else self._prepare_placed_uncoupled(name, work, placement, ())
                )
                if len(cache) >= _STAGE_PREPARATION_CACHE_LIMIT:
                    cache.clear()
                cache[ukey] = prepared
                return prepared
        return self._prepare(
            name, work, latency=latency, caps=caps, acquire=acquire, free=free,
            placement=placement, coupled=coupled, priority=priority,
            cohort_size=cohort_size, cohort_widened=cohort_widened,
            cohort_members=cohort_members,
            latency_from_first_service=latency_from_first_service,
        )

    def _prepare(
        self, name, work=None, *, latency=0.0, caps=None, acquire=(), free=(),
        placement=None, coupled=False, priority=0, cohort_size=1,
        cohort_widened=None, cohort_members=None,
        latency_from_first_service=False, static_id: int = 0,
    ) -> _PreparedTask:
        """Validate a template against this engine's physical resource binding."""
        if (work is None and latency == 0 and caps is None and acquire == ()
                and free == () and placement is None and not coupled
                and cohort_size == 1 and not cohort_widened and not cohort_members
                and not latency_from_first_service):
            return _EMPTY_TASK
        hbm_accounting: tuple[tuple[str, float, float, float, float | None, str, float, str, float, str], ...] = ()
        if work and self._hbm_bus_aliases:
            work, caps, hbm_accounting = _expand_hbm_bus_work(self._hbm_bus_aliases, name, work, caps)
            if not coupled and hbm_accounting and _is_pure_coupled_hbm_work(work, hbm_accounting):
                coupled = True
        if (work is not None and not caps
                and not acquire and not free
                and not coupled and latency == 0.0 and not priority
                and cohort_size == 1 and not cohort_widened and not cohort_members
                and not latency_from_first_service):
            if placement is None and work.__class__ is dict:
                return self._prepare_plain_work(name, work, hbm_accounting)
            if (placement is not None and type(latency) in (float, int)
                    and type(priority) is int and type(cohort_size) is int):
                return self._prepare_placed_uncoupled(
                    name, work, placement, hbm_accounting, static_id=static_id,
                )
        demands = {key: float(value) for key, value in (work or {}).items() if value}
        for key, value in demands.items():
            if key not in self.resources:
                raise ExecutionError(f"resource_missing:{key}")
            if value < 0 or not math.isfinite(value):
                raise ExecutionError(f"invalid_work:{name}:{key}")
        if latency < 0 or not math.isfinite(latency):
            raise ExecutionError(f"invalid_latency:{name}")
        if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
            raise ExecutionError(f"invalid_task_priority:{name}")
        if not isinstance(cohort_size, int) or isinstance(cohort_size, bool) or cohort_size < 1:
            raise ExecutionError(f"invalid_cohort_size:{name}")
        limits = dict(caps or {})
        if any(value <= 0 or not math.isfinite(value) for value in limits.values()):
            raise ExecutionError(f"invalid_supply_rate:{name}")
        placement, placement_stable = self._validate_placement(placement)
        norm_widened, norm_members = _normalize_cohort_symmetry(
            name, placement, cohort_size, cohort_widened, cohort_members,
        )
        allocation_demands = {}
        seen_acquire: set[str] = set()
        for allocation in acquire:
            if allocation.name in seen_acquire:
                raise ExecutionError(f"allocation_duplicate:{allocation.name}")
            seen_acquire.add(allocation.name)
            if allocation.size < 0 or not math.isfinite(allocation.size):
                raise ExecutionError(f"invalid_allocation:{allocation.name}")
            if allocation.pool not in self.pools:
                raise ExecutionError(f"pool_missing:{allocation.pool}")
            if allocation.size > self.pools[allocation.pool]:
                raise ExecutionError(f"allocation_exceeds_capacity:{allocation.name}")
            if not allocation.units or len(set(allocation.units)) != len(allocation.units) or any(
                unit < 0 or unit >= self.pool_units[allocation.pool] for unit in allocation.units
            ):
                raise ExecutionError(f"buffer_unit_out_of_range:{allocation.pool}")
            for unit in allocation.units:
                key = allocation.pool, unit
                allocation_demands[key] = allocation_demands.get(key, 0.0) + allocation.size
        acq_t = acquire if acquire.__class__ is tuple else tuple(acquire)
        free_t = free if free.__class__ is tuple else tuple(free)
        demands_t = tuple(allocation_demands.items())
        demand_keys = (frozenset(allocation_demands)
                       if allocation_demands else _EMPTY_FROZENSET)
        acq_pools = (frozenset(a.pool for a in acq_t)
                     if acq_t else _EMPTY_FROZENSET)
        fits_empty = (all(amt <= self.pools[k[0]] for k, amt in demands_t)
                      if demands_t else True)
        c_demand = None
        c_cap_limit = math.inf
        c_solo_rates = None
        c_res_info = None
        s_rates = None
        u_res_info = None
        if (coupled and demands
                and all(self.resources[k].policy != "fifo" for k in demands)):
            c_demand, c_cap_limit, c_solo_rates, c_res_info = self._prepare_coupled_service(
                demands, placement, limits, cohort_size, norm_widened)
        elif not coupled and demands:
            s_rates = {}
            u_res_info = {}
            for k, d_val in demands.items():
                res = self.resources[k]
                r_rate = res.rate
                uk = placement.get(k) if placement else None
                if res.policy == "fifo":
                    s_rate = (r_rate / res.units * len(uk)) if uk is not None else r_rate
                elif uk is not None:
                    s_rate = r_rate / res.units * len(uk)
                else:
                    s_rate = r_rate
                cap_val = limits.get(k) if limits else None
                if cap_val is not None and cap_val < s_rate:
                    s_rate = cap_val
                s_rates[k] = s_rate
                u_cap = r_rate / res.units
                uc_u = res.units if uk is None else len(uk)
                denom_u = u_cap * uc_u
                scale_u = d_val / denom_u if denom_u > 0.0 else 0.0
                if u_res_info is not None and scale_u > 0.0 and math.isfinite(scale_u):
                    amt_u = d_val / scale_u
                    pu_u = amt_u if uc_u == 1 else amt_u / uc_u
                    lim_u = math.inf
                    if cap_val is not None and amt_u:
                        cl_u = cap_val / amt_u
                        if cl_u < lim_u:
                            lim_u = cl_u
                    solo_lim = lim_u
                    if pu_u:
                        c_lim_u = u_cap / pu_u
                        if c_lim_u < solo_lim:
                            solo_lim = c_lim_u
                    fu_u = solo_lim if 0.0 < solo_lim < math.inf else 0.0
                    u_res_info[k] = (d_val, amt_u, pu_u, lim_u, amt_u * fu_u, uk)
                else:
                    u_res_info = None
        partial_placed_res = (
            frozenset(
                key for key, units in placement.items()
                if key in demands and len(units) < self.resources[key].units
            )
            if placement else _EMPTY_FROZENSET
        )
        static_stable = (
            placement_stable
            and type(latency) in (float, int)
            and type(coupled) is bool
            and type(priority) is int
            and type(cohort_size) is int
            and all(type(k) is str for k in demands)
            and all(type(k) is str and type(v) in (float, int) for k, v in limits.items())
            and all(type(w) is str for w in norm_widened)
        )
        exp_units = 0
        eff_units = 0
        for key in demands:
            uk = placement.get(key) if placement else None
            if uk is not None:
                nu = len(uk)
                exp_units += nu
                eff_units += nu
            else:
                eff_units += self.resources[key].units
        return _PreparedTask(
            demands, latency, limits, acq_t, free_t,
            placement, coupled, demands_t,
            priority, cohort_size, (), (), acq_pools, static_id, fits_empty,
            c_demand, c_cap_limit, c_solo_rates, demand_keys,
            s_rates, c_res_info, u_res_info,
            _compute_pool_unit_masks(acq_t), norm_widened, norm_members,
            bool(latency_from_first_service), self._hbm_bus_epoch, hbm_accounting,
            partial_placed_res, static_stable, exp_units, eff_units)

    def _validate_placement(self, raw_placement) -> tuple[dict[str, tuple[int, ...]], bool]:
        if not raw_placement:
            return _EMPTY_PLACEMENT, True
        is_proxy = raw_placement.__class__ is MappingProxyType
        if is_proxy:
            hit = self._validated_placements.get(id(raw_placement))
            if hit is not None and hit[0] is raw_placement:
                return hit[1], hit[2]
        resources = self.resources
        validated: dict[str, tuple[int, ...]] = {}
        stable = True
        for key, raw_units in raw_placement.items():
            res = resources.get(key)
            if res is None:
                raise ExecutionError(f"resource_missing:{key}")
            if type(key) is not str:
                stable = False
            units = raw_units if raw_units.__class__ is tuple else tuple(raw_units)
            n_u = len(units)
            if n_u == 1:
                u0 = units[0]
                if u0 < 0 or u0 >= res.units:
                    raise ExecutionError(f"resource_unit_out_of_range:{key}")
                if type(u0) is not int:
                    stable = False
                validated[key] = units
            else:
                if not n_u or len(set(units)) != n_u or any(
                    unit < 0 or unit >= res.units for unit in units
                ):
                    raise ExecutionError(f"resource_unit_out_of_range:{key}")
                if any(type(unit) is not int for unit in units):
                    stable = False
                validated[key] = tuple(sorted(units))
        if is_proxy:
            cache = self._validated_placements
            if len(cache) >= _VALIDATED_PLACEMENT_CACHE_LIMIT:
                cache.clear()
            cache[id(raw_placement)] = (raw_placement, validated, stable)
        return validated, stable

    def _prepare_placed_uncoupled(
        self, name, work, raw_placement, hbm_accounting, *, static_id: int = 0,
    ) -> _PreparedTask:
        placement, placement_stable = self._validate_placement(raw_placement)
        resources = self.resources
        demands: dict[str, float] = {}
        s_rates: dict[str, float] = {}
        u_res_info: dict[str, tuple] | None = {}
        partial_list: list[str] = []
        keys_str = True
        exp_units = 0
        eff_units = 0
        for key, value in work.items():
            if not value:
                continue
            val = float(value)
            res = resources.get(key)
            if res is None:
                raise ExecutionError(f"resource_missing:{key}")
            if val < 0 or not math.isfinite(val):
                raise ExecutionError(f"invalid_work:{name}:{key}")
            if type(key) is not str:
                keys_str = False
            demands[key] = val
            r_rate = res.rate
            r_units = res.units
            uk = placement.get(key)
            if uk is not None:
                uc_u = len(uk)
                exp_units += uc_u
                eff_units += uc_u
                s_rate = r_rate / r_units * uc_u
                if uc_u < r_units:
                    partial_list.append(key)
            else:
                uc_u = r_units
                eff_units += r_units
                s_rate = r_rate
            s_rates[key] = s_rate
            u_cap = r_rate / r_units
            denom_u = u_cap * uc_u
            scale_u = val / denom_u if denom_u > 0.0 else 0.0
            if u_res_info is not None and scale_u > 0.0 and math.isfinite(scale_u):
                amt_u = val / scale_u
                pu_u = amt_u if uc_u == 1 else amt_u / uc_u
                c_lim_u = (u_cap / pu_u) if pu_u else math.inf
                fu_u = c_lim_u if 0.0 < c_lim_u < math.inf else 0.0
                u_res_info[key] = (val, amt_u, pu_u, math.inf, amt_u * fu_u, uk)
            else:
                u_res_info = None
        partial_res = frozenset(partial_list) if partial_list else _EMPTY_FROZENSET
        return _PreparedTask(
            demands, 0.0, _EMPTY_CAPS, (), (),
            placement, False, (), 0, 1, (), (), _EMPTY_FROZENSET, static_id, True,
            None, math.inf, None, _EMPTY_FROZENSET,
            s_rates or None, None, u_res_info or None,
            (), (), (), False, self._hbm_bus_epoch, hbm_accounting,
            partial_res, placement_stable and keys_str, exp_units, eff_units,
        )

    def _prepare_plain_work(self, name, work, hbm_accounting) -> _PreparedTask:
        # Ordinary finite work on declared resources: the same checks as
        # the general path, without building the four unused structures.
        resources = self.resources
        demands = {}
        s_rates = {}
        u_res_info = {}
        eff_units = 0
        for key, value in work.items():
            if not value:
                continue
            value = float(value)
            res = resources.get(key)
            if res is None:
                raise ExecutionError(f"resource_missing:{key}")
            if value < 0 or not math.isfinite(value):
                raise ExecutionError(f"invalid_work:{name}:{key}")
            demands[key] = value
            r_rate = res.rate
            eff_units += res.units
            s_rates[key] = r_rate
            u_cap = r_rate / res.units
            scale_u = value / r_rate
            if scale_u > 0.0 and math.isfinite(scale_u):
                amt_u = value / scale_u
                pu_u = amt_u if res.units == 1 else amt_u / res.units
                c_lim_u = (u_cap / pu_u) if pu_u else math.inf
                fu_u = c_lim_u if 0.0 < c_lim_u < math.inf else 0.0
                if u_res_info is not None:
                    u_res_info[key] = (
                        value, amt_u, pu_u, math.inf, amt_u * fu_u, None,
                    )
            else:
                u_res_info = None
        return _PreparedTask(
            demands, 0.0, _EMPTY_CAPS, (), (),
            _EMPTY_PLACEMENT, False, (), 0, 1,
            solo_rates=s_rates or None,
            uncoupled_res_info=u_res_info or None,
            hbm_bus_epoch=self._hbm_bus_epoch,
            hbm_accounting=hbm_accounting,
            static_stable=all(type(k) is str for k in demands),
            explicit_units=0,
            effective_units=eff_units)

    def _prepare_coupled_service(self, demands, placement, limits, cohort_size, norm_widened):
        c_demand = None
        c_cap_limit = math.inf
        c_solo_rates = None
        c_res_info = None
        scale0 = 0.0
        for k, d in demands.items():
            res = self.resources[k]
            decl = placement.get(k) if placement else None
            n_u = res.units if decl is None else len(decl)
            denom = (res.rate / res.units) * n_u
            if cohort_size > 1 and k not in norm_widened:
                denom *= cohort_size
            s_k = d / denom if denom > 0.0 else math.inf
            if s_k > scale0:
                scale0 = s_k
        if scale0 > 0.0 and math.isfinite(scale0):
            c_demand = {k: d / scale0 for k, d in demands.items()}
            if limits:
                for k, amt in c_demand.items():
                    cap_val = limits.get(k)
                    if cap_val is not None and amt:
                        cl = cap_val / amt
                        if cl < c_cap_limit:
                            c_cap_limit = cl
            c_info_dict = {}
            c_solo_limit = c_cap_limit
            for k, amt in c_demand.items():
                res = self.resources[k]
                u_cap = res.rate / res.units
                uk = placement.get(k) if placement else None
                u_cnt = res.units if uk is None else len(uk)
                pu = amt if u_cnt == 1 else amt / u_cnt
                cl = (u_cap / pu) if pu else math.inf
                c_info_dict[k] = (amt, u_cap, pu, cl, uk)
                if cl < c_solo_limit:
                    c_solo_limit = cl
            c_items_tuple = tuple(c_demand.items())
            c_excl_dict = {}
            for s_k in c_demand:
                ex_lim = c_cap_limit
                for o_k, (_, _, _, cl_o, _) in c_info_dict.items():
                    if o_k != s_k and cl_o < ex_lim:
                        ex_lim = cl_o
                c_excl_dict[s_k] = ex_lim
            c_res_info = (c_info_dict, c_items_tuple, c_excl_dict)
            if 0.0 < c_solo_limit < math.inf:
                c_solo_rates = tuple(
                    (k, amt * c_solo_limit) for k, amt in c_items_tuple
                )
        return c_demand, c_cap_limit, c_solo_rates, c_res_info

    @staticmethod
    def clone_prepared_scoped(static: _PreparedTask,
                              acquire: tuple[Allocation, ...],
                              free: tuple[Allocation, ...]) -> _PreparedTask:
        """Clone a validated static Stage plan with freshly scoped buffer names."""
        return _PreparedTask(
            static.demand, static.latency, static.caps, acquire, free,
            static.placement, static.coupled, static.allocation_demands,
            static.priority, static.cohort_size, static.acquire_assigns,
            static.free_pairs, static.acquire_pools, static.static_id,
            static.fits_empty_pools, static.coupled_demand,
            static.coupled_cap_limit, static.coupled_solo_rates,
            static.demand_keys, static.solo_rates,
            static.coupled_res_info, static.uncoupled_res_info,
            static.pool_unit_masks, static.cohort_widened, static.cohort_members,
            static.latency_from_first_service, static.hbm_bus_epoch,
            static.hbm_accounting, static.partial_placed_resources,
            static.static_stable, static.explicit_units, static.effective_units)

    def prepared_stage(self, stage, acquire=(), free=()):
        """Validate one immutable Stage declaration, caching its static part.

        A repeated tile re-declares the same work/caps/placement geometry under
        a new task name; only the scoped allocation identities change. The
        static declaration is therefore cached by its structural signature and
        each occurrence validates its own buffers against the live pool.
        """
        id_cache = self._stage_id_prepared
        lookup_stage = getattr(stage, "_proto_stage", stage)
        hit = id_cache.get(id(lookup_stage))
        # The identity cache is a weak map: a completed tile must not stay
        # alive just because its Stage address was remembered.  A dead weakref
        # also distinguishes a genuine hit from a recycled ``id``.
        if hit is not None and hit[0]() is lookup_stage:
            static = hit[1]
        else:
            static = self._prepare_static_stage(lookup_stage)
        if acquire is static.acquire and free is static.free:
            return static
        if not acquire and not free:
            if not static.acquire and not static.free:
                return static
            return _PreparedTask(
                static.demand, static.latency, static.caps, (), (),
                static.placement, static.coupled, (),
                static.priority, static.cohort_size, (), (),
                _EMPTY_FROZENSET, 0, True, static.coupled_demand,
                static.coupled_cap_limit, static.coupled_solo_rates,
                _EMPTY_FROZENSET, static.solo_rates,
                static.coupled_res_info, static.uncoupled_res_info, (),
                static.cohort_widened, static.cohort_members,
                static.latency_from_first_service,
                static.hbm_bus_epoch, static.hbm_accounting,
                static.partial_placed_resources, static.static_stable,
                static.explicit_units, static.effective_units)
        if acquire.__class__ is not tuple:
            acquire = tuple(acquire)
        if free.__class__ is not tuple:
            free = tuple(free)
        if not acquire:
            return _PreparedTask(
                static.demand, static.latency, static.caps, (), free,
                static.placement, static.coupled, (),
                static.priority, static.cohort_size, (), (),
                _EMPTY_FROZENSET, 0, True, static.coupled_demand,
                static.coupled_cap_limit, static.coupled_solo_rates,
                _EMPTY_FROZENSET, static.solo_rates,
                static.coupled_res_info, static.uncoupled_res_info, (),
                static.cohort_widened, static.cohort_members,
                static.latency_from_first_service,
                static.hbm_bus_epoch, static.hbm_accounting,
                static.partial_placed_resources, static.static_stable,
                static.explicit_units, static.effective_units)
        static_acq = static.acquire
        n_acq = len(acquire)
        if n_acq == len(static_acq):
            if n_acq == 1:
                a0 = acquire[0]
                s0 = static_acq[0]
                if (a0.size == s0.size and a0.pool == s0.pool
                        and (a0.units is s0.units or a0.units == s0.units)):
                    return self.clone_prepared_scoped(static, acquire, free)
            elif n_acq == 2:
                a0, a1 = acquire
                s0, s1 = static_acq
                if (a0.name != a1.name
                        and a0.size == s0.size and a0.pool == s0.pool
                        and (a0.units is s0.units or a0.units == s0.units)
                        and a1.size == s1.size and a1.pool == s1.pool
                        and (a1.units is s1.units or a1.units == s1.units)):
                    return self.clone_prepared_scoped(static, acquire, free)
            else:
                same_shape = True
                for a_item, s_item in zip(acquire, static_acq):
                    if (a_item.size != s_item.size or a_item.pool != s_item.pool
                            or (a_item.units is not s_item.units
                                and a_item.units != s_item.units)):
                        same_shape = False
                        break
                if same_shape and len({a.name for a in acquire}) == n_acq:
                    return self.clone_prepared_scoped(static, acquire, free)
        if n_acq == 1:
            return self._prepare_stage_single_allocation(static, acquire, free)
        pools = self.pools
        pool_units = self.pool_units
        demands = {}
        seen: set[str] = set()
        for allocation in acquire:
            name = allocation.name
            if name in seen:
                raise ExecutionError(f"allocation_duplicate:{name}")
            seen.add(name)
            size = allocation.size
            if size < 0 or not math.isfinite(size):
                raise ExecutionError(f"invalid_allocation:{name}")
            pool = allocation.pool
            if pool not in pools:
                raise ExecutionError(f"pool_missing:{pool}")
            if size > pools[pool]:
                raise ExecutionError(f"allocation_exceeds_capacity:{name}")
            units = allocation.units
            if (not units or len(set(units)) != len(units)
                    or any(unit < 0 or unit >= pool_units[pool]
                           for unit in units)):
                raise ExecutionError(f"buffer_unit_out_of_range:{pool}")
            for unit in units:
                pair = (pool, unit)
                demands[pair] = demands.get(pair, 0.0) + size
        demands_t = tuple(demands.items())
        acq_pools = frozenset(a.pool for a in acquire)
        fits_empty = all(amt <= pools[k[0]] for k, amt in demands_t)
        return _PreparedTask(
            static.demand, static.latency, static.caps, acquire, free,
            static.placement, static.coupled, demands_t,
            static.priority, static.cohort_size, (), (),
            acq_pools, 0, fits_empty, static.coupled_demand,
            static.coupled_cap_limit, static.coupled_solo_rates,
            frozenset(demands), static.solo_rates,
            static.coupled_res_info, static.uncoupled_res_info,
            _compute_pool_unit_masks(acquire),
            static.cohort_widened, static.cohort_members,
            static.latency_from_first_service,
            hbm_bus_epoch=static.hbm_bus_epoch,
            hbm_accounting=static.hbm_accounting,
            partial_placed_resources=static.partial_placed_resources,
            static_stable=static.static_stable,
            explicit_units=static.explicit_units,
            effective_units=static.effective_units)

    def _prepare_stage_single_allocation(self, static, acquire, free) -> _PreparedTask:
        allocation = acquire[0]
        size = allocation.size
        if size < 0 or not math.isfinite(size):
            raise ExecutionError(f"invalid_allocation:{allocation.name}")
        pool = allocation.pool
        pools = self.pools
        if pool not in pools:
            raise ExecutionError(f"pool_missing:{pool}")
        if size > pools[pool]:
            raise ExecutionError(f"allocation_exceeds_capacity:{allocation.name}")
        units = allocation.units
        pu = self.pool_units[pool]
        if len(units) == 1:
            u0 = units[0]
            if u0 < 0 or u0 >= pu:
                raise ExecutionError(f"buffer_unit_out_of_range:{pool}")
            demands_tuple = (((pool, u0), size),)
        else:
            if (not units or len(set(units)) != len(units)
                    or any(unit < 0 or unit >= pu for unit in units)):
                raise ExecutionError(f"buffer_unit_out_of_range:{pool}")
            demands_tuple = tuple(((pool, unit), size) for unit in units)
        p_masks_1 = (((pool, _units_to_mask(units)),)
                     if size > 0 else None)
        return _PreparedTask(
            static.demand, static.latency, static.caps, acquire, free,
            static.placement, static.coupled, demands_tuple,
            static.priority, static.cohort_size, (), (),
            frozenset((pool,)), 0, True, static.coupled_demand,
            static.coupled_cap_limit, static.coupled_solo_rates,
            frozenset(k for k, _ in demands_tuple), static.solo_rates,
            static.coupled_res_info, static.uncoupled_res_info,
            p_masks_1, static.cohort_widened, static.cohort_members,
            static.latency_from_first_service,
            hbm_bus_epoch=static.hbm_bus_epoch,
            hbm_accounting=static.hbm_accounting,
            partial_placed_resources=static.partial_placed_resources,
            static_stable=static.static_stable,
            explicit_units=static.explicit_units,
            effective_units=static.effective_units)

    def _prepare_static_stage(self, lookup_stage) -> _PreparedTask:
        id_cache = self._stage_id_prepared
        stg_acq = lookup_stage.acquire
        stg_free = lookup_stage.free
        cohort_size = getattr(lookup_stage, "cohort_size", 1)
        cohort_widened = getattr(lookup_stage, "cohort_widened", ())
        raw_members = getattr(lookup_stage, "cohort_members", ())
        lffs = bool(getattr(lookup_stage, "latency_from_first_service", False))
        if (not stg_acq and not stg_free and not lookup_stage.caps
                and lookup_stage.latency == 0.0 and not lookup_stage.coupled
                and lookup_stage.priority == 0 and cohort_size == 1
                and not cohort_widened and not raw_members and not lffs
                and not self._hbm_bus_aliases
                and type(lookup_stage.latency) in (float, int)
                and type(lookup_stage.priority) is int
                and type(cohort_size) is int):
            ukey = self._uncoupled_fast_key(lookup_stage.work, lookup_stage.placement)
            if ukey is not None:
                cache = self._stage_prepared
                static = cache.get(ukey)
                if static is None:
                    static = (
                        self._prepare_plain_work(lookup_stage.name, lookup_stage.work, ())
                        if not lookup_stage.placement
                        else self._prepare_placed_uncoupled(
                            lookup_stage.name, lookup_stage.work,
                            lookup_stage.placement, (),
                        )
                    )
                    if len(cache) >= _STAGE_PREPARATION_CACHE_LIMIT:
                        cache.clear()
                    cache[ukey] = static
                if len(id_cache) >= _STAGE_PREPARATION_CACHE_LIMIT:
                    id_cache.clear()
                id_cache[id(lookup_stage)] = (weakref.ref(lookup_stage), static)
                return static
        members_sig = (
            tuple(sorted(raw_members.items()))
            if isinstance(raw_members, Mapping) else tuple(raw_members)
        )
        if not stg_acq:
            acq_sig = ()
        elif len(stg_acq) == 1:
            a0 = stg_acq[0]
            acq_sig = ((a0.pool, a0.size, a0.units),)
        elif len({a.name for a in stg_acq}) == len(stg_acq):
            acq_sig = tuple((a.pool, a.size, a.units) for a in stg_acq)
        else:
            acq_sig = tuple((a.name, a.pool, a.size, a.units)
                            for a in stg_acq)
        free_sig = (tuple((a.pool, a.size, a.units) for a in stg_free)
                    if stg_free else ())
        hbm_sig = ()
        if lookup_stage.work and self._hbm_bus_aliases:
            hbm_sig = tuple(
                (base, _classify_hbm_split(base, lookup_stage.name, lookup_stage.work, float(amt))[0])
                for base, bus in self._hbm_bus_aliases.items()
                if (amt := lookup_stage.work.get(base))
                and bus.row_r not in lookup_stage.work
                and bus.row_w not in lookup_stage.work
            )
        proto_cache = self._stage_proto_prepared
        proto_key = (id(lookup_stage.work), id(lookup_stage.caps),
                     id(lookup_stage.placement),
                     lookup_stage.latency,
                     lookup_stage.coupled, lookup_stage.priority,
                     cohort_size, cohort_widened, members_sig,
                     acq_sig, free_sig, lffs, hbm_sig)
        phit = proto_cache.get(proto_key)
        if (phit is not None and phit[0] is lookup_stage.work
                and phit[1] is lookup_stage.caps
                and phit[2] is lookup_stage.placement):
            static_base = phit[3]
        else:
            key = (tuple(lookup_stage.work.items()),
                   lookup_stage.latency, tuple(lookup_stage.caps.items()),
                   tuple((name, units)
                         for name, units in lookup_stage.placement.items()),
                   lookup_stage.coupled, lookup_stage.priority, cohort_size,
                   cohort_widened, members_sig, acq_sig, free_sig, lffs, hbm_sig)
            cache = self._stage_prepared
            static_base = cache.get(key)
            if static_base is None:
                self._next_static_id += 1
                static_base = self._prepare(
                    lookup_stage.name, lookup_stage.work,
                    latency=lookup_stage.latency, caps=lookup_stage.caps,
                    acquire=stg_acq, free=stg_free,
                    placement=lookup_stage.placement,
                    coupled=lookup_stage.coupled,
                    priority=lookup_stage.priority,
                    cohort_size=cohort_size,
                    cohort_widened=cohort_widened or None,
                    cohort_members=raw_members or None,
                    latency_from_first_service=lffs,
                    static_id=self._next_static_id)
                if len(cache) >= _STAGE_PREPARATION_CACHE_LIMIT:
                    cache.clear()
                cache[key] = static_base
            if len(proto_cache) >= _STAGE_PREPARATION_CACHE_LIMIT:
                proto_cache.clear()
            proto_cache[proto_key] = (
                lookup_stage.work, lookup_stage.caps,
                lookup_stage.placement, static_base)
        if (stg_acq is static_base.acquire
                and stg_free is static_base.free):
            static = static_base
        else:
            static = _PreparedTask(
                static_base.demand, static_base.latency, static_base.caps,
                stg_acq, stg_free, static_base.placement,
                static_base.coupled, static_base.allocation_demands,
                static_base.priority, static_base.cohort_size,
                static_base.acquire_assigns, static_base.free_pairs,
                static_base.acquire_pools, static_base.static_id,
                static_base.fits_empty_pools, static_base.coupled_demand,
                static_base.coupled_cap_limit,
                static_base.coupled_solo_rates,
                static_base.demand_keys, static_base.solo_rates,
                static_base.coupled_res_info,
                static_base.uncoupled_res_info,
                static_base.pool_unit_masks,
                static_base.cohort_widened, static_base.cohort_members,
                static_base.latency_from_first_service,
                static_base.hbm_bus_epoch, static_base.hbm_accounting,
                static_base.partial_placed_resources,
                static_base.static_stable,
                static_base.explicit_units, static_base.effective_units)
        if len(id_cache) >= _STAGE_PREPARATION_CACHE_LIMIT:
            id_cache.clear()
        id_cache[id(lookup_stage)] = (weakref.ref(lookup_stage), static)
        return static

    def prepare_stage(self, name, work=None, *, latency=0.0, caps=None,
                      acquire=(), free=(), placement=None, coupled=False,
                      priority=0, cohort_size=1, cohort_widened=None,
                      cohort_members=None, latency_from_first_service=False):
        return self._prepare(
            name, work, latency=latency, caps=caps, acquire=acquire,
            free=free, placement=placement, coupled=coupled, priority=priority,
            cohort_size=cohort_size, cohort_widened=cohort_widened,
            cohort_members=cohort_members,
            latency_from_first_service=latency_from_first_service)

    def submit_prepared(self, prepared, name: str, *, after=(), release=None,
                        owner="", metadata=None, on_complete=None,
                        on_start=None, memory=None, memory_pools=None,
                        memory_bindings=None) -> Event:
        return self._submit_prepared(
            prepared, name, after=after, release=release, owner=owner,
            metadata=metadata, on_complete=on_complete, on_start=on_start,
            memory=memory, memory_pools=memory_pools,
            memory_bindings=memory_bindings)
