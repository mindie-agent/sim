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
import re
from typing import Callable, Mapping, Sequence
import weakref

from .engine_types import (
    Allocation, Event, ExecutionError, _EMPTY_CAPS, _EMPTY_FROZENSET,
    _EMPTY_PLACEMENT, _EMPTY_TASK, _PreparedTask, _compute_pool_unit_masks,
    _units_to_mask,
)
from .memory_physics import HbmBusSpec

_READ_PIPE_PREFIXES = frozenset({"mte2", "mte2_aic", "mte2_aiv"})
_WRITE_PIPE_PREFIXES = frozenset({"mte3", "fixpipe"})
_READ_NAME_TOKENS = frozenset({
    "read", "load", "fetch", "prefetch", "gather", "reload", "in", "input", "counts",
})
_WRITE_NAME_TOKENS = frozenset({
    "write", "store", "spill", "writeback", "evict", "scatter", "out", "output",
})
_NAME_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _hbm_bus_alias_terms(
    bus, base: str, name: str, work: Mapping[str, float], amount: float,
) -> tuple[float, float]:
    """Resolve directional ``(row_r, row_w)`` work via ``bus.terms_for``."""
    suffix = "@" + base.split("@", 1)[1] if "@" in base else ""
    read_pipe = 0.0
    write_pipe = 0.0
    for key, value in work.items():
        if not value or key == base:
            continue
        if suffix and "@" in key and not key.endswith(suffix):
            continue
        prefix = key.split("@", 1)[0]
        if prefix in _READ_PIPE_PREFIXES:
            read_pipe += float(value)
        elif prefix in _WRITE_PIPE_PREFIXES:
            write_pipe += float(value)
    if read_pipe > 0.0 and write_pipe > 0.0:
        share = read_pipe / (read_pipe + write_pipe)
        tr_r, tr_w = bus.terms_for("read", amount * share)
        tw_r, tw_w = bus.terms_for("write", amount * (1.0 - share))
        return (
            tr_r.ms_per_progress + tw_r.ms_per_progress,
            tr_w.ms_per_progress + tw_w.ms_per_progress,
        )
    if read_pipe > 0.0:
        tr_r, tr_w = bus.terms_for("read", amount)
        return tr_r.ms_per_progress, tr_w.ms_per_progress
    if write_pipe > 0.0:
        tw_r, tw_w = bus.terms_for("write", amount)
        return tw_r.ms_per_progress, tw_w.ms_per_progress
    tokens = {tok for tok in _NAME_SPLIT_RE.split(name.lower()) if tok}
    has_r = bool(tokens & _READ_NAME_TOKENS)
    has_w = bool(tokens & _WRITE_NAME_TOKENS)
    if has_r and not has_w:
        tr_r, tr_w = bus.terms_for("read", amount)
        return tr_r.ms_per_progress, tr_w.ms_per_progress
    if has_w and not has_r:
        tw_r, tw_w = bus.terms_for("write", amount)
        return tw_r.ms_per_progress, tw_w.ms_per_progress
    tr_r, tr_w = bus.terms_for("read", amount)
    tw_r, tw_w = bus.terms_for("write", amount)
    return (
        max(tr_r.ms_per_progress, tw_r.ms_per_progress),
        max(tr_w.ms_per_progress, tw_w.ms_per_progress),
    )


def _expand_hbm_bus_work(
    aliases: Mapping[str, object], name: str,
    work: Mapping[str, float], caps: Mapping[str, float] | None,
) -> tuple[Mapping[str, float], Mapping[str, float] | None]:
    """Mirror shipping scalar HBM demands onto bound directional bus rows."""
    if not any(base in work and work[base] for base in aliases):
        return work, caps
    expanded = dict(work)
    expanded_caps = dict(caps) if caps else None
    for base, bus in aliases.items():
        amount = expanded.get(base)
        if not amount or bus.row_r in expanded or bus.row_w in expanded:
            continue
        amt = float(amount)
        if amt <= 0.0 or not math.isfinite(amt):
            continue
        t_r, t_w = _hbm_bus_alias_terms(bus, base, name, expanded, amt)
        if t_r > 0.0:
            expanded[bus.row_r] = t_r
        if t_w > 0.0:
            expanded[bus.row_w] = t_w
        if expanded_caps is not None and base in expanded_caps:
            cap_val = float(expanded_caps[base])
            if cap_val > 0.0 and math.isfinite(cap_val):
                if t_r > 0.0:
                    expanded_caps.setdefault(bus.row_r, t_r * cap_val / amt)
                if t_w > 0.0:
                    expanded_caps.setdefault(bus.row_w, t_w * cap_val / amt)
    return expanded, expanded_caps


def _discover_hbm_bus_aliases(resources: Mapping, aliases: dict) -> bool:
    """Register default symmetric bus aliases for pre-existing ``:r``/``:w`` rows."""
    discovered = False
    for name, res_r in resources.items():
        if not name.endswith(":r") or res_r.rate != 1.0:
            continue
        base = name[:-2]
        row_w = base + ":w"
        if base in aliases or base not in resources or row_w not in resources:
            continue
        base_res = resources[base]
        res_w = resources[row_w]
        if base_res.rate <= 0.0 or res_w.rate != 1.0:
            continue
        aliases[base] = HbmBusSpec(
            0, float(base_res.rate), float(base_res.rate), 0.0,
            row_r=name, row_w=row_w,
        )
        discovered = True
    return discovered


def _retrofit_hbm_bus_tasks(
    tasks: Mapping, native_slots: Mapping, base: str, bus, *,
    replacing: bool = False, include_native: bool = False,
) -> bool:
    """Bind already-submitted shipping HBM tasks onto ``bus`` directional rows."""
    deferred = False
    for seq, task in tasks.items():
        rem = task.remaining.get(base, 0.0)
        if rem <= 0.0:
            continue
        if not replacing and (bus.row_r in task.remaining or bus.row_w in task.remaining):
            continue
        if not include_native and seq in native_slots:
            deferred = True
            continue
        dem = task.demand.get(base, rem)
        t_r, t_w = _hbm_bus_alias_terms(bus, base, task.event.name, task.demand, rem)
        d_r, d_w = _hbm_bus_alias_terms(bus, base, task.event.name, task.demand, dem)
        new_demand = {k: v for k, v in task.demand.items() if k not in (bus.row_r, bus.row_w)}
        if d_r > 0.0:
            new_demand[bus.row_r] = d_r
        if d_w > 0.0:
            new_demand[bus.row_w] = d_w
        task.demand = new_demand
        if t_r > 0.0:
            task.remaining[bus.row_r] = t_r
        else:
            task.remaining.pop(bus.row_r, None)
        if t_w > 0.0:
            task.remaining[bus.row_w] = t_w
        else:
            task.remaining.pop(bus.row_w, None)
        if base in task.caps:
            cap_val = float(task.caps[base])
            new_caps = {k: v for k, v in task.caps.items() if k not in (bus.row_r, bus.row_w)}
            if cap_val > 0.0 and math.isfinite(cap_val):
                if t_r > 0.0:
                    new_caps[bus.row_r] = t_r * cap_val / rem
                if t_w > 0.0:
                    new_caps[bus.row_w] = t_w * cap_val / rem
            task.caps = new_caps
        task.prepared = None
    return deferred


def _refresh_prepared_for_hbm_bus(engine, prepared: _PreparedTask, name: str) -> _PreparedTask:
    """Re-prepare cached shipping tasks whenever HBM directional bindings changed."""
    aliases = engine._hbm_bus_aliases
    if not aliases or prepared.hbm_bus_epoch == engine._hbm_bus_epoch or not prepared.demand:
        return prepared
    if not any(base in prepared.demand for base in aliases):
        return prepared
    drop = {
        row for base, bus in aliases.items()
        if base in prepared.demand for row in (bus.row_r, bus.row_w)
    }
    raw_demand = {k: v for k, v in prepared.demand.items() if k not in drop}
    raw_caps = (
        {k: v for k, v in prepared.caps.items() if k not in drop}
        if prepared.caps else None
    )
    return engine._prepare(
        name, raw_demand,
        latency=prepared.latency,
        caps=raw_caps,
        acquire=prepared.acquire,
        free=prepared.free,
        placement=prepared.placement or None,
        coupled=prepared.coupled,
        priority=prepared.priority,
        cohort_size=prepared.cohort_size,
        cohort_widened=prepared.cohort_widened or None,
        cohort_members=dict(prepared.cohort_members) if prepared.cohort_members else None,
        latency_from_first_service=prepared.latency_from_first_service,
    )


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
            prepared = self._prepare(
                name, work, latency=latency, caps=caps, acquire=acquire, free=free,
                placement=placement, coupled=coupled, priority=priority,
                cohort_size=cohort_size, cohort_widened=cohort_widened,
                cohort_members=cohort_members,
                latency_from_first_service=latency_from_first_service,
            )
        return self._submit_prepared(
            prepared, name, after=after, release=release, owner=owner,
            metadata=metadata, on_complete=on_complete, on_start=on_start,
            memory=memory, memory_pools=memory_pools,
            memory_bindings=memory_bindings, native_hook=native_hook,
            deliveries=deliveries,
        )

    def _prepare(
        self, name, work=None, *, latency=0.0, caps=None, acquire=(), free=(),
        placement=None, coupled=False, priority=0, cohort_size=1,
        cohort_widened=None, cohort_members=None,
        latency_from_first_service=False,
    ) -> _PreparedTask:
        """Validate a template against this engine's physical resource binding."""
        if (work is None and latency == 0 and caps is None and acquire == ()
                and free == () and placement is None and not coupled
                and cohort_size == 1 and not cohort_widened and not cohort_members
                and not latency_from_first_service):
            return _EMPTY_TASK
        if work and self._hbm_bus_aliases:
            work, caps = _expand_hbm_bus_work(self._hbm_bus_aliases, name, work, caps)
        if (work is not None and work.__class__ is dict and not caps
                and not acquire and not free and placement is None
                and not coupled and latency == 0.0 and not priority
                and cohort_size == 1 and not cohort_widened and not cohort_members
                and not latency_from_first_service):
            # Ordinary finite work on declared resources: the same checks as
            # the general path, without building the four unused structures.
            resources = self.resources
            demands = {}
            s_rates = {}
            u_res_info = {}
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
                hbm_bus_epoch=self._hbm_bus_epoch)
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
        placement = {key: tuple(units) for key, units in (placement or {}).items()}
        for key, units in placement.items():
            if key not in self.resources:
                raise ExecutionError(f"resource_missing:{key}")
            if not units or len(set(units)) != len(units) or any(
                unit < 0 or unit >= self.resources[key].units for unit in units
            ):
                raise ExecutionError(f"resource_unit_out_of_range:{key}")
        placement = {key: tuple(sorted(units)) for key, units in placement.items()}
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
        return _PreparedTask(
            demands, latency, limits, acq_t, free_t,
            placement, coupled, demands_t,
            priority, cohort_size, (), (), acq_pools, 0, fits_empty,
            c_demand, c_cap_limit, c_solo_rates, demand_keys,
            s_rates, c_res_info, u_res_info,
            _compute_pool_unit_masks(acq_t), norm_widened, norm_members,
            bool(latency_from_first_service), self._hbm_bus_epoch)

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
            static.latency_from_first_service, static.hbm_bus_epoch)

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
            cohort_size = getattr(lookup_stage, "cohort_size", 1)
            cohort_widened = getattr(lookup_stage, "cohort_widened", ())
            raw_members = getattr(lookup_stage, "cohort_members", ())
            members_sig = (
                tuple(sorted(raw_members.items()))
                if isinstance(raw_members, Mapping) else tuple(raw_members)
            )
            stg_acq = lookup_stage.acquire
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
            stg_free = lookup_stage.free
            free_sig = (tuple((a.pool, a.size, a.units) for a in stg_free)
                        if stg_free else ())
            lffs = bool(getattr(lookup_stage, "latency_from_first_service", False))
            proto_cache = self._stage_proto_prepared
            proto_key = (id(lookup_stage.work), id(lookup_stage.caps),
                         id(lookup_stage.placement),
                         lookup_stage.name, lookup_stage.latency,
                         lookup_stage.coupled, lookup_stage.priority,
                         cohort_size, cohort_widened, members_sig,
                         acq_sig, free_sig, lffs)
            phit = proto_cache.get(proto_key)
            if (phit is not None and phit[0] is lookup_stage.work
                    and phit[1] is lookup_stage.caps
                    and phit[2] is lookup_stage.placement):
                static_base = phit[3]
            else:
                key = (lookup_stage.name, tuple(lookup_stage.work.items()),
                       lookup_stage.latency, tuple(lookup_stage.caps.items()),
                       tuple((name, units)
                             for name, units in lookup_stage.placement.items()),
                       lookup_stage.coupled, lookup_stage.priority, cohort_size,
                       cohort_widened, members_sig, acq_sig, free_sig, lffs)
                cache = self._stage_prepared
                static_base = cache.get(key)
                if static_base is None:
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
                        latency_from_first_service=lffs)
                    object.__setattr__(static_base, "static_id", id(static_base))
                    if len(cache) >= 16384:
                        cache.clear()
                    cache[key] = static_base
                if len(proto_cache) >= 16384:
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
                    static_base.latency_from_first_service)
            if len(id_cache) >= 16384:
                id_cache.clear()
            id_cache[id(lookup_stage)] = (weakref.ref(lookup_stage), static)
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
                static.latency_from_first_service)
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
                static.latency_from_first_service)
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
                static.latency_from_first_service)
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
            hbm_bus_epoch=static.hbm_bus_epoch)

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
