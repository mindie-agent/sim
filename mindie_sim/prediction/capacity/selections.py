"""Selected allocations and shared cache-page holders.

[Role] Bind selected allocation plans and effects and retain shared pages.
[Invariants] The physical backing and its generation remain associated with each
  live holder. Releasing one holder preserves the other owners.
[Timeouts & Locks] Bounded by selected allocations and live shared pages. No locks or I/O.
[Side-Effect Policy] Updates selected and shared holders and records release events.
"""
from __future__ import annotations

from mindie_sim.catalog.cache_pages import Allocation, AllocationPlan

from .contracts import _union_bytes


class _MemorySelectionsMixin:
    def _install_selected(
        self, replica: int, stage: int, items: dict[str, Allocation], *,
        staged_released: set[tuple[int, int, str]] | None = None,
    ) -> bool:
        plan = AllocationPlan(tuple(items.values()))
        # Resolve once to preserve alias-cycle/pool-overcommit validation; the
        # unified physical refs below decide the actual byte union.
        plan.resolve()
        refs = self._selected_refs(replica, stage, items)
        old_names = set(self.selected_items.get((replica, stage), {}))
        drop = {(replica, "selected", stage, name) for name in old_names}
        extra = [
            ((replica, "selected", stage, name), (ref,))
            for name, ref in refs.items()
        ]
        hbm, host = self._state_usage(
            extra_holders=extra, drop_holders=drop)
        if not self._usage_budget_ok(hbm, host):
            return False
        for name in old_names - set(refs):
            self._set_ref((replica, "selected", stage, name), ())
        for name, ref in refs.items():
            self._set_ref((replica, "selected", stage, name), (ref,))
        self.selected_items[(replica, stage)] = dict(items)
        self.selected_plans[(replica, stage)] = plan
        self.selected_physical = {
            key: value for key, value in self.selected_physical.items()
            if not (key[0] == replica and key[1] == stage)
        }
        for name, ref in refs.items():
            self.selected_physical[(replica, stage, ref.physical_key)] = (
                _union_bytes(ref.ranges))
        if staged_released is not None:
            self.selected_released = set(staged_released)
        self._sync_usage()
        self.budget_version += 1
        for name in list(self.selected_released):
            if (name[0] == replica and name[1] == stage
                    and name[2] not in items):
                self.selected_released.discard(name)
        return True

    def attach_allocation_plan(
        self, plan: AllocationPlan, *, replica: int = 0, stage: int = 0,
        replace: bool = False,
    ) -> bool:
        """Attach/replace a selected implementation's alias/pool allocations.

        ``replace=False`` is an incremental acquire: existing live acquisitions
        remain and duplicate names are rejected.  ``replace=True`` treats the
        supplied plan as a complete replacement and commits only its peak
        difference.  All checks precede the single commit.
        """
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not 0 <= stage < p.pp:
            raise ValueError("memory_selected_plan_index_out_of_range")
        incoming = {item.name: item for item in plan.allocations}
        if len(incoming) != len(plan.allocations):
            return False
        if replace:
            items = incoming
        else:
            items = dict(self.selected_items.get((replica, stage), {}))
            for name, item in incoming.items():
                if name in items:
                    return False
                items[name] = item
        if not self._install_selected(replica, stage, items):
            return False
        discard_names = items if replace else incoming
        for name in discard_names:
            self.selected_released.discard((replica, stage, name))
        return True

    @staticmethod
    def _binding_report(binding: dict) -> dict:
        row = dict(binding)
        event = row.get("event")
        if event is not None and not isinstance(event, (str, int, float, bool, type(None))):
            row["event"] = getattr(event, "sequence", repr(event))
        return row

    @staticmethod
    def _effect_is_onchip(effect) -> bool:
        pool = str(getattr(effect, "pool", "")).strip().lower()
        return pool in {"l0a", "l0b", "l0c", "l1", "l2", "ub", "smp", "aic",
                        "aiv", "onchip", "local"}

    def _onchip_budget_ok(
        self, replica: int, stage: int, staged_onchip: dict,
    ) -> bool:
        """Check staged on-chip effects against every declared die & engine pool cap."""
        dies = self._stage_dies(replica, stage)
        stage_entries = [
            rec for (rep, st, _name), rec in staged_onchip.items()
            if rep == replica and st == stage
        ]
        if not stage_entries:
            return True
        engine_pools = (
            getattr(self.runtime_state, "pools", None)
            if self.runtime_state is not None else None
        )
        engine_pool_units = (
            getattr(self.runtime_state, "pool_units", None)
            if self.runtime_state is not None else None
        )
        engine_caps: dict[str, float] = {}
        engine_units: dict[str, int] = {}
        if isinstance(engine_pools, dict):
            for k, val in engine_pools.items():
                norm_k = str(k).strip().lower()
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    engine_caps[norm_k] = float(val)
                elif hasattr(val, "capacity"):
                    engine_caps[norm_k] = float(getattr(val, "capacity"))
                    if hasattr(val, "units"):
                        engine_units[norm_k] = max(1, int(getattr(val, "units") or 1))
        if isinstance(engine_pool_units, dict):
            for k, val in engine_pool_units.items():
                if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                    engine_units[str(k).strip().lower()] = int(val)
        for die_idx, die in enumerate(dies):
            buffers = {
                str(k).strip().lower(): float(v)
                for k, v in dict(getattr(die, "buffers", ()) or ()).items()
            }
            unit_counts = {
                str(k).strip().lower(): int(v)
                for k, v in dict(getattr(die, "buffer_units", ()) or ()).items()
            }
            default_units = max(1, int(getattr(die, "compute_units", 1) or 1))
            usage: dict[tuple[str, int], float] = {}
            for rec in stage_entries:
                rec_rank = rec.get("rank")
                if rec_rank is not None and self._account_rank(rec_rank) != die_idx:
                    continue
                pool_norm = str(rec.get("pool", "")).strip().lower()
                size = float(rec.get("size", 0.0))
                units = tuple(int(u) for u in (rec.get("units") or (0,)))
                hw_cap = buffers.get(pool_norm)
                eng_cap = engine_caps.get(pool_norm)
                caps = [c for c in (hw_cap, eng_cap) if c is not None]
                if not caps:
                    continue
                cap = min(caps)
                unit_bounds = []
                if hw_cap is not None:
                    unit_bounds.append(max(1, int(unit_counts.get(pool_norm, default_units))))
                if eng_cap is not None:
                    unit_bounds.append(max(1, int(engine_units.get(pool_norm, default_units if hw_cap is not None else 1))))
                unit_limit = min(unit_bounds)
                for u_idx in units:
                    if not 0 <= u_idx < unit_limit:
                        return False
                    used = usage.get((pool_norm, u_idx), 0.0) + size
                    if used > cap + max(1e-9, abs(cap) * 1e-12):
                        return False
                    usage[(pool_norm, u_idx)] = used
        return True

    def _record_step_peaks(self, step_peaks: list) -> None:
        p = self.case.runtime.parallel
        for step_hbm, step_host in step_peaks:
            for rep in range(p.dp):
                self.host_dram_peak[rep] = max(self.host_dram_peak[rep], float(step_host[rep]))
                for st in range(p.pp):
                    row = step_hbm[rep][st]
                    for rk, used in enumerate(row):
                        self.peak_rank[rep][st][rk] = max(
                            self.peak_rank[rep][st][rk], float(used),
                        )
                    self.peak_combined[rep][st] = max(
                        self.peak_combined[rep][st], max(row, default=0.0),
                    )

    def attach_allocation_effects(
        self, effects, *, replica: int = 0, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
    ) -> bool:
        """Apply a selected program's acquire/release list atomically.

        Every intermediate acquire step is validated against HBM/host and
        declared on-chip die pool capacities before committing.  Rejected
        transactions mutate no items, on-chip effects, release markers or peaks.
        """
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not 0 <= stage < p.pp:
            raise ValueError("memory_selected_plan_index_out_of_range")
        old_names = set(self.selected_items.get((replica, stage), {}))
        old_drop = {(replica, "selected", stage, name) for name in old_names}
        items = dict(self.selected_items.get((replica, stage), {}))
        bindings = {}
        staged_onchip = dict(self.onchip_effects)
        staged_released = set(self.selected_released)
        step_peaks: list[tuple[list, list]] = []
        for effect in effects:
            name = f"selected:{getattr(effect, 'name', '')}"
            action = getattr(effect, "action", "")
            if action == "release":
                staged_onchip.pop((replica, stage, name), None)
                items = self._release_selected_item(
                    replica, stage, items, name, released=staged_released,
                )
                continue
            if action != "acquire":
                return False
            if getattr(effect, "unit", "bytes") != "bytes":
                return False
            if self._effect_is_onchip(effect):
                if (replica, stage, name) in staged_onchip:
                    return False
                staged_onchip[(replica, stage, name)] = {
                    "pool": getattr(effect, "pool", ""),
                    "size": float(getattr(effect, "size", 0.0)),
                    "units": tuple(int(u) for u in (getattr(effect, "units", (0,)) or (0,))),
                    "rank": getattr(effect, "rank", None),
                    "lifetime": getattr(effect, "lifetime", ""),
                    "owner": owner, "storage": storage,
                    "ranges": tuple(ranges) if ranges is not None else (),
                    "generation": generation, "event": event,
                }
                if not self._onchip_budget_ok(replica, stage, staged_onchip):
                    return False
                continue
            if name in items:
                return False
            items[name] = Allocation(
                name, float(getattr(effect, "size", 0.0)), "workspace",
                note=f"pool={getattr(effect, 'pool', '')};"
                     f"lifetime={getattr(effect, 'lifetime', '')};"
                     f"condition={getattr(effect, 'condition', '')}",
                physical_key=getattr(effect, "physical_key", None),
                rank=getattr(effect, "rank", None),
                storage=storage,
                tier=getattr(effect, "tier", "hbm"),
                generation=generation,
                ranges=ranges,
            )
            bindings[name] = {
                "owner": owner, "storage": storage,
                "ranges": tuple(ranges) if ranges is not None else (),
                "generation": generation, "event": event,
                "pool": getattr(effect, "pool", ""),
                "lifetime": getattr(effect, "lifetime", ""),
                "condition": getattr(effect, "condition", ""),
            }
            AllocationPlan(tuple(items.values())).resolve()
            step_refs = self._selected_refs(replica, stage, items)
            step_extra = [
                ((replica, "selected", stage, ref_name), (ref,))
                for ref_name, ref in step_refs.items()
            ]
            step_hbm, step_host = self._state_usage(
                extra_holders=step_extra, drop_holders=old_drop,
            )
            if not self._usage_budget_ok(step_hbm, step_host):
                return False
            step_peaks.append((step_hbm, step_host))
        if not self._install_selected(
            replica, stage, items, staged_released=staged_released,
        ):
            return False
        self.onchip_effects = staged_onchip
        self._record_step_peaks(step_peaks)
        for name, binding in bindings.items():
            if name in items:
                self.effect_bindings[(replica, stage, name)] = binding
                self.selected_released.discard((replica, stage, name))
        for binding_key in list(self.effect_bindings):
            if (binding_key[0] == replica and binding_key[1] == stage
                    and binding_key[2] not in items):
                self.effect_bindings.pop(binding_key, None)
        return True

    def _release_selected_item(
        self, replica: int, stage: int, items: dict[str, Allocation], name: str, *,
        released: set[tuple[int, int, str]] | None = None,
    ) -> dict[str, Allocation]:
        """Release a name while aliases keep their physical backing live."""
        if name not in items:
            return items
        released_set = self.selected_released if released is None else released
        target = items[name]
        replaced = dict(items)
        aliases = [value.name for value in replaced.values()
                   if value.live and value.alias_of == name and value.name != name]
        if aliases:
            # The alias is the remaining reader of the backing allocation.
            released_set.add((replica, stage, name))
            return replaced
        del replaced[name]
        if target.alias_of:
            root = target.alias_of
            root_released = (replica, stage, root) in released_set
            other_aliases = [value for value in replaced.values()
                             if value.live and value.alias_of == root]
            if root_released and not other_aliases:
                replaced.pop(root, None)
                released_set.discard((replica, stage, root))
        return replaced

    def release_selected_plan(
        self, replica: int = 0, stage: int = 0, name: str | None = None,
    ) -> bool:
        key = (replica, stage)
        items = dict(self.selected_items.get(key, {}))
        if not items:
            return False
        staged_released = set(self.selected_released)
        if name is None:
            items = {}
            staged_released = {
                record for record in staged_released
                if not (record[0] == replica and record[1] == stage)
            }
        else:
            if name not in items:
                return False
            items = self._release_selected_item(
                replica, stage, items, name, released=staged_released,
            )
        before_hbm, before_host = self._state_usage()
        if not self._install_selected(
            replica, stage, items, staged_released=staged_released,
        ):
            return False
        after_hbm, after_host = self._state_usage()
        released = self._freed_replica_hbm(replica, before_hbm, after_hbm) + max(
            0.0, float(before_host[replica]) - float(after_host[replica]),
        )
        if name is not None:
            self.release_events.append({
                "kind": "selected", "replica": replica, "stage": stage,
                "name": name, "released_bytes": released,
            })
        for binding_key in list(self.effect_bindings):
            if (binding_key[0] == replica and binding_key[1] == stage
                    and binding_key[2] not in items):
                self.effect_bindings.pop(binding_key, None)
        return True

    def release_selected_allocation(
        self, name: str, *, replica: int = 0, stage: int = 0,
    ) -> bool:
        return self.release_selected_plan(replica, stage, name=name)

    def _shared_page_key(
        self, replica: int, rank: int | None, physical: str,
        storage: str | None, tier: str,
    ) -> tuple:
        account_rank = self._account_rank(rank)
        storage_norm = str(storage or "")
        tier_norm = str(tier or "hbm").lower()
        full_key = (replica, physical, account_rank, storage_norm, tier_norm)
        if full_key in self.shared_pages:
            return full_key
        base_key = (replica, physical)
        if account_rank == 0:
            existing = self.shared_pages.get(base_key)
            if existing is None or (
                self._account_rank(existing.get("rank")) == 0
                and str(existing.get("storage") or "") == storage_norm
                and str(existing.get("tier") or "hbm").lower() == tier_norm
            ):
                return base_key
        return full_key

    def retain_shared_pages(
        self, replica: int, owner: str, stages, *, version: int = 0,
        physical_key: str | None = None, storage: str | None = None,
        rank: int | None = None, tier: str = "hbm", stage: int = 0,
    ) -> bool:
        """Retain a physical shared-page owner used by another request.

        The first retain charges the page range on its physical rank/storage/tier;
        subsequent readers on that exact physical backing only bump the reference
        count, while distinct physical ranks retain independent allocations.
        """
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not owner:
            raise ValueError("memory_shared_pages_index_invalid")
        ranges = tuple(float(value) for value in stages)
        if len(ranges) != p.pp or any(value < 0.0 for value in ranges):
            raise ValueError("memory_shared_pages_range_invalid")
        physical = str(physical_key or owner)
        key = self._shared_page_key(replica, rank, physical, storage, tier)
        existing = self.shared_pages.get(key)
        if existing is not None:
            if ranges != existing["stages"]:
                return False
            existing["owners"][str(owner)] = existing["owners"].get(str(owner), 0) + 1
            existing["refs"] += 1
            existing["version"] = max(existing["version"], int(version))
            self.budget_version += 1
            return True
        record = {
            "stages": ranges, "version": int(version), "refs": 1,
            "owners": {str(owner): 1}, "storage": storage,
            "rank": rank, "tier": tier, "charged": False,
        }
        holder = (replica, "shared", key)
        refs = self._shared_ref(replica, physical, record)
        hbm, host = self._state_usage(extra_holders=[(holder, refs)])
        if not self._usage_budget_ok(hbm, host):
            return False
        self.shared_pages[key] = record
        self._set_ref(holder, refs)
        self._sync_usage()
        self.budget_version += 1
        return True

    def release_shared_pages(self, replica: int, owner: str) -> bool:
        released = False
        for key, existing in list(self.shared_pages.items()):
            if key[0] != replica or str(owner) not in existing.get("owners", {}):
                continue
            released = True
            existing["owners"][str(owner)] -= 1
            if existing["owners"][str(owner)] <= 0:
                del existing["owners"][str(owner)]
            existing["refs"] -= 1
            if existing["refs"] > 0:
                self.budget_version += 1
                continue
            before_hbm, before_host = self._state_usage()
            self._set_ref((replica, "shared", key), ())
            del self.shared_pages[key]
            after_hbm, after_host = self._state_usage()
            freed = self._freed_replica_hbm(replica, before_hbm, after_hbm) + max(
                0.0, float(before_host[replica]) - float(after_host[replica]),
            )
            self._sync_usage()
            self.release_events.append({
                "kind": "shared", "replica": replica, "owner": owner,
                "physical_key": key[1], **existing,
                "released_bytes": freed,
            })
            self.budget_version += 1
        return released
