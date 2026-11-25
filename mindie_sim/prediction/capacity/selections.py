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
        self, replica: int, stage: int, items: dict[str, Allocation],
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
        for name in items:
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

    def attach_allocation_effects(
        self, effects, *, replica: int = 0, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
    ) -> bool:
        """Apply a selected program's acquire/release list atomically.

        The name, size, unit, pool, lifetime and condition are retained from
        the existing typed effect.  This ledger only decides the byte budget;
        T04 remains the owner of the actual pool mechanics.  On-chip effects
        are staged in the same transaction, so a rejected attach changes no
        effect either.
        """
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not 0 <= stage < p.pp:
            raise ValueError("memory_selected_plan_index_out_of_range")
        items = dict(self.selected_items.get((replica, stage), {}))
        bindings = {}
        staged_onchip = dict(self.onchip_effects)
        for effect in effects:
            name = f"selected:{getattr(effect, 'name', '')}"
            action = getattr(effect, "action", "")
            if action == "release":
                staged_onchip.pop((replica, stage, name), None)
                items = self._release_selected_item(replica, stage, items, name)
                continue
            if action != "acquire":
                return False
            if getattr(effect, "unit", "bytes") != "bytes":
                return False
            if self._effect_is_onchip(effect):
                staged_onchip[(replica, stage, name)] = {
                    "pool": getattr(effect, "pool", ""),
                    "size": float(getattr(effect, "size", 0.0)),
                    "lifetime": getattr(effect, "lifetime", ""),
                    "owner": owner, "storage": storage,
                    "ranges": tuple(ranges) if ranges is not None else (),
                    "generation": generation, "event": event,
                }
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
        if not self._install_selected(replica, stage, items):
            return False
        self.onchip_effects = staged_onchip
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
        self, replica: int, stage: int, items: dict[str, Allocation], name: str,
    ) -> dict[str, Allocation]:
        """Release a name while aliases keep their physical backing live."""
        if name not in items:
            return items
        target = items[name]
        replaced = dict(items)
        aliases = [value.name for value in replaced.values()
                   if value.alias_of == name and value.name != name]
        if aliases:
            # The alias is the remaining reader of the backing allocation.
            self.selected_released.add((replica, stage, name))
            return replaced
        del replaced[name]
        if target.alias_of:
            root = target.alias_of
            root_released = (replica, stage, root) in self.selected_released
            other_aliases = [value for value in replaced.values()
                             if value.alias_of == root]
            if root_released and not other_aliases:
                replaced.pop(root, None)
        return replaced

    def release_selected_plan(
        self, replica: int = 0, stage: int = 0, name: str | None = None,
    ) -> bool:
        key = (replica, stage)
        items = dict(self.selected_items.get(key, {}))
        if not items:
            return False
        if name is None:
            items = {}
            self.selected_released = {
                record for record in self.selected_released
                if not (record[0] == replica and record[1] == stage)
            }
        else:
            if name not in items:
                return False
            items = self._release_selected_item(replica, stage, items, name)
        before = self._replica_occupied(replica)
        if not self._install_selected(replica, stage, items):
            return False
        released = max(0.0, before - self._replica_occupied(replica))
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

    def retain_shared_pages(
        self, replica: int, owner: str, stages, *, version: int = 0,
        physical_key: str | None = None, storage: str | None = None,
        rank: int | None = None, tier: str = "hbm", stage: int = 0,
    ) -> bool:
        """Retain a physical shared-page owner used by another request.

        The first retain charges the page range; subsequent readers only bump
        the reference count.  ``physical_key`` identifies the underlying page
        owner, so two request/string owners pointing at the same physical range
        are charged once and keep independent references.  SERVICE-CORE
        supplies owner, range and version; this ledger never derives them from
        a logical prefix length.
        """
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not owner:
            raise ValueError("memory_shared_pages_index_invalid")
        ranges = tuple(float(value) for value in stages)
        if len(ranges) != p.pp or any(value < 0.0 for value in ranges):
            raise ValueError("memory_shared_pages_range_invalid")
        physical = str(physical_key or owner)
        key = (replica, physical)
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
        holder = (replica, "shared", physical)
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
            before = self._replica_occupied(replica)
            self._set_ref((replica, "shared", key[1]), ())
            del self.shared_pages[key]
            self._sync_usage()
            self.release_events.append({
                "kind": "shared", "replica": replica, "owner": owner,
                "physical_key": key[1], **existing,
                "released_bytes": max(0.0, before - self._replica_occupied(replica)),
            })
            self.budget_version += 1
        return released
