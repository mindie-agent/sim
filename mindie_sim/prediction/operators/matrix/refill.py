"""Independent finite refill and consumption windows on the common engine.

[Role] Prefetch panels while a continuous consumer stream uses their contents.
[Invariants] Refill slots retire only after all panel consumers complete.
  Consumer slots and stage streams span panels; no panel barrier flushes them.
  Buffers and all resource service belong to the caller's existing Engine.
[Timeouts & Locks] O(refill_slots + consumer_slots) active state; no I/O.
[Side-Effect Policy] Submits tasks and releases allocations on that engine only.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RefillPipeline:
    count: int
    refill_slots: int
    consumer_slots: int
    panel: Callable
    group_size: int | None = None

    def start(self, engine, name, ready, *, owner=""):
        from mindie_sim.prediction.program import scope_allocations

        complete = engine.future(name + ":complete")
        scoped = lambda items: scope_allocations(name, items)
        panels, tails, panel_events = {}, {}, {}
        live_slots, slot_released = {}, {}
        issued, retired, next_panel, active, max_lag = 0, 0, 0, 0, 1
        load_tail = ready

        def finish():
            if retired == self.count and not active:
                tails.clear()
                panel_events.clear()
                slot_released.clear()
                engine.resolve(complete)

        def prune_history(current_index):
            cutoff = current_index - max(self.refill_slots, self.consumer_slots, max_lag)
            for old_idx in tuple(panel_events):
                if old_idx < cutoff and old_idx not in panels:
                    del panel_events[old_idx]

        def prefetch():
            nonlocal issued, load_tail
            while (issued < self.count
                   and (issued % self.refill_slots) not in live_slots
                   and (self.group_size is None
                        or issued < (retired // self.group_size + 1) * self.group_size)):
                index = issued
                slot_id = index % self.refill_slots
                issued += 1
                load, steps, release = self.panel(index)
                load_deps = [load_tail]
                prior_release = slot_released.get(slot_id)
                if prior_release is not None:
                    load_deps.append(prior_release)
                loaded = load.submit(engine, f"{name}:panel{index}:load",
                                     tuple(dict.fromkeys(load_deps)), scoped, owner)
                load_tail = loaded
                live_slots[slot_id] = index
                panels[index] = [loaded, iter(steps), release, 0, []]

        def retire_panel(index, release_event):
            nonlocal retired
            del panels[index]
            slot_id = index % self.refill_slots
            live_slots.pop(slot_id, None)
            slot_released[slot_id] = release_event
            retired += 1
            prefetch()
            consume()
            finish()

        def retire_step(event, index):
            nonlocal active
            active -= 1
            state = panels.get(index)
            if state is not None and event in state[4]:
                state[4].remove(event)
            consume()
            finish()

        def consume():
            nonlocal next_panel, active, max_lag
            while active < self.consumer_slots and next_panel in panels:
                index = next_panel
                loaded, steps, release, part, pending_parts = panels[index]
                # The producer supplies a final-item flag; only the live panel
                # iterator and current consumer window are retained.
                item = next(steps, None)
                if item is None:
                    next_panel += 1
                    prune_history(index)
                    engine.submit(
                        f"{name}:panel{index}:release",
                        after=tuple(pending_parts) or (loaded,),
                        free=scoped(release) if release else (), owner=owner,
                        on_complete=lambda event, index=index:
                        retire_panel(index, event))
                    continue
                stages, last = item
                panels[index][3] += 1
                if last:
                    next_panel += 1
                active += 1
                local, previous = {}, dict(tails)
                panel_map = panel_events.setdefault(index, {})
                for stage in stages:
                    for _, lag in stage.previous_lags:
                        if lag > max_lag:
                            max_lag = lag
                    deps = [loaded]
                    deps.extend(local[key] for key in stage.after)
                    deps.extend(previous[key] for key in stage.previous if key in previous)
                    for dep_name, lag in stage.previous_lags:
                        target = index - lag
                        if target >= 0:
                            prior_panel = panel_events.get(target)
                            if prior_panel is not None and dep_name in prior_panel:
                                deps.append(prior_panel[dep_name])
                    if stage.name in tails:
                        deps.append(tails[stage.name])
                    event = stage.submit(engine, f"{name}:panel{index}:part{part}:{stage.name}",
                                         tuple(dict.fromkeys(deps)), scoped, owner)
                    local[stage.name] = tails[stage.name] = panel_map[stage.name] = event
                part_done = engine.submit(
                    f"{name}:panel{index}:part{part}:retire",
                    after=tuple(local.values()) or (loaded,), owner=owner,
                    on_complete=lambda event, index=index:
                    retire_step(event, index))
                pending_parts.append(part_done)
                if last:
                    prune_history(index)
                    engine.submit(
                        f"{name}:panel{index}:release",
                        after=tuple(pending_parts),
                        free=scoped(release) if release else (), owner=owner,
                        on_complete=lambda event, index=index:
                        retire_panel(index, event))

        def begin(_):
            prefetch()
            consume()
            finish()

        engine.submit(name + ":begin", after=(ready,), on_complete=begin, owner=owner)
        return complete
