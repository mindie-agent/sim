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
        from .program import scope_allocations

        complete = engine.future(name + ":complete")
        scoped = lambda items: scope_allocations(name, items)
        panels, tails = {}, {}
        issued, retired, next_panel, active = 0, 0, 0, 0
        load_tail = ready

        def finish():
            if retired == self.count and not active:
                tails.clear()
                engine.resolve(complete)

        def prefetch():
            nonlocal issued, load_tail
            while (issued < self.count and len(panels) < self.refill_slots
                   and (self.group_size is None
                        or issued < (retired // self.group_size + 1) * self.group_size)):
                index = issued
                issued += 1
                load, steps, release = self.panel(index)
                loaded = load.submit(engine, f"{name}:panel{index}:load",
                                     (load_tail,), scoped, owner)
                load_tail = loaded
                panels[index] = [loaded, iter(steps), release, 0]

        def retire_panel(index):
            nonlocal retired
            del panels[index]
            retired += 1
            prefetch()

        def retire_step(_, index, last):
            nonlocal active
            active -= 1
            if last:
                retire_panel(index)
            consume()
            finish()

        def consume():
            nonlocal next_panel, active
            while active < self.consumer_slots and next_panel in panels:
                index = next_panel
                loaded, steps, release, part = panels[index]
                # The producer supplies a final-item flag; only the live panel
                # iterator and current consumer window are retained.
                stages, last = next(steps)
                panels[index][3] += 1
                if last:
                    next_panel += 1
                active += 1
                local, previous = {}, dict(tails)
                for stage in stages:
                    deps = [loaded]
                    deps.extend(local[key] for key in stage.after)
                    deps.extend(previous[key] for key in stage.previous if key in previous)
                    if stage.name in tails:
                        deps.append(tails[stage.name])
                    event = stage.submit(engine, f"{name}:panel{index}:part{part}:{stage.name}",
                                         tuple(deps), scoped, owner)
                    local[stage.name] = tails[stage.name] = event
                engine.submit(
                    f"{name}:panel{index}:part{part}:retire", after=tuple(local.values()),
                    free=scoped(release) if last else (), owner=owner,
                    on_complete=lambda event, index=index, last=last:
                    retire_step(event, index, last))

        def begin(_):
            prefetch()
            consume()
            finish()

        engine.submit(name + ":begin", after=(ready,), on_complete=begin, owner=owner)
        return complete
