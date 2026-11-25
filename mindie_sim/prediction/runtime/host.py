"""Product host-submit FIFO cursors and captured-replay boundaries.

[Role] Track one physical host FIFO's ordered launch cursor independently of
  device completion.
[Invariants] Host progress is bounded only by actual host dependencies,
  submit work and declared queue capacity. Device completion never gates host
  progress, and host lookahead never changes timing.
[Timeouts & Locks] No waits; callers submit real host tasks through the Engine.
[Side-Effect Policy] Mutates only the cursor record it owns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HostCursor:
    fifo_role: str
    occurrence_uid: str
    queue_capacity: int | None = None
    captured: bool = False
    submitted: int = 0
    host_completed: int = 0
    active_launches: int = 0
    host_dependency_ports: tuple[str, ...] = ()
    retired: bool = False
    tail_event: Any = None

    def can_submit(self) -> bool:
        if self.retired:
            return False
        return self.queue_capacity is None or self.active_launches < self.queue_capacity

    def note_submit(self) -> int:
        if not self.can_submit():
            raise ValueError("host_fifo_queue_full")
        self.submitted += 1
        self.active_launches += 1
        return self.submitted - 1

    def note_host_done(self, sequence: int, event: Any = None) -> int:
        """Advance the host cursor at host completion, not device completion."""
        self.host_completed = max(self.host_completed, int(sequence) + 1)
        self.active_launches = max(0, self.active_launches - 1)
        if self.active_launches == 0 and (event is None or self.tail_event is event):
            self.tail_event = None
        return self.host_completed

    def retire(self) -> None:
        self.retired = True


def cursor_for(state, fifo_role: str, occurrence_uid: str, spec) -> HostCursor:
    key = str(fifo_role)
    cursor = state.host_fifos.get(key)
    if cursor is None:
        cursor = HostCursor(
            fifo_role=key, occurrence_uid=occurrence_uid,
            queue_capacity=(None if spec is None
                            else getattr(spec, "queue_capacity", None)),
            captured=bool(getattr(spec, "captured", False)),
            host_dependency_ports=tuple(
                getattr(spec, "host_dependency_ports", ()) or ()),
        )
        state.host_fifos[key] = cursor
    return cursor
