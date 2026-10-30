"""Finite address-dependent transfer queues on the shared engine.

[Role] Issue scalar transfers with bounded outstanding requests and address hazards.
[Invariants] Requests to an occupied address granule wait for its response.
  Queue slots release on response, not issue. Payload, issue service and response
  latency are distinct. Identical independent units may share one representative
  trace, with their physical issue work and memory traffic conserved.
[Timeouts & Locks] Virtual events only; O(outstanding requests) live address state.
[Side-Effect Policy] Submits tasks only to the caller's engine.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class QueuedStore:
    rows: int
    cols: int
    row_stride_bytes: int
    element_bytes: int
    offset_bytes: int
    participants: int
    issue_resource: str
    issue_rate: float
    memory_resource: str
    granule_bytes: int
    outstanding: int
    response_ms: float
    row_control_ms: float = 0.0
    issue_placement: tuple[int, ...] = ()

    def start(self, engine, name, ready, *, owner=""):
        if self.outstanding < 1 or self.granule_bytes < 1:
            raise ValueError("transaction_queue_requires_positive_capacity_and_granule")
        count = self.rows * self.cols
        if not count:
            return engine.submit(name + ":empty", after=(ready,), owner=owner)
        complete = engine.future(name + ":queue_complete")
        last_by_granule = {}
        issue_tail = ready
        issued, pending = 0, 0

        def issue():
            nonlocal issued, pending, issue_tail
            index = issued
            issued += 1
            pending += 1
            row, col = divmod(index, self.cols)
            address = self.offset_bytes + row * self.row_stride_bytes + col * self.element_bytes
            keys = tuple(range(address // self.granule_bytes,
                               (address + self.element_bytes - 1) // self.granule_bytes + 1))
            submitted = engine.submit(
                f"{name}:request{index}:issue", {self.issue_resource: self.participants},
                after=(issue_tail,), caps={self.issue_resource: self.issue_rate * self.participants},
                placement={self.issue_resource: self.issue_placement} if self.issue_placement else None,
                owner=owner,
            )
            issue_tail = submitted
            if col == self.cols - 1 and self.row_control_ms:
                issue_tail = engine.submit(f"{name}:row{row}:control", after=(submitted,),
                                           latency=self.row_control_ms, owner=owner)
            dependencies = (submitted,) + tuple(last_by_granule[key] for key in keys
                                               if key in last_by_granule)

            def retire(event):
                nonlocal pending
                pending -= 1
                for key in keys:
                    if last_by_granule.get(key) is event:
                        del last_by_granule[key]
                if issued < count:
                    issue()
                elif not pending:
                    engine.submit(name + ":drain", after=(issue_tail,),
                                  on_complete=lambda _: engine.resolve(complete), owner=owner)

            response = engine.submit(
                f"{name}:request{index}:response",
                {self.memory_resource: self.element_bytes * self.participants},
                after=dependencies, latency=self.response_ms, on_complete=retire, owner=owner,
            )
            for key in keys:
                last_by_granule[key] = response

        for _ in range(min(self.outstanding, count)):
            issue()
        return complete
