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
    issue_quota_resource: str = ""

    def start(self, engine, name, ready, *, owner=""):
        outstanding = int(self.outstanding)
        granule = int(self.granule_bytes)
        rows = int(self.rows)
        cols = int(self.cols)
        offset = int(self.offset_bytes)
        row_stride = int(self.row_stride_bytes)
        elem_bytes = int(self.element_bytes)
        if outstanding < 1 or granule < 1:
            raise ValueError("transaction_queue_requires_positive_capacity_and_granule")
        count = rows * cols
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
            row, col = divmod(index, cols)
            address = offset + row * row_stride + col * elem_bytes
            keys = tuple(range(address // granule,
                               (address + elem_bytes - 1) // granule + 1))
            issue_work = {self.issue_resource: self.participants}
            if self.issue_quota_resource:
                issue_work[self.issue_quota_resource] = self.participants
            submitted = engine.submit(
                f"{name}:request{index}:issue", issue_work,
                after=(issue_tail,), caps={self.issue_resource: self.issue_rate * self.participants},
                placement={self.issue_resource: self.issue_placement} if self.issue_placement else None,
                coupled=bool(self.issue_quota_resource),
                owner=owner,
            )
            issue_tail = submitted
            if col == cols - 1 and self.row_control_ms:
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
                f"{name}:store:request{index}:response",
                {self.memory_resource: elem_bytes * self.participants},
                after=dependencies, latency=self.response_ms, on_complete=retire, owner=owner,
            )
            for key in keys:
                last_by_granule[key] = response

        for _ in range(min(outstanding, count)):
            issue()
        return complete
