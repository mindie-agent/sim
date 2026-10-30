"""Finite credit-controlled transfers on the common execution engine.

[Role] Separate serialized transmission, response delay and queue capacity.
[Invariants] A transaction holds a slot through its post-service response.
  One invocation serializes its own transmissions; other invocations contend
  through Engine. Paths are independent, with no inferred store-forward hops.
  Each path conserves rounded wire bytes, including its final transaction.
[Timeouts & Locks] Virtual events only; O(outstanding) tasks per active path.
[Side-Effect Policy] Submits work only to the caller's Engine.
"""
from dataclasses import dataclass
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class FiniteTransfer:
    work: Mapping[str, float]
    transaction_bytes: int = 1
    outstanding: int | None = None
    response_ms: float = 0.0

    def __post_init__(self):
        if (not isinstance(self.transaction_bytes, int) or self.transaction_bytes < 1
                or (self.outstanding is not None
                    and (not isinstance(self.outstanding, int) or self.outstanding < 1))):
            raise ModelingError("transfer_requires_positive_integer_granule_and_slots")
        if not math.isfinite(self.response_ms) or self.response_ms < 0:
            raise ModelingError("transfer_response_delay_invalid")
        if any(not math.isfinite(value) or value < 0 for value in self.work.values()):
            raise ModelingError("transfer_work_invalid")

    def start(self, engine, name, ready, *, owner=""):
        paths = []
        for resource, amount in self.work.items():
            if not amount:
                continue
            if engine.resources[resource].unit != "bytes":
                raise ModelingError(f"transfer_requires_byte_resource:{resource}")
            count = math.ceil(amount / self.transaction_bytes)
            paths.append(self._path(engine, f"{name}:{resource}", ready,
                                    resource, count, owner))
        return engine.submit(name + ":complete", after=tuple(paths) or (ready,), owner=owner)

    def _path(self, engine, name, ready, resource, count, owner):
        # With no credit stalls, a continuous service has the same last-byte
        # completion as the serialized transactions, without expanding packets.
        if self.outstanding is None or (
                not self.response_ms and engine.resources[resource].policy == "fair"):
            service = engine.submit(
                name + ":service", {resource: count * self.transaction_bytes},
                after=(ready,), owner=owner)
            return engine.submit(name + ":response", after=(service,),
                                 latency=self.response_ms, owner=owner)

        complete = engine.future(name + ":drained")
        service_tail = ready
        issued, retired = 0, 0

        def retire(_):
            nonlocal retired
            retired += 1
            if retired == count:
                engine.resolve(complete)
            elif issued < count:
                issue()

        def issue():
            nonlocal issued, service_tail
            index = issued
            issued += 1
            service_tail = engine.submit(
                f"{name}:transaction{index}:service", {resource: self.transaction_bytes},
                after=(service_tail,), owner=owner)
            engine.submit(
                f"{name}:transaction{index}:response", after=(service_tail,),
                latency=self.response_ms, on_complete=retire, owner=owner)

        def begin(_):
            for _ in range(min(count, self.outstanding)):
                issue()

        engine.submit(name + ":begin", after=(ready,), on_complete=begin, owner=owner)
        return complete
