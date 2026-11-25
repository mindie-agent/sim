"""Cache path physical work and executable programs.

[Role] Translate resolved cache work into copy, conversion, fill and write stages.
[Invariants] Transfers and staging allocations retain their own dependencies and
  lifetimes.
[Timeouts & Locks] Finite phase graphs and transfer windows on the caller's Engine.
[Side-Effect Policy] Submits cache work and transient allocations to that Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from mindie_sim.schema.common import ModelingError

from ..engine import Allocation, Engine, Event
from ..program import KernelProgram, Stage
from ..collectives.transfer import FiniteTransfer
from .contracts import A5_LINE_BYTES, CacheAccessResult


@dataclass
class AccessSummary:
    """Aggregated access work for compute and capacity consumers."""

    accesses: int = 0
    logical_bytes: int = 0
    gm_bytes: int = 0
    onchip_bytes: int = 0
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    direct_write_bytes: int = 0
    cache_read_bytes: int = 0
    staging_peak_bytes: int = 0
    resident_peak_bytes: int = 0

    @property
    def hbm_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    def add(self, result: CacheAccessResult) -> "AccessSummary":
        self.accesses += 1
        self.logical_bytes += result.logical_bytes
        self.gm_bytes += result.gm_bytes
        self.onchip_bytes += result.onchip_bytes
        self.hbm_read_bytes += result.hbm_read_bytes
        self.hbm_write_bytes += result.hbm_write_bytes
        self.direct_write_bytes += result.direct_write_bytes
        self.cache_read_bytes += result.cache_read_bytes
        self.staging_peak_bytes = max(self.staging_peak_bytes, result.staging_bytes)
        self.resident_peak_bytes = max(self.resident_peak_bytes, result.resident_bytes)
        return self

    def to_dict(self) -> dict:
        return {
            "accesses": self.accesses, "logical_bytes": self.logical_bytes,
            "gm_bytes": self.gm_bytes, "onchip_bytes": self.onchip_bytes,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes, "hbm_bytes": self.hbm_bytes,
            "direct_write_bytes": self.direct_write_bytes,
            "cache_read_bytes": self.cache_read_bytes,
            "staging_peak_bytes": self.staging_peak_bytes,
            "resident_peak_bytes": self.resident_peak_bytes,
        }


@dataclass(frozen=True)
class CacheStage:
    """One shared physical stage of the resolved cache data path."""

    label: str
    resource: str | None = None
    amount: float = 0.0
    after: tuple[str, ...] = ()
    publish: bool = False
    terminal: bool = False


def _result_line_bytes(result: CacheAccessResult) -> int:
    if result.hbm_read_lines:
        return result.hbm_read_bytes // result.hbm_read_lines
    if result.hbm_write_lines:
        return result.hbm_write_bytes // result.hbm_write_lines
    return result.line_bytes


def _round_hbm_bytes(amount: int | float, transaction_bytes: int) -> float:
    """Round an external HBM transfer up to the next-level transaction granule."""
    if not amount:
        return 0.0
    if type(amount) is int:
        count = (amount + transaction_bytes - 1) // transaction_bytes
    else:
        count = math.ceil(amount / transaction_bytes)
    return float(count * transaction_bytes)


def cache_work_stages(result: CacheAccessResult, *,
                      hbm_resource: str = "hbm",
                      onchip_resource: str = "mte2",
                      write_resource: str | None = None,
                      vector_resource: str = "vector",
                      source_bytes: int = 0,
                      source_residency: str = "on_chip",
                      convert_ops: float = 0.0,
                      transaction_bytes: int | None = None) -> tuple[CacheStage, ...]:
    """One explicit immutable phase DAG consumed by both adapters.

    Every stage names its real predecessors; there is no implicit ``ready``
    sibling. The chain is source -> conversion -> request read/RMW -> physical
    drain (when a dirty slot is reused) -> publish/stage -> direct writeback ->
    retire. Only absent phases are collapsed.
    """
    if transaction_bytes is None:
        transaction_bytes = _result_line_bytes(result)
    if type(transaction_bytes) is not int or transaction_bytes < 1:
        raise ModelingError("cache_work_stages_transaction_bytes_invalid")
    stages: list[CacheStage] = []
    labels: set[str] = set()

    def add(label, resource=None, amount=0.0, *, after, publish=False,
            terminal=False):
        predecessors = tuple(after)
        if not predecessors:
            raise ModelingError(f"cache_stage_requires_predecessor:{label}")
        if label in labels:
            raise ModelingError(f"cache_stage_duplicate:{label}")
        for predecessor in predecessors:
            if predecessor != "ready" and predecessor not in labels:
                raise ModelingError(
                    f"cache_stage_predecessor_missing:{label}:{predecessor}")
        labels.add(label)
        stages.append(CacheStage(
            label, resource, float(amount) if amount else 0.0,
            predecessors, publish, terminal))
        return label

    basis = ("ready",)
    if source_bytes:
        if source_residency != "on_chip":
            label = add(
                "source_hbm", hbm_resource,
                _round_hbm_bytes(source_bytes, transaction_bytes), after=basis)
            basis = (label,)
        label = add("source_onchip", onchip_resource, source_bytes, after=basis)
        basis = (label,)
    if convert_ops:
        label = add("convert", vector_resource, convert_ops, after=basis)
        basis = (label,)
    drain_bytes = result.writeback_bytes
    direct_bytes = result.direct_write_bytes
    if result.mode == "read":
        if drain_bytes:
            # A miss may replace a dirty slot. The old bytes leave HBM before
            # the fill overwrites that physical slot.
            label = add(
                "drain", hbm_resource,
                _round_hbm_bytes(drain_bytes, transaction_bytes), after=basis)
            basis = (label,)
        if result.hbm_read_bytes:
            fill = add(
                "fill", hbm_resource,
                _round_hbm_bytes(result.hbm_read_bytes, transaction_bytes),
                after=basis)
            basis = (add("publish", after=(fill,), publish=True),)
        deliver = add("deliver", onchip_resource, result.onchip_bytes,
                      after=basis)
        add("retire", after=(deliver,), terminal=True)
    else:
        if result.hbm_read_bytes:
            label = add(
                "rmw", hbm_resource,
                _round_hbm_bytes(result.hbm_read_bytes, transaction_bytes),
                after=basis)
            basis = (label,)
        if drain_bytes:
            label = add(
                "drain", hbm_resource,
                _round_hbm_bytes(drain_bytes, transaction_bytes), after=basis)
            basis = (label,)
        stage = add("stage", write_resource or onchip_resource,
                    result.onchip_bytes, after=basis, publish=True)
        terminal = (stage,)
        if direct_bytes:
            label = add(
                "writeback", hbm_resource,
                _round_hbm_bytes(direct_bytes, transaction_bytes),
                after=terminal)
            terminal = (label,)
        add("retire", after=terminal, terminal=True)

    # The graph is closed, acyclic in declaration order, has exactly one
    # terminal runner and every stage is reachable from the adapter's ready.
    reachable = {"ready"}
    for stage in stages:
        if any(predecessor not in reachable for predecessor in stage.after):
            raise ModelingError(f"cache_stage_graph_open:{stage.label}")
        reachable.add(stage.label)
    terminal_stages = [stage for stage in stages if stage.terminal]
    if len(terminal_stages) != 1:
        raise ModelingError(
            "cache_stage_graph_terminal_count:" + str(len(terminal_stages)))
    terminal_label = terminal_stages[0].label
    successors = {stage.label: 0 for stage in stages}
    for stage in stages:
        for predecessor in stage.after:
            if predecessor in successors:
                successors[predecessor] += 1
    if any(count == 0 and label != terminal_label
           for label, count in successors.items()):
        raise ModelingError("cache_stage_graph_orphan")
    return tuple(stages)


@dataclass(frozen=True)
class CacheTransfer:
    """Finite byte transfer with the same credit/response parameters for both
    cache adapters. This is a transport declaration, not a T08 grant."""

    work: tuple[tuple[str, float], ...]
    transaction_bytes: int = A5_LINE_BYTES
    outstanding: int | None = None
    response_ms: float = 0.0

    def __post_init__(self):
        if type(self.transaction_bytes) is not int or self.transaction_bytes < 1:
            raise ModelingError("cache_transfer_transaction_bytes_invalid")
        if self.outstanding is not None and (
                type(self.outstanding) is not int or self.outstanding < 1):
            raise ModelingError("cache_transfer_outstanding_invalid")
        if (not isinstance(self.response_ms, (int, float))
                or isinstance(self.response_ms, bool) or self.response_ms < 0):
            raise ModelingError("cache_transfer_response_invalid")

    def start(self, engine: Engine, name: str, ready: Event, *,
              owner: str = "") -> Event:
        return FiniteTransfer(
            dict(self.work), transaction_bytes=self.transaction_bytes,
            outstanding=self.outstanding, response_ms=self.response_ms,
        ).start(engine, name, ready, owner=owner)


@dataclass(frozen=True)
class CacheAccessProgram:
    """Executable access path for callers that own a Stage/Program boundary.

    The cache resolves state when the access is built; this program submits the
    resolved physical path. It is the runnable form that includes ready delay,
    source read/conversion, staging allocation and all HBM work (read fill plus
    any eviction writeback), which a graph envelope cannot express by bytes
    alone. ``outstanding``/``response_ms`` are the request's real transfer
    parameters and are applied to every HBM phase by the same ``CacheTransfer``
    used by the event-time adapter.
    """

    result: CacheAccessResult
    hbm_resource: str = "hbm"
    onchip_resource: str = "mte2"
    pool: str | None = None
    source_required: bool = False
    source: Event | None = None
    source_bytes: int = 0
    convert_ops: float = 0.0
    source_residency: str = "on_chip"
    vector_resource: str = "vector"
    write_resource: str | None = None
    outstanding: int = 0
    response_ms: float = 0.0
    transaction_bytes: int | None = None

    def __post_init__(self):
        if type(self.outstanding) is not int or self.outstanding < 0:
            raise ModelingError("cache_access_program_outstanding_invalid")
        if (not isinstance(self.response_ms, (int, float))
                or isinstance(self.response_ms, bool) or self.response_ms < 0):
            raise ModelingError("cache_access_program_response_invalid")
        if self.transaction_bytes is None:
            object.__setattr__(
                self, "transaction_bytes", _result_line_bytes(self.result))
        elif type(self.transaction_bytes) is not int or self.transaction_bytes < 1:
            raise ModelingError("cache_access_program_transaction_bytes_invalid")

    def start(self, engine: Engine, name: str, ready: Event, *,
              source: Event | None = None, owner: str = "") -> Event:
        source = self.source if source is None else source
        if self.source_required and source is None:
            raise ModelingError("cache_access_program_source_missing")
        if source is not None and not isinstance(source, Event):
            raise ModelingError("cache_access_program_source_invalid")
        if not isinstance(engine, Engine):
            raise ModelingError("cache_access_program_requires_engine")
        if self.result.staging_bytes and not self.pool:
            raise ModelingError("cache_access_program_requires_staging_pool")
        if (self.result.hbm_read_bytes or self.result.hbm_write_bytes) \
                and self.hbm_resource not in engine.resources:
            raise ModelingError(f"cache_access_hbm_resource_missing:{self.hbm_resource}")
        if self.result.onchip_bytes and self.onchip_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_onchip_resource_missing:{self.onchip_resource}")
        if self.source_bytes and self.source_residency == "unresolved" \
                and self.hbm_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_hbm_resource_missing:{self.hbm_resource}")
        if (self.source_bytes or self.convert_ops) \
                and self.onchip_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_onchip_resource_missing:{self.onchip_resource}")
        if self.convert_ops and self.vector_resource not in engine.resources:
            raise ModelingError(
                f"cache_access_vector_resource_missing:{self.vector_resource}")
        staging = ()
        if self.result.staging_bytes:
            if self.pool not in engine.pools:
                raise ModelingError(f"cache_access_pool_missing:{self.pool}")
            staging = (Allocation(name + ":staging", self.pool,
                                  self.result.staging_bytes),)
        # Build and preflight the complete phase graph before creating the
        # source-join task or any Stage. No Engine mutation precedes this.
        items = cache_work_stages(
            self.result, hbm_resource=self.hbm_resource,
            onchip_resource=self.onchip_resource,
            write_resource=(self.write_resource
                            or self.onchip_resource),
            vector_resource=self.vector_resource,
            source_bytes=self.source_bytes,
            source_residency=self.source_residency,
            convert_ops=self.convert_ops,
            transaction_bytes=self.transaction_bytes)
        for item in items:
            if item.resource is not None and item.resource not in engine.resources:
                raise ModelingError(
                    f"cache_access_resource_missing:{item.resource}")
            if (item.resource == self.hbm_resource
                    and (self.outstanding or self.response_ms)
                    and engine.resources[item.resource].unit != "bytes"):
                raise ModelingError(
                    f"cache_access_hbm_resource_not_bytes:{self.hbm_resource}")
        start_ready = ready
        if source is not None:
            start_ready = engine.submit(name + ":source_join",
                                        after=(ready, source), owner=owner)
        stages = [Stage("ready", latency=self.result.ready_ms,
                        acquire=staging)]
        for item in items:
            work = ({item.resource: item.amount} if item.resource else {})
            if (item.resource == self.hbm_resource
                    and (self.outstanding or self.response_ms)):
                work = {}
                transfer = CacheTransfer(
                    ((item.resource, item.amount),),
                    transaction_bytes=self.transaction_bytes,
                    outstanding=self.outstanding or None,
                    response_ms=self.response_ms)
                stages.append(Stage(item.label, after=item.after,
                                    program=transfer))
            else:
                stages.append(Stage(item.label, work, after=item.after))
        if staging:
            # Hold the staging slot through the last physical effect, not just
            # through the visible copy.
            stages[-1] = replace(stages[-1], free=staging)
        return KernelProgram(tuple(stages)).start(engine, name, start_ready,
                                                  owner=owner)
