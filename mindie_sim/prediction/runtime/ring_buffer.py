"""Three-event distance-d RingBuffer lowering for on-chip and workspace slots.

[Role] Lower producer/multi-consumer ring-buffered slot generations into exact
  ``publish``, ``read_done[consumer]``, and ``last_use`` causal events with
  distance-``d`` credit back-edges, plus flat integer records for C11.
[Invariants] Slot ``k % d`` at generation ``k // d`` can never be overwritten
  before ``last_use(k - d)`` completes across all consumers (async copy,
  downstream core, live-out). Physical slot capacity grants are distinct from
  per-generation write-after-visibility events.
[Timeouts & Locks] Pure O(generations * consumers) construction; no I/O or locks.
[Side-Effect Policy] Mutates only the owning ``RingBuffer`` builder state.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True, slots=True)
class SlotGenerationId:
    """Unique physical ring-slot generation identity."""

    occurrence: int
    die: int
    owner: int
    allocation: str
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class GenerationEvents:
    """Three-event lifecycle record for generation ``k`` on a ring buffer."""

    index: int
    slot_gen: SlotGenerationId
    credit: str
    write: str
    publish: str
    read_done: tuple[str, ...]
    last_use: str


@dataclass(frozen=True, slots=True)
class SlotGenRecord:
    """Flat integer representation mirroring C11 ``msim_slot_gen``."""

    occurrence: int
    generation: int
    die: int
    owner: int
    pool: int
    slot: int
    publish: int
    last_use: int


@dataclass(frozen=True, slots=True)
class DepRecord:
    """Flat integer dependency edge mirroring C11 ``msim_dep``."""

    predecessor: int
    successor: int


@dataclass(frozen=True, slots=True)
class _NodeSpec:
    node_id: int
    name: str
    kind: str
    after: tuple[str, ...]
    work: Mapping[str, float]
    placement: Mapping[str, tuple[int, ...]]
    latency_ms: float
    coupled: bool
    slot: int | None = None
    acquire_slot: bool = False
    release_slot: bool = False


class RingBuffer:
    """Distance-``d`` ring buffer with explicit ``publish``, ``read_done``, ``last_use``."""

    def __init__(
        self,
        *,
        occurrence: int = 0,
        die: int = 0,
        owner: int = 0,
        pool: str = "ub",
        allocation: str = "ring",
        slot_bytes: int = 0,
        depth: int = 2,
        pool_id: int = 0,
    ) -> None:
        if type(depth) is not int or depth < 1:
            raise ModelingError(f"ring_buffer_depth_invalid:{depth!r}")
        if type(slot_bytes) is not int or slot_bytes < 0:
            raise ModelingError(f"ring_buffer_slot_bytes_invalid:{slot_bytes!r}")
        if type(occurrence) is not int or occurrence < 0:
            raise ModelingError(f"ring_buffer_occurrence_invalid:{occurrence!r}")
        if type(die) is not int or die < 0 or type(owner) is not int or owner < 0:
            raise ModelingError("ring_buffer_owner_invalid")
        if not allocation or not pool:
            raise ModelingError("ring_buffer_identity_required")
        self.occurrence = occurrence
        self.die = die
        self.owner = owner
        self.pool = pool
        self.pool_id = int(pool_id)
        self.allocation = allocation
        self.slot_bytes = slot_bytes
        self.depth = depth
        self._nodes: list[_NodeSpec] = []
        self._node_ids: dict[str, int] = {}
        self._generations: dict[int, GenerationEvents] = {}
        self._initial_credits: dict[int, str] = {}
        self._current_k: int = 0

    def slot(self, k: int) -> int:
        if type(k) is not int or k < 0:
            raise ModelingError(f"ring_buffer_index_invalid:{k!r}")
        return k % self.depth

    def generation(self, k: int) -> int:
        if type(k) is not int or k < 0:
            raise ModelingError(f"ring_buffer_index_invalid:{k!r}")
        return k // self.depth

    def _add_node(
        self,
        name: str,
        kind: str,
        after: tuple[str, ...],
        *,
        work: Mapping[str, float] | None = None,
        placement: Mapping[str, tuple[int, ...]] | None = None,
        latency_ms: float = 0.0,
        coupled: bool = False,
        slot: int | None = None,
        acquire_slot: bool = False,
        release_slot: bool = False,
    ) -> str:
        if name in self._node_ids:
            raise ModelingError(f"ring_buffer_duplicate_event:{name}")
        nid = len(self._nodes)
        self._node_ids[name] = nid
        self._nodes.append(_NodeSpec(
            node_id=nid,
            name=name,
            kind=kind,
            after=tuple(after),
            work=dict(work or {}),
            placement={k: tuple(v) for k, v in (placement or {}).items()},
            latency_ms=float(latency_ms),
            coupled=bool(coupled),
            slot=slot,
            acquire_slot=acquire_slot,
            release_slot=release_slot,
        ))
        return name

    def initial_credit(self, slot: int) -> str:
        """Reserve physical slot capacity once at ring initialization."""
        if not (0 <= slot < self.depth):
            raise ModelingError(f"ring_buffer_slot_out_of_range:{slot}")
        existing = self._initial_credits.get(slot)
        if existing is not None:
            return existing
        name = f"{self.allocation}:slot{slot}:credit0"
        self._add_node(
            name, "credit", (),
            slot=slot, acquire_slot=(self.slot_bytes > 0),
        )
        self._initial_credits[slot] = name
        return name

    def last_use(self, k: int) -> str:
        """Return the recorded ``last_use`` event for earlier generation ``k``."""
        if k not in self._generations:
            raise ModelingError(f"ring_buffer_missing_generation:{k}")
        return self._generations[k].last_use

    def issue_producer(
        self,
        producer: str,
        *,
        after: str,
        work: Mapping[str, float] | None = None,
        placement: Mapping[str, tuple[int, ...]] | None = None,
        latency_ms: float = 0.0,
        coupled: bool = False,
        extra_after: Sequence[str] = (),
    ) -> str:
        """Emit producer write stage for the current generation after slot credit."""
        name = f"{self.allocation}:g{self._current_k}:write:{producer}"
        deps = tuple(dict.fromkeys((after, *extra_after)))
        return self._add_node(
            name, "write", deps,
            work=work, placement=placement, latency_ms=latency_ms, coupled=coupled,
        )

    def publish_after_visibility(
        self,
        write: str,
        *,
        visibility_latency_ms: float = 0.0,
    ) -> str:
        """Emit explicit write-after-visibility ``publish`` marker for generation."""
        name = f"{self.allocation}:g{self._current_k}:publish"
        return self._add_node(name, "publish", (write,), latency_ms=visibility_latency_ms)

    def issue_consumer(
        self,
        consumer: str,
        *,
        after: str,
        work: Mapping[str, float] | None = None,
        placement: Mapping[str, tuple[int, ...]] | None = None,
        latency_ms: float = 0.0,
        coupled: bool = False,
        extra_after: Sequence[str] = (),
    ) -> str:
        """Emit one consumer's ``read_done`` stage after generation ``publish``."""
        name = f"{self.allocation}:g{self._current_k}:read_done:{consumer}"
        deps = tuple(dict.fromkeys((after, *extra_after)))
        return self._add_node(
            name, "read_done", deps,
            work=work, placement=placement, latency_ms=latency_ms, coupled=coupled,
        )

    def join(self, events: tuple[str, ...]) -> str:
        """Emit the ``last_use`` join across all consumer ``read_done`` events."""
        if not events:
            raise ModelingError("ring_buffer_join_requires_events")
        name = f"{self.allocation}:g{self._current_k}:last_use"
        return self._add_node(name, "last_use", tuple(dict.fromkeys(events)))

    def record(
        self,
        k: int,
        publish: str,
        done: tuple[str, ...],
        last_use: str,
        *,
        credit: str = "",
        write: str = "",
    ) -> GenerationEvents:
        slot = self.slot(k)
        gen = self.generation(k)
        ev = GenerationEvents(
            index=k,
            slot_gen=SlotGenerationId(
                occurrence=self.occurrence,
                die=self.die,
                owner=self.owner,
                allocation=self.allocation,
                slot=slot,
                generation=gen,
            ),
            credit=credit or (
                self._initial_credits[slot] if k < self.depth else self._generations[k - self.depth].last_use
            ),
            write=write,
            publish=publish,
            read_done=tuple(done),
            last_use=last_use,
        )
        self._generations[k] = ev
        return ev

    @property
    def generations(self) -> tuple[GenerationEvents, ...]:
        return tuple(self._generations[k] for k in sorted(self._generations))

    def to_c_records(self) -> tuple[tuple[SlotGenRecord, ...], tuple[DepRecord, ...]]:
        """Export flat integer ``msim_slot_gen`` and ``msim_dep`` tables for C11."""
        slot_gens: list[SlotGenRecord] = []
        for ev in self.generations:
            sg = ev.slot_gen
            slot_gens.append(SlotGenRecord(
                occurrence=sg.occurrence,
                generation=sg.generation,
                die=sg.die,
                owner=sg.owner,
                pool=self.pool_id,
                slot=sg.slot,
                publish=self._node_ids[ev.publish],
                last_use=self._node_ids[ev.last_use],
            ))
        deps: list[DepRecord] = []
        for node in self._nodes:
            for pred_name in node.after:
                deps.append(DepRecord(
                    predecessor=self._node_ids[pred_name],
                    successor=node.node_id,
                ))
        return tuple(slot_gens), tuple(deps)

    def to_stages(self, *, release_slots_at_end: bool = True) -> tuple[Any, ...]:
        """Lower all recorded generations into immutable ``Stage`` DAG entries."""
        from ..program.stages import Stage
        from .engine_types import Allocation

        stages: list[Stage] = []
        slot_allocs: dict[int, Allocation] = {}
        last_use_by_slot: dict[int, str] = {}
        for ev in self.generations:
            last_use_by_slot[ev.slot_gen.slot] = ev.last_use

        for node in self._nodes:
            acq: tuple[Allocation, ...] = ()
            if node.acquire_slot and node.slot is not None and self.slot_bytes > 0:
                alloc = Allocation(
                    name=f"{self.allocation}:slot{node.slot}",
                    pool=self.pool,
                    size=float(self.slot_bytes),
                    units=(self.owner,),
                )
                slot_allocs[node.slot] = alloc
                acq = (alloc,)
            stages.append(Stage(
                name=node.name,
                work=node.work,
                after=node.after,
                latency=node.latency_ms,
                acquire=acq,
                placement=node.placement,
                coupled=node.coupled,
                keep_empty=bool(acq),
            ))

        if release_slots_at_end and slot_allocs:
            for slot, alloc in sorted(slot_allocs.items()):
                after_ev = last_use_by_slot.get(slot, self._initial_credits[slot])
                stages.append(Stage(
                    name=f"{self.allocation}:slot{slot}:release",
                    after=(after_ev,),
                    free=(alloc,),
                    keep_empty=True,
                ))
        return tuple(stages)


def lower_generation(
    ring: RingBuffer,
    k: int,
    producer: str,
    consumers: tuple[str, ...],
    *,
    producer_work: Mapping[str, float] | None = None,
    producer_placement: Mapping[str, tuple[int, ...]] | None = None,
    producer_latency_ms: float = 0.0,
    producer_coupled: bool = False,
    visibility_latency_ms: float = 0.0,
    consumer_works: Mapping[str, Mapping[str, float]] | None = None,
    consumer_placements: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None,
    consumer_latencies_ms: Mapping[str, float] | None = None,
    consumer_coupled: Mapping[str, bool] | None = None,
    producer_extra_after: Sequence[str] = (),
    consumer_extra_after: Mapping[str, Sequence[str]] | None = None,
) -> GenerationEvents:
    """Lower generation ``k`` onto ``ring`` with distance-``d`` credit back-edge."""
    ring._current_k = k
    slot = ring.slot(k)
    credit = (
        ring.initial_credit(slot)
        if k < ring.depth else ring.last_use(k - ring.depth)
    )
    write = ring.issue_producer(
        producer,
        after=credit,
        work=producer_work,
        placement=producer_placement,
        latency_ms=producer_latency_ms,
        coupled=producer_coupled,
        extra_after=producer_extra_after,
    )
    publish = ring.publish_after_visibility(
        write, visibility_latency_ms=visibility_latency_ms,
    )
    c_works = consumer_works or {}
    c_places = consumer_placements or {}
    c_lats = consumer_latencies_ms or {}
    c_coup = consumer_coupled or {}
    c_extra = consumer_extra_after or {}
    done = tuple(
        ring.issue_consumer(
            consumer,
            after=publish,
            work=c_works.get(consumer),
            placement=c_places.get(consumer),
            latency_ms=c_lats.get(consumer, 0.0),
            coupled=c_coup.get(consumer, False),
            extra_after=c_extra.get(consumer, ()),
        )
        for consumer in consumers
    )
    last_use = ring.join(done or (publish,))
    return ring.record(k, publish, done, last_use, credit=credit, write=write)


__all__ = [
    "DepRecord",
    "GenerationEvents",
    "RingBuffer",
    "SlotGenRecord",
    "SlotGenerationId",
    "lower_generation",
]
