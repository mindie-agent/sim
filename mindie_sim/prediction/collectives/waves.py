"""Bounded executable startup waves for generic all-to-all collectives.

[Role]
  Realize the declared startup model as an executable physical schedule: each
  wave owns a contiguous, balanced range of cyclic peer offsets, all sends in a
  wave start concurrently (bounded by the declared per-rank inflight window),
  and a rank enters its next wave only after its own sends and receives for the
  previous wave complete.  Every hop delay and all peer payloads are preserved.

[Invariants]
  * ``S`` in ``[1, G-1]`` partitions all ``G-1`` offsets exactly once; ``S=G-1``
    recovers serial peer phases and ``S=1`` permits all-peer concurrency.
  * No global barrier is introduced; only per-rank wave completion is a
    dependency.  Routing, framing, useful/wire bytes and receive caps are the
    existing topology ones.
  * Explicit ``steps`` controls ``S``; a source-bound recipe wins and is never
    silently replaced by this generic program.
  * Inflight work is bounded by the declared window and lazy offset iteration;
    no unlimited fabric bandwidth is invented.

[Timeouts & Locks]
  Pure construction plus Engine event submission; no I/O or locks.

[Side-Effect Policy]
  Submits causal Engine work only; returns immutable plan/program records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.communication import MessageFlow


REL_TOL = 1e-12
ABS_TOL = 1e-9
REL_TOL = 1e-12
ABS_TOL = 1e-9
OFFSET_FIRST = 1
DEFAULT_MAX_INFLIGHT: int | None = None
GENERIC_RECIPE = "startup_waves"
ALLTOALL_FAMILIES = frozenset({"alltoall", "alltoallv"})
NONLEGACY_STARTUP_MODES = frozenset({"ring", "log2", "concurrent", "topology"})


def validate_wave_rounds(group_size: int, rounds: int) -> int:
    """Validate ``S`` against the nonempty cyclic-offset range."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ModelingError(f"collective_group_size_invalid:{group_size}")
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise ModelingError(f"communication_round_count_invalid:{rounds}")
    if group_size <= 1 or rounds < 1 or rounds > group_size - 1:
        raise ModelingError(f"communication_round_count_invalid:{rounds}")
    return int(rounds)


def wave_ranges(group_size: int, rounds: int) -> tuple[tuple[int, int], ...]:
    """Contiguous balanced inclusive cyclic-offset ranges, one per wave."""
    validate_wave_rounds(group_size, rounds)
    offsets = group_size - 1
    return tuple(
        (OFFSET_FIRST + wave * offsets // rounds,
         (wave + 1) * offsets // rounds)
        for wave in range(rounds)
    )


def wave_offsets(group_size: int, rounds: int):
    """Lazily iterate ``(wave, offset)`` pairs in ascending offset order."""
    for wave, (start, end) in enumerate(wave_ranges(group_size, rounds)):
        for offset in range(start, end + 1):
            yield wave, offset


def startup_waves_recipe(op, recipe, runtime_mode):
    """Resolve the same generic replacement before and after preparation."""
    if (not getattr(recipe, "source_bound", False)
            and not getattr(recipe, "native", False)
            and not getattr(recipe, "algorithm", None)
            and startup_waves_selected(op, runtime_mode)):
        return RECIPE
    return recipe


def startup_waves_selected(op: Any, runtime_mode: str) -> bool:
    """True for an unbound generic all-to-all with waves or explicit steps."""
    if getattr(op, "source_boundary", ""):
        return False
    metadata = dict(getattr(op, "metadata", ()) or ())
    if metadata.get("implementation") or metadata.get("hccl_selection"):
        return False
    family = str(metadata.get("collective", op.family)).strip().lower()
    if family not in ALLTOALL_FAMILIES:
        return False
    steps = metadata.get("steps")
    if isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1:
        mode_raw = metadata.get("collective_startup_mode", runtime_mode)
        norm = {"direct": "concurrent"}.get(str(mode_raw).strip().lower(),
                                            str(mode_raw).strip().lower())
        if norm in NONLEGACY_STARTUP_MODES:
            return True
    mode = metadata.get("collective_startup_mode", runtime_mode)
    if not isinstance(mode, str):
        return False
    normalized = {"direct": "concurrent"}.get(mode.strip().lower(), mode.strip().lower())
    return normalized in NONLEGACY_STARTUP_MODES


@dataclass(frozen=True)
class WaveGroup:
    """One all-to-all group executed as bounded startup waves."""

    ranks: tuple[int, ...]
    volumes: tuple[tuple[int, float], ...]
    startup_rounds: int
    topology: object = field(repr=False, compare=False)
    peer_traffic: object | None = field(default=None, repr=False, compare=False)
    max_inflight_per_rank: int | None = DEFAULT_MAX_INFLIGHT

    def _payload(self, source_index: int, destination_index: int, count: int) -> float:
        if self.peer_traffic is not None:
            return self.peer_traffic.payload(source_index, destination_index)
        if count <= 1:
            return 0.0
        return dict(self.volumes)[self.ranks[source_index]] / (count - 1)

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        from mindie_sim.prediction.collectives import _ready_by_rank

        initial = _ready_by_rank(self.ranks, ready_by_rank)
        count = len(self.ranks)
        if count <= 1 or not any(dict(self.volumes).values()):
            return initial
        rounds = validate_wave_rounds(count, self.startup_rounds)
        if self.peer_traffic is not None:
            if self.peer_traffic.size != count:
                raise ModelingError("pairwise_collective_peer_group_size_mismatch")
            for index, rank in enumerate(self.ranks):
                if not math.isclose(
                        self.peer_traffic.row_total(index),
                        dict(self.volumes)[rank], rel_tol=REL_TOL, abs_tol=ABS_TOL):
                    raise ModelingError(
                        f"pairwise_collective_peer_row_total_mismatch:{rank}")
        waves = wave_ranges(count, rounds)
        complete = {rank: engine.future(f"{name}:rank{rank}:complete")
                    for rank in self.ranks}
        wave_index = [0] * count
        send_pending = [0] * count
        recv_pending = [0] * count
        pending = [{} for _ in range(count)]
        queued: list[list[int]] = [[] for _ in range(count)]
        window = self.max_inflight_per_rank

        def parse_wave(index: int) -> tuple[int, int]:
            return waves[wave_index[index]]

        def retire(index: int) -> None:
            if wave_index[index] >= len(waves):
                event = complete[self.ranks[index]]
                if not event.done:
                    engine.resolve(event)

        def enter_next(index: int) -> None:
            if queued[index] or send_pending[index] or recv_pending[index]:
                return
            wave_index[index] += 1
            if wave_index[index] >= len(waves):
                retire(index)
                return
            begin(index)

        def deliver(index: int, wave: int, event) -> None:
            del event
            if wave < wave_index[index]:
                raise ModelingError("startup_wave_receive_wave_regressed")
            if wave > wave_index[index]:
                pending[index][wave] = pending[index].get(wave, 0) + 1
                return
            recv_pending[index] -= 1
            if recv_pending[index] < 0:
                raise ModelingError("startup_wave_receive_underflow")
            if recv_pending[index] == 0:
                pending[index].pop(wave, None)
            enter_next(index)

        def send_done(index: int, event) -> None:
            del event
            send_pending[index] -= 1
            if send_pending[index] < 0:
                raise ModelingError("startup_wave_send_underflow")
            if window is None or send_pending[index] < window:
                issue(index)
            enter_next(index)

        def issue(index: int) -> None:
            if not queued[index]:
                return
            send_pending[index] += 1
            source = self.ranks[index]
            wave = wave_index[index]
            offset = queued[index].pop(0)
            destination_index = (index + offset) % count
            destination = self.ranks[destination_index]
            payload = self._payload(index, destination_index, count)
            prefix = f"{name}:rank{source}:wave{wave}:offset{offset}"
            dependencies = [initial[source]]
            if payload:
                dependencies.append(initial[destination])
            ready = engine.submit(
                prefix + ":ready", after=tuple(dependencies), owner=owner,
                metadata={"source_rank": source,
                          "destination_rank": destination,
                          "wave": wave, "offset": offset})
            received_after = []
            if payload:
                message = self.topology.route_message(MessageFlow(
                    source, destination, payload))
                previous = ready
                for hop_index, hop in enumerate(message.hops):
                    service = engine.submit(
                        f"{prefix}:hop{hop_index}:service", hop.demand,
                        after=(previous,), owner=owner, coupled=True,
                        on_complete=((lambda arrival: send_done(index, arrival))
                                     if hop_index == 0 else None),
                    )
                    previous = engine.submit(
                        f"{prefix}:hop{hop_index}:arrival", after=(service,),
                        latency=hop.latency_ms, owner=owner)
                received_after.append(previous)
                if not message.hops:
                    engine.submit(prefix + ":done", after=(ready,), owner=owner,
                                  on_complete=lambda arrival: send_done(index, arrival))
            else:
                engine.submit(prefix + ":done", after=(ready,), owner=owner,
                              on_complete=lambda arrival: send_done(index, arrival))
            engine.submit(
                prefix + ":received",
                after=tuple(received_after) + (initial[destination],),
                owner=owner,
                on_complete=lambda arrival: deliver(destination_index, wave, arrival),
            )

        def begin(index: int) -> None:
            start, end = parse_wave(index)
            offsets = tuple(range(start, end + 1))
            recv_pending[index] = len(offsets) - int(
                pending[index].pop(wave_index[index], 0))
            queued[index] = list(offsets)
            if recv_pending[index] < 0:
                raise ModelingError("startup_wave_receive_underflow")
            limit = len(offsets) if window is None else min(window, len(offsets))
            for _ in range(limit):
                issue(index)
            enter_next(index)

        for index in range(count):
            begin(index)
        return complete


@dataclass(frozen=True)
class WavePlan:
    family: str
    groups: tuple[WaveGroup, ...]
    startup_rounds: int | None = None

    @property
    def ranks(self):
        return tuple(sorted({rank for group in self.groups for rank in group.ranks}))

    def start(self, engine, name, ready, *, owner=""):
        return self.start_participants(
            engine, name, {rank: ready for rank in self.ranks}, owner=owner)

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        events = {}
        for index, group in enumerate(self.groups):
            part = group.start_participants(
                engine, f"{name}:group{index}", ready_by_rank, owner=owner)
            events.update(part)
        return events

    def validate_path_model(self, path_model) -> None:
        if path_model is not None:
            raise ModelingError(
                f"startup_waves_path_model_unsupported:{path_model}")


@dataclass(frozen=True)
class StartupWaves:
    """Generic executable recipe for a declared nonlegacy startup model."""

    algorithm = "startup_waves"

    def lower(self, context):
        from mindie_sim.prediction.program import KernelProgram, Stage

        if any(amount for role, amount in context.native_work.items()
               if role != "link"):
            raise ModelingError("startup_waves_requires_transport_only_work")
        if context.collective_plan is None:
            if not context.native_work.get("link"):
                return KernelProgram((), minimum_ms=context.minimum_ms)
            raise ModelingError("startup_waves_requires_bound_topology")
        plan = context.collective_plan()
        rounds = getattr(plan, "startup_rounds", None)
        if rounds is None:
            raise ModelingError("startup_waves_requires_resolved_startup_rounds")
        options = getattr(context, "transport_options", None)
        window = getattr(options, "max_inflight_per_rank", None)
        groups = tuple(
            WaveGroup(
                group.ranks, group.volumes, rounds, group.topology,
                peer_traffic=group.peer_traffic,
                max_inflight_per_rank=window,
            )
            for group in getattr(plan, "groups", ())
        )
        return KernelProgram(
            (Stage("startup_waves", program=WavePlan(
                getattr(plan, "family", "alltoall"), groups, rounds)),),
            minimum_ms=context.minimum_ms)


RECIPE = StartupWaves()
