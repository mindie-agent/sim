"""Executable startup waves: partitioning, bounded concurrency, no extra startup."""
from __future__ import annotations

import pytest

from mindie_sim.prediction.collective_waves import (
    WaveGroup, WavePlan, wave_offsets, wave_ranges,
)
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from issue34_helpers import hardware


def topology(ranks=4):
    return Topology(hardware(ranks), ParallelSpec(tp=ranks))


def run_group(ranks, rounds, *, volumes=None):
    topo = topology(ranks)
    volumes = volumes or tuple((rank, 30.0) for rank in range(ranks))
    group = WaveGroup(tuple(range(ranks)), volumes, rounds, topo)
    engine = Engine(topo.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in range(ranks)}
    done = group.start_participants(engine, f"waves{ranks}-{rounds}", ready)
    engine.advance(until=tuple(done.values()), first=False)
    return engine, done


def test_wave_ranges_partition_every_offset_exactly_once():
    for group in (2, 3, 4, 5, 8):
        for rounds in range(1, group):
            ranges = wave_ranges(group, rounds)
            offsets = [offset for _wave, offset in wave_offsets(group, rounds)]
            assert offsets == list(range(1, group))
            assert len(ranges) == rounds
            assert all(start <= end for start, end in ranges)
            for left, right in zip(ranges, ranges[1:]):
                assert left[1] < right[0]
    with pytest.raises(ModelingError, match="communication_round_count_invalid"):
        wave_ranges(4, 4)


def test_concurrent_and_serial_waves_conserve_all_peer_bytes():
    serial_engine, serial_done = run_group(4, 3)
    concurrent_engine, concurrent_done = run_group(4, 1)
    assert all(event.done for event in serial_done.values())
    assert all(event.done for event in concurrent_done.values())
    serial_bytes = sum(value for name, value in serial_engine.serviced.items()
                       if name.startswith("link:"))
    concurrent_bytes = sum(value for name, value in concurrent_engine.serviced.items()
                           if name.startswith("link:"))
    assert serial_bytes == pytest.approx(concurrent_bytes)
    assert serial_bytes == pytest.approx(4 * 30.0)
    assert serial_engine.now == pytest.approx(concurrent_engine.now)


def test_wave_window_bounds_inflight_and_preserves_payloads():
    engine, done = run_group(4, 1)
    assert all(event.done for event in done.values())
    total = sum(value for name, value in engine.serviced.items()
                if name.startswith("link:"))
    assert total == pytest.approx(120.0)


def test_reverse_variable_traffic_preserves_each_direction():
    from mindie_sim.prediction.peer_traffic import PeerTraffic

    topo = topology(3)
    peer = PeerTraffic.from_mapping({
        "kind": "matrix",
        "payload_bytes": [[0, 5, 7], [3, 0, 11], [13, 17, 0]],
        "direction": "forward",
    }, 3)
    volumes = tuple((rank, peer.row_total(rank)) for rank in range(3))
    group = WaveGroup(tuple(range(3)), volumes, 1, topo, peer_traffic=peer)
    engine = Engine(topo.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in range(3)}
    done = group.start_participants(engine, "reverse", ready)
    engine.advance(until=tuple(done.values()), first=False)
    assert all(event.done for event in done.values())
    total = sum(value for name, value in engine.serviced.items()
                if name.startswith("link:"))
    assert total == pytest.approx(56.0)


def test_zero_payload_idle_peer_still_rendezvous():
    topo = topology(3)
    group = WaveGroup((0, 1, 2), ((0, 0.0), (1, 60.0), (2, 0.0)), 1, topo)
    engine = Engine(topo.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in range(3)}
    done = group.start_participants(engine, "idle", ready)
    engine.advance(until=tuple(done.values()), first=False)
    assert all(event.done for event in done.values())
    assert engine.serviced.get("link:fabric", 0.0) == pytest.approx(60.0)
