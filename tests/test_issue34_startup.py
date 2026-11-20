"""AllToAll startup-model resolution and preserved physical work."""
from __future__ import annotations

import pytest

from mindie_sim.prediction.collective_startup import resolve_collective_startup
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from _support.builders import hardware


def exchange(volume=30.0, *, family="alltoall", steps=None, mode=None):
    metadata = [("group_axis", "tp")]
    if steps is not None:
        metadata.append(("steps", steps))
    if mode is not None:
        metadata.append(("collective_startup_mode", mode))
    return OpSpec("exchange", family, "prefill", 0, 0, (), (1,), (1,), (1,), (1,),
                  link_bytes=volume, metadata=tuple(metadata))


def startup(group, *, steps=None, runtime="legacy", protocols=None, family="alltoall"):
    metadata = {} if steps is None else {"steps": steps}
    return resolve_collective_startup(
        family=family, group_size=group, metadata=metadata,
        runtime_mode=runtime, topology_protocols=protocols or {})


def test_log2_and_legacy_startup_arithmetic():
    assert startup(64).startup_rounds == 63
    assert startup(128).startup_rounds == 127
    assert startup(64, runtime="log2").startup_rounds == 6
    assert startup(128, runtime="log2").startup_rounds == 7
    assert startup(3, runtime="log2").startup_rounds == 2
    assert startup(5, runtime="log2").startup_rounds == 3
    assert startup(8, runtime="concurrent").startup_rounds == 1
    assert startup(8, runtime="concurrent").peer_phases == 7
    assert startup(64, runtime="topology",
                   protocols={"alltoall": "log2"}).origin == "topology_protocol"
    with pytest.raises(ModelingError, match="collective_startup_protocol_undeclared"):
        startup(64, runtime="topology")
    with pytest.raises(ModelingError, match="collective_startup_family_unsupported"):
        resolve_collective_startup(family="allreduce", group_size=4, metadata={},
                                   runtime_mode="legacy", topology_protocols={})


@pytest.mark.parametrize("bad", [0, 4, -1, True, 1.5, "2", float("nan")])
def test_invalid_steps_are_rejected_before_zero_volume(bad):
    with pytest.raises(ModelingError, match="communication_round_count_invalid"):
        startup(4, steps=bad)
    topo = Topology(hardware(4), ParallelSpec(tp=4))
    with pytest.raises(ModelingError, match="communication_round_count_invalid"):
        topo.accounting(exchange(0.0, steps=bad), 0)


def test_single_rank_has_no_legal_steps_and_is_a_zero_noop():
    assert startup(1).startup_rounds == 0 and startup(1).peer_phases == 0
    with pytest.raises(ModelingError, match="communication_round_count_invalid"):
        startup(1, steps=1)
    topo = Topology(hardware(1), ParallelSpec(tp=1))
    work = topo.accounting(exchange(0.0), 0)
    assert work.messages == 0 and work.setup_ms == 0


def test_steps_change_only_startup_and_conserve_peer_work():
    topo = Topology(hardware(4), ParallelSpec(tp=4))
    one = topo.accounting(exchange(30.0, steps=1), 0)
    three = topo.accounting(exchange(30.0, steps=3), 0)
    assert sum(one.useful_by_rank.values()) == sum(three.useful_by_rank.values())
    assert sum(one.wire_by_resource.values()) == sum(three.wire_by_resource.values())
    assert one.messages == 12 == three.messages
    assert three.setup_ms >= one.setup_ms
    # G=4 explicit steps=2 is a legal startup model but the physical message
    # set remains all twelve peer flows.
    two = topo.accounting(exchange(30.0, steps=2), 0)
    assert two.messages == 12
    assert sum(two.useful_by_rank.values()) == sum(one.useful_by_rank.values())


def test_legacy_default_remains_serial_peer_phases():
    topo = Topology(hardware(4), ParallelSpec(tp=4))
    legacy = topo.accounting(exchange(30.0), 0)
    explicit = topo.accounting(exchange(30.0, steps=3), 0)
    assert legacy.setup_ms == pytest.approx(explicit.setup_ms)
    assert sum(legacy.useful_by_rank.values()) == sum(explicit.useful_by_rank.values())
