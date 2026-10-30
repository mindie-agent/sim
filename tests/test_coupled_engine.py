"""Physical work, backpressure and external contention of compact pipelines."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource


def test_coupled_progress_cannot_read_ahead_while_compute_is_contended():
    engine = Engine([Resource("read", 10), Resource("compute", 10)])
    done = engine.submit("steady", {"read": 100, "compute": 100}, coupled=True)
    blocker = engine.submit("other-compute", {"compute": 50})
    engine.advance(horizon=5)
    assert engine.serviced == pytest.approx({"read": 25, "compute": 50})
    engine.advance(until=(blocker,))
    assert blocker.time == pytest.approx(10)
    assert engine.serviced == pytest.approx({"read": 50, "compute": 100})
    engine.advance(until=(done,))
    assert done.time == pytest.approx(15)
    assert engine.serviced == pytest.approx({"read": 100, "compute": 150})


def test_complementary_resource_flows_reclaim_the_capacity_of_each_other():
    engine = Engine([Resource("read", 10), Resource("compute", 10)])
    left = engine.submit("left", {"read": 10, "compute": 100}, coupled=True)
    right = engine.submit("right", {"read": 100, "compute": 10}, coupled=True)
    engine.advance(until=(left, right), first=False)
    assert left.time == pytest.approx(11)
    assert right.time == pytest.approx(11)
    assert engine.serviced == pytest.approx({"read": 110, "compute": 110})


def test_coupled_unit_ownership_and_external_release_change_progress_causally():
    engine = Engine([Resource("read", 20, units=2), Resource("compute", 20, units=2)])
    placement = {"read": (0,), "compute": (0,)}
    done = engine.submit("steady", {"read": 100, "compute": 100},
                         placement=placement, coupled=True, release=3)
    other = engine.submit("other", {"compute": 40}, release=5,
                          placement={"compute": (0,)})
    independent = engine.submit("independent", {"compute": 100},
                                placement={"compute": (1,)})
    engine.advance(horizon=3)
    assert engine.serviced["read"] == 0
    engine.advance(until=(done, other, independent), first=False)
    assert independent.time == pytest.approx(10)
    assert other.time == pytest.approx(13)
    assert done.time == pytest.approx(17)
    assert engine.summary()["live_tasks"] == 0
    assert engine.serviced == pytest.approx({"read": 100, "compute": 240})


def test_fifo_eligibility_does_not_consume_other_coupled_resources_early():
    engine = Engine([Resource("queue", 1, policy="fifo"), Resource("read", 10)])
    earlier = engine.submit("earlier", {"queue": 5})
    later = engine.submit("later", {"queue": 2, "read": 20}, coupled=True)
    engine.advance(until=(earlier,))
    assert engine.serviced == pytest.approx({"queue": 5, "read": 0})
    engine.advance(until=(later,))
    assert later.time == pytest.approx(7)


@pytest.mark.parametrize('right_owners', ((1, 3), (3, 1)))
def test_capped_flows_and_independent_components_share_only_their_owners(right_owners):
    # Only two of four owners participate: 60 read and 40 compute units/ms.
    # The capped flow takes (12, 4). Compute then splits its remaining 36
    # between the second coupled flow and the ordinary compute component.
    # The ordinary read component can reclaim the final 30 read units/ms.
    engine = Engine([Resource('read', 120, units=4), Resource('compute', 80, units=4)])
    placement = {'read': (1, 3), 'compute': (1, 3)}
    left = engine.submit('capped', {'read': 60, 'compute': 20},
                         placement=placement, caps={'read': 12}, coupled=True)
    right = engine.submit('coupled', {'read': 60, 'compute': 60}, coupled=True,
                          placement={'read': right_owners, 'compute': right_owners})
    ordinary = engine.submit('independent-components', {'read': 20, 'compute': 20},
                             placement=placement)
    # A disjoint owner cannot affect the rates or lend its unused capacity.
    separate = engine.submit('other-owner', {'read': 100}, placement={'read': (0,)})
    engine._settle()
    rates = engine._rates()
    assert rates['read'] == pytest.approx({left.sequence: 12, right.sequence: 18,
                                           ordinary.sequence: 30, separate.sequence: 30})
    assert rates['compute'] == pytest.approx({left.sequence: 4, right.sequence: 18,
                                              ordinary.sequence: 18})
    engine.advance(horizon=0.5)
    assert engine.serviced == pytest.approx({'read': 45, 'compute': 20})
    engine.advance()
    assert engine.serviced == pytest.approx({'read': 240, 'compute': 100})
    assert engine.summary()['live_tasks'] == 0
