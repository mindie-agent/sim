"""Independent contention and residency counterexamples for physical units."""
import pytest

from mindie_sim.prediction.engine import Allocation, Engine, ExecutionError, Resource
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage
from mindie_sim.prediction.ownership import unit_rates


def test_disjoint_owner_groups_redistribute_capped_supply_only_inside_the_group():
    caps = {"a": float("inf"), "b": 30, "c": float("inf"), "d": 10000, "e": 40}
    placements = {"a": (0, 1), "b": (0, 1), "c": (2,),
                  "d": (4, 5, 6), "e": (4, 5, 6)}
    rates = unit_rates(800, 8, caps, placements)
    assert rates == pytest.approx({"a": 170, "b": 30, "c": 100, "d": 260, "e": 40})
    # Physical units 3 and 7 stay idle even though a and d could consume more.
    assert sum(rates.values()) == 600
    permuted = {key: tuple(reversed(units)) if key in {"b", "e"} else units
                for key, units in placements.items()}
    assert unit_rates(800, 8, caps, permuted) == pytest.approx(rates)


def test_unpinned_and_partial_overlap_keep_the_joint_per_unit_constraint():
    rates = unit_rates(800, 8, {"wide": float("inf"), "pair": float("inf"), "cap": 25},
                       {"pair": (0, 1), "cap": (2,)})
    # The shared units bind at wide/8 + pair/2 = 100; the limited third
    # consumer cannot transfer unused supply from unit2 to either shared unit.
    assert rates == pytest.approx({"wide": 160, "pair": 160, "cap": 25})
    rates = unit_rates(400, 4, {"a": 20, "b": float("inf"), "c": float("inf")},
                       {"a": (0, 1), "b": (1, 2), "c": (2, 3)})
    assert rates == pytest.approx({"a": 20, "b": 100, "c": 100})


def test_grouped_owner_release_reallocates_only_its_physical_partition():
    engine = Engine([Resource("copy", 800, units=8)])
    fast = engine.submit("fast", {"copy": 170}, placement={"copy": (0, 1)})
    capped = engine.submit("capped", {"copy": 60}, caps={"copy": 30},
                           placement={"copy": (0, 1)})
    other = engine.submit("other", {"copy": 300}, placement={"copy": (4, 5, 6)})
    engine.advance(until=(fast, capped, other), first=False)
    assert (fast.time, capped.time, other.time) == pytest.approx((1, 2, 1))
    assert engine.serviced["copy"] == 530
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


@pytest.mark.parametrize("placements,expected", [
    (((0,), (1,)), 1),
    (((0,), (0,)), 2),
])
def test_idle_units_cannot_supply_work_owned_by_one_unit(placements, expected):
    engine = Engine([Resource("vector", 400, units=4)])
    events = [engine.submit(str(i), {"vector": 100}, placement={"vector": units})
              for i, units in enumerate(placements)]
    engine.advance(until=events, first=False)
    assert engine.now == pytest.approx(expected)
    assert engine.serviced["vector"] == 200


def test_overlapping_groups_share_the_intersection_and_conserve_total_work():
    engine = Engine([Resource("matrix", 200, units=2)])
    group = engine.submit("pair", {"matrix": 200}, placement={"matrix": (0, 1)})
    single = engine.submit("single", {"matrix": 100}, placement={"matrix": (1,)})
    engine.advance(until=(group, single), first=False)
    # Both advance at 200/3 until the single finishes at 1.5ms; the pair
    # has 100 work left and finishes at 2ms using both units.
    assert single.time == pytest.approx(1.5)
    assert group.time == pytest.approx(2)
    assert engine.serviced["matrix"] == 300


@pytest.mark.parametrize("owners,expected", [((0, 1), 1), ((0, 0), 2)])
def test_buffers_are_independent_per_unit_and_never_aggregated(owners, expected):
    engine = Engine([], pools={"ub": 64}, pool_units={"ub": 4})
    events = []
    for index, unit in enumerate(owners):
        allocation = Allocation(f"resident{index}", "ub", 64, (unit,))
        events.append(engine.submit(str(index), latency=1, acquire=(allocation,),
                                    free=(allocation,)))
    engine.advance(until=events, first=False)
    assert engine.now == expected
    assert engine.peak_allocated["ub"] == 64
    assert engine.allocated["ub"] == 0
    assert not engine.unit_allocated
    with pytest.raises(ExecutionError, match="allocation_exceeds_capacity"):
        engine.submit("cannot_borrow", acquire=(Allocation("large", "ub", 65, (0,)),))


def test_nested_pipeline_preserves_unit_identity_and_releases_every_live_allocation():
    engine = Engine([Resource("vector", 200, units=2)],
                    pools={"ub": 64}, pool_units={"ub": 2})
    def tile(index):
        allocation = Allocation(f"tile{index}", "ub", 64, (1,))
        return (Stage("resident", acquire=(allocation,), free=(allocation,),
                      program=KernelProgram((Stage(
                          "work", {"vector": 100}, placement={"vector": (1,)}),))),)
    end = PipelineProgram(200, 2, tile).start(engine, "pipeline", engine.submit("ready"))
    engine.advance(until=(end,))
    assert end.time == pytest.approx(200)
    assert engine.serviced["vector"] == 20000
    assert engine.summary()["peak_allocated_per_unit"] == {"ub:unit1": 64}
    assert not engine.allocations and not engine.unit_allocated
    assert engine.peak_tasks < 30


def test_cross_iteration_handoff_allows_prefetch_but_protects_reused_output():
    engine = Engine([Resource("copy", 1), Resource("matrix", 1), Resource("vector", 1)],
                    trace_limit=100)
    def tile(index):
        return (Stage("load", {"copy": 1}, previous=("matrix",)),
                Stage("matrix", {"matrix": 1}, after=("load",), previous=("output",)),
                Stage("output", {"vector": 4}, after=("matrix",)))
    end = PipelineProgram(2, 2, tile).start(engine, "head", engine.submit("ready"))
    engine.advance(until=(end,))
    # First: load0[0,1], matrix0[1,2], output0[2,6].
    # Second: load1[2,3] overlaps output0, matrix1 waits until6, output1[7,11].
    assert end.time == 11
    assert engine.serviced == {"copy": 2, "matrix": 2, "vector": 8}
