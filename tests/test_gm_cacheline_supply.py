"""A5 generic 128B HBM line supply, not a scalar GetValue latency price."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.schema.hardware import Hardware

ROLE = "gm_hbm_cacheline_128b"


def _run(lines, *, owner=0, sku="Ascend950DT_9582"):
    d = Hardware.from_mapping({"backend": "ascend", "sku": sku,
                               "num_devices": 1, "device": {}}).dies[0]
    rates = {name: float(rate) / 1000.0 for name, rate in d.rates}
    units = {name: int(d.units(name)) for name in rates}
    engine = Engine([Resource(name, rate, units=units[name])
                     for name, rate in rates.items()])
    context = RecipeContext(None, {}, {name: name for name in rates},
                            rates, 64, resource_units=units)
    ready = engine.submit("ready", release=0)
    work = Work(resources={ROLE: lines})
    demand = context.bind(work)
    placement = context.owned(owner, 64, work.resources)
    done = KernelProgram((Stage("line", demand, placement=placement),)).start(
        engine, "line", ready)
    engine.advance(until=(done,))
    return float(done.time - ready.time) * 1e6, engine.serviced, d


def test_a5_hbm_128b_line_is_a_generic_supply_not_scalar_latency():
    elapsed, serviced, d = _run(1)
    assert elapsed == pytest.approx(1e9 / 488281250, rel=1e-12)
    assert serviced.get(ROLE) == 1 and sum(serviced.values()) == 1
    assert float(dict(d.rates)[ROLE]) == 31250000000.0
    assert d.units(ROLE) == 64
    assert dict(d.work_units)[ROLE] == "cachelines"
    # The 1025-line count is bounded only as HBM line transactions; this is
    # not a claim that a dependent GetValue/pointer chain completes in 2.1us.
    assert _run(1025)[0] == pytest.approx(1025e9 / 488281250, rel=1e-12)
    for unpriced in ("scalar_gm_access", "scalar_alu_issue"):
        assert unpriced not in dict(d.rates)
