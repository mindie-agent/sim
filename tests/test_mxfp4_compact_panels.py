"""MXFP4 panel compression retains periodic copies and physical bottlenecks."""
from copy import deepcopy

import pytest

from mindie_sim import operator_costs
from mindie_sim.prediction.mxfp4_grouped import recipe
from mindie_sim.prediction.program import RecipeBook
from test_mxfp4_grouped import source_case, gate, physical_op


def price(raw, op, compact):
    recipes = RecipeBook(families={"grouped_gemm": recipe(compact_panels=compact)})
    return operator_costs(raw, (op,), recipes=recipes, input_ready_ms=7)["operators"][0]


@pytest.mark.parametrize("rows,n,k", [((1, 0, 2), 320, 320), ((17,), 4608, 5120),
                                       ((1, 1), 5120, 2304), ((3,), 160, 7680)])
def test_periodic_activation_and_scale_refills_conserve_actual_service(source_case, gate, rows, n, k):
    op = physical_op(gate, rows, n, k)
    detailed, compact = (price(source_case, op, value) for value in (False, True))
    assert compact["serviced"] == pytest.approx(detailed["serviced"])
    assert not compact["service_before_input"]
    assert compact["live_tasks"] == 0
    assert not any(compact["remaining_buffers"].values())
    assert compact["peak_buffers_per_unit"] == detailed["peak_buffers_per_unit"]
    # Fixpipe rate is destination bytes/cycle for F32->BF16. HBM and
    # Fixpipe both publish 2 bytes per real output, not FP32 source bytes.
    assert compact["serviced"]["fixpipe@r0:s0"] == 2 * sum(rows) * n


@pytest.mark.parametrize("role", ("hbm", "mte2_aic", "mte2_aiv", "mte1", "mte3", "vector", "fixpipe"))
def test_compact_pipeline_still_responds_to_each_scarce_physical_pipe(source_case, gate, role):
    values = []
    for factor in (.5, 1, 2):
        raw = deepcopy(source_case)
        raw["machine"]["device"]["rates"] = {
            **{r: 1e18 for r in ("hbm", "mte2_aic", "mte2_aiv", "mte1", "mte3", "vector", "fixpipe")},
            role: 1e8 * factor}
        values.append(price(raw, physical_op(gate, (1, 1), 160, 5120), True)["duration_ms"])
    assert values[0] > values[1] > values[2]


@pytest.mark.parametrize("n,k", ((4608, 5120), (5120, 2304)))
def test_uniform_and_ragged_owner_waves_keep_half_hbm_fidelity(source_case, gate, n, k):
    from mindie_sim.prediction.structure import parse_case
    raw = deepcopy(source_case)
    baseline = dict(parse_case(raw).machine.dies[0].rates)["hbm"]
    raw["machine"]["device"]["rates"] = {"hbm": baseline / 2}
    op = physical_op(gate, (1,) * 6 + (0,) * 42, n, k)
    detailed, compact = (price(raw, op, value) for value in (False, True))
    # Whole-die nominal supply lost uniform-owner backpressure; a permanent
    # 1/32 share overconstrained the two 128/160-column phase classes. Equal
    # source schedules share nominal refill supply, with real inter-class
    # competition still performed by the Engine.
    assert abs(compact["duration_ms"] / detailed["duration_ms"] - 1) < .10
    assert compact["serviced"] == pytest.approx(detailed["serviced"])
