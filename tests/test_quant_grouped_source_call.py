"""Public source callback controls with independently counted physical work."""
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.call_builders import build_call
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.common import ModelingError


ROOT = Path(__file__).resolve().parents[1]


def declaration():
    return {
        "builder": "quantized_grouped_source",
        "arguments": {
            "expert_rows": [3, 0, 3, 0], "k": 64, "n": 256,
            "tiling": {"base_m": 16, "base_n": 128, "base_k": 32, "used_cores": 32},
        },
    }


def test_real_callback_replay_preserves_output_publication_private_scales_and_ready_time():
    case = json.loads((ROOT / "examples/calls/quantized-grouped-small.json").read_text())
    call = build_call(declaration())
    report = call_costs(case, call.ops, recipes=call.recipes, mode="graph",
                        invocations=2, input_ready_ms=5, trace_limit=1000)
    work = report["execution"]["work"]
    assert work["flops"] == 2 * (2 * 6 * 256 * 64)
    assert work["vector_ops"] == 2 * 6144
    assert work["hbm_bytes"] == 2 * 38328
    service = report["execution"]["serviced"]
    for role, amount in {
        "memory": 38328, "mte2_aic": 33536, "mte1": 33536,
        "mte2_aiv": 4144, "fixpipe": 8192, "mte3": 3072,
    }.items():
        assert service[role + "@r0:s0"] == pytest.approx(2 * amount)
    assert report["single_call"]["source_boundaries"] == {"kernel": 1}
    assert report["operator_occurrences"] == 2
    assert report["complete_ms"] > 5
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert not any(report["execution"]["live_allocated"].values())
    assert report["parameters"]["calibration"] == "none"


def test_source_callback_binding_cannot_be_replaced_by_fast_generic_phase_recipe():
    call = build_call(declaration())
    generic = RecipeBook.from_mapping({"programs": {"matmul": {"kind": "phase_matmul"}}})
    with pytest.raises(ModelingError, match="source_plan_binding_required"):
        generic.select(call.ops[0])
    # Keeping the source binding and adding a generic override still runs the source.
    generic.recipes.update(call.recipes.recipes)
    assert generic.select(call.ops[0]) is call.recipes.select(call.ops[0])


def test_analytical_callback_remains_available_without_source_tiling():
    source = declaration()
    source["builder"] = "quantized_grouped"
    del source["arguments"]["tiling"]
    call = build_call(source)
    assert not call.ops[0].plan_required
    assert call.ledger["identified_historical_compiled_kernel"] is None
    assert call.ledger["matrix"]["core_recipe"] == "mapped_matmul"


def test_source_tiling_192_retains_logical_work_with_declared_inner_divisor_mapping():
    raw = declaration()
    raw["arguments"]["tiling"].update(base_m=192, base_n=192)
    call = build_call(raw)
    case = json.loads((ROOT / "examples/calls/quantized-grouped-small.json").read_text())
    report = call_costs(case, call.ops, recipes=call.recipes)
    assert report["execution"]["work"]["flops"] == 2 * 6 * 256 * 64
    assert report["execution"]["serviced"]["fixpipe@r0:s0"] == 8192
    assert call.ledger["matrix"]["inner_mapping"]["compute_m"] == 64
    assert call.ledger["matrix"]["inner_mapping"]["compute_n"] == 64
    assert not any(report["execution"]["live_allocated"].values())
