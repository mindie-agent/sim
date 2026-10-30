"""Full-call argument binding and source-derived device work controls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim.observations.gdn_binding import bind_observation
from mindie_sim.observations.historical import call_observations
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.call_gdn import make_call, _gating_plan
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operators import native_causal_conv_prefill as convolution
from mindie_sim.prediction.verification import verify_operator
from tools.run_operator_cases import control_case


def observed_calls():
    protocol = json.loads(Path("tests/fixtures/observations/gdn-call-protocol.json").read_text())
    return {row["identity"]["conditions"]["role"]: row for row in call_observations(protocol)}


def test_historical_fp32_gating_has_two_fp32_outputs_and_ignores_timing_values():
    observed = observed_calls()["gating"]
    first = bind_observation(observed)
    changed = deepcopy(observed)
    changed["samples"] = [{"duration_ms": 1e12}]
    changed["statistics"] = {"sample_median_ms": 1e12}
    changed["canary"] = {"ok": False}
    second = bind_observation(changed)
    assert first["status"] == second["status"] == "bound"
    assert first["call"].ops == second["call"].ops
    assert first["call"].ledger == second["call"].ledger
    assert first["call"].ledger["outputs"] == {
        "g": {"shape": [1, 4096, 48], "dtype": "fp32"},
        "beta": {"shape": [1, 4096, 48], "dtype": "fp32"},
    }
    assert first["observation_boundary"] == "device_event_operator_call"
    changed["identity"]["conditions"]["dtype"] = "bf16"
    assert bind_observation(changed)["status"] == "semantic_mismatch"


def test_masked_gating_launch_keeps_parameter_loads_and_padded_arithmetic():
    call = make_call("gating", {"rows": 1, "heads": 9})
    result = verify_operator(control_case(units=2, split_copy=True),
                             call.ops[0], _gating_plan, input_ready_ms=3)
    assert result["status"] == "passed"
    # Four AIVs all execute two8-head column tiles. Only one row and nine
    # heads read a/b or write outputs; every AIV reads the head parameters.
    work = result["result"]["operator"]
    assert work["mte2_bytes"] == 1 * 9 * 8 + 4 * 9 * 8
    assert work["mte3_bytes"] == 1 * 9 * 8
    assert work["hbm_bytes"] == work["mte2_bytes"] + work["mte3_bytes"]
    assert work["vector_ops"] == 4 * 2 * (13 * 64 * 8 + 2 * 8)
    assert result["input_dependency_obeyed"]
    assert result["buffers_released"]


def test_gating_call_graph_matches_independent_operator_work_without_measurements():
    call = make_call("gating", {"rows": 129, "heads": 17, "phase": "decode"})
    raw = control_case(units=2, split_copy=True)
    graph = graph_costs(raw, call.ops, recipes=call.recipes)
    assert graph["boundary"] == "device_graph"
    assert graph["operator_occurrences"] == 1
    row = graph["isolated"]["operators"][0]
    assert row["declared_service"] == pytest.approx(row["serviced"])
    assert graph["duration_ms"] == pytest.approx(row["duration_ms"])
    assert graph["parameters"]["calibration"] == "none"
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0


def test_historical_conv_call_retains_bias_device_flags_and_real_state_rows():
    bound = bind_observation(observed_calls()["conv_prefill"])
    assert bound["status"] == "bound"
    op = bound["call"].ops[0]
    metadata = dict(op.metadata)
    assert metadata["bias"] is True
    assert metadata["has_initial_state"] == (False,)
    assert metadata["cache_indices"] == (1,)
    assert metadata["state_rows"] == 2
    assert metadata["state_history_slots"] == 3
    tiling = convolution.host_tiling(op)
    assert tiling["has_init_state_workspace"] == 1
    # The full call's declared tensors may be small as well; no measurement
    # presence or original model label participates in this execution.
    call = make_call("conv_prefill", {"batch": 2, "tokens": 3, "conv_width": 16, "kernel": 4})
    graph = graph_costs(control_case(units=2), call.ops, recipes=call.recipes)
    assert graph["operator_occurrences"] == 1
    row = graph["isolated"]["operators"][0]
    assert row["declared_service"] == pytest.approx(row["serviced"])
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0


def test_other_wrappers_remain_visible_instead_of_becoming_kernel_only_pairs():
    rows = observed_calls()
    for role in ("scan_decode", "conv_update"):
        bound = bind_observation(rows[role])
        assert bound["status"] == "semantic_mismatch"
        assert bound["call"] is None
        assert bound["mismatches"]
        assert bound["observation_boundary"] == rows[role]["boundary"]
    # The standalone scan prefill now has its own complete-call composite, but
    # the original reference failure remains on its source observation.
    assert bind_observation(rows["scan_prefill"])["status"] == "bound"
    assert rows["scan_prefill"]["canary"]["ok"] is False


def test_modified_producer_cannot_claim_reviewed_binding_or_gate_standalone_simulation():
    observed = deepcopy(observed_calls()["gating"])
    observed["identity"]["producer_sha256"] = "modified-producer"
    binding = bind_observation(observed)
    assert binding["status"] == "semantic_mismatch"
    assert binding["call"] is None
    assert "producer source" in binding["mismatches"][0]
    call = make_call("gating", {"rows": 1, "heads": 9})
    raw = control_case(units=2, split_copy=True)
    raw["runtime"]["submission"] = {"scheduler_ms": 1000, "completion_ms": 1000}
    report = call_costs(raw, call.ops, recipes=call.recipes)
    control = call_costs(control_case(units=2, split_copy=True), call.ops, recipes=call.recipes)
    assert report["duration_ms"] == pytest.approx(control["duration_ms"])
    assert report["boundary"] == "device_graph"
    assert report["parameters"]["calibration"] == "none"
    assert report["single_call"]["expanded_operators"] == 1
