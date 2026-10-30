"""Ordinary body execution selects the supported caller and preserves opt-outs."""
from copy import deepcopy

import pytest

from mindie_sim import operator_costs, predict, target_operators
from mindie_sim.prediction.body_defaults import attention_defaults
from mindie_sim.prediction.structure import parse_case


def case(sku):
    return {"machine": {"backend": "ascend", "sku": sku, "num_devices": 8},
            "model": {"preset": "qwen3.6-27b", "quantization": "w8a8", "layers": 1},
            "runtime": {"tp": 8},
            "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]}}


@pytest.mark.parametrize("sku", ("Ascend910_9362", "Ascend950DT_9582"))
@pytest.mark.parametrize("query", (1, 4))
def test_plain_body_executes_caller_copies_and_separate_source_programs(sku, query):
    raw = case(sku)
    before = deepcopy(raw)
    result = predict(raw, scope="body", tokens=query, context=128, sequences=2)
    recipes = [row["recipe"] for row in result["timeline"]]
    assert recipes.count("ascend_causal_conv1d_update") == 1
    assert recipes.count("ascend_gdn_decode_gating") == 1
    assert recipes.count("decode_q_l2norm") == recipes.count("decode_k_l2norm") == 1
    assert sum(row["family"] == "tensor_copy" for row in result["timeline"]) == 9
    assert result["execution"]["serviced"]["mte2_aiv@r0:s0"] > 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert result["default_mechanisms"]["selections"][0]["selection"] == "default_installed"
    assert raw == before


def test_explicit_program_and_unsupported_precision_keep_their_existing_execution():
    raw = case("Ascend950DT_9582")
    raw["runtime"]["kernels"] = {"families": {"gdn_gating": {"kind": "envelope"}}}
    explicit = predict(raw, scope="body", tokens=4, context=128)
    assert not any(row["recipe"] == "ascend_gdn_decode_gating" for row in explicit["timeline"])
    assert explicit["default_mechanisms"]["selections"][0]["reason"] == "explicit_implementation_preserved"
    raw = case("Ascend950DT_9582")
    raw["model"]["activation_dtype"] = "fp16"
    other = predict(raw, scope="body", tokens=4, context=128)
    assert other["body_forward_ms"] > 0
    assert other["default_mechanisms"]["selections"][0]["reason"] == "gdn_requires_bf16_inputs_fp32_state"


def test_prefill_does_not_install_a_decode_caller_and_explicit_options_retain_priority():
    raw = case("Ascend950DT_9582")
    parsed, report = attention_defaults(parse_case(raw), "prefill", tokens=4)
    assert "gdn_decode" not in parsed.runtime.kernels.get("lowerings", {})
    assert report["selections"][0]["reason"] == "decode_profile_only"
    binding = {"implementation": "vllm_ascend_gdn_decode", "options": {"gating_programs": 8}}
    raw["runtime"]["kernels"] = {"lowerings": {"gdn_decode": binding}}
    selected, report = attention_defaults(parse_case(raw), "decode", tokens=4)
    options = selected.runtime.kernels["lowerings"]["gdn_decode"]["options"]
    assert options["gating_programs"] == 8
    assert "recurrent_implementation" not in options
    assert options["gating_beta_store"]["from_hardware"] is True
    assert binding == {"implementation": "vllm_ascend_gdn_decode", "options": {"gating_programs": 8}}
    assert report["selections"][0]["reason"] == "explicit_implementation_preserved"


def test_explicit_per_request_recurrence_survives_body_caller_selection():
    raw = case("Ascend950DT_9582")
    raw["runtime"]["kernels"] = {"recipes": {"recurrent_scan0": {"kind": "envelope"}}}
    result = predict(raw, scope="body", tokens=4, context=128, sequences=2)
    recipes = [row["recipe"] for row in result["timeline"]]
    assert "recurrent_scan0" in recipes
    assert "ascend_gdn_decode_gating" not in recipes
    assert result["default_mechanisms"]["selections"][0]["reason"] == "explicit_implementation_preserved"
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_a5_body_defaults_execute_source_recurrence_and_identified_store_queue():
    raw = case("Ascend950DT_9582")
    ops = target_operators(raw, scope="body", phase="decode", tokens=4, sequences=12, context=128)
    gated, = [op for op in ops if op.recipe == "ascend_gdn_decode_gating"]
    recurrent, = [op for op in ops if op.recipe == "ascend_recurrent_gdn_fp32_state"]
    assert dict(recurrent.metadata)["recurrence_selection"] == "arch35_fp32"
    first = operator_costs(raw, ops)["operators"]
    baseline, = [row for row in first if row["operator"]["op_id"] == gated.op_id]
    slower = deepcopy(raw)
    rate = dict(parse_case(raw).machine.dies[0].rates)["scalar_store_issue"]
    slower["machine"]["device"] = {"rates": {"scalar_store_issue": rate / 100}}
    changed = operator_costs(slower, ops)["operators"]
    slow, = [row for row in changed if row["operator"]["op_id"] == gated.op_id]
    assert slow["duration_ms"] > baseline["duration_ms"]
    assert slow["serviced"] == pytest.approx(baseline["serviced"], rel=1e-12)
    report = predict(raw, scope="body", tokens=4, sequences=12, context=128)
    assert report["execution"]["serviced"]["scalar_store_issue@r0:s0"] > 0
    assert sum(row["family"] == "allreduce" for row in report["timeline"]) >= 2
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_source_recurrence_domain_and_primitive_do_not_transfer_to_unmeasured_hardware():
    a3, _ = attention_defaults(parse_case(case("Ascend910_9362")), "decode", tokens=4)
    assert not a3.runtime.kernels["lowerings"]["gdn_decode"]["options"]
    wider, _ = attention_defaults(parse_case(case("Ascend950DT_9582")), "decode", tokens=9)
    assert "recurrent_implementation" not in wider.runtime.kernels["lowerings"]["gdn_decode"]["options"]
    from mindie_sim.schema.hardware import Hardware
    hardware = parse_case(case("Ascend950DT_9582")).machine
    assert Hardware.from_mapping(hardware.to_dict()).dies == hardware.dies


def test_reused_target_uses_current_hardware_store_response_without_changing_work():
    raw = case("Ascend950DT_9582")
    ops = target_operators(raw, scope="body", phase="decode", tokens=4, sequences=2, context=128)
    changed = deepcopy(raw)
    changed["machine"]["device"] = {"store_queues": {"scalar_store_issue": {"response_ms": .001}}}
    fresh = target_operators(changed, scope="body", phase="decode", tokens=4, sequences=2, context=128)
    def gating_cost(hardware, target):
        return next(row for row in operator_costs(hardware, target)["operators"]
                    if row["operator"]["recipe"] == "ascend_gdn_decode_gating")
    prior, reused, rebuilt = gating_cost(raw, ops), gating_cost(changed, ops), gating_cost(changed, fresh)
    assert reused["duration_ms"] == rebuilt["duration_ms"] > prior["duration_ms"]
    assert reused["serviced"] == pytest.approx(prior["serviced"], rel=1e-12)
    assert dict(dict(parse_case(changed).machine.dies[0].store_queues)["scalar_store_issue"])["outstanding"] == 8
    changed["machine"]["device"]["work_units"] = {"scalar_store_issue": "bytes"}
    with pytest.raises(ValueError, match="native_work_unit_conflict:scalar_store_issue"):
        parse_case(changed)
