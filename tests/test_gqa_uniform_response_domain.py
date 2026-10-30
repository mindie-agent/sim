"""Homogeneous native responses must not swallow ragged GQA batches."""
from copy import deepcopy

from mindie_sim import operator_costs
from mindie_sim.prediction.attention import gqa
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import RequestView
from test_attention_decode import hardware


def operation(raw, requests):
    case = parse_case(raw)
    plan = case.model.layer_plans[0]
    views = tuple(RequestView(str(i), q, k, True, "decode")
                  for i, (q, k) in enumerate(requests))
    ops = gqa(prefix="gqa", plan=plan, views=views, stage=0, deps="ready",
              model=case.model, parallel=case.runtime.parallel, quant=case.model.quant,
              runtime=case.runtime,
              mode="decode", padded_tokens=sum(q for q, _ in requests),
              cache_ids={}, stage_of=lambda _: 0).ops
    return next(op for op in ops if op.recipe == "flash_attention_decode")


def test_native_homogeneous_response_matches_all_requests_or_retains_ragged_work():
    raw = hardware()
    raw["runtime"]["kernels"] = {"recipes": {"flash_attention_decode": {
        "kind": "staged_gqa_decode", "target_cores": 4}}, "service_regimes": [{
        "id": "homogeneous-control",
        "match": {"recipe": "flash_attention_decode",
                  "metadata.gqa_decode.uniform_query_tokens": 4},
        "coordinates": {"context": {"path": "metadata.gqa_decode.uniform_context_tokens"}},
        "domain": {"context": {"min": 32, "max": 32}},
        "response": {"fixed_us": 1000},
        "execution": {"kind": "joint_native", "resource": "native_call_service",
                      "reference_rate": 1e9, "native_kernel": "SyntheticFAI"}}]}
    device = raw["machine"]["device"]
    device["rates"]["native_call_service"] = 1e9
    device["resource_units"]["native_call_service"] = 1
    device["work_units"] = {"native_call_service": "busy_ns"}
    plain = deepcopy(raw)
    plain["runtime"]["kernels"]["service_regimes"] = []
    for requests in ([(4, 32), (4, 32)], [(4, 32), (4, 64)], [(4, 32), (1, 32)]):
        op = operation(raw, requests)
        result = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
        control = operator_costs(plain, (op,), input_ready_ms=7)["operators"][0]
        if requests[0] == requests[1]:
            assert result["duration_ms"] == 1
            assert result["program"] == "JointNativeService"
            assert result["analytical_reference_work"]
        else:
            assert result["duration_ms"] == control["duration_ms"]
            assert result["serviced"] == control["serviced"]
            assert result["program"] != "JointNativeService"
        assert not result["service_before_input"] and result["live_tasks"] == 0
