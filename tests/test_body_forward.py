"""Device-only target forward keeps physical geometry and cohort traffic."""
from copy import deepcopy

import pytest

from mindie_sim import predict, predict_pd, target_graphs
from mindie_sim.__main__ import parser, run
from mindie_sim.schema.common import ModelingError
from test_prediction_system import case
from test_prediction_deployment import pd_case


def test_body_excludes_host_and_step_wait_but_retains_slow_memory():
    raw = case(output=1)
    options = dict(scope="body", tokens=4, context=64, sequences=3)
    base = predict(raw, **options)
    raw["machine"]["device"]["launch_us"] = 20000
    raw["runtime"]["submission"] = {"scheduler_ms": 10, "completion_ms": 10}
    raw["workload"]["requests"][0]["arrival_ms"] = 1000
    assert predict(raw, **options)["body_forward_ms"] == base["body_forward_ms"]
    raw["machine"]["device"]["rates"]["memory"] /= 100
    slower = predict(raw, **options)
    assert slower["body_forward_ms"] > base["body_forward_ms"]
    assert slower["execution"]["work"] == base["execution"]["work"]
    assert all(r["family"] not in {"embedding", "logits", "sampling", "token_commit"}
               for r in base["timeline"])
    assert base["execution"]["submission"]["host_service_ms"] == 0


def test_body_retains_full_capture_padding_and_balanced_domain():
    raw = case()
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[8], graph_capture_mode="full")
    body = target_graphs(raw, phase="decode", tokens=1, scope="body")[0]
    model = target_graphs(raw, phase="decode", tokens=1, scope="model")[0]
    a = next(op for op in body if op.op_id.endswith("norm1"))
    b = next(op for op in model if op.op_id == a.op_id)
    assert a.local_shape == b.local_shape == (8, 32)
    assert a.vector_ops == b.vector_ops == 1280
    report = predict(raw, scope="body")
    assert report["replay_scope"] == "full"
    assert report["execution"]["submission"]["graph_replays"] == 0
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["runtime"]["routing"] = {"policy": "balanced"}
    padded = predict(raw, scope="body")
    assert padded["body_forward_ms"] > 0
    expert = next(op for op in target_graphs(raw, phase="decode", tokens=1, scope="body")[0]
                  if op.family == "grouped_gemm")
    counts = dict(expert.metadata)["routing_realization"]
    assert sum(counts["global_expert_rows"]) == 8 * 2
    assert (counts["valid_local_tokens"], counts["padded_local_tokens"]) == (1, 8)


def test_body_complete_ep_cohort_retains_network_service_and_pp_dependencies():
    raw = case(dp=2, tp=1, ep=2, pp=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["runtime"]["routing"] = {"policy": "balanced"}
    options = dict(scope="body", sequences=4, context=32)
    fast = predict(raw, **options)
    raw["machine"]["links"][0]["capacity_bytes_per_ms"] /= 2
    slow = predict(raw, **options)
    assert fast["geometry"]["participating_replicas"] == [0, 1]
    assert fast["geometry"]["valid_query_tokens"] == 8
    assert fast["execution"]["serviced"]["link:fabric"] > 0
    assert slow["body_forward_ms"] > fast["body_forward_ms"]
    assert slow["execution"]["serviced"] == fast["execution"]["serviced"]
    assert any(r["family"] == "pp_p2p" for r in fast["timeline"])
    assert fast["execution"]["live_tasks"] == fast["execution"]["live_programs"] == 0


def test_pd_body_has_independent_pool_spans_without_handoff_or_acceptance_fit():
    raw = pd_case()
    raw["pools"]["decode"]["machine"]["device"]["rates"]["memory"] /= 100
    result = predict_pd(raw, scope="body", context=32, tokens=4, sequences=2)
    assert result == predict(raw, scope="body", context=32, tokens=4, sequences=2)
    assert result["body_forward_ms"]["decode"] > result["body_forward_ms"]["prefill"]
    modified = deepcopy(raw)
    modified["connector"]["bytes_per_s"] /= 1000
    assert predict_pd(modified, scope="body", context=32, tokens=4,
                      sequences=2)["body_forward_ms"] == result["body_forward_ms"]
    for pool in result["pools"].values():
        assert pool["geometry"]["tokens_per_sequence"] == 4
        assert pool["execution"]["live_tasks"] == pool["execution"]["live_programs"] == 0


def test_predict_cli_exposes_body_geometry():
    args = parser().parse_args(["predict", "examples/quickstart.json", "--scope", "body",
                               "--phase", "decode", "--tokens", "4", "--sequences", "2",
                               "--context", "2048", "--event-limit", "0"])
    result = run(args)
    assert result["body_forward_ms"] > 0
    assert result["geometry"]["valid_query_tokens"] == 8
    assert result["execution"]["submission"]["included"] is False


def test_standalone_draft_is_not_reported_as_a_target_body():
    raw = case()
    raw["model"] = {"preset": "kimi-k3-dspark"}
    with pytest.raises(ModelingError, match="draft_requires_target_model"):
        predict(raw, scope="body")
