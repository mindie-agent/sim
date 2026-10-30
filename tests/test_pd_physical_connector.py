"""Public P/D predictions conserve payload, wire and representative HBM work."""
from copy import deepcopy

import pytest

from mindie_sim import predict_pd
from test_prediction_deployment import pd_case
from test_prediction_system import case, linear_work


def physical_case(**link):
    raw = pd_case()
    raw["connector"] = {"link": {"name": "handoff", "capacity_bytes_per_ms": 128,
                                   **link}}
    return raw


@pytest.mark.parametrize("factor", [.5, 1, 2])
def test_public_pd_physical_and_legacy_bandwidth_scan(factor):
    physical = physical_case(capacity_bytes_per_ms=128 * factor)
    legacy = pd_case()
    legacy["connector"]["bytes_per_s"] *= factor
    explicit = predict_pd(physical, lower=linear_work)
    original = predict_pd(legacy, lower=linear_work)
    assert explicit["requests"] == original["requests"]
    assert explicit["requests"][0]["finish_ms"] == 5 + 3 / factor
    assert explicit["execution"]["handoff_bytes"] == 384
    assert explicit["execution"]["connector"]["wire_bytes"] == 384
    assert explicit["execution"]["serviced"]["pd:link:handoff"] == 384


@pytest.mark.parametrize("nested", [False, True])
def test_pd_framing_and_switch_capacity_are_one_physical_transmission(nested):
    link = dict(transaction_bytes=256, alignment_bytes=256, protocol_overhead_bytes=64,
                switch_hops=2, switch_hop_latency_us=500,
                switch_capacity_bytes_per_ms=64)
    raw = physical_case(**link) if nested else pd_case()
    if not nested:
        raw["connector"].update(link)
    result = predict_pd(raw, lower=linear_work)
    # One 384-byte payload occupies two 320-byte transactions. The same wire
    # payload consumes link and switch capacity concurrently, not two sends.
    assert result["requests"][0]["finish_ms"] == 5 + 1 + 640 / 64
    execution = result["execution"]
    assert execution["handoff_bytes"] == 384
    assert execution["connector"]["wire_bytes"] == 640
    assert execution["connector"]["transactions"] == 2
    for pool in ("prefill", "decode"):
        assert execution["serviced"][f"{pool}:memory@r0:s0"] == 384
    transfer = next(event for event in result["events"] if event["task"] == "pd:r:wire")
    assert transfer["start_ms"] == 4
    assert transfer["end_ms"] == 14


def test_pd_full_duplex_endpoint_capacity_is_shared_across_ready_handoffs():
    raw = physical_case(semantics="per_port")
    raw["workload"]["requests"] = [
        {"id": name, "prompt_tokens": 3, "output_tokens": 3}
        for name in ("a", "b")]
    for pool in raw["pools"].values():
        pool["runtime"]["dp"] = 2
        pool["machine"]["num_devices"] = 2
        pool["machine"]["links"][0]["members"] = [0, 1]
    result = predict_pd(raw, lower=linear_work)
    assert [row["finish_ms"] for row in result["requests"]] == [11, 11]
    service = result["execution"]["serviced"]
    assert service["pd:link:handoff:tx:port0"] == 768
    assert service["pd:link:handoff:rx:port1"] == 768
    assert service["pd:link:handoff:rx:port0"] == service["pd:link:handoff:tx:port1"] == 0
    assert result["execution"]["connector"]["wire_bytes"] == 768


def test_pd_destination_layout_keeps_each_representative_stage_memory_once():
    raw = physical_case(transaction_bytes=256, protocol_overhead_bytes=64)
    # Heterogeneous pool layouts preserve the declared representative-memory
    # approximation: prefill TP1/PP1 and decode TP2/PP2 share logical cache size.
    destination = case(tp=2, pp=2)
    raw["pools"]["decode"] = {key: deepcopy(destination[key]) for key in ("machine", "runtime")}
    result = predict_pd(raw, lower=linear_work)
    service = result["execution"]["serviced"]
    assert result["execution"]["handoff_bytes"] == 384
    assert service["prefill:memory@r0:s0"] == 384
    assert service["decode:memory@r0:s0"] == 96
    assert service["decode:memory@r0:s1"] == 96
    assert result["execution"]["connector"]["wire_bytes"] == 640


def test_pd_framed_handoff_waits_for_source_readiness_and_destination_admission():
    raw = physical_case(transaction_bytes=256, protocol_overhead_bytes=64,
                        switch_hops=2, switch_hop_latency_us=500)
    raw["workload"]["requests"] = [
        {"id": name, "prompt_tokens": 3, "output_tokens": 3, "arrival_ms": 7}
        for name in ("a", "b")]
    raw["pools"]["prefill"]["runtime"]["max_num_batched_tokens"] = 3
    raw["pools"]["decode"]["runtime"]["max_num_seqs"] = 1
    result = predict_pd(raw, lower=linear_work)
    assert [row["first_token_abs_ms"] for row in result["requests"]] == [10, 13]
    assert [row["finish_ms"] for row in result["requests"]] == [18, 26]
    transfers = {event["task"]: event for event in result["events"] if event["task"] in {"pd:a:wire", "pd:b:wire"}}
    assert transfers["pd:a:wire"]["start_ms"] == 11
    assert transfers["pd:b:wire"]["start_ms"] == 19
    assert result["execution"]["connector"]["wire_bytes"] == 1280
    assert all(value == 0 for pool in result["memory"].values()
               for stage in pool["live_cache_bytes_per_replica_stage"] for value in stage)
