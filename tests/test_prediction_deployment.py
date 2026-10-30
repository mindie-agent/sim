"""Deployment oracles independent of transformer timing estimates."""
from copy import deepcopy

import pytest

from mindie_sim import Case, predict
from mindie_sim.prediction.capacity import Memory
from test_prediction_system import case, linear_work


def pd_case(requests=None):
    raw = case(prompt=3, output=3, requests=requests)
    raw["pools"] = {
        "prefill": {"machine": deepcopy(raw["machine"]), "runtime": deepcopy(raw["runtime"])},
        "decode": {"machine": deepcopy(raw["machine"]), "runtime": deepcopy(raw["runtime"])},
    }
    raw["connector"] = {"bytes_per_s": 128000, "latency_us": 0}
    return raw


def test_pd_handoff_delays_decode_and_conserves_bytes():
    result = predict(pd_case(), lower=linear_work)
    row = result["requests"][0]
    assert row["ttft_ms"] == 3
    # 2 layers * 2 KV planes * 2 heads * 8 elements * 2 bytes * 3 tokens.
    assert result["execution"]["handoff_bytes"] == 384
    assert row["finish_ms"] == 8
    assert row["tpot_ms"] == 2.5
    assert all(v == 0 for pool in result["memory"].values()
               for stage in pool["live_cache_bytes_per_replica_stage"] for v in stage)


def test_pd_pools_overlap_and_destination_admission_applies_backpressure():
    value = pd_case([{"id": key, "prompt_tokens": 3, "output_tokens": 3} for key in ("a", "b")])
    value["pools"]["prefill"]["runtime"]["max_num_batched_tokens"] = 3
    value["pools"]["decode"]["runtime"]["max_num_seqs"] = 1
    result = predict(value, lower=linear_work)
    assert [row["first_token_abs_ms"] for row in result["requests"]] == [3, 6]
    assert [row["finish_ms"] for row in result["requests"]] == [8, 13]
    assert result["execution"]["serviced"]["pd_link"] == 768
    assert result["metrics"]["output_tps"] == pytest.approx(6000 / 13)


def test_async_scheduler_preparation_overlaps_actual_device_work():
    raw = case(prompt=4, output=3)
    raw["runtime"]["submission"] = {"scheduler_ms": .5}
    synchronous = predict(raw, lower=linear_work)
    raw["runtime"]["async_scheduling"] = True
    asynchronous = predict(raw, lower=linear_work)
    assert synchronous["metrics"]["makespan_ms"] == 7.5
    assert asynchronous["metrics"]["makespan_ms"] == 6.5
    assert asynchronous["execution"]["work"] == synchronous["execution"]["work"]
    assert asynchronous["execution"]["live_tasks"] == asynchronous["execution"]["live_programs"] == 0


def test_source_cache_pressure_waits_for_inflight_handoff_before_failing_admission():
    value = pd_case([{"id": key, "prompt_tokens": 3, "output_tokens": 3} for key in ("a", "b")])
    pool = value["pools"]["prefill"]
    pool["runtime"].update(kv_block_size=1, max_num_batched_tokens=3,
                           gpu_memory_utilization=1, memory={"derive_workspace": False})
    parsed = Case.from_mapping({**pool, "model": value["model"], "workload": value["workload"]})
    pool["machine"]["device"]["memory_bytes"] = int(Memory(parsed).weights[0] + 4 * 128)
    result = predict(value, lower=linear_work)
    assert result["status"] == "modeled"
    assert [r["first_token_abs_ms"] for r in result["requests"]] == [3, 9]
    assert [r["finish_ms"] for r in result["requests"]] == [8, 14]


def test_pd_handoff_services_both_memories_even_on_an_uncontended_fast_link():
    value = pd_case()
    value["connector"]["bytes_per_s"] = 1e12
    for pool in value["pools"].values():
        pool["machine"]["device"]["rates"]["memory"] = 128000
    result = predict(value, lower=linear_work)
    assert result["requests"][0]["finish_ms"] == 8
    assert result["execution"]["serviced"]["prefill:memory@r0:s0"] == 384
    assert result["execution"]["serviced"]["decode:memory@r0:s0"] == 384
