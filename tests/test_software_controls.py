"""Software and hardware changes must alter the actual execution program."""
from copy import deepcopy

from mindie_sim import predict
from test_prediction_system import case


def test_hardware_rate_change_alters_time_and_conserves_logical_work():
    raw = case(prompt=8, output=3)
    before = predict(raw)
    raw["machine"]["device"]["compute_peaks"]["bf16"] /= 4
    after = predict(raw)
    assert after["metrics"]["makespan_ms"] > before["metrics"]["makespan_ms"]
    assert after["execution"]["work"] == before["execution"]["work"]
    raw = case(prompt=8, output=3)
    raw["machine"]["device"]["rates"]["memory"] /= 4
    slower_memory = predict(raw)
    assert slower_memory["metrics"]["makespan_ms"] > before["metrics"]["makespan_ms"]


def test_full_graph_replay_reduces_submissions_and_preserves_padding_work():
    raw = case(prompt=4, output=3)
    raw["machine"]["device"]["launch_us"] = 1000
    eager = predict(raw)
    raw["runtime"].update(graph_capture_keys=[1, 2, 4], graph_mode="auto")
    graph = predict(raw)
    assert graph["execution"]["work"] == eager["execution"]["work"]
    assert graph["requests"][0]["ttft_ms"] == eager["requests"][0]["ttft_ms"]
    assert graph["requests"][0]["tpot_ms"] < eager["requests"][0]["tpot_ms"]
    assert graph["execution"]["serviced"]["host@r0:s0"] < eager["execution"]["serviced"]["host@r0:s0"]
