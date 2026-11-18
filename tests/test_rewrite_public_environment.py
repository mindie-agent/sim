"""S-owned standalone public environment receipt tests.

[Role] Prove the hardware-only public levels (operator/graph/native call) use
  one real Engine, report actual event boundaries and conserve declared work
  without a model, request or target duration.
[Invariants] ``duration_ms`` is complete-start of the real submission; declared
  hbm/vector work is serviced exactly once; a delayed external input shifts the
  start boundary without inventing service; repeated invocations own one charge
  each.
[Side-Effect Policy] Remote CPU-only virtual engine; no devices or I/O.
"""
from __future__ import annotations

import pytest

#: A5 sustainable DRAM service is declared at 0.8316 of the raw HBM peak.
A5_DRAM_SERVICE_SCALE = 1 / 0.8316

from mindie_sim.backends.ascend import materialize
from mindie_sim.public.api import (
    simulate_graph,
    simulate_native_call,
    simulate_operator,
)
from mindie_sim.public.environment import ExecutionEnvironment
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec

SKU = "Ascend950DT_9582"


def _hardware():
    return Hardware.from_mapping(materialize({"sku": SKU}))


def _op(identifier, *, deps=(), hbm=64.0, vector=64.0, stage=0):
    return OpSpec(
        op_id=identifier, family="vector", phase="decode", stage=stage,
        layer=0, requests=(), semantic_shape=(64,), local_shape=(64,),
        padded_shape=(64,), launch_shape=(64,), dtype="bf16",
        deps=tuple(deps), hbm_bytes=hbm, vector_ops=vector)


def test_public_operator_report_uses_the_submission_boundaries():
    environment = ExecutionEnvironment.from_hardware(_hardware())
    result = simulate_operator(environment, (_op("a"),))
    row = result["operators"][0]
    assert result["schema"] == "mindie-sim.operators/1"
    assert row["ready_ms"] == 0.0
    assert row["complete_ms"] == pytest.approx(row["duration_ms"])
    assert row["serviced"]["memory@r0:s0"] == pytest.approx(64.0 * A5_DRAM_SERVICE_SCALE)
    assert row["serviced"]["vector@r0:s0"] == pytest.approx(64.0)
    assert set(row["serviced"]) == {"memory@r0:s0", "vector@r0:s0"}
    assert result["timeline"][0]["complete_ms"] == pytest.approx(
        row["complete_ms"])
    assert result["operator_sum_ms"] == pytest.approx(row["duration_ms"])
    assert isinstance(row["runtime_uid"], str) and row["runtime_uid"]


def test_public_operator_input_ready_delays_service_not_duration():
    environment = ExecutionEnvironment.from_hardware(_hardware())
    baseline = simulate_operator(environment, (_op("a"),))
    delayed = simulate_operator(environment, (_op("a"),), input_ready_ms=5.0)
    assert delayed["operators"][0]["ready_ms"] == pytest.approx(5.0)
    assert delayed["operators"][0]["complete_ms"] >= 5.0
    assert delayed["operator_sum_ms"] == pytest.approx(
        baseline["operator_sum_ms"])


def test_public_graph_serializes_a_real_dependency_and_conserves_work():
    environment = ExecutionEnvironment.from_hardware(_hardware())
    first = _op("a")
    second = _op("b", deps=("a",))
    result = simulate_graph(environment, (first, second))
    assert result["operator_occurrences"] == 2
    assert result["operator_events"] == 2
    assert result["zero_work_joins"] == []
    assert isinstance(result["runtime_uid"], str) and result["runtime_uid"]
    assert result["duration_ms"] > 0.0
    assert result["complete_ms"] == pytest.approx(
        result["charge_ready_ms"] + result["duration_ms"])
    by_id = {row["physical_id"]: row for row in result["timeline"]}
    assert by_id["a"]["complete_ms"] <= by_id["b"]["ready_ms"]
    assert result["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(128.0 * A5_DRAM_SERVICE_SCALE)
    assert result["execution"]["serviced"]["vector@r0:s0"] == pytest.approx(128.0)


def test_public_graph_external_time_is_the_release_boundary():
    environment = ExecutionEnvironment.from_hardware(_hardware())
    result = simulate_graph(
        environment, (_op("a", deps=("producer",)),),
        external_inputs={"producer": 5.0})
    assert result["charge_ready_ms"] >= 5.0
    assert result["timeline"][0]["ready_ms"] >= 5.0


def test_public_native_call_charges_each_invocation_once():
    environment = ExecutionEnvironment.from_hardware(_hardware())
    single = simulate_native_call(
        environment, (_op("a"),), invocations=1, mode="eager")
    twice = simulate_native_call(
        environment, (_op("a"),), invocations=2, mode="eager")
    assert twice["physical_operators"] == 1
    assert twice["expanded_operators"] == 2
    assert twice["operator_occurrences"] == 2
    assert isinstance(twice["runtime_uid"], str) and twice["runtime_uid"]
    assert twice["duration_ms"] == pytest.approx(
        2 * single["duration_ms"])
    assert len(twice["timeline"]) == 2
    assert twice["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(128.0 * A5_DRAM_SERVICE_SCALE)
    assert twice["execution"]["serviced"]["vector@r0:s0"] == pytest.approx(128.0)
