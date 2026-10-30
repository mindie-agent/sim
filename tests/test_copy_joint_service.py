"""Copy joint capacity retains bytes, owner counts and inclusive call timing."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.source_program_service import SourceProgramService

ROOT = Path(__file__).resolve().parents[1]
PACKET = json.loads((ROOT / "data/observations/a5-copy-joint-service-20261030/parameters.json").read_text())


def case(scale=1):
    raw = json.loads((ROOT / "examples/source-families/glm-routing-state-control.json").read_text())
    arch = deepcopy(PACKET["compute_arch"])
    for service in arch["instruction_services"].values():
        service["per_owner_rate"] *= scale
    raw["machine"].setdefault("device", {})["compute_arch"] = arch
    raw["runtime"]["kernels"]["lowerings"]["conv_state_copy"]["options"][
        "joint_program_service"] = PACKET["joint_program_service"]
    return raw


def copy(raw, batch=16, direction="gather"):
    return next(op for op in target_graphs(raw, scope="body", phase="decode", tokens=4,
        context=4096, sequences=batch)[0] if op.family == "conv_state_copy"
        and dict(op.metadata)["direction"] == direction)


@pytest.mark.parametrize("direction", ("gather", "scatter"))
def test_joint_busy_supply_scans_do_not_change_source_counts_or_bytes(direction):
    amounts = []
    for scale in (.5, 1, 2):
        raw = case(scale)
        op = copy(raw, direction=direction)
        row = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
        info = row["operator"]["metadata"]["program_execution"]
        expected = max(p["owner_busy_ns"] + 18 * p["program_busy_ns"]
                       for p in PACKET["joint_program_service"]["directions"][direction].values())
        assert row["duration_ms"] == pytest.approx(expected / 1e6 / scale)
        assert sum(info["source_programs_per_owner"]) == info["source_programs"] == 16 * 72
        assert info["metadata_loads"] == 3 * 16 * 72
        assert info["payload_read_bytes"] == info["payload_write_bytes"] == 16 * 6 * 3072 * 2
        assert not row["service_before_input"]
        assert not row["live_tasks"] and not any(row["remaining_buffers"].values())
        assert max(map(abs, row["service_delta"].values()), default=0) < 1e-6
        amounts.append(row["declared_service"])
    assert amounts[0] == amounts[1] == amounts[2]


def test_whole_call_response_is_inclusive_and_is_not_added_to_pipe_busy():
    raw = case()
    op = copy(raw)
    raw["runtime"]["kernels"]["service_regimes"] = [{
        "id": "inclusive-copy-control", "match": {"recipe": op.recipe},
        "response": {"fixed_us": 20}}]
    row = operator_costs(raw, (op,))["operators"][0]
    assert row["duration_ms"] == pytest.approx(.020)


def test_joint_busy_owners_overlap_but_same_owner_contends_without_idle_borrowing():
    joint = SourceProgramService(PACKET["joint_program_service"]["directions"]["gather"])
    roles = tuple(joint.work(2).resources)
    rates = {r: 64e6 for r in roles}
    units = {r: 64 for r in roles}
    context = RecipeContext(None, {}, {r: r for r in roles}, rates, 32, resource_units=units)
    expected = max(joint.work(2).resources.values()) / 1e6
    for owners, multiplier in (((0,), 1), ((0, 1), 1), ((0, 0), 2)):
        engine = Engine([Resource(r, rates[r], units=64) for r in roles])
        ready = engine.submit("late", latency=7)
        done = [KernelProgram((Stage("reserve"), joint.stage(context, 2, owner, 64))).start(
            engine, f"call{i}", ready) for i, owner in enumerate(owners)]
        engine.advance(until=tuple(done), first=False)
        assert max(e.time for e in done) - 7 == pytest.approx(expected * multiplier)


def test_unmatched_state_shape_keeps_analytic_work_and_reports_joint_domain():
    raw = case()
    from dataclasses import replace
    op = copy(raw)
    op = replace(op, local_shape=(16, 6, 3071))
    row = operator_costs(raw, (op,))["operators"][0]
    info = row["operator"]["metadata"]["program_execution"]
    assert not info["joint_pipe_busy_ns"]
    assert info["joint_program_service_basis"] == "analytic_outside_joint_program_contract"
    assert info["payload_read_bytes"] == 16 * 6 * 3071 * 2
