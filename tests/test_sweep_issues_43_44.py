"""Regression tests for topology resize (#44) and streaming sweeps (#43).

[Role] Exercise public symbol compatibility, sanitized transfer-channel
  projection, deterministic streamed identity, checkpoint repair, static
  pre-graph rejection, replica-equivalent DP scaling and bounded timeouts.
[Invariants] Every assertion is behavioral: die counts, link semantics,
  candidate identity, stages and exact DP scaling.
[Timeouts & Locks] Tier-0 in-process cases only; no device or network access.
[Side-Effect Policy] Writes only under pytest's ``tmp_path``.
"""
from __future__ import annotations

from copy import deepcopy
import json
import time

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.prediction.structure import parse_case
from mindie_sim.sweep import (
    enumerate_parallel_candidates,
    reconstruct_sweep_summary,
    solve_hardware_sla_throughput,
    solve_sla_sizing,
    sweep_topologies,
    sweep_topologies_iter,
    sweep_topology_grid,
    sweep_topology_grid_iter,
)
from mindie_sim.sweep_resize import materialize_machine_for_devices


def _case(model="deepseek-v3.2", hardware="a2", tp=2, dp=2, ep=2,
          layers=4, **extra):
    return compose_case(model, hardware=hardware, tp=tp, dp=dp, ep=ep,
                        layers=layers, **extra)


@pytest.mark.parametrize("sku", ["Ascend950DT_9582", "Ascend950PR_9579"])
def test_a5_resize_preserves_declared_link_semantics(sku):
    case = _case("qwen3.8-flash-next", sku, tp=4, dp=1, ep=4, layers=4,
                 num_devices=4)
    parsed = parse_case(case)
    for target in (1, 2, 8):
        machine = materialize_machine_for_devices(parsed, case, target)
        assert len(machine.dies) == target
        if target == 1:
            assert machine.links == ()
            continue
        intra = [link for link in machine.links
                 if link.channel == "d2d_intra_node"]
        assert intra and all(link.semantics == "per_pair" for link in intra)
        assert intra[0].capacity_bytes_per_ms == pytest.approx(47.5e6)
        assert intra[0].effective_peak_capacity_bytes_per_ms == pytest.approx(56.0e6)
        assert intra[0].bw_utilization == pytest.approx(47.5 / 56.0)
        assert intra[0].latency_us == pytest.approx(12.0)
        if target > machine.ranks_per_node:
            inter = [link for link in machine.links
                     if link.channel == "d2d_inter_node"]
            assert inter and all(link.semantics == "per_port" for link in inter)
            assert inter[0].capacity_bytes_per_ms == pytest.approx(50.0e6)


@pytest.mark.parametrize("sku", ["Ascend910B1", "Ascend910_9362"])
def test_a2_a3_resize_preserves_per_port_semantics(sku):
    case = _case("deepseek-v3.2", sku, tp=2, dp=2, ep=2, layers=4,
                 num_devices=4)
    parsed = parse_case(case)
    machine = materialize_machine_for_devices(parsed, case, 8)
    assert len(machine.dies) == 8
    intra = [link for link in machine.links if link.channel == "d2d_intra_node"]
    assert intra and all(link.semantics == "per_port" for link in intra)


def test_resize_8_to_2_4_16_64_and_legacy_channel_roundtrip():
    case = _case("qwen3.8-flash-next", "Ascend950DT_9582", tp=4, dp=1,
                 ep=4, layers=4, num_devices=4, ranks_per_node=4)
    parsed = parse_case(case)
    for target in (2, 4, 16, 64):
        machine = materialize_machine_for_devices(parsed, case, target)
        assert len(machine.dies) == target
    legacy = deepcopy(case)
    channels = legacy["machine"]["transfer_channels"]
    channels["d2d_intra_node"] = dict(channels["d2d_intra_node"])
    channels["d2d_intra_node"].update({
        "semantics": "per_port", "channel_semantics": "unknown",
        "effective_uni_bw_gbps": 47.5,
    })
    channels["d2d_intra_package"] = dict(channels["d2d_intra_package"])
    channels["d2d_intra_package"].update({
        "channel_semantics": "none", "effective_uni_bw_gbps": None,
    })
    legacy_parsed = parse_case(legacy)
    for target in (1, 2, 8):
        machine = materialize_machine_for_devices(
            legacy_parsed, legacy, target)
        assert len(machine.dies) == target
        intra = [link for link in machine.links
                 if link.channel == "d2d_intra_node"]
        if intra:
            assert intra[0].semantics == "per_port"


def test_issue_43_public_symbols_and_stream_identity():
    import mindie_sim
    for name in ("sweep_topologies", "sweep_topology_grid",
                 "sweep_topologies_iter", "sweep_topology_grid_iter",
                 "reconstruct_sweep_summary", "solve_sla_sizing",
                 "solve_hardware_sla_throughput"):
        assert hasattr(mindie_sim, name)
    assert solve_hardware_sla_throughput is solve_sla_sizing
    case = _case()
    candidates = [{"dp": 2, "tp": 2, "ep": 2},
                  {"dp": 4, "tp": 1, "ep": 4}]
    rows = list(sweep_topologies_iter(
        case, candidates=candidates, num_devices=4, sla_ms=50.0,
        context=512, phase="decode"))
    assert all(len(row["candidate_id"]) == 32 for row in rows)
    assert len({row["candidate_id"] for row in rows}) == len(rows)
    assert reconstruct_sweep_summary(rows) == sweep_topologies(
        case, candidates=candidates, num_devices=4, sla_ms=50.0,
        context=512, phase="decode")


def test_issue_43_resume_repairs_truncated_tail(tmp_path):
    case = _case()
    candidates = [{"dp": 2, "tp": 2, "ep": 2},
                  {"dp": 4, "tp": 1, "ep": 4}]
    path = tmp_path / "sweep.jsonl"
    first = list(sweep_topologies_iter(
        case, candidates=candidates, num_devices=4, sla_ms=50.0,
        context=512, output_jsonl=path))
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"candidate_id": "truncated')
    resumed = list(sweep_topologies_iter(
        case, candidates=candidates, num_devices=4, sla_ms=50.0,
        context=512, output_jsonl=path, resume=True))
    assert len(resumed) == len(first)
    assert all(row.get("resumed_from_checkpoint") is True for row in resumed)
    assert [row["candidate_id"] for row in resumed] == [
        row["candidate_id"] for row in first]
    actual = reconstruct_sweep_summary(resumed)["candidates"]
    for row in actual:
        row.pop("resumed_from_checkpoint", None)
    assert actual == reconstruct_sweep_summary(first)["candidates"]


def test_static_validation_rejects_without_graph(monkeypatch):
    from mindie_sim.prediction import efficiency
    case = _case("qwen3.5-35b-a3b", "a2", tp=2, dp=2, ep=2, layers=4)
    candidates = [
        {"dp": 3, "tp": 2, "ep": 2},          # world-size mismatch
        {"dp": 1, "tp": 3, "ep": 1},          # head indivisibility
        {"dp": 1, "tp": 2, "ep": 3},          # EP does not divide lane plane
        {"dp": 1, "tp": 4, "ep": 4, "cp": 8, "dcp": 8},  # DCP not divisor
    ]
    calls = []
    monkeypatch.setattr(
        efficiency, "target_graphs",
        lambda *args, **kwargs: calls.append(1) or (_ for _ in ()).throw(
            AssertionError("graph lowered for a statically invalid candidate")))
    rows = list(sweep_topologies_iter(
        case, candidates=candidates, num_devices=4, sla_ms=50.0,
        context=512, phase="decode"))
    assert calls == []
    assert len(rows) == len(candidates)
    for row in rows:
        assert row["evaluation_stage"] == "static_validation"
        assert row["feasible"] is False
        assert row["rejection_reason"]


def test_capacity_precheck_rejects_before_memory(monkeypatch):
    from mindie_sim import sweep_engine
    case = _case()
    monkeypatch.setattr(sweep_engine, "stage_weight_bytes",
                        lambda *args, **kwargs: [1.0e30])
    rows = list(sweep_topologies_iter(
        case, candidates=[{"dp": 2, "tp": 2, "ep": 2}],
        num_devices=4, sla_ms=50.0, context=512))
    assert len(rows) == 1
    assert rows[0]["evaluation_stage"] == "static_validation"
    assert rows[0]["feasibility_status"] == "capacity_infeasible"
    assert rows[0]["rejection_reason"].startswith("capacity_exceeded:")


def test_replica_equivalence_scales_dp_exactly():
    case = _case("deepseek-v3.2", "a2", tp=2, dp=2, ep=2, layers=4)
    report = sweep_topologies(
        case, num_devices_grid=(2, 4, 8), sla_ms=50.0, context=512,
        phase="decode", allow_ep=True)
    assert report["cache_stats"]["model_spec_reuses"] > 0
    assert report["cache_stats"]["replica_equivalence_hits"] > 0
    rows = [row for row in report["candidates"]
            if row["parallel"].get("tp") == 2
            and row["parallel"].get("ep") == 2 and row["feasible"]]
    by_dp = {row["parallel"]["dp"]: row for row in rows}
    assert by_dp[1]["step_ms_at_b1"] == pytest.approx(
        by_dp[2]["step_ms_at_b1"], rel=1e-12)
    assert by_dp[1]["step_ms_at_b1"] == pytest.approx(
        by_dp[4]["step_ms_at_b1"], rel=1e-12)
    assert by_dp[2]["tps_feasible"] == pytest.approx(
        2.0 * by_dp[1]["tps_feasible"], rel=1e-12)
    assert by_dp[4]["tps_feasible"] == pytest.approx(
        4.0 * by_dp[1]["tps_feasible"], rel=1e-12)


def test_two_stage_screening_marks_estimates_and_upgrades_winners():
    case = _case()
    candidates = [{"dp": 2, "tp": 2, "ep": 2},
                  {"dp": 4, "tp": 1, "ep": 4}]
    rows = list(sweep_topologies_iter(
        case, candidates=candidates, num_devices=4, sla_ms=50.0,
        context=512, phase="decode", two_stage=True))
    screened = [row for row in rows if row["evaluation_stage"] == "b1_screen"]
    full = [row for row in rows if row["evaluation_stage"] == "full_sla"]
    assert screened and all(row["is_screening_estimate"] is True
                            for row in screened)
    assert full and all(row["is_screening_estimate"] is False for row in full)
    report = reconstruct_sweep_summary(rows)
    assert report["screening_summary"]["two_stage"] is True


def test_candidate_timeout_and_outer_deadline(monkeypatch):
    case = _case()
    candidates = [{"dp": 2, "tp": 2, "ep": 2},
                  {"dp": 4, "tp": 1, "ep": 4}]
    from mindie_sim import sweep as sweep_module

    real_full = sweep_module.full_row
    calls = {"n": 0}

    def slow_first(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("candidate_timeout")
        return real_full(*args, **kwargs)

    monkeypatch.setattr(sweep_module, "full_row", slow_first)
    rows = list(sweep_topologies_iter(
        case, candidates=candidates, num_devices=4, sla_ms=50.0,
        context=512, candidate_timeout_s=5.0))
    assert any(row["status"] == "timeout" for row in rows)
    assert any(row.get("evaluation_stage") != "timeout" for row in rows)
    with pytest.raises(TimeoutError):
        list(sweep_topologies_iter(
            case, candidates=candidates, num_devices=4, sla_ms=50.0,
            context=512, deadline_monotonic=time.monotonic() - 1.0))
