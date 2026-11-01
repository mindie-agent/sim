"""Tier-0 tests for HBM capacity and async_scheduling delayed-release memory modeling.

[Role] Verify closed-world HBM accounting for async_scheduling=True vs False across
  Memory, Scheduler discrete-event execution, and SLA concurrency solving.
[Invariants] async_scheduling=False preserves synchronous workspace and immediate
  KV release; async_scheduling=True accounts for ping-pong step workspace,
  in-flight lookahead token reservation, and 1-step delayed KV release.
[Timeouts & Locks] Pure in-memory Tier-0 tests (<5s); no I/O or external processes.
[Side-Effect Policy] Mutates only local test cases.
"""
from __future__ import annotations

import pytest

from mindie_sim.prediction.api import predict
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    _vram_max_sequences_per_dp,
    solve_hardware_sla_throughput,
)
from mindie_sim.prediction.structure import parse_case
from test_prediction_system import case, linear_work


def test_memory_accounts_for_async_workspace_and_delayed_release_policy():
    raw_sync = case(prompt=16, output=8)
    raw_sync["runtime"]["async_scheduling"] = False
    mem_sync = Memory(parse_case(raw_sync), workspace_tokens=16)

    raw_async = case(prompt=16, output=8)
    raw_async["runtime"]["async_scheduling"] = True
    mem_async = Memory(parse_case(raw_async), workspace_tokens=16)

    assert mem_sync.async_delayed_release_steps == 0
    assert mem_sync.async_in_flight_lookahead_steps == 0
    assert mem_sync.async_workspace == [0.0]

    assert mem_async.async_delayed_release_steps == 1
    assert mem_async.async_in_flight_lookahead_steps == 1
    assert mem_async.async_workspace[0] > 0.0
    assert mem_async.workspace[0] == pytest.approx(
        mem_sync.workspace[0] + mem_async.async_workspace[0]
    )
    assert mem_async.capacity[0][0] == pytest.approx(
        mem_sync.capacity[0][0] - mem_async.async_workspace[0]
    )


def test_scheduler_delayed_kv_release_under_async_scheduling_delays_admission_by_one_step():
    # Machine HBM fits at most 12 tokens total (12 * 128 bytes).
    # r1 (prompt=4, output=2) and r2 (prompt=4, output=4) take 4+4=8 tokens at prefill,
    # so r3 (prompt=5, output=2) cannot fit at t=0 (8+5=13 > 12) and waits until r1 finishes.
    requests = [
        {"id": "r1", "prompt_tokens": 4, "output_tokens": 2, "arrival_ms": 0.0},
        {"id": "r2", "prompt_tokens": 4, "output_tokens": 4, "arrival_ms": 0.0},
        {"id": "r3", "prompt_tokens": 5, "output_tokens": 2, "arrival_ms": 0.0},
    ]
    base = case(requests=requests)
    base["runtime"].update(
        kv_block_size=1,
        max_num_seqs=3,
        max_num_batched_tokens=12,
        enable_chunked_prefill=False,
        gpu_memory_utilization=1.0,
        memory={"derive_workspace": False},
    )
    parsed = parse_case(base)
    # 12 tokens * 128 bytes/token = 1536 bytes of KV capacity
    base["machine"]["device"]["memory_bytes"] = int(Memory(parsed).weights[0] + 12 * 128)

    sync_report = predict(base, lower=linear_work)
    assert sync_report["status"] == "modeled"
    sync_r3 = next(r for r in sync_report["requests"] if r["request_id"] == "r3")
    assert sync_report["memory"]["peak_delayed_release_cache_bytes_per_replica_stage"] == [[0.0]]

    async_case = case(requests=requests)
    async_case["runtime"].update(
        kv_block_size=1,
        max_num_seqs=3,
        max_num_batched_tokens=12,
        enable_chunked_prefill=False,
        gpu_memory_utilization=1.0,
        async_scheduling=True,
        memory={"derive_workspace": False},
    )
    async_case["machine"]["device"]["memory_bytes"] = base["machine"]["device"]["memory_bytes"]

    async_report = predict(async_case, lower=linear_work)
    assert async_report["status"] == "modeled"
    async_r3 = next(r for r in async_report["requests"] if r["request_id"] == "r3")

    # Under async_scheduling=True, r1's KV blocks remain pinned during the step immediately
    # following r1's completion while r2 is still active, delaying r3's admission by 1 step.
    assert async_r3["admission_ms"] > sync_r3["admission_ms"]
    assert async_report["memory"]["peak_delayed_release_cache_bytes_per_replica_stage"][0][0] > 0.0
    assert async_report["memory"]["live_cache_bytes_per_replica_stage"] == [[0.0]]


def test_scheduler_stalls_rather_than_failing_when_single_slot_is_pinned_by_delayed_release():
    requests = [
        {"id": "r1", "prompt_tokens": 3, "output_tokens": 2, "arrival_ms": 0.0},
        {"id": "dp1_1", "prompt_tokens": 3, "output_tokens": 2, "arrival_ms": 0.0},
        {"id": "r2", "prompt_tokens": 3, "output_tokens": 2, "arrival_ms": 0.0},
        {"id": "dp1_2", "prompt_tokens": 3, "output_tokens": 2, "arrival_ms": 10.0},
    ]
    raw = case(dp=2, requests=requests)
    raw["runtime"].update(
        kv_block_size=1,
        max_num_seqs=2,
        max_num_batched_tokens=4,
        enable_chunked_prefill=False,
        gpu_memory_utilization=1.0,
        async_scheduling=True,
        submission={"scheduler_ms": 1.0},
        memory={"derive_workspace": False, "async_delayed_release_steps": 3},
    )
    parsed = parse_case(raw)
    raw["machine"]["device"]["memory_bytes"] = int(Memory(parsed).weights[0] + 4 * 128)

    report = predict(raw, lower=linear_work)
    assert report["status"] == "modeled"
    assert [r["status"] for r in report["requests"]] == ["served", "served", "served", "served"]
    r1_finish = next(r for r in report["requests"] if r["request_id"] == "r1")["finish_ms"]
    r2_admit = next(r for r in report["requests"] if r["request_id"] == "r2")["admission_ms"]
    # A 3-step delayed release with scheduler_ms=1.0 advances DP0 by 3.0 ms without waiting for DP1's arrival at 10.0 ms.
    assert r1_finish + 2.0 <= r2_admit < 10.0
    assert report["memory"]["live_cache_bytes_per_replica_stage"] == [[0.0], [0.0]]


def test_speculative_lookahead_does_not_exceed_remaining_request_output():
    from dataclasses import replace
    from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec

    raw = case(prompt=4, output=4)
    raw["runtime"].update(
        kv_block_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=8,
        gpu_memory_utilization=1.0,
        mtp_acceptance_length=3.0,
        metadata={"draft_mode": "mtp", "draft_proposal_tokens": 2, "draft_accept_tokens": 2},
        memory={"derive_workspace": False},
    )
    parsed = parse_case(raw)
    draft = DraftSpec(
        kind="mtp", layers=1, hidden_size=32, intermediate_size=64,
        num_heads=4, num_kv_heads=2, head_dim=8, q_width=32, kv_width=16,
        vocab_size=128, target_shared_embedding=True, target_shared_lm_head=True,
    )
    parsed = replace(parsed, model=replace(parsed.model, auxiliary=AuxiliarySpec(mtp=draft)))
    mem_sync = Memory(parsed)
    # 7 tokens total (prompt=4 + 3 verification tokens on the final step)
    cap_bytes = int(mem_sync.weights[0] + mem_sync.cache_bytes(7)[0])
    parsed_sync = replace(
        parsed,
        machine=replace(
            parsed.machine,
            dies=tuple(replace(d, memory_bytes=cap_bytes) for d in parsed.machine.dies),
        ),
    )

    sync_res = predict(parsed_sync, lower=linear_work)
    assert sync_res["status"] == "modeled"

    parsed_async = replace(parsed_sync, runtime=replace(parsed_sync.runtime, async_scheduling=True))
    async_res = predict(parsed_async, lower=linear_work)
    assert async_res["status"] == "modeled"


def test_analytical_async_vram_bound_accounts_for_full_batch_wave_retention():
    raw = case(requests=[{"id": str(i), "prompt_tokens": 4, "output_tokens": 4} for i in range(8)])
    raw["runtime"].update(
        max_num_seqs=2,
        kv_block_size=1,
        gpu_memory_utilization=1.0,
        async_scheduling=True,
        memory={
            "derive_workspace": False,
            "async_in_flight_lookahead_steps": 0,
            "async_delayed_release_steps": 10,
        },
    )
    mem = Memory(parse_case(raw))
    # 24 tokens capacity (3 requests of 8 tokens): cannot sustain b=2 when delayed_steps=10
    raw["machine"]["device"]["memory_bytes"] = int(mem.weights[0] + 24 * 128)
    parsed = parse_case(raw)
    assert _vram_max_sequences_per_dp(parsed, context=4, tokens_per_seq=1, output_budget=4) == 1


def test_sla_solver_quantifies_concurrency_gain_from_disabling_async_scheduling():
    raw = case(prompt=16, output=8)
    raw["runtime"].update(
        kv_block_size=16,
        max_num_seqs=64,
        max_num_batched_tokens=256,
        gpu_memory_utilization=1.0,
        async_scheduling=True,
    )
    parsed_sync = parse_case({**raw, "runtime": {**raw["runtime"], "async_scheduling": False}})
    mem_sync = Memory(parsed_sync, workspace_tokens=8)
    kv_per_seq = mem_sync.cache_bytes(64 + 16, padded=True)[0]
    # Provision HBM so synchronous mode fits 8 sequences, while async mode (which pins >=1
    # delayed-release sequence + async workspace + lookahead token) fits fewer than 8 sequences.
    raw["machine"]["device"]["memory_bytes"] = int(
        mem_sync.weights[0] + mem_sync.workspace[0] + 8 * kv_per_seq + 64
    )

    b_sync = _vram_max_sequences_per_dp(
        parse_case(raw), context=64, tokens_per_seq=1, output_budget=16, async_scheduling=False
    )
    b_async = _vram_max_sequences_per_dp(
        parse_case(raw), context=64, tokens_per_seq=1, output_budget=16, async_scheduling=True
    )
    assert b_sync == 8
    assert b_async < b_sync

    sla_async = solve_hardware_sla_throughput(
        raw, sla_ms=50.0, context=64, output_budget=16, scenarios=[{"id": "baseline", "label": "baseline"}]
    )
    summary = sla_async["async_scheduling_memory"]
    assert summary["configured_async_scheduling"] is True
    assert summary["b_vram_max_sync"] == b_sync
    assert summary["b_vram_max_async"] == b_async
    assert summary["delta_b_vram_from_disabling_async"] == b_sync - b_async
    assert summary["disabling_async_unlocks_concurrency"] is True

    raw_sync = {**raw, "runtime": {**raw["runtime"], "async_scheduling": False}}
    sla_sync = solve_hardware_sla_throughput(
        raw_sync, sla_ms=50.0, context=64, output_budget=16, scenarios=[{"id": "baseline", "label": "baseline"}]
    )
    assert sla_sync["scenarios"][0]["b_feasible"] > sla_async["scenarios"][0]["b_feasible"]
    assert sla_sync["scenarios"][0]["tps_feasible"] > sla_async["scenarios"][0]["tps_feasible"]
