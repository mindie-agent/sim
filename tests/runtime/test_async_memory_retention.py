"""Physical memory retained by asynchronous consumers and scheduler steps.

[Role] Follow resident KV, host offload and staging through resize, retirement,
  delayed release, scheduling and SLA sizing.
[Invariants] A live consumer owns its physical byte range until completion.
  Admission and peak reports share the bytes of concurrent owners by tier and rank.
[Timeouts & Locks] Scheduler predictions run in daemon threads with a bounded wait.
  Other cases use local in-memory ledgers and analytical calculations.
[Side-Effect Policy] Only local cases, ledgers and prediction results change.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import replace

import pytest

from _support.memory_cases import budget_memory, case, draft_case, linear_work, sparse_mla_case
from mindie_sim.prediction.api import predict
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.efficiency import _vram_max_sequences_per_dp, solve_sla_sizing
from mindie_sim.prediction.structure import parse_case

pytestmark = pytest.mark.tier0

BYTES_PER_KV_ROW = 128
OFFLOAD_POLICY = MemoryPolicy(derive_workspace=False, kvc_ddr_ratio=0.5)
SPARSE_OFFLOAD = {
    "implementation": "vllm_ascend_sparse_kv_offload",
    "options": {"dram_size_per_dp_gb": 0.02, "keep_device_kv_cache": False},
}


def _offloaded_memory():
    return Memory(parse_case(sparse_mla_case(context_tokens=4096, offloading=SPARSE_OFFLOAD)))


def _predict_bounded(parsed):
    result = Future()

    def run():
        try:
            result.set_result(predict(parsed, lower=linear_work))
        except Exception as exc:
            result.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return result.result(timeout=15)


def test_joint_hbm_host_grant_is_atomic_across_domains():
    memory = _offloaded_memory()
    # One domain over budget: no HBM or host grant is committed.
    assert not memory.grant_batch(
        0, [("joint:hbm", 8192)], host_bytes=9_000_000)
    assert memory.live_grants(0, 0) == 0.0
    assert memory.granted_host == [0.0]
    assert memory.grant_bindings == {}
    # Both domains fit: commit both; release clears both.
    assert memory.grant_batch(
        0, [("joint:hbm", 4096)], host_bytes=4096)
    assert memory.live_grants(0, 0) == 4096
    assert memory.granted_host == [4096.0]
    assert memory.release_grants(0, ["joint:hbm"]) == 4096
    assert memory.granted_host == [0.0]
    assert memory.grant_bindings == {}


def test_staging_and_grant_share_peak_without_double_counting():
    memory = budget_memory(896, policy=OFFLOAD_POLICY)
    assert memory.capacity == [[896.0]]
    assert memory.resize(0, "r", 8)
    assert memory.live == [[512.0]]
    assert memory.staging_live == [256.0]
    assert memory.hold_for_async_consumer(0, "r", "reader")
    # 512 resident + 256 staging + 256 extra > 896.
    assert memory.grant_batch(0, [("reader-extra", 256)]) is False
    assert memory.live_grants(0, 0) == 0.0
    assert memory.peak_combined[0][0] == 768.0
    # A separately completed run can release both resident and staging, then
    # a standalone 896 grant fits; the historical peak is max(768, 896).
    memory.force_release(0, "r")
    assert memory.grant_batch(0, [("standalone", 896)])
    assert memory.peak_combined[0][0] == 896.0
    assert memory.to_dict()["peak_resident_bytes_per_replica_stage"] == [[
        int(memory.weights[0] + 896)]]
    assert memory.release_grants(0, ["standalone"]) == 896


def test_kv_offload_fixed_plan_matches_resize_residency_and_staging():
    memory = budget_memory(896, policy=OFFLOAD_POLICY)
    assert memory.capacity == [[896.0]]
    assert memory.cache_bytes(8) == (1024.0,)
    assert memory._max_layer_cache_bytes(8) == (512.0,)
    assert memory.resize(0, "r", 8) is True
    assert memory.live == [[512.0]]
    # Peak and admission share resident + staging, not gross KV.
    assert memory.peak == [[768.0]]
    assert memory.peak_kv_resident == [[512.0]]
    assert memory.peak_kv_staging == [[256.0]]
    report = memory.allocation_report(8)
    assert report["kv_offload"]["gross_kv_bytes_per_stage_rank"] == [1024.0]
    assert report["kv_offload"]["hbm_resident_kv_bytes_per_stage_rank"] == [512.0]
    assert report["kv_offload"]["staging_bytes_per_stage_rank"] == [256.0]
    assert report["peak_resident_bytes_per_replica_stage"] == [[
        int(memory.weights[0] + 768)]]
    assert report["rank_plans"][0][0]["peak_bytes"] == int(memory.weights[0] + 768)
    assert report["headroom_bytes_per_replica_stage"] == [[128.0]]
    totals = memory.to_dict()["allocation_totals_at_max_model_len"]
    assert totals["cache_gross_bytes_per_stage_rank"] == [memory.cache_bytes(
        memory.case.runtime.max_model_len)[0]]
    assert totals["cache_hbm_resident_bytes_per_stage_rank"][0] + \
        totals["cache_staging_bytes_per_stage_rank"][0] < \
        totals["cache_gross_bytes_per_stage_rank"][0]


def test_live_request_keeps_cache_after_its_consumers_complete():
    parsed = parse_case(case())
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, "r", 4)
    held = memory.live[0][0]
    assert memory.hold_for_async_consumer(0, "r", "attention")
    assert memory.hold_for_async_consumer(0, "r", "collective")
    assert memory.pending_async_consumers(0, "r") == ("attention", "collective")
    assert memory.release_async_consumer(0, "r", "attention") is True
    assert memory.live == [[held]]
    assert memory.pending_async_consumers(0, "r") == ("collective",)
    # The request is still live: its current consumer completing is not a
    # request retirement and must not free committed KV for the next step.
    assert memory.release_async_consumer(0, "r", "collective") is True
    assert memory.live == [[held]]
    assert memory.owners[(0, "r")] == tuple(memory.cache_bytes(4))
    memory.release(0, "r")
    assert memory.live == [[0.0]]
    assert memory.release_async_consumer(0, "r", "collective") is False


def test_async_hold_prevents_resize_from_reclaiming_physical_backing():
    memory = budget_memory(1024)
    assert memory.capacity == [[1024.0]]
    assert memory.resize(0, "r", 8)
    assert memory.live == [[1024.0]]
    assert memory.hold_for_async_consumer(0, "r", "attention")
    # Logical demand shrinks, but the consumer still owns the old backing.
    assert memory.resize(0, "r", 4)
    assert memory.logical_owners[(0, "r")] == (512.0,)
    assert memory.owners[(0, "r")] == (1024.0,)
    assert memory.live == [[1024.0]]
    assert not memory.resize(0, "other", 1)
    assert memory.release_async_consumer(0, "r", "attention") is True
    assert memory.owners[(0, "r")] == (512.0,)
    assert memory.live == [[512.0]]
    assert memory.resize(0, "other", 1)
    assert memory.live == [[640.0]]

    # Two consumers with different registered ranges: the union stays until
    # both complete; completion only reclaims the no-longer-referenced tail.
    memory2 = budget_memory(1024)
    assert memory2.resize(0, "r", 8)
    assert memory2.hold_for_async_consumer(0, "r", "first")
    assert memory2.resize(0, "r", 4)
    assert memory2.hold_for_async_consumer(0, "r", "second")
    # The second consumer registered only the shrunken 512 B range; it must
    # not inherit the first consumer's old 1024 B backing.
    assert memory2.release_async_consumer(0, "r", "first") is True
    assert memory2.owners[(0, "r")] == (512.0,)
    assert memory2.live == [[512.0]]
    assert memory2.release_async_consumer(0, "r", "second") is True
    assert memory2.owners[(0, "r")] == (512.0,)
    assert memory2.live == [[512.0]]


def test_sparse_host_dram_shrink_uses_the_same_hold_union_and_admission():
    memory = _offloaded_memory()
    assert memory.host_dram_capacity == [8891924.0]
    assert memory.host_dram_cache_bytes(2048) == 4718592.0
    assert memory.host_dram_cache_bytes(1024) == 2359296.0
    # Empty ledger, same budget as resize: max span is 3840, not 4096.
    assert memory.admissible_tokens(0, "r", 4096, granularity=128) == 3840
    assert memory.live == [[0.0]]
    assert memory.host_dram_live == [0.0]
    assert memory.owners == {}
    assert memory.resize(0, "r", 2048)
    assert memory.host_dram_live == [4718592.0]
    assert memory.hold_for_async_consumer(0, "r", "consumer")
    assert memory.resize(0, "r", 1024)
    assert memory.logical_host_owners[(0, "r")] == 2359296.0
    assert memory.host_dram_owners[(0, "r")] == 4718592.0
    assert memory.host_dram_live == [4718592.0]
    # 4,718,592 + 4,718,592 = 9,437,184 > 8,891,924.  A failed resize must
    # not commit HBM or host changes partially.
    before_live = [list(row) for row in memory.live]
    before_host = list(memory.host_dram_live)
    assert not memory.resize(0, "other", 2048)
    assert memory.live == before_live
    assert memory.host_dram_live == before_host
    assert memory.release_async_consumer(0, "r", "consumer")
    assert memory.host_dram_live == [2359296.0]
    assert memory.resize(0, "other", 2048)
    assert memory.host_dram_live == [7077888.0]


def test_retire_removes_own_demand_and_keeps_only_other_consumer_ranges():
    memory = budget_memory(1024)
    assert memory.resize(0, "r", 4)
    assert memory.hold_for_async_consumer(0, "r", "A")
    assert memory.resize(0, "r", 8)
    assert memory.hold_for_async_consumer(0, "r", "B")
    memory.release(0, "r")
    assert memory.is_retired(0, "r")
    assert memory.logical_owners[(0, "r")] == (0.0,)
    # B still owns its 1024 B range, so another request cannot yet allocate.
    assert not memory.resize(0, "other", 4)
    assert memory.release_async_consumer(0, "r", "B")
    assert memory.owners[(0, "r")] == (512.0,)
    assert memory.live == [[512.0]]
    # A's 512 B remains live; a second 512 B request fits in the 1024 B budget.
    assert memory.resize(0, "other", 4)
    assert memory.live == [[1024.0]]
    assert memory.release_async_consumer(0, "r", "A")
    assert memory.live == [[512.0]]


def test_async_retired_request_waits_for_pending_consumer_and_delayed_expiry():
    parsed = parse_case(case())
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, "r", 4)
    held = memory.live[0][0]
    assert memory.hold_for_async_consumer(0, "r", "attention")
    # Retire before the consumer's completion. The owner remains resident.
    memory.release(0, "r")
    assert memory.is_retired(0, "r")
    assert memory.live == [[held]]
    assert memory.pending_async_consumers(0, "r") == ("attention",)
    assert memory.release_async_consumer(0, "r", "attention") is True
    assert memory.live == [[0.0]]
    assert not memory.is_retired(0, "r")

    # A delayed lookahead expiry is also not a force-retire: pending holds stay.
    deferred = Memory(parsed, MemoryPolicy(
        derive_workspace=False, async_delayed_release_steps=1))
    assert deferred.resize(0, "r", 4)
    assert deferred.hold_for_async_consumer(0, "r", "attention")
    deferred.defer_release(0, "r", 1)
    deferred.step_delayed_releases(0)
    assert deferred.is_retired(0, "r")
    assert deferred.live == [[held]]
    assert deferred.pending_async_consumers(0, "r") == ("attention",)
    assert deferred.delayed_owners == {}
    assert deferred.release_async_consumer(0, "r", "attention") is True
    assert deferred.live == [[0.0]]

    # Cancellation/rollback has an explicit force path.
    forced = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert forced.resize(0, "r", 4)
    assert forced.hold_for_async_consumer(0, "r", "attention")
    forced.force_release(0, "r")
    assert forced.live == [[0.0]]
    assert forced.pending_async_consumers(0, "r") == ()


def test_retirement_reclaims_unreferenced_tail_before_consumer_completion():
    memory = budget_memory(1024)
    assert memory.resize(0, "r", 4)
    assert memory.hold_for_async_consumer(0, "r", "reader")
    assert memory.resize(0, "r", 8)
    memory.release(0, "r")
    assert memory.occupied_bytes(0, 0) == 512.0
    assert memory.resize(0, "other", 4)
    assert memory.occupied_bytes(0, 0) == 1024.0
    assert memory.release_async_consumer(0, "r", "reader")
    assert memory.occupied_bytes(0, 0) == 512.0


def test_pipeline_partitions_budget_their_own_cache_and_staging():
    memory = budget_memory(512, pp=2, policy=OFFLOAD_POLICY)
    assert memory.capacity == [[512.0, 512.0]]
    assert memory.resize(0, "r", 8)
    assert memory.live == [[256.0, 256.0]]
    assert memory.staging_rank[0] == [[256.0], [256.0]]
    assert memory.peak_combined == [[512.0, 512.0]]
    assert not memory.grant_batch(0, [("stage-extra", 256)], stage=1)
    assert memory.occupied_bytes(0, 1) == 512.0


def test_host_grants_and_request_resize_share_occupancy():
    memory = _offloaded_memory()
    assert memory.grant_batch(0, [("h", 0)], host_bytes=3_000_000)
    assert memory.resize(0, "r", 2048)
    assert memory.resize(0, "r", 2048)
    assert memory.host_occupied_bytes(0) == 7_718_592.0
    memory.release_grants(0, ["h"])
    assert memory.host_occupied_bytes(0) == 4_718_592.0
    memory.release(0, "r")
    assert memory.host_occupied_bytes(0) == 0.0


def test_memory_accounts_for_async_workspace_and_delayed_release_policy():
    raw = case(prompt=16, output=8)
    parsed = parse_case(raw)
    sync = Memory(parsed, workspace_tokens=16)
    asynchronous = Memory(replace(parsed, runtime=replace(parsed.runtime, async_scheduling=True)),
                          workspace_tokens=16)

    assert (sync.async_delayed_release_steps, sync.async_in_flight_lookahead_steps) == (0, 0)
    assert sync.async_workspace == [0.0]
    assert (asynchronous.async_delayed_release_steps, asynchronous.async_in_flight_lookahead_steps) == (1, 1)
    extra = asynchronous.async_workspace[0]
    assert extra > 0
    assert asynchronous.workspace[0] == pytest.approx(sync.workspace[0] + extra)
    assert asynchronous.capacity[0][0] == pytest.approx(sync.capacity[0][0] - extra)


def test_scheduler_delayed_release_defers_admission_by_one_step():
    # 12 KV rows fit. At arrival the first two requests take 8 rows; the
    # waiting request needs 5. It can start when the first request releases.
    requests = [
        {"id": "first", "prompt_tokens": 4, "output_tokens": 2},
        {"id": "active", "prompt_tokens": 4, "output_tokens": 4},
        {"id": "waiting", "prompt_tokens": 5, "output_tokens": 2},
    ]
    raw = case(requests=requests)
    raw["runtime"].update(
        kv_block_size=1, max_num_seqs=3, max_num_batched_tokens=12,
        enable_chunked_prefill=False, gpu_memory_utilization=1.0, memory={"derive_workspace": False},
    )
    raw["machine"]["device"]["memory_bytes"] = int(Memory(parse_case(raw)).weights[0] + 12 * BYTES_PER_KV_ROW)
    reports = {}
    for async_scheduling in (False, True):
        parsed = parse_case({**raw, "runtime": {**raw["runtime"], "async_scheduling": async_scheduling}})
        report = _predict_bounded(parsed)
        assert report["status"] == "modeled"
        reports[async_scheduling] = report
    sync, asynchronous = reports[False], reports[True]
    sync_waiting = next(row for row in sync["requests"] if row["request_id"] == "waiting")
    async_waiting = next(row for row in asynchronous["requests"] if row["request_id"] == "waiting")
    assert sync["memory"]["peak_delayed_release_cache_bytes_per_replica_stage"] == [[0.0]]
    # Eight prefill tokens and a two-token decode take 10 ms. The remaining
    # active request takes one more millisecond before the delayed release.
    assert sync_waiting["admission_ms"] == 10.0
    assert async_waiting["admission_ms"] - sync_waiting["admission_ms"] == 1.0
    assert asynchronous["memory"]["peak_delayed_release_cache_bytes_per_replica_stage"][0][0] > 0.0
    assert asynchronous["memory"]["live_cache_bytes_per_replica_stage"] == [[0.0]]


def test_scheduler_stalls_while_delayed_release_pins_its_only_slot():
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
    raw["machine"]["device"]["memory_bytes"] = int(Memory(parsed).weights[0] + 4 * BYTES_PER_KV_ROW)

    report = _predict_bounded(raw)
    assert report["status"] == "modeled"
    assert [r["status"] for r in report["requests"]] == ["served", "served", "served", "served"]
    r1_finish = next(r for r in report["requests"] if r["request_id"] == "r1")["finish_ms"]
    r2_admit = next(r for r in report["requests"] if r["request_id"] == "r2")["admission_ms"]
    # Scheduler waits count toward delayed release while the other replica
    # waits for its next arrival at 10 ms.
    assert r1_finish + 2.0 <= r2_admit < 10.0
    assert report["memory"]["live_cache_bytes_per_replica_stage"] == [[0.0], [0.0]]


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_speculative_lookahead_stays_within_remaining_request_output(async_scheduling):
    parsed = draft_case()
    # This proposal accepts both draft tokens. Only four output tokens are
    # requested, so the final verification must fit within seven KV rows.
    request = replace(parsed.workload.requests[0], output_tokens=4)
    workload = replace(parsed.workload, requests=(request,))
    metadata = dict(parsed.runtime.metadata, draft_accept_tokens=2)
    runtime = replace(parsed.runtime, kv_block_size=1, max_num_seqs=1,
                      max_num_batched_tokens=8, gpu_memory_utilization=1.0,
                      async_scheduling=async_scheduling, mtp_acceptance_length=3.0,
                      metadata=tuple(metadata.items()),
                      memory={"derive_workspace": False})
    parsed = replace(parsed, runtime=runtime, workload=workload)
    memory = Memory(parsed)
    budget = int(memory.weights[0] + memory.cache_bytes(7)[0])
    dies = tuple(replace(die, memory_bytes=budget) for die in parsed.machine.dies)
    report = _predict_bounded(replace(parsed, machine=replace(parsed.machine, dies=dies)))
    assert report["status"] == "modeled"
    served, = report["requests"]
    assert served["generated_tokens"] == 4
    assert served["speculative"]["proposed_tokens"] == served["speculative"]["accepted_tokens"] == 2
    # Two target layers and one draft layer each store 64 bytes per row.
    assert report["memory"]["peak_cache_bytes_per_replica_stage"] == [[7 * 3 * 64]]


def test_analytical_async_vram_bound_accounts_for_retention_of_the_full_batch():
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
    raw["machine"]["device"]["memory_bytes"] = int(mem.weights[0] + 24 * BYTES_PER_KV_ROW)
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

    sla_async = solve_sla_sizing(
        raw,
        sla_ms=50.0,
        context=64,
        output_budget=16,
        max_sequences_per_dp=8,
        scenarios=[{"id": "baseline", "label": "baseline"}],
    )
    summary = sla_async["async_scheduling_memory"]
    assert summary["configured_async_scheduling"] is True
    assert summary["b_vram_max_sync"] == b_sync
    assert summary["b_vram_max_async"] == b_async
    assert summary["delta_b_vram_from_disabling_async"] == b_sync - b_async
    assert summary["disabling_async_unlocks_concurrency"] is True

    raw_sync = {**raw, "runtime": {**raw["runtime"], "async_scheduling": False}}
    sla_sync = solve_sla_sizing(
        raw_sync,
        sla_ms=50.0,
        context=64,
        output_budget=16,
        max_sequences_per_dp=8,
        scenarios=[{"id": "baseline", "label": "baseline"}],
    )
    assert sla_sync["scenarios"][0]["b_feasible"] > sla_async["scenarios"][0]["b_feasible"]
    assert sla_sync["scenarios"][0]["tps_feasible"] > sla_async["scenarios"][0]["tps_feasible"]


def test_request_consumer_retry_requires_the_same_handle():
    memory = budget_memory(1024)
    assert memory.resize(0, "r", 4)
    assert memory.hold_for_async_consumer(0, "r", "consumer")
    version = memory.budget_version
    assert memory.hold_for_async_consumer(0, "r", "consumer")
    assert memory.budget_version == version
    assert memory.pending_async_consumers(0, "r") == ("consumer",)
    assert not memory.hold_for_async_consumer(0, "r", "consumer", generation=3)
    assert memory.pending_async_consumers(0, "r") == ("consumer",)
    assert not memory.release_async_consumer(0, "r", "consumer", rank=0)
    assert memory.release_async_consumer(0, "r", "consumer")


def test_grant_and_reader_release_each_require_their_own_event():
    memory = budget_memory(4096)
    grant_event, reader_event = object(), object()
    grant = dict(name="G", bytes=4096, physical_key="P", storage="S",
                 ranges=(0, 4096), generation=7, event=grant_event)
    assert memory.grant_batch(0, [grant])
    assert memory.hold_for_async_consumer(0, "G", "reader", grant="G", generation=7, event=reader_event)
    assert memory.release_grant(0, "G", generation=7, event=grant_event)
    assert memory.occupied_bytes(0, 0) == 4096
    assert not memory.release_async_consumer(0, "G", "reader", generation=7, event=object())
    assert memory.occupied_bytes(0, 0) == 4096
    assert memory.release_async_consumer(0, "G", "reader", generation=7, event=reader_event)
    assert memory.occupied_bytes(0, 0) == 0

    # A grant without an event requires an event-free release.
    assert memory.grant_batch(0, [dict(name="N", bytes=4096, physical_key="Q", ranges=(0, 4096))])
    assert not memory.release_grant(0, "N", event=object())
    assert memory.occupied_bytes(0, 0) == 4096
    assert memory.release_grant(0, "N")
    assert memory.occupied_bytes(0, 0) == 0


@pytest.mark.parametrize("backend", ("python", "c"))
def test_evicted_dirty_writeback_blocks_overlapping_reads_until_delivered(backend):
    from mindie_sim.prediction.runtime.engine import Engine, Resource
    from mindie_sim.prediction.runtime.memory_physics import (
        AddressRun, AddressRunPlan, AdmissionMemoryView, CACHE_POLICY_DISABLE, HbmBusSpec,
        MemoryAccessPattern, ReuseKey, VisibilityContract, bind_memory_access,
    )

    size = 64
    bus = HbmBusSpec(0, 64.0, 16.0, 0.0)
    view = AdmissionMemoryView({0: size}, granule_bytes=size)
    eng = Engine([Resource("mte3", 32.0, units=1), Resource("mte2_aiv", 64.0, units=1),
                  Resource("l2", 64.0, units=1)], backend=backend, trace_limit=64)
    eng.add_hbm_bus(bus)
    key = ReuseKey(0, "X", 1, "s0")
    runs = AddressRunPlan((AddressRun(0, 1, size, size),))
    p_wr = MemoryAccessPattern("X", 1, 0, 0, "write", "mte3", size, runs, key,
                               visibility=VisibilityContract(scope="die", private_workspace=True))
    p_rd = MemoryAccessPattern("X", 1, 0, 0, "read", "mte2_aiv", size, runs, key,
                               visibility=VisibilityContract(scope="die", private_workspace=True))
    p_unc = MemoryAccessPattern("X", 1, 0, 0, "read", "mte2_aiv", size, runs, None, cache_policy=CACHE_POLICY_DISABLE)
    b_wr = bind_memory_access(p_wr, view, hbm_bus=bus, copy_hit_bytes_per_ms=32.0, copy_miss_bytes_per_ms=32.0,
                              l2_bytes_per_ms=64.0, pipe_row_id="mte3", l2_row_id="l2", commit=False)
    b_rd = bind_memory_access(p_rd, view, hbm_bus=bus, copy_hit_bytes_per_ms=64.0, copy_miss_bytes_per_ms=64.0,
                              l2_bytes_per_ms=64.0, pipe_row_id="mte2_aiv", l2_row_id="l2", commit=False)
    b_unc = bind_memory_access(p_unc, view, hbm_bus=bus, copy_hit_bytes_per_ms=64.0, copy_miss_bytes_per_ms=64.0,
                               l2_bytes_per_ms=64.0, pipe_row_id="mte2_aiv", l2_row_id="l2", commit=False)
    ev_w = eng.submit_bound_access("wr", b_wr, release=0.0, memory_state=view, pattern=p_wr)
    eng.advance(until=(ev_w,))
    eng.advance(horizon=2.5)
    view.grant_reservation(0, "full", size)
    ev_r1 = eng.submit_bound_access("r1", b_rd, release=3.0, memory_state=view, pattern=p_rd)
    ev_r2 = eng.submit_bound_access("r2", b_rd, release=4.1, memory_state=view, pattern=p_rd)
    ev_r3 = eng.submit_bound_access("r3", b_unc, release=4.5, memory_state=view, pattern=p_unc)
    eng.advance(until=(ev_r1, ev_r2, ev_r3), first=False)
    wb_end = next(r["end_ms"] for r in eng.trace if "dirty_writeback" in r["task"])
    assert ev_r1.time >= wb_end and ev_r2.time >= wb_end and ev_r3.time >= wb_end
    assert eng.delivered[bus.ctr_w] == float(size)


def test_split_periodic_residency_runs_stay_memory_bounded():
    import tracemalloc
    from mindie_sim.prediction.runtime.memory_physics import AddressRun, AddressRunPlan
    from mindie_sim.templates.contracts import TensorRangeObligation, _split_entry_residency

    half, stride, sparse, sector = 2_097_152, 4097, 4_194_336, 32
    cov = AddressRunPlan((AddressRun(0, half, sector, sparse), AddressRun(half * sparse, half, sector, sparse))).sector_coverage()
    r_a = AddressRun(0, half, sector, stride)
    cases = (
        (TensorRangeObligation(0, "rows", 0, (), "hbm_cold", "device", (r_a, AddressRun(half * stride, half, sector, stride))),
         {"l2_warm": 1024, "hbm_cold": 134_216_704}),
        (TensorRangeObligation(0, "rows", 0, (), "hbm_cold", "device", (r_a, AddressRun((half + 1) * stride, half - 1, sector, stride))),
         {"l2_warm": 1024, "hbm_cold": 134_216_672}),
    )
    for entry, expected in cases:
        tracemalloc.start()
        try:
            split = tuple(_split_entry_residency(entry, cov))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 8 * 1024**2
        counts = {item.initial_residency: sum(hi - lo for lo, hi in item.ranges)
                  + sum(run.rows * run.row_bytes for run in item.address_runs) for item in split}
        assert counts == expected


def test_reparsing_media_case_with_text_workload_clears_vision_work():
    import json
    from dataclasses import replace
    from pathlib import Path
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.schema.model import WorkloadSpec
    from mindie_sim.schema.workload import IterationPlan, RequestView

    raw = json.loads(Path("examples/quickstart.json").read_text(encoding="utf-8"))
    raw["model"].update(
        layers=1,
        vision=dict(
            kind="vit", layers=1, hidden_size=8, num_heads=1,
            intermediate_size=16, qkv_width=24, patch_size=2, out_hidden_size=512,
        ),
    )
    raw["workload"] = {"requests": [dict(id="r", prompt_tokens=5, media_tokens=4, output_tokens=1)]}
    image_case = parse_case(raw)
    assert dict(image_case.runtime.metadata).get("media_tokens_by_request") == {"r": 4}
    text_workload = WorkloadSpec.from_mapping({"requests": [dict(id="r", prompt_tokens=5, output_tokens=1)]})
    plan = IterationPlan((RequestView("r", 5, 0, True, "prefill"),))
    reused = parse_case(replace(image_case, workload=text_workload))
    fresh = parse_case({**raw, "workload": text_workload.to_dict()})
    assert "media_tokens_by_request" not in dict(reused.runtime.metadata)
    for case in (reused, fresh):
        ops = Frontend()(plan, case.model, case.runtime, case.machine).ops
        assert sum(op.flops for op in ops if op.op_id.startswith("media0:")) == 0

    # Pure-model assembly without concrete workload requests preserves explicit media_tokens_by_request.
    from mindie_sim.prediction.structure import parse_model

    pure_raw = {
        "model": raw["model"],
        "machine": raw["machine"],
        "runtime": {**raw["runtime"], "metadata": {"media_tokens_by_request": {"model0:0": 4}}},
    }
    pure_case = parse_model(pure_raw)
    assert dict(pure_case.runtime.metadata).get("media_tokens_by_request") == {"model0:0": 4}
    pure_plan = IterationPlan((RequestView("model0:0", 5, 0, True, "prefill"),))
    pure_ops = Frontend()(pure_plan, pure_case.model, pure_case.runtime, pure_case.machine).ops
    assert sum(op.flops for op in pure_ops if op.op_id.startswith("media0:")) > 0
