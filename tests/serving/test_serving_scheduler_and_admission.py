"""Decode progress probes and full serving scheduler/pipeline assembly.

[Role]
  Establish the public serving contracts of the cohort driver: decode progress
  windows and capacity probes, per-replica graph padding, multi-microbatch
  aggregation, PP/DP binding identity, selected allocations, chain delivery and
  the SERVICE-CORE commit/feedback boundary.

[Invariants]
  * Probe geometry distinguishes valid request rows, capture buckets and
    physical padding without inventing KV or global batch.
  * Every microbatch/replica commits its own original slot and state version;
    the driver waits only its own delivery tail, never an external timer.
  * Selected allocations are acquired once, refused explicitly when impossible,
    and never issue model work on a permanent failure.

[Timeouts & Locks]
  Tier 0: bounded in-memory cohort runs; no device, network, sleep or file I/O.

[Side-Effect Policy]
  Caller-owned Memory/Scheduler/Execution only; writes pytest temporary state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import pytest
from mindie_sim import target_operators
from mindie_sim.prediction.execution import dispatch
from mindie_sim.prediction.api import predict, run_cohort, simulate_service
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.execution.decode_probe import decode_windows, probe_capacity
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.prediction.sharding.sequence_parallel import DECLARATION
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import AllocationEffect
from mindie_sim.schema.workload import IterationPlan, RequestView
from _support.serving_cases import (
    binding_lower, capture_binding, case, held_decode_scheduler, linear_work, pd_case,
    grant_bytes_for_headroom, memory_with_free_budget, pipeline_case, pp2_binding_case,
    replica_schedulers, scheduler_for, serving_execution, slow_second_request, speculative_case,
)
pytestmark = pytest.mark.tier0
def test_sampled_verification_windows_follow_output_progress_without_acceptance_scaling():
    windows = decode_windows(126000, 3000, query_tokens=4)
    assert [row["context"] for row in windows] == [126000, 127498, 128996]
    assert [row["kv_length"] for row in windows] == [126004, 127502, 129000]
    assert all(row["tokens"] == 4 for row in windows)
    assert [row["context"] for row in decode_windows(126000, 3000)] == [ 126000, 127499, 128999]
    observed = decode_windows(131201, 64, query_tokens=6, output_offsets=[0, 11, 37, 58])
    assert [row["kv_length"] for row in observed] == [131207, 131218, 131244, 131265]
def test_ep_idle_ranks_keep_weights_without_inventing_kv_or_global_batch():
    raw = case(dp=4, ep=4, prompt=126000)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["runtime"]["max_model_len"] = 129000
    one = probe_capacity(raw, context=126000, tokens=1, sequences=[1, 0, 0, 0])
    full = probe_capacity(raw, context=126000, tokens=1, sequences=1)
    assert one["sequences_in_cohort"] == 1 and full["sequences_in_cohort"] == 4
    assert [row["cache_bytes"] for row in one["ranks"]] == [ one["ranks"][0]["cache_bytes"], 0, 0, 0]
    assert all(row["weights_bytes"] > 0 for row in one["ranks"])
    assert sum(row["cache_bytes"] for row in full["ranks"]) == ( 4 * one["ranks"][0]["cache_bytes"])
    independent = probe_capacity(case(dp=4), context=64, tokens=1, sequences=2)
    assert independent["sequences_in_cohort"] == 2
    assert independent["unprobed_replicas"] == [1, 2, 3]
def test_padded_decode_workspace_uses_actual_rows_and_cache_uses_valid_sequences():
    raw = case(prompt=126000)
    raw["runtime"].update(max_model_len=129000, max_num_batched_tokens=8192,
                          graph_mode="auto", graph_capture_mode="full",
                          graph_capture_keys=[16])
    first = probe_capacity(raw, context=126000, tokens=1, sequences=3)
    before = deepcopy(raw)
    raw["workload"]["requests"][0]["prompt_tokens"] = 1
    second = probe_capacity(raw, context=126000, tokens=1, sequences=3)
    assert first == second
    assert first["padded_query_tokens_by_replica"] == {0: 16}
    memory = Memory(parse_case(before), workspace_tokens=16)
    assert first["ranks"][0]["workspace_bytes"] == memory.workspace[0]
    assert first["ranks"][0]["cache_bytes"] == 3 * memory.cache_bytes(126001)[0]
    next_page = probe_capacity(raw, context=126128, tokens=1, sequences=3)
    assert next_page["ranks"][0]["cache_bytes"] > first["ranks"][0]["cache_bytes"]
def test_long_context_capacity_failure_does_not_gate_analytical_body_execution():
    raw = case()
    raw["runtime"].update(max_model_len=129000, max_num_seqs=64)
    raw["machine"]["device"]["memory_bytes"] = 1000000
    baseline = predict(raw, scope="body", context=126000, sequences=2)
    assert baseline["status"] == "modeled" and baseline["body_forward_ms"] > 0
    assert baseline["capacity"]["status"] == "outside_declared_capacity"
    assert any(row["constraint"] == "resident_memory" for row in baseline["capacity"]["violations"])
    raw["machine"]["device"]["memory_bytes"] = 1024 ** 3
    larger = predict(raw, scope="body", context=126000, sequences=2)
    assert larger["capacity"]["status"] == "within_declared_capacity"
    assert larger["body_forward_ms"] == baseline["body_forward_ms"]
def test_capacity_retains_pipeline_and_replica_limits_without_truncating_windows():
    raw = case(pp=2, dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["runtime"].update(max_model_len=128000, max_num_seqs=2, max_num_batched_tokens=8)
    result = probe_capacity(raw, context=128996, tokens=4, sequences=[3, 0])
    assert result["kv_length"] == 129000 and len(result["ranks"]) == 4
    constraints = {row["constraint"] for row in result["violations"]}
    assert {"max_model_len", "max_num_seqs", "max_num_batched_tokens"} <= constraints
    for row in result["ranks"]:
        assert row["resident_bytes"] == pytest.approx( row["weights_bytes"] + row["workspace_bytes"] + row["reserved_bytes"] + row["cache_bytes"])
        assert row["headroom_bytes"] == row["cache_capacity_bytes"] - row["cache_bytes"]
def test_valid_requests_capture_bucket_and_padding_are_distinct_public_geometry():
    raw = case(tp=16, ep=16)
    raw["model"].update(
        hidden_size=128, num_heads=16, num_kv_heads=2, intermediate_size=256,
        ffn="moe", num_experts=16, top_k=2, moe_intermediate_size=64,
        layer_plans=[{"index": i, "attention": "gqa", "ffn": "moe", "num_heads": 16,
                      "num_kv_heads": 2, "head_dim": 8, "num_experts": 16, "top_k": 2,
                      "moe_intermediate_size": 64, "residual_block_size": 2}
                     for i in range(2)])
    raw["runtime"].update(graph_mode="auto", graph_capture_mode="full",
                          graph_capture_keys=[8], routing={"policy": "balanced"},
                          kernels={"sequence_parallel": dict(DECLARATION,
                                                             padding="right_to_tp")})
    report = predict(raw, scope="body", tokens=1, sequences=6, context=32)
    geometry = report["geometry"]
    assert geometry["valid_query_tokens_by_replica"] == {0: 6}
    assert geometry["graph_query_tokens_by_replica"] == {0: 8}
    assert geometry["padded_query_tokens_by_replica"] == {0: 16}
    assert report["replay_scope_by_replica"] == {0: "full"}
    assert report["capacity"]["graph_query_tokens_by_replica"] == {0: 8}
    assert report["capacity"]["ranks"][0]["cache_bytes"] == ( 6 * report["capacity"]["cache_bytes_per_sequence_stage"][0])
@pytest.mark.parametrize("scope", ("body", "model"))
def test_native_response_axes_distinguish_capture_bucket_from_active_attention_rows(scope):
    raw = case()
    raw["model"]["quant"] = {"name": "int8", "weight_format": "int8",
                             "activation_format": "int8"}
    raw["runtime"].update(graph_mode="auto", graph_capture_mode="full",
                          graph_capture_keys=[8])
    first = target_operators(raw, phase="decode", tokens=1, sequences=6,
                             context=32, scope=scope)
    raw["runtime"]["graph_capture_keys"] = [16]
    second = target_operators(raw, phase="decode", tokens=1, sequences=6,
                              context=32, scope=scope)
    attention = [op for op in first if op.family == "attention"]
    other = [op for op in second if op.family == "attention"]
    assert attention and len(attention) == len(other)
    assert any(op.recipe == "dynamic_quantize" for op in first)
    assert [op.local_shape for op in attention] == [op.local_shape for op in other]
    for operators, physical in ((first, 8), (second, 16)):
        assert all(dict(op.metadata)["graph_active_query_rows"] == 6 for op in operators)
        assert all(dict(op.metadata)["graph_physical_query_rows"] == physical for op in operators)
def test_simulate_service_aggregates_every_microbatch_and_its_fallback():
    result = simulate_service(pipeline_case(), event_limit=8)
    assembly = result["execution"]["assembly"]
    assert result["execution"]["iterations"] == 2
    assert (assembly["microbatches"], assembly["declared_commits"], assembly["fallback_commits"]) == (2, 4, 0)
    assert assembly["host_submits"] > 0
    assert all(row["status"] == "served" for row in result["requests"])
    feedback = result["execution"]["scheduler_feedback"][0]
    assert feedback["done"] is True
    assert all(item["status"] == "done" for item in feedback["requests"])
    fallback = simulate_service(pipeline_case(), lower=linear_work, event_limit=8)
    assembly = fallback["execution"]["assembly"]
    assert (assembly["declared_commits"], assembly["fallback_commits"], assembly["host_submits"]) == (0, 4, 4)
    events = fallback["execution"]["step_summary"]["total"]["request_events"]
    assert events["commits"] == 0 and events["fallback_completions"] == 4
def test_simulate_service_low_critical_and_overload_share_one_loop():
    requests = [{"id": "a", "prompt_tokens": 1, "output_tokens": 1},
                {"id": "b", "prompt_tokens": 1, "output_tokens": 1}]
    assert simulate_service(case(requests=requests), event_limit=4)["status"] == "modeled"
    critical = case(requests=requests)
    critical["runtime"]["max_num_seqs"] = 1
    result = simulate_service(critical, event_limit=4)
    assert result["status"] == "modeled" and result["execution"]["iterations"] >= 2
    overload = simulate_service(case(), memory_policy=MemoryPolicy(
        workspace_bytes=2 * 1024 ** 3), event_limit=4)
    assert overload["status"] == "infeasible" and overload["memory"]["status"] == "oom"
    assert overload["requests"][0]["generated_tokens"] == 0
    assert overload["production_entry"] == "simulate_service"
def test_framework_fragment_and_actual_transfer_share_link_and_hbm_work():
    raw = case(prompt=1, output=1)
    baseline = simulate_service(raw)
    raw["runtime"]["serving_io"] = {
        "host_prepare_ms": 0.1, "host_result_ms": 0.1,
        "h2d": {"bytes_per_s": 1000, "fixed_bytes": 64},
        "d2h": {"bytes_per_s": 1000, "fixed_bytes": 64}}
    parsed = parse_case(raw)
    execution = serving_execution(parsed, Memory(parsed), include_io=True)
    hbm = execution.stage_names[0, 0]["hbm"]
    fragments = execution.serving_io.fragments(h2d_hbm=hbm, d2h_hbm=hbm)
    by_name = {fragment.name: fragment for fragment in fragments}
    serviced = simulate_service(raw)["execution"]["serviced"]
    for name, resource in (("input_transfer", "h2d@r0"), ("output_transfer", "d2h@r0")):
        work = {item.resource: item.fixed for item in by_name[name].work}
        assert work[resource] == work[hbm] == serviced[resource] == 64
    assert serviced[hbm] - baseline["execution"]["serviced"][hbm] == pytest.approx(2 * 64)
    occurrence_raw = case(requests=[{"id": "r", "prompt_tokens": 1,
                                     "output_tokens": 1}])
    occurrence_raw["runtime"]["serving_io"] = {
        "host_prepare_ms": 0.1, "host_result_ms": 0.1, "client_delivery_ms": 1}
    observed = simulate_service(occurrence_raw, event_limit=4)
    assert len(observed["framework_occurrences"]) == 1
    occurrence = observed["framework_occurrences"][0]
    assert occurrence["prefix"] == "r0:i0"
    assert occurrence["host_prepare_ms"] is not None
    assert occurrence["deliveries_ms"]["r"] == occurrence["client_propagation_ms"]
def test_run_cohort_drains_its_own_delivery_tail_and_ignores_external_timer():
    raw = case(prompt=1, output=1)
    raw["runtime"]["serving_io"] = {"client_delivery_ms": 1000}
    parsed = parse_case(raw)
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    execution = serving_execution(parsed, memory, include_io=True, bind_memory=False)
    run = run_cohort(parsed, execution, [scheduler], memory, Frontend(), trace_limit=0)
    assert run["stop_reason"] == "drained" and scheduler.done is True
    assert run["serving_end"] >= 1000
    assert execution.engine.summary()["live_tasks"] == 0
    assert execution.engine.summary()["live_programs"] == 0
    raw = case(requests=[{"id": "a", "prompt_tokens": 1, "output_tokens": 1}])
    raw["runtime"]["serving_io"] = {"host_result_ms": 1, "client_delivery_ms": 5}
    parsed = parse_case(raw)
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    execution = serving_execution(parsed, memory, include_io=True, bind_memory=False)
    timer = execution.engine.submit("external_timer", {}, latency=1000)
    run = run_cohort(parsed, execution, [scheduler], memory, linear_work, trace_limit=0)
    assert run["stop_reason"] == "drained" and scheduler.done is True
    assert run["serving_end"] >= 6 and timer.done is False
def test_simulate_service_consumes_caller_memory_and_reuses_committed_grants():
    raw = case(prompt=1, output=1)
    parsed = parse_case(raw)
    memory = Memory(parsed)
    effect = AllocationEffect.acquire("workspace", "hbm", 4096)
    result = simulate_service(raw, memory=memory, selected_allocations=[effect],
                              event_limit=0)
    assert result["status"] == "modeled"
    assert "selected:workspace" in memory.selected_items[(0, 0)]
    assert memory.selected_items[(0, 0)]["selected:workspace"].bytes >= 4096
    memory = Memory(parse_case(raw))
    first = simulate_service(raw, memory=memory,
                             selected_allocations=[AllocationEffect.acquire(
                                 "a_ws", "hbm", 4096)], lower=linear_work)
    assert first["status"] == "modeled" and memory.selected_reserved[0][0] == 4096
    second = simulate_service(raw, memory=memory,
                              selected_allocations=[AllocationEffect.acquire(
                                  "b_ws", "hbm", 4096)], lower=linear_work)
    assert second["status"] == "modeled"
    assert {"selected:a_ws", "selected:b_ws"} <= set(memory.selected_items[(0, 0)])
    assert memory.selected_reserved[0][0] == 8192
def test_window_keeps_original_output_and_does_not_submit_unselected_successor():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 3, "chain_id": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1, "gap_ms": 2}])
    parsed = parse_case(raw)
    selected = replace(parsed, workload=replace(parsed.workload, requests=parsed.workload.requests[:1]))
    result = simulate_service(selected, window={"a": {"computed": 4}}, token_quotas={"a": 2}, lower=linear_work)
    row, = result["requests"]
    assert (row["request_id"], row["output_tokens"], row["generated_tokens"]) == ("a", 3, 2)
    assert row["status"] == "unserved" and len(result["traces"]) == 2
def test_global_cap_reserves_before_async_dp_emits_only_two_first_steps():
    raw = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 4},
        {"id": "b", "prompt_tokens": 10, "output_tokens": 4}])
    result = predict(raw, lower=linear_work, max_model_steps=2)
    assert result["execution"]["logical_steps"] == 2
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["a"]["generated_tokens"] == rows["b"]["generated_tokens"] == 1
def test_run_cohort_rejects_conflicting_memory_ledger_and_preserves_caller_work():
    raw = case(prompt=1, output=1)
    parsed = parse_case(raw)
    first, second = Memory(parsed), Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), first, 0)
    execution = serving_execution(parsed, first, include_io=True)
    prior = execution.engine.submit("prior_work", {}, latency=10)
    completed = execution.engine.summary()["completed_tasks"]
    version = scheduler.states[0].state_version
    with pytest.raises(ModelingError, match="execution_memory_ledger_conflict"):
        run_cohort(parsed, execution, [scheduler], second, linear_work)
    assert execution.memory is first and execution.engine.memory is first
    assert scheduler.memory is first and scheduler.states[0].state_version == version
    assert prior.done is False
    assert execution.engine.summary()["completed_tasks"] == completed
    run = run_cohort(parsed, execution, [scheduler], first, linear_work)
    assert run["stop_reason"] == "drained" and scheduler.done is True
def test_selected_allocation_preconditions_and_release_wait():
    raw = case(prompt=1, output=1)
    parsed = parse_case(raw)
    permanent = Memory(parsed)
    capacity = permanent.capacity[0][0]
    with pytest.raises(ModelingError, match="permanent_capacity"):
        simulate_service(raw, memory=permanent, selected_allocations=[
            AllocationEffect.acquire("too_large", "hbm", capacity * 2)])
    assert permanent.selected_items.get((0, 0), {}) == {} and permanent.live[0][0] == 0
    waiting = Memory(parsed)
    assert waiting.grant_batch(0, [("hold", waiting.capacity[0][0] - 4096)])
    with pytest.raises(ModelingError, match="waiting_for_release"):
        simulate_service(raw, memory=waiting, selected_allocations=[
            AllocationEffect.acquire("needs_release", "hbm", 8192)])
    assert "selected:needs_release" not in waiting.selected_items.get((0, 0), {})
    assert waiting.release_grants(0, ["hold"]) > 0
    result = simulate_service(raw, memory=waiting, selected_allocations=[
        AllocationEffect.acquire("needs_release", "hbm", 8192)], lower=linear_work)
    assert result["status"] == "modeled"
    assert "selected:needs_release" in waiting.selected_items[(0, 0)]
    waiting2 = Memory(parsed)
    hold = grant_bytes_for_headroom(waiting2, 4096.0)
    executable = serving_execution(parsed, waiting2, include_io=True)
    release = executable.engine.submit(
        "release_hold", {}, latency=5,
        on_complete=lambda _event: waiting2.release_grants(0, ["hold"]))
    assert waiting2.grant_batch(0, [{"name": "hold", "bytes": hold, "event": release}])
    scheduler = Scheduler(parsed, list(parsed.workload.requests), waiting2, 0)
    run = run_cohort(parsed, executable, [scheduler], waiting2, linear_work,
                     selected_allocations=[AllocationEffect.acquire(
                         "workspace", "hbm", 8192)])
    assert run["selected_allocation_attached"] is True and run["model_steps"] >= 1
    assert executable.engine.now >= 5
    assert "selected:workspace" in waiting2.selected_items[(0, 0)]
    assert scheduler.done is True
def test_selected_allocation_permanent_refusal_issues_zero_model_work():
    raw = case(prompt=1, output=1)
    parsed = parse_case(raw)
    memory = Memory(parsed)
    hold = grant_bytes_for_headroom(memory, 4096.0)
    executable = serving_execution(parsed, memory, include_io=True)
    release = executable.engine.submit("release_part", {}, latency=5)
    first = min(2048.0, hold)
    assert memory.grant_batch(0, [{"name": "hold_a", "bytes": first, "event": release}])
    if hold > first:
        assert memory.grant_batch(0, [{"name": "hold_b", "bytes": hold - first}])
    executable.engine.submit(
        "release_part_callback", after=(release,),
        on_complete=lambda _event: memory.release_grants(0, ["hold_a"]))
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    with pytest.raises(ModelingError, match="waiting_for_release.*permanent=1"):
        run_cohort(parsed, executable, [scheduler], memory, linear_work,
                   selected_allocations=[AllocationEffect.acquire(
                       "workspace", "hbm", 8192)])
    assert scheduler.states[0].iterations == 0 and scheduler.states[0].generated == 0
    assert executable.work["flops"] == 0
def test_service_microbatches_drain_and_start_from_real_prefill():
    full = simulate_service(pipeline_case(), trace_limit=0)
    assert full["execution"]["assembly"]["microbatches"] == 2
    assert full["execution"]["cohort_stop"] == "drained"
    raw = case(requests=[{"id": "r", "prompt_tokens": 4, "output_tokens": 2}])
    result = simulate_service(raw)
    assert result["execution"]["assembly"]["declared_commits"] == 2
    assert [step["mode"] for step in result["traces"]] == ["prefill", "decode"]
def test_full_capture_idle_participant_keeps_padding_work_without_fake_progress():
    raw = case(dp=2, ep=2, output=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[1],
                          graph_capture_mode="full")
    result = simulate_service(raw, event_limit=0)
    assert result["status"] == "modeled"
    assert result["requests"][0]["generated_tokens"] == 2
    assert result["execution"]["iterations"] >= 2
    assert result["execution"]["live_tasks"] == 0
    assert result["execution"]["live_programs"] == 0
    assert result["execution"]["assembly"]["declared_commits"] >= 1
@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("quota", [1, 2])
def test_decode_quota_preserves_each_request_output_and_context(dp, quota):
    for request_ids in (("b",), ("a", "b")):
        raw = case(dp=dp, requests=[{"id": key, "prompt_tokens": 4, "output_tokens": 3}
                                    for key in request_ids])
        raw["runtime"]["max_num_seqs"] = 2
        result = simulate_service(raw, window={key: {"computed": 4} for key in request_ids},
                                  token_quotas=dict.fromkeys(request_ids, quota))
        assert result["status"] == "incomplete" and result["execution"]["cohort_stop"] == "token_quota"
        assert result["execution"]["assembly"]["declared_commits"] == quota * len(request_ids)
        assert [row["request_id"] for row in result["requests"]] == list(request_ids)
        for row in result["requests"]:
            assert (row["output_tokens"], row["generated_tokens"], row["visible_context_tokens"]) == (3, quota, 4 + quota)
            assert row["status"] == "unserved"
def test_quota_keeps_original_output_and_chain_successor_at_zero():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 3, "chain_id": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1}])
    raw["runtime"]["max_num_seqs"] = 2
    result = simulate_service(raw, window={"a": {"computed": 4}, "b": {"computed": 1}},
                              token_quotas={"a": 2, "b": 2}, lower=linear_work)
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["a"]["generated_tokens"] == 2 and rows["a"]["output_tokens"] == 3
    assert rows["a"]["status"] == "unserved"
    assert rows["b"]["generated_tokens"] == 0 and rows["b"]["admission_ms"] is None
    assert rows["b"]["output_tokens"] == 1 and rows["b"]["status"] == "unserved"
    assert result["execution"]["cohort_stop"] == "token_quota"
    assert result["execution"]["assembly"]["fallback_commits"] == 2
def test_service_respects_arrivals_and_single_seat_admission():
    raw = case(requests=[
        {"id": "r0", "prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 0},
        {"id": "r1", "prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 1000}])
    raw["runtime"]["max_num_seqs"] = 1
    result = simulate_service(raw, max_model_steps=2)
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["r0"]["admission_ms"] == 0 and rows["r1"]["admission_ms"] >= 1000
def test_simulate_service_pd_uses_shared_two_pool_engine_path():
    result = simulate_service(pd_case(), lower=linear_work, event_limit=8)
    assert result["execution"]["serviced"]["pd_link"] > 0
def test_feedback_uses_committed_state_not_emission_count():
    result = simulate_service(case(requests=[{"id": "a", "prompt_tokens": 2,
                                              "output_tokens": 2}]), event_limit=4)
    row = result["requests"][0]
    feedback_all = result["execution"]["scheduler_feedback"][0]
    feedback = feedback_all["requests"][0]
    assert feedback["generated_tokens"] == row["generated_tokens"] == 2
    assert feedback["visible_context_tokens"] == row["visible_context_tokens"] == 3
    assert feedback_all["capacity"]["kv_capacity_bytes_by_stage"][0] > 0
    assert feedback_all["capacity"]["headroom_bytes_by_stage"][0] >= 0
def test_pipeline_microbatches_bind_original_slots_and_replica_identity():
    run, seen = capture_binding(parse_case(pp2_binding_case(1)), binding_lower,
                                 {"a": 3, "b": 7, "c": 11, "d": 15})
    rebound = {op_id.split(":", 1)[1]: dict(metadata["state_accesses"][0])
               for _replica, op_id, metadata in seen}
    assert {key: record["request_slot"] for key, record in rebound.items()} == { "a": 0, "b": 1, "c": 2, "d": 3}
    assert {key: record["state_version"] for key, record in rebound.items()} == { "a": 3, "b": 7, "c": 11, "d": 15}
    assert all(record["request"] == key and record["reader"] == "op:" + key for key, record in rebound.items())
    assert {tuple(sorted(item["slots"])) for item in run["assembly"]["batch_bindings"]} == { ("a", "c"), ("b", "d")}
    run, seen = capture_binding(parse_case(pp2_binding_case(2)), binding_lower,
                                 {"a": 2, "b": 5, "c": 8, "d": 13})
    identity = {(replica, op_id.split(":", 1)[1],
                 dict(metadata["state_accesses"][0])["request_slot"],
                 dict(metadata["state_accesses"][0])["state_version"])
                for replica, op_id, metadata in seen}
    assert identity == {(0, "a", 0, 2), (0, "c", 1, 8), (1, "b", 0, 5), (1, "d", 1, 13)}
    per_replica = {}
    for item in run["assembly"]["batch_bindings"]:
        per_replica.setdefault(item["replica"], []).append(item["slots"])
    assert per_replica[0] == [{"a": 0}, {"c": 1}]
    assert per_replica[1] == [{"b": 0}, {"d": 1}]
def test_padding_plan_publishes_no_request_commit_or_delivery_row():
    parsed = parse_case(case())
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    execution = serving_execution(parsed, memory, include_io=False)
    padding = RequestView("serving0:padding", 4, 0, False, "decode", padding=True,
                          draft_verify=False, draft_prepare=False)
    steps = dispatch.submit_batch(parsed, [scheduler], {0: IterationPlan((padding,),
                                                                       iteration=0)},
                                  execution, Frontend(), 0)
    step = steps[0]
    assert step.commits == {} and step.declared_commits == {}
    assert step.fallback_commits == {} and step.host_results == {}
    assert step.client_deliveries == {} and scheduler.states[0].generated == 0
def test_chain_delivery_and_two_service_deliveries_drain_at_eleven():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1, "gap_ms": 2},
        {"id": "c", "prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 1000}])
    raw["runtime"]["serving_io"] = {"client_delivery_ms": 6}
    full_raw = deepcopy(raw)
    result = predict(raw, lower=linear_work, max_model_steps=1)
    rows = {row["request_id"]: row for row in result["requests"]}
    assert result["execution"]["cohort_stop"] == "max_model_steps"

    assert rows["b"]["admission_ms"] is None
    assert rows["c"]["admission_ms"] is None
    full = predict(full_raw, lower=linear_work, max_model_steps=2)
    rows = {row["request_id"]: row for row in full["requests"]}
    assert rows["a"]["boundaries"]["client_delivery"]["finish_ms"] == 7
    assert rows["b"]["admission_ms"] == 9 and rows["b"]["finish_ms"] == 10
    assert rows["c"]["generated_tokens"] == 0
    assert full["execution"]["cohort_stop"] == "max_model_steps"
    raw = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 5, "output_tokens": 1}])
    raw["runtime"]["serving_io"] = {"client_delivery_ms": 6}
    parsed = parse_case(raw)
    memory = Memory(parsed)
    schedulers = replica_schedulers(parsed, memory)
    execution = serving_execution(parsed, memory, include_io=True)
    timer = execution.engine.submit("unrelated_timer", {}, latency=1000)
    run = run_cohort(parsed, execution, schedulers, memory, linear_work)
    assert run["stop_reason"] == "drained" and run["serving_end"] == 11
    assert all(scheduler.done for scheduler in schedulers) and timer.done is False
def test_release_wakeup_uses_five_ms_not_blocked_dp_or_unrelated_timer():
    raw = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1}])
    parsed = parse_case(raw)
    memory = Memory(parsed)
    schedulers = replica_schedulers(parsed, memory)
    execution = serving_execution(parsed, memory, include_io=False)
    timer = execution.engine.submit("unrelated_timer", {}, latency=1000)
    hold = grant_bytes_for_headroom(memory, 1.0)
    release = execution.engine.submit(
        "l0_last_read", {}, latency=5,
        on_complete=lambda _event: memory.release_grants(0, ["l0_hold"]))
    assert memory.grant_batch(0, [{"name": "l0_hold", "bytes": hold, "event": release}])
    run = run_cohort(parsed, execution, schedulers, memory, slow_second_request)
    assert run["stop_reason"] == "drained"
    assert schedulers[0].states[0].admission_ms == 5
    assert schedulers[0].states[0].finish_ms == 6
    assert schedulers[1].states[0].finish_ms == 20 and timer.done is False
def test_long_delivery_does_not_block_independent_next_request():
    raw = case(requests=[{"id": "a", "prompt_tokens": 1, "output_tokens": 1},
                         {"id": "b", "prompt_tokens": 1, "output_tokens": 1}])
    raw["runtime"].update(max_num_seqs=1, serving_io={"client_delivery_ms": 1000})
    rows = {row["request_id"]: row for row in simulate_service(raw, event_limit=0)["requests"]}
    assert rows["b"]["finish_ms"] < rows["a"]["boundaries"]["client_delivery"]["finish_ms"]

def test_direct_memory_last_read_reauths_the_same_request_as_the_wrapper():
    def recompute(release):
        scheduler, memory = held_decode_scheduler(prompt=4, max_batch_tokens=4, generated=1)
        assert scheduler.schedule(0) is None
        assert scheduler.last_reason == "waiting_for_release"
        assert scheduler.states[1].waiting_for_release is True
        # Re-admit the affected request first; this is an explicit service order,
        # not a second admission rule.
        scheduler.order = lambda state: (state.request.request_id == "a",)
        release(scheduler, memory)
        plan = scheduler.schedule(1.0)
        assert plan is not None and plan.requests[0].request_id == "b"
        assert (plan.requests[0].phase, plan.requests[0].q_len) == ("prefill", 4)
        assert scheduler.emissions(plan) == {}
        assert memory.owners[(0, "b")] == (512.0,)
        scheduler.commit(1.0, {})

        # The other request really completed and releases its same-pool capacity.
        scheduler.by_id["a"].finish_ms = 1.0
        scheduler._live_ids.discard("a")
        memory.release(0, "a")

        decode = scheduler.schedule(2.0)
        assert decode is not None
        assert (decode.requests[0].phase, decode.requests[0].q_len) == ("decode", 1)
        assert scheduler.emissions(decode) == {"b": 1}
        assert memory.owners[(0, "b")] == (640.0,)
        scheduler.commit(2.0, {"b": 2.0})
        assert scheduler.states[1].generated == 2
    recompute(lambda scheduler, memory: memory.release_async_consumer(0, "b", "reader"))
    recompute(lambda scheduler, memory: scheduler.release_state_consumer("b", "reader"))

def test_twelve_kib_retirement_keeps_unfinished_reader_and_reauths_after_release():
    raw = case(requests=[{"id": request, "prompt_tokens": 1, "output_tokens": 5} for request in ("L0", "L1", "L2")])
    parsed, memory = memory_with_free_budget(raw, 12288, weights_only=True)
    for request in ("L0", "L1", "L2"):
        assert memory.resize(0, request, 32)
    scheduler = scheduler_for(parsed, memory)
    assert scheduler.hold_state_consumer("L0", "reader", storage="gqa_kv", ranges=((0, 32),), generation=7, event="reader:event")
    scheduler.retire_state("L0")
    assert memory.is_retired(0, "L0")
    assert memory.live == [[12288.0]]
    assert not memory.resize(0, "L3", 32)
    assert scheduler.release_state_consumer("L0", "wrong") is False
    assert memory.live == [[12288.0]]
    assert scheduler.release_state_consumer("L0", "reader") is True
    assert memory.live == [[8192.0]]
    assert memory.resize(0, "L3", 32)
    assert memory.live == [[12288.0]]
    for request in ("L1", "L2", "L3"):
        memory.release(0, request)
    assert memory.live == [[0.0]]

def test_confirmed_cancel_may_force_release_a_held_request():
    scheduler = scheduler_for(speculative_case(trace=(), output=1))
    assert scheduler.schedule(0) is not None
    assert scheduler.hold_state_consumer("r", "reader")
    scheduler.retire_state("r", force=True)
    assert (0, "r") not in scheduler.memory.owners
    assert scheduler.memory.pending_async_consumers(0, "r") == ()
    assert scheduler.feedback(0)["pending_retired_requests"] == []

def test_async_reader_holds_cross_two_service_iterations_and_last_use_release():
    scheduler = scheduler_for(speculative_case(trace=(), output=2))
    assert scheduler.schedule(0) is not None
    scheduler.commit(1.0, {"r": 1.0})
    assert scheduler.hold_state_consumer("r", "pd-reader") is True
    assert scheduler.schedule(1.0) is not None
    scheduler.commit(2.0, {"r": 2.0})
    assert scheduler.states[0].finish_ms == 2.0
    assert scheduler.memory.is_retired(0, "r")
    assert (0, "r") in scheduler.memory.owners
    assert scheduler.feedback(1.0)["pending_retired_requests"] == ["r"]
    assert scheduler.release_state_consumer("r", "pd-reader") is True
    assert (0, "r") not in scheduler.memory.owners
