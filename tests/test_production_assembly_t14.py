"""Production cohort-loop assembly through api.simulate_service.

These tests exercise the existing predict cohort loop, shared Memory/Scheduler/
Execution, dispatcher aggregation across every microbatch, and the SERVICE-CORE
commit/feedback boundary. They use finite CPU fixtures only.
"""
import pytest

from mindie_sim.compose import assemble
from mindie_sim.prediction.api import predict, run_cohort, simulate_service
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import AllocationEffect
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from test_prediction_system import case, linear_work
from test_prediction_deployment import pd_case


def pipeline_case():
    raw = case(pp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 2},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 2},
    ])
    raw["runtime"]["pipeline_microbatches"] = 2
    return raw


def test_simulate_service_aggregates_every_microbatch_not_only_first():
    result = simulate_service(pipeline_case(), event_limit=8)
    assert result["production_entry"] == "simulate_service"
    assert result["execution"]["iterations"] == 2
    assembly = result["execution"]["assembly"]
    assert assembly["microbatches"] == 2
    assert assembly["declared_commits"] == 4
    assert assembly["fallback_commits"] == 0
    assert assembly["host_submits"] > 0
    assert all(row["status"] == "served" for row in result["requests"])
    feedback = result["execution"]["scheduler_feedback"][0]
    assert feedback["done"] is True
    assert all(item["status"] == "done" for item in feedback["requests"])


def test_simulate_service_aggregates_fallback_and_submit_for_custom_lower():
    result = simulate_service(pipeline_case(), lower=linear_work, event_limit=8)
    assembly = result["execution"]["assembly"]
    assert assembly["declared_commits"] == 0
    assert assembly["fallback_commits"] == 4
    assert assembly["host_submits"] == 4
    assert result["execution"]["step_summary"]["total"]["request_events"]["commits"] == 0
    assert result["execution"]["step_summary"]["total"]["request_events"][
        "fallback_completions"] == 4


def test_simulate_service_low_critical_and_overload_share_one_loop():
    low = simulate_service(case(requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1},
    ]), event_limit=4)
    assert low["status"] == "modeled"

    critical_raw = case(requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1},
    ])
    critical_raw["runtime"]["max_num_seqs"] = 1
    critical = simulate_service(critical_raw, event_limit=4)
    assert critical["status"] == "modeled"
    assert critical["execution"]["iterations"] >= 2

    overload = simulate_service(
        case(), memory_policy=MemoryPolicy(workspace_bytes=2 * 1024 ** 3),
        event_limit=4,
    )
    assert overload["status"] == "infeasible"
    assert overload["memory"]["status"] == "oom"
    assert overload["requests"][0]["generated_tokens"] == 0
    assert overload["production_entry"] == "simulate_service"


def test_framework_fragment_and_actual_transfer_share_link_and_hbm_work():
    raw = case()
    raw["runtime"]["serving_io"] = {
        "host_prepare_ms": 0.1, "host_result_ms": 0.1,
        "h2d": {"bytes_per_s": 1000, "fixed_bytes": 64},
        "d2h": {"bytes_per_s": 1000, "fixed_bytes": 64},
    }
    result = assemble(raw, scope="framework", iterations=1,
                      emitted_tokens={"r": 1})
    by_name = {item["name"]: item for item in result["framework_fragments"]}
    for name in ("input_transfer", "output_transfer"):
        work = {item["resource"]: item["fixed"] for item in by_name[name]["work"]}
        assert work["h2d@r0" if name == "input_transfer" else "d2h@r0"] == 64
        assert work["memory@r0:s0"] == 64
    assert result["execution"]["serviced"]["h2d@r0"] == 64
    assert result["execution"]["serviced"]["d2h@r0"] == 64
    assert result["execution"]["serviced"]["memory@r0:s0"] >= 64


def test_direct_run_cohort_full_window_drains_its_delivery_callbacks():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    raw["runtime"]["serving_io"] = {"client_delivery_ms": 1000}
    parsed = parse_case(raw)
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True,
                          memory_policy=memory.policy)
    execution.engine.now = 0
    run = run_cohort(parsed, execution, [scheduler], memory, Frontend(),
                     trace_limit=0)
    assert run["stop_reason"] == "drained"
    assert scheduler.done is True
    assert run["serving_end"] >= 1000
    assert execution.engine.summary()["live_tasks"] == 0
    assert execution.engine.summary()["live_programs"] == 0


def test_simulate_service_consumes_callers_same_memory_and_selected_allocation():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    parsed = parse_case(raw)
    memory = Memory(parsed)
    effect = AllocationEffect.acquire("workspace", "hbm", 4096)
    result = simulate_service(raw, memory=memory,
                              selected_allocations=[effect], event_limit=0)
    assert result["status"] == "modeled"
    assert "selected:workspace" in memory.selected_items[(0, 0)]
    assert memory.selected_items[(0, 0)]["selected:workspace"].bytes >= 4096


def test_window_keeps_original_output_and_does_not_submit_unselected_successor():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 3, "chain_id": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1,
         "gap_ms": 2},
    ])
    result = assemble(
        raw, scope="service",
        requests=[{"id": "a", "prompt_tokens": 4, "output_tokens": 3}],
        phase="decode", context=4, iterations=1, tokens=2,
        lower=linear_work,
    )
    assert [row["request_id"] for row in result["requests"]] == ["a"]
    assert result["requests"][0]["generated_tokens"] == 2
    assert result["window"]["original_output_tokens"] == {"a": 3}
    assert result["window"]["unfinished_request_ids"] == ["a"]
    assert len(result["steps"]) == 2


def test_global_cap_reserves_before_async_dp_emits_only_two_first_steps():
    raw = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 4},
        {"id": "b", "prompt_tokens": 10, "output_tokens": 4},
    ])
    result = predict(raw, lower=linear_work, max_model_steps=2)
    assert result["execution"]["logical_steps"] == 2
    assert result["execution"]["model_steps"] == 2
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["a"]["generated_tokens"] == 1
    assert rows["b"]["generated_tokens"] == 1


def test_direct_run_cohort_waits_only_its_own_delivery_tail_not_external_timer():
    from tests.test_prediction_system import linear_work
    raw = case(requests=[{"id": "a", "prompt_tokens": 1, "output_tokens": 1}])
    raw["runtime"]["serving_io"] = {"host_result_ms": 1,
                                    "client_delivery_ms": 5}
    parsed = parse_case(raw)
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True,
                          memory_policy=memory.policy)
    execution.engine.now = 0
    timer = execution.engine.submit("external_timer", {}, latency=1000)
    run = run_cohort(parsed, execution, [scheduler], memory, linear_work,
                     trace_limit=0)
    assert run["stop_reason"] == "drained"
    assert scheduler.done is True
    assert run["serving_end"] >= 6
    assert timer.done is False



def test_framework_occurrences_use_the_same_bounded_online_observer():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    raw["runtime"]["serving_io"] = {
        "host_prepare_ms": 0.1, "host_result_ms": 0.1,
        "client_delivery_ms": 1,
    }
    result = simulate_service(raw, event_limit=4)
    assert len(result["framework_occurrences"]) == 1
    occurrence = result["framework_occurrences"][0]
    assert occurrence["prefix"] == "r0:i0"
    assert occurrence["host_prepare_ms"] is not None
    assert occurrence["deliveries_ms"]["r"] == occurrence["client_propagation_ms"]


def test_run_cohort_rejects_conflicting_memory_owner_identity():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    parsed = parse_case(raw)
    first = Memory(parsed)
    second = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), first, 0)
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True,
                          memory=first, memory_policy=first.policy)
    execution.engine.now = 0
    with pytest.raises(ModelingError, match="execution_memory_ledger_conflict"):
        run_cohort(parsed, execution, [scheduler], second, linear_work)


def test_selected_allocation_precondition_distinguishes_permanent_and_waiting():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    parsed = parse_case(raw)
    permanent = Memory(parsed)
    capacity = permanent.capacity[0][0]
    with pytest.raises(ModelingError, match="permanent_capacity"):
        simulate_service(raw, memory=permanent, selected_allocations=[
            AllocationEffect.acquire("too_large", "hbm", capacity * 2)])
    assert permanent.selected_items.get((0, 0), {}) == {}
    assert permanent.live[0][0] == 0

    waiting = Memory(parsed)
    assert waiting.grant_batch(0, [("hold", waiting.capacity[0][0] - 4096)])
    with pytest.raises(ModelingError, match="waiting_for_release"):
        simulate_service(raw, memory=waiting, selected_allocations=[
            AllocationEffect.acquire("needs_release", "hbm", 8192)])
    assert "selected:needs_release" not in waiting.selected_items.get((0, 0), {})
    assert waiting.release_grants(0, ["hold"]) > 0
    result = simulate_service(raw, memory=waiting, selected_allocations=[
        AllocationEffect.acquire("needs_release", "hbm", 8192)],
        lower=linear_work)
    assert result["status"] == "modeled"
    assert "selected:needs_release" in waiting.selected_items[(0, 0)]


def test_run_cohort_accepts_caller_owned_execution_memory_scheduler_and_lowerer():
    parsed = parse_case(case())
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    execution = Execution(
        parsed, CostModel(parsed), include_serving_io=True,
        memory_policy=memory.policy,
    )
    execution.engine.now = 0
    run = run_cohort(
        parsed, execution, [scheduler], memory,
        Frontend(), trace_limit=0,
    )
    assert run["serving_end"] >= 0
    assert run["model_steps"] >= 1
    assert run["stop_reason"] == "drained"
    assert run["engine"] is execution.engine
    assert execution.memory is memory
    assert execution.memory_ledger is memory
    assert scheduler.done is True


def test_assemble_service_uses_the_same_cohort_driver_and_accepts_multi_dp():
    full = assemble(pipeline_case(), full_service=True, trace_limit=0)
    assert full["assemble_backend"] == "cohort_driver"
    assert full["production_entry"] == "simulate_service"
    assert full["execution"]["assembly"]["microbatches"] == 2

    multi = assemble(case(dp=2), scope="service", trace_limit=0)
    assert multi["assemble_backend"] == "cohort_driver"
    assert multi["execution"]["cohort_stop"] == "drained"
    assert multi["execution"]["model_steps"] >= 1


def test_full_capture_idle_ep_participant_keeps_padding_work_without_fake_progress():
    raw = case(dp=2, ep=2, output=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2,
                        moe_intermediate_size=32)
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[1],
                          graph_capture_mode="full")
    result = simulate_service(raw, event_limit=0)
    assert result["status"] == "modeled"
    assert result["requests"][0]["generated_tokens"] == 2
    assert result["execution"]["iterations"] >= 2
    assert result["execution"]["live_tasks"] == 0
    assert result["execution"]["live_programs"] == 0
    # The idle participant's physical capture work remains in the same run;
    # only the logical emission map is empty for it.
    assert result["execution"]["assembly"]["declared_commits"] >= 1


def test_assemble_explicit_request_override_is_same_for_dp1_and_dp2():
    base = case(requests=[{"id": "a", "prompt_tokens": 4, "output_tokens": 4}])
    override = {"id": "b", "prompt_tokens": 4, "output_tokens": 3}
    for dp in (1, 2):
        raw = case(dp=dp, requests=[{"id": "a", "prompt_tokens": 4,
                                     "output_tokens": 4}])
        result = assemble(raw, scope="service", requests=[override],
                          phase="decode", context=4, iterations=1, tokens=2)
        assert result["assemble_backend"] == "cohort_driver"
        assert result["window"]["request_ids"] == ["b"]
        assert result["window"]["max_model_steps"] is None
        assert result["window"]["per_request_token_quota"] == 2
        assert result["window"]["unfinished_request_ids"] == ["b"]
        assert result["execution"]["model_steps"] == 2
        assert result["execution"]["cohort_stop"] == "token_quota"
        assert [row["request_id"] for row in result["requests"]] == ["b"]
        row = result["requests"][0]
        assert row["generated_tokens"] == 2
        assert row["visible_context_tokens"] == 6
        assert result["assembly"]["declared_commits"] == 2
        assert result["status"] == "incomplete"


def test_explicit_submission_selection_changes_real_launch_work():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    with_submit = assemble(raw, scope="service", iterations=1)
    without_submit = assemble(raw, scope="service", iterations=1,
                              include_submission=False)
    assert with_submit["execution"]["assembly"]["host_submits"] > 0
    assert without_submit["execution"]["assembly"]["host_submits"] == 0
    assert with_submit["execution"]["assembly"]["declared_commits"] == 1


def test_default_full_service_starts_from_real_prefill():
    raw = case(requests=[{"id": "r", "prompt_tokens": 4, "output_tokens": 2}])
    result = assemble(raw, scope="service", trace_limit=0)
    assert result["execution"]["cohort_stop"] == "drained"
    assert result["execution"]["assembly"]["declared_commits"] == 2
    assert [step["mode"] for step in result["steps"]] == ["prefill", "decode"]
    assert result["steps"][0]["valid_tokens"] == 4
    assert result["steps"][1]["valid_tokens"] == 1
    row = result["requests"][0]
    assert row["generated_tokens"] == 2
    assert row["visible_context_tokens"] == 5


def test_two_active_requests_keep_per_request_decode_quota_on_dp1_and_dp2():
    for dp in (1, 2):
        raw = case(dp=dp, requests=[
            {"id": "a", "prompt_tokens": 4, "output_tokens": 3},
            {"id": "b", "prompt_tokens": 4, "output_tokens": 3},
        ])
        raw["runtime"]["max_num_seqs"] = 2
        result = assemble(raw, scope="service", phase="decode", context=4,
                          iterations=1, tokens=2)
        assert result["window"]["per_request_token_quota"] == 2
        assert result["execution"]["assembly"]["declared_commits"] == 4
        assert result["execution"]["model_steps"] >= 2
        rows = {row["request_id"]: row for row in result["requests"]}
        for request_id in ("a", "b"):
            assert rows[request_id]["generated_tokens"] == 2
            assert rows[request_id]["visible_context_tokens"] == 6
        assert result["window"]["unfinished_request_ids"] == ["a", "b"]


def test_default_service_drains_while_explicit_one_step_is_a_window():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 2}])
    full = assemble(raw, scope="service", trace_limit=0)
    assert full["execution"]["cohort_stop"] == "drained"
    assert full["requests"][0]["generated_tokens"] == 2

    one = assemble(raw, scope="service", iterations=1, trace_limit=0)
    assert one["execution"]["cohort_stop"] == "token_quota"
    assert one["execution"]["model_steps"] == 1
    assert one["requests"][0]["generated_tokens"] == 1
    assert one["window"]["per_request_token_quota"] == 1
    assert one["window"]["unfinished_request_ids"] == ["r"]
    assert one["status"] == "incomplete"


def test_pp2_full_service_keeps_all_microbatch_commits():
    raw = case(pp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 2},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 2},
    ])
    raw["runtime"]["pipeline_microbatches"] = 2
    result = assemble(raw, scope="service", full_service=True, trace_limit=0)
    assert result["execution"]["assembly"]["declared_commits"] == 4
    assert result["execution"]["assembly"]["fallback_commits"] == 0
    assert result["execution"]["cohort_stop"] == "drained"
    assert [row["generated_tokens"] for row in result["requests"]] == [2, 2]


def test_assemble_default_service_respects_arrivals_and_single_seat_admission():
    raw = case(requests=[
        {"id": "r0", "prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 0},
        {"id": "r1", "prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 1000},
    ])
    raw["runtime"]["max_num_seqs"] = 1
    result = assemble(raw, scope="service", phase="prefill", iterations=2)
    assert result["assemble_backend"] == "cohort_driver"
    assert result["execution"]["cohort_stop"] == "max_model_steps"
    assert result["execution"]["model_steps"] == 2
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["r0"]["admission_ms"] == 0
    assert rows["r1"]["admission_ms"] >= 1000
    assert rows["r0"]["generated_tokens"] == rows["r1"]["generated_tokens"] == 1


def test_long_delivery_does_not_block_independent_next_request():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1},
    ])
    raw["runtime"]["max_num_seqs"] = 1
    raw["runtime"]["serving_io"] = {"client_delivery_ms": 1000}
    result = simulate_service(raw, event_limit=0)
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["b"]["finish_ms"] < rows["a"]["boundaries"]["client_delivery"]["finish_ms"]


def test_explicit_decode_window_reports_unfinished_state_via_driver():
    result = assemble(case(requests=[
        {"id": "r", "prompt_tokens": 4, "output_tokens": 2},
    ]), scope="service", phase="decode", context=4, iterations=1, tokens=1)
    assert result["assemble_backend"] == "cohort_driver"
    assert result["execution"]["cohort_stop"] == "token_quota"
    assert result["status"] == "incomplete"
    assert result["window"]["unfinished_request_ids"] == ["r"]
    row = result["requests"][0]
    assert row["generated_tokens"] == 1
    assert row["visible_context_tokens"] == 5
    assert row["output_tokens"] == 2
    assert row["status"] == "unserved"


def test_simulate_service_pd_uses_shared_two_pool_engine_path():
    result = simulate_service(pd_case(), lower=linear_work, event_limit=8)
    assert result["production_entry"] == "simulate_service"
    assert result["status"] == "modeled"
    assert result["execution"]["serviced"]["pd_link"] > 0
    assert result["execution"]["live_tasks"] == 0
    assert result["execution"]["live_programs"] == 0


def test_feedback_uses_committed_state_not_emission_count():
    result = simulate_service(case(requests=[
        {"id": "a", "prompt_tokens": 2, "output_tokens": 2},
    ]), event_limit=4)
    row = result["requests"][0]
    feedback_all = result["execution"]["scheduler_feedback"][0]
    feedback = feedback_all["requests"][0]
    assert feedback["generated_tokens"] == row["generated_tokens"] == 2
    assert feedback["visible_context_tokens"] == row["visible_context_tokens"] == 3
    assert feedback["state_version"] == row["state_version"]
    assert feedback_all["capacity"]["kv_capacity_bytes_by_stage"][0] > 0
    assert feedback_all["capacity"]["headroom_bytes_by_stage"][0] >= 0


def _binding_lower(plan, model, runtime, machine, dispatch_binding=None):
    """Cached-template producer: records carry stale slot/version on purpose."""
    from mindie_sim.schema.operators import OpSpec, IterationWork
    ops = []
    for view in plan.requests:
        record = {
            "request": "template", "direction": "read", "storage": "kv:L0",
            "bytes": 16, "span": (0, 16), "request_slot": 0,
            "state_version": 0, "reader": "template", "template": True,
        }
        shape = (1,)
        ops.append(OpSpec(
            f"op:{view.request_id}", "vector", view.phase, 0, 0,
            (view.request_id,), shape, shape, shape, shape, flops=1e6,
            vector_ops=1e6, metadata=(("state_accesses", (record,)),)))
    tokens = sum(view.q_len for view in plan.requests)
    return IterationWork(tuple(ops), plan, 1, tokens, tokens, None,
                         "body", "test_binding")


def _capture_binding(parsed, lower, versions):
    memory = Memory(parsed)
    schedulers = [
        Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    ] if parsed.runtime.parallel.dp == 1 else None
    if schedulers is None:
        groups = [[] for _ in range(parsed.runtime.parallel.dp)]
        for index, request in enumerate(sorted(
                parsed.workload.requests, key=lambda r: (r.arrival_ms, r.request_id))):
            groups[index % parsed.runtime.parallel.dp].append(request)
        schedulers = [Scheduler(parsed, group, memory, replica)
                      for replica, group in enumerate(groups)]
    by_id = {state.request.request_id: state
             for scheduler in schedulers for state in scheduler.states}
    for request_id, version in versions.items():
        by_id[request_id].state_version = int(version)
    seen = []

    def capture(op, replica, recipe, ready, complete):
        seen.append((replica, op.op_id, dict(op.metadata)))

    execution = Execution(
        parsed, CostModel(parsed), include_serving_io=False,
        memory=memory, memory_policy=memory.policy, on_operator=capture)
    execution.engine.now = 0
    run = run_cohort(parsed, execution, schedulers, memory, lower,
                     trace_limit=0)
    return run, seen


def _pp2_binding_case(dp):
    raw = case(dp=dp, pp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "c", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "d", "prompt_tokens": 1, "output_tokens": 1},
    ])
    raw["runtime"]["pipeline_microbatches"] = 2
    return raw


def test_pp2_microbatches_bind_original_slots_and_versions_not_local_positions():
    parsed = parse_case(_pp2_binding_case(dp=1))
    run, seen = _capture_binding(parsed, _binding_lower,
                                 {"a": 3, "b": 7, "c": 11, "d": 15})
    rebound = {}
    for _replica, op_id, metadata in seen:
        record = dict(metadata["state_accesses"][0])
        rebound[op_id.split(":", 1)[1]] = record
    # Original plan slots are [0,2] and [1,3], not local [0,1] positions.
    assert {request: record["request_slot"]
            for request, record in rebound.items()} == {
                "a": 0, "b": 1, "c": 2, "d": 3}
    assert {request: record["state_version"]
            for request, record in rebound.items()} == {
                "a": 3, "b": 7, "c": 11, "d": 15}
    assert all(record["request"] == request
               for request, record in rebound.items())
    assert all(record["reader"] == "op:" + request
               for request, record in rebound.items())
    assert all(record["microbatch"] in {0, 1} for record in rebound.values())
    bindings = run["assembly"]["batch_bindings"]
    assert {tuple(sorted(item["slots"])) for item in bindings} == {
        ("a", "c"), ("b", "d")}


def test_dp2_pp2_binding_keeps_each_replicas_real_allocation_identity():
    parsed = parse_case(_pp2_binding_case(dp=2))
    run, seen = _capture_binding(parsed, _binding_lower,
                                 {"a": 2, "b": 5, "c": 8, "d": 13})
    identity = {(replica, op_id.split(":", 1)[1],
                 dict(metadata["state_accesses"][0])["request_slot"],
                 dict(metadata["state_accesses"][0])["state_version"])
                for replica, op_id, metadata in seen}
    assert identity == {
        (0, "a", 0, 2), (0, "c", 1, 8),
        (1, "b", 0, 5), (1, "d", 1, 13),
    }
    per_replica = {}
    for item in run["assembly"]["batch_bindings"]:
        per_replica.setdefault(item["replica"], []).append(item["slots"])
    # Each DP replica has its own two-request plan; PP2 splits it into two
    # one-request microbatches with that replica's real slot 0/1.
    assert per_replica[0] == [{"a": 0}, {"c": 1}]
    assert per_replica[1] == [{"b": 0}, {"d": 1}]


def test_padding_plan_publishes_no_request_commit_or_delivery_row():
    from mindie_sim.prediction import dispatch
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.schema.workload import IterationPlan, RequestView

    parsed = parse_case(case())
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    execution = Execution(parsed, CostModel(parsed), include_serving_io=False,
                          memory=memory, memory_policy=memory.policy)
    execution.engine.now = 0
    padding = RequestView(
        "serving0:padding", 4, 0, False, "decode", padding=True,
        draft_verify=False, draft_prepare=False)
    plan = IterationPlan((padding,), iteration=0)
    steps = dispatch.submit_batch(
        parsed, [scheduler], {0: plan}, execution, Frontend(), 0)
    step = steps[0]
    assert step.commits == {}
    assert step.declared_commits == {}
    assert step.fallback_commits == {}
    assert step.host_results == {}
    assert step.client_deliveries == {}
    assert scheduler.states[0].generated == 0


def test_run_cohort_memory_conflict_preserves_caller_work_and_bindings():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    parsed = parse_case(raw)
    first = Memory(parsed)
    second = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), first, 0)
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True,
                          memory=first, memory_policy=first.policy)
    execution.engine.now = 0
    prior = execution.engine.submit("prior_work", {}, latency=10)
    completed = execution.engine.summary()["completed_tasks"]
    version = scheduler.states[0].state_version
    with pytest.raises(ModelingError, match="execution_memory_ledger_conflict"):
        run_cohort(parsed, execution, [scheduler], second, linear_work)
    assert execution.memory is first
    assert execution.engine.memory is first
    assert scheduler.memory is first
    assert scheduler.states[0].state_version == version
    assert prior.done is False
    assert execution.engine.summary()["completed_tasks"] == completed
    # The same caller binding still runs normally after the failed attempt.
    run = run_cohort(parsed, execution, [scheduler], first, linear_work)
    assert run["stop_reason"] == "drained"
    assert scheduler.done is True


def test_selected_allocation_waits_for_own_release_then_admits_same_run():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    parsed = parse_case(raw)
    memory = Memory(parsed)
    capacity = memory.capacity[0][0]
    used = (memory.live[0][0]
            + memory.staging_live[0]
            + memory.granted[0][0]
            + memory.shared_reserved[0][0]
            + memory.selected_reserved[0][0])
    hold = max(0.0, capacity - used - 4096.0)
    executable = Execution(parsed, CostModel(parsed), include_serving_io=True,
                           memory=memory, memory_policy=memory.policy)
    executable.engine.now = 0
    release = executable.engine.submit(
        "release_hold", {}, latency=5,
        on_complete=lambda _event: memory.release_grants(0, ["hold"]))
    assert memory.grant_batch(0, [{
        "name": "hold", "bytes": hold, "event": release}])
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    from mindie_sim.schema.operators import AllocationEffect
    effect = AllocationEffect.acquire("workspace", "hbm", 8192)
    run = run_cohort(
        parsed, executable, [scheduler], memory, linear_work,
        selected_allocations=[effect])
    assert run["selected_allocation_attached"] is True
    assert run["model_steps"] >= 1
    assert executable.engine.now >= 5
    assert "selected:workspace" in memory.selected_items[(0, 0)]
    assert scheduler.done is True


def test_selected_allocation_permanent_refusal_issues_zero_model_work():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    parsed = parse_case(raw)
    memory = Memory(parsed)
    capacity = memory.capacity[0][0]
    used = (memory.live[0][0]
            + memory.staging_live[0]
            + memory.granted[0][0]
            + memory.shared_reserved[0][0]
            + memory.selected_reserved[0][0])
    hold = max(0.0, capacity - used - 4096.0)
    executable = Execution(parsed, CostModel(parsed), include_serving_io=True,
                           memory=memory, memory_policy=memory.policy)
    executable.engine.now = 0
    release = executable.engine.submit("release_part", {}, latency=5)
    # Releasing one 2 KiB grant makes 6 KiB free; 8 KiB is still impossible.
    first = min(2048.0, hold)
    assert memory.grant_batch(0, [{
        "name": "hold_a", "bytes": first, "event": release}])
    if hold > first:
        assert memory.grant_batch(0, [{
            "name": "hold_b", "bytes": hold - first}])
    executable.engine.submit(
        "release_part_callback", after=(release,),
        on_complete=lambda _event: memory.release_grants(0, ["hold_a"]))
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory, 0)
    from mindie_sim.schema.operators import AllocationEffect
    with pytest.raises(ModelingError, match="waiting_for_release.*permanent=1"):
        run_cohort(
            parsed, executable, [scheduler], memory, linear_work,
            selected_allocations=[
                AllocationEffect.acquire("workspace", "hbm", 8192)])
    assert scheduler.states[0].iterations == 0
    assert scheduler.states[0].generated == 0
    assert executable.work["flops"] == 0


def test_chain_a_to_b_waits_real_delivery_and_short_window_stops_its_edges():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1,
         "gap_ms": 2},
        {"id": "c", "prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 1000},
    ])
    raw["runtime"]["serving_io"] = {"client_delivery_ms": 6}
    result = predict(raw, lower=linear_work, max_model_steps=1)
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["a"]["boundaries"]["client_delivery"]["finish_ms"] == 7
    assert rows["b"]["admission_ms"] is None
    assert rows["c"]["admission_ms"] is None
    assert result["execution"]["cohort_stop"] == "max_model_steps"

    full_raw = case(requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1,
         "gap_ms": 2},
        {"id": "c", "prompt_tokens": 1, "output_tokens": 1, "arrival_ms": 1000},
    ])
    full_raw["runtime"]["serving_io"] = {"client_delivery_ms": 6}
    full = predict(full_raw, lower=linear_work, max_model_steps=2)
    full_rows = {row["request_id"]: row for row in full["requests"]}
    assert full_rows["a"]["boundaries"]["client_delivery"]["finish_ms"] == 7
    assert full_rows["b"]["admission_ms"] == 9
    assert full_rows["b"]["finish_ms"] == 10
    assert full_rows["c"]["generated_tokens"] == 0
    assert full["execution"]["cohort_stop"] == "max_model_steps"


def test_two_service_deliveries_drain_at_eleven_and_leave_external_timer():
    raw = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 5, "output_tokens": 1},
    ])
    raw["runtime"]["serving_io"] = {"client_delivery_ms": 6}
    parsed = parse_case(raw)
    memory = Memory(parsed)
    groups = [[], []]
    for index, request in enumerate(sorted(
            parsed.workload.requests, key=lambda r: (r.arrival_ms, r.request_id))):
        groups[index % 2].append(request)
    schedulers = [Scheduler(parsed, group, memory, replica)
                  for replica, group in enumerate(groups)]
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True,
                          memory=memory, memory_policy=memory.policy)
    execution.engine.now = 0
    timer = execution.engine.submit("unrelated_timer", {}, latency=1000)
    run = run_cohort(parsed, execution, schedulers, memory, linear_work)
    assert run["stop_reason"] == "drained"
    assert run["serving_end"] == 11
    assert all(scheduler.done for scheduler in schedulers)
    assert timer.done is False


def test_quota_keeps_original_output_tokens_and_chain_successor_at_zero():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 3, "chain_id": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1},
    ])
    raw["runtime"]["max_num_seqs"] = 2
    result = assemble(raw, scope="service", phase="decode", context=4,
                      iterations=1, tokens=2, lower=linear_work)
    rows = {row["request_id"]: row for row in result["requests"]}
    assert rows["a"]["generated_tokens"] == 2
    assert rows["a"]["output_tokens"] == 3
    assert rows["a"]["status"] == "unserved"
    assert rows["b"]["generated_tokens"] == 0
    assert rows["b"]["admission_ms"] is None
    assert result["window"]["original_output_tokens"] == {"a": 3, "b": 1}
    assert result["window"]["unfinished_request_ids"] == ["a", "b"]
    assert result["execution"]["cohort_stop"] == "token_quota"
    assert result["execution"]["assembly"]["fallback_commits"] == 2


def test_two_contexts_share_memory_and_only_new_issue_adds_its_grant():
    raw = case(requests=[{"id": "r", "prompt_tokens": 1, "output_tokens": 1}])
    parsed = parse_case(raw)
    memory = Memory(parsed)
    from mindie_sim.schema.operators import AllocationEffect
    first = simulate_service(
        raw, memory=memory,
        selected_allocations=[AllocationEffect.acquire("a_ws", "hbm", 4096)],
        lower=linear_work)
    assert first["status"] == "modeled"
    assert memory.selected_reserved[0][0] == 4096
    second = simulate_service(
        raw, memory=memory,
        selected_allocations=[AllocationEffect.acquire("b_ws", "hbm", 4096)],
        lower=linear_work)
    assert second["status"] == "modeled"
    # A's already committed object and grant are retained; B adds its own.
    assert "selected:a_ws" in memory.selected_items[(0, 0)]
    assert "selected:b_ws" in memory.selected_items[(0, 0)]
    assert memory.selected_reserved[0][0] == 8192


def test_release_wakeup_uses_five_ms_not_blocked_dp_or_unrelated_timer():
    raw = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 1},
    ])
    parsed = parse_case(raw)
    memory = Memory(parsed)
    groups = [[], []]
    for index, request in enumerate(sorted(
            parsed.workload.requests, key=lambda r: (r.arrival_ms, r.request_id))):
        groups[index % 2].append(request)
    schedulers = [Scheduler(parsed, group, memory, replica)
                  for replica, group in enumerate(groups)]
    execution = Execution(parsed, CostModel(parsed), include_serving_io=False,
                          memory=memory, memory_policy=memory.policy)
    execution.engine.now = 0
    timer = execution.engine.submit("unrelated_timer", {}, latency=1000)
    capacity = memory.capacity[0][0]
    used = (memory.live[0][0] + memory.staging_live[0]
            + memory.granted[0][0] + memory.shared_reserved[0][0]
            + memory.selected_reserved[0][0])
    hold = max(0.0, capacity - used - 1.0)
    release = execution.engine.submit(
        "l0_last_read", {}, latency=5,
        on_complete=lambda _event: memory.release_grants(0, ["l0_hold"]))
    assert memory.grant_batch(0, [{
        "name": "l0_hold", "bytes": hold, "event": release}])

    def slow_b(plan, model, runtime, machine):
        from mindie_sim.schema.operators import OpSpec, IterationWork
        tokens = sum(view.q_len for view in plan.requests)
        scale = 20.0 if any(view.request_id == "b" for view in plan.requests) else 1.0
        shape = (tokens,)
        op = OpSpec("work", "vector", "body", 0, 0,
                    tuple(view.request_id for view in plan.requests),
                    shape, shape, shape, shape, flops=tokens * 1e6 * scale,
                    vector_ops=tokens * 1e6 * scale, dtype="bf16")
        return IterationWork((op,), plan, 1, tokens, tokens, None, "body",
                             "test_slow")

    run = run_cohort(parsed, execution, schedulers, memory, slow_b)
    assert run["stop_reason"] == "drained"
    a = schedulers[0].states[0]
    b = schedulers[1].states[0]
    assert a.admission_ms == 5
    assert a.finish_ms == 6
    assert b.finish_ms == 20
    assert timer.done is False
