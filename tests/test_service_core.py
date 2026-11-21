"""Scheduler/state feedback core for model-service assembly.

[Role]
  Establish that scheduled batches reserve and then commit versioned request
  state, that speculative acceptance changes later work and KV volume, that a
  transient async lookahead reservation cannot unserved an otherwise feasible
  request, and that the production A5 serving entry observes the same state.

[Invariants]
  * A committed state version is visible only after the batch that produced it;
    ``pending_context`` is never read as committed context.
  * Emissions are the integer commits the same acceptance policy applies later.
  * Rejected speculative rows remain reserved during verification and are
    released only by commit; accepted rows extend the next context.
  * KV feedback is derived from the live cache ledger, not from a measured
    latency or a token history.
  * The production entry is exercised on a real A5 case, not a fabricated
    service engine.

[Timeouts & Locks]
  Tier 0 scheduler arithmetic plus one bounded A5 serving run; no I/O, device,
  weights or locks.

[Side-Effect Policy]
  Local cases and local ledgers only; the A5 case is built in memory.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim import predict
from mindie_sim.compose import compose_case
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.prediction.body_defaults import attention_defaults
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import RequestView
from test_draft_and_media import draft_case
from test_prediction_system import case


class _TinyMemory:
    """Capacity double with explicit lookahead reservation targets.

    The values are abstract bytes; only admission ordering and visibility are
    under test.  The rejected target is the one the scheduler asks for when it
    adds a next-step lookahead, while the real row it must execute is smaller.
    """

    async_in_flight_lookahead_steps = 2
    async_delayed_release_steps = 0

    def __init__(self, capacity=5, rejected=(6,), reject_once=()):
        self.owners = {}
        self.live = [[0.0]]
        self.capacity = [[float(capacity)]]
        self.rejected = set(rejected)
        self.reject_once = set(reject_once)

    def resize(self, replica, request, tokens):
        if tokens in self.reject_once:
            self.reject_once.remove(tokens)
            return False
        if tokens in self.rejected or tokens > self.capacity[replica][0]:
            return False
        self.owners[replica, request] = (float(tokens),)
        self.live[replica][0] = float(tokens)
        return True

    def release(self, replica, request):
        sizes = self.owners.pop((replica, request), (0.0,))
        self.live[replica][0] = max(0.0, self.live[replica][0] - sum(sizes))

    def has_delayed_releases(self, replica):
        return False

    def step_delayed_releases(self, replica):
        return None

    def flush_delayed_releases(self, replica=None):
        return None


def _draft_case(trace=(0, 2), output=4, prompt=4):
    raw = case(requests=[{"id": "r", "prompt_tokens": prompt, "output_tokens": output}])
    raw["runtime"]["metadata"] = {
        "draft_mode": "mtp", "draft_proposal_tokens": 3,
        "draft_accept_trace": list(trace), "draft_accept_tokens": 1,
    }
    return parse_case(raw)


def test_scheduler_commits_versioned_state_from_integer_acceptance():
    parsed = _draft_case()
    memory = Memory(parsed)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
    state = scheduler.states[0]

    plan = scheduler.schedule(0)
    assert plan is not None and scheduler.last_reason == "scheduled"
    assert plan.requests[0].phase == "prefill"
    prefill_view = plan.requests[0]
    assert state.state_version == 0
    assert (state.pending_context, state.reserved_tokens) == (4, 4)
    assert scheduler.emissions(plan) == {"r": 1}
    assert scheduler.feedback(0)["requests"][0]["status"] == "in_flight"

    scheduler.commit(1.0, {"r": 1.0})
    assert state.state_version == 1
    assert (state.computed, state.generated) == (4, 1)
    assert (state.pending_context, state.reserved_tokens) == (None, 0)

    plan = scheduler.schedule(1.0)
    assert plan is not None
    view = plan.requests[0]
    assert view is not prefill_view
    assert (view.phase, view.context, view.draft_proposal_tokens) == ("decode", 4, 2)
    assert scheduler.emissions(plan) == {"r": 1}
    assert (state.pending_context, state.reserved_tokens) == (7, 3)
    scheduler.commit(2.0, {"r": 2.0})
    assert (state.computed, state.generated, state.state_version) == (5, 2, 2)
    assert (state.accepted, state.rolled_back) == (0, 2)

    plan = scheduler.schedule(2.0)
    assert plan is not None
    assert plan.requests[0].draft_proposal_tokens == 1
    assert scheduler.emissions(plan) == {"r": 2}
    scheduler.commit(3.0, {"r": 3.0})
    assert (state.computed, state.generated, state.state_version) == (7, 4, 3)
    assert (state.accepted, state.rolled_back) == (1, 2)
    assert state.finish_ms == 3.0

    feedback = scheduler.feedback(3.0, include_done=True)
    request = feedback["requests"][0]
    assert request["status"] == "done"
    assert request["visible_context_tokens"] == 7
    assert request["generated_tokens"] == 4
    assert request["state_version"] == 3
    assert request["pending_context_tokens"] is None
    assert feedback["in_flight_requests"] == []
    assert feedback["admission"]["infeasible_requests"] == []
    assert feedback["capacity"]["owner_bytes_by_request"] == {}


def test_padding_lane_never_emits_or_touches_request_progress():
    parsed = _draft_case(trace=(), output=2)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    plan = scheduler.schedule(0)
    assert plan is not None
    active = plan.requests[0]
    padding = RequestView("serving1:padding", 4, 0, False, "decode",
                          padding=True, draft_verify=False)
    mixed = replace(plan, requests=(active, padding))
    assert scheduler.emissions(mixed) == {"r": 1}
    pure = replace(plan, requests=(padding,))
    assert scheduler.emissions(pure) == {}
    before = [state.state_version for state in scheduler.states]
    scheduler.commit(1.0, {"r": 1.0})
    assert [state.state_version for state in scheduler.states] == [value + 1 for value in before]
    assert all(state.finish_ms is None for state in scheduler.states)
    ghost = replace(plan, requests=(replace(active, request_id="ghost"),))
    with pytest.raises(Exception, match="scheduler_emission_request_not_registered"):
        scheduler.emissions(ghost)


def test_chunked_prefill_commits_only_at_the_sampling_boundary():
    raw = case(requests=[{"id": "r", "prompt_tokens": 8, "output_tokens": 2}])
    raw["runtime"]["enable_chunked_prefill"] = True
    raw["runtime"]["max_num_batched_tokens"] = 4
    parsed = parse_case(raw)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    state = scheduler.states[0]

    first = scheduler.schedule(0)
    assert first is not None and first.requests[0].q_len == 4
    assert first.requests[0].samples is False
    assert scheduler.emissions(first) == {}
    scheduler.commit(1.0, {})
    assert (state.computed, state.generated, state.state_version) == (4, 0, 1)

    second = scheduler.schedule(1.0)
    assert second is not None and second.requests[0].q_len == 4
    assert second.requests[0].samples is True
    assert scheduler.emissions(second) == {"r": 1}
    scheduler.commit(2.0, {"r": 2.0})
    assert (state.computed, state.generated, state.state_version) == (8, 1, 2)

    decode = scheduler.schedule(2.0)
    assert decode is not None and decode.requests[0].phase == "decode"
    assert decode.requests[0].context == 8


def test_lookahead_reservation_falls_back_before_unserving_a_request():
    raw = case(requests=[{"id": "r", "prompt_tokens": 3, "output_tokens": 3}])
    parsed = parse_case(raw)
    memory = _TinyMemory(capacity=6, rejected=(), reject_once=(5,))
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
    state = scheduler.states[0]

    plan = scheduler.schedule(0)
    assert plan is not None and plan.requests[0].q_len == 3
    scheduler.commit(1.0, {"r": 1.0})
    assert state.computed == 3

    # Decode would reserve one produced row plus one lookahead row, but only
    # the produced row fits.  The batch must execute, not be marked unserved.
    plan = scheduler.schedule(1.0)
    assert plan is not None
    assert plan.requests[0].q_len == 1
    assert state.reserved_tokens == 1
    scheduler.commit(2.0, {"r": 2.0})

    plan = scheduler.schedule(2.0)
    assert plan is not None
    scheduler.commit(3.0, {"r": 3.0})
    assert scheduler.last_reason == "scheduled"
    assert state.finish_ms == 3.0
    assert state.reason is None
    assert scheduler.feedback(3.0)["admission"]["capacity_or_budget_exhausted"] is False


def test_capacity_feedback_reports_real_infeasibility_not_a_timing_floor():
    parsed = _draft_case(trace=(), output=2)
    memory = _TinyMemory(capacity=2, rejected=())
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)

    assert scheduler.schedule(0) is None
    assert scheduler.last_reason == "capacity_or_budget_exhausted"
    feedback = scheduler.feedback(0)
    assert feedback["admission"]["capacity_or_budget_exhausted"] is True
    assert feedback["admission"]["infeasible_requests"] == ["r"]
    assert feedback["requests"][0]["reason"] == "insufficient_cache_or_token_budget"
    assert feedback["capacity"]["kv_capacity_bytes_by_stage"] == [2.0]
    assert feedback["capacity"]["headroom_bytes_by_stage"] == [2.0]


def test_non_speculative_decode_cannot_commit_two_ungenerated_tokens():
    parsed = _draft_case(trace=(), output=2)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    scheduler.schedule(0)
    scheduler.commit(1.0, {"r": 1.0})
    plan = scheduler.schedule(1.0)
    assert plan is not None and plan.requests[0].q_len == 1
    bad = replace(plan, requests=(replace(plan.requests[0], q_len=2),))
    with pytest.raises(Exception, match="non_speculative_decode_forward_must_commit_one_token"):
        scheduler.emissions(bad)
    with pytest.raises(Exception, match="non_speculative_decode_forward_must_commit_one_token"):
        scheduler.by_id["r"].commit(bad.requests[0], 2.0)


def _memory_with_free_budget(raw, extra_bytes):
    """Build the real mounted Memory with exactly ``extra_bytes`` free."""
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    base = parse_case(raw)
    base_memory = Memory(base, MemoryPolicy(derive_workspace=False))
    capacity = int(base_memory.fixed_per_rank[0] + extra_bytes)
    parsed = replace(base, machine=replace(
        base.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in base.machine.dies)))
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert memory.capacity[0][0] == pytest.approx(extra_bytes)
    return parsed, memory


def test_transient_copy_grant_waits_for_actual_release_then_admits():
    parsed, memory = _memory_with_free_budget(
        case(prompt=1, output=1), 12288)
    # The copy grant occupies all free headroom until its real release at t=1.
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
    assert scheduler.grant_state([("copy", 12288.0)]) is True
    emitted = scheduler.schedule(0)
    assert emitted is None
    assert scheduler.last_reason == "waiting_for_release"
    assert scheduler.states[0].reason is None
    assert scheduler.states[0].generated == 0
    assert scheduler.states[0].state_version == 0
    assert memory.granted[0][0] == 12288.0
    assert scheduler.release_state_grants(("copy",)) == pytest.approx(12288.0)
    scheduler.now = 1.0
    plan = scheduler.schedule(1.0)
    assert plan is not None
    assert scheduler.last_reason == "scheduled"
    assert scheduler.emissions(plan) == {"r": 1}
    assert scheduler.states[0].generated == 0
    scheduler.commit(1.0, {"r": 1.0})
    assert scheduler.states[0].generated == 1


def test_twelve_kib_retirement_keeps_unfinished_reader_and_reauths_after_release():
    raw = case(requests=[
        {"id": "L0", "prompt_tokens": 1, "output_tokens": 5},
        {"id": "L1", "prompt_tokens": 1, "output_tokens": 5},
        {"id": "L2", "prompt_tokens": 1, "output_tokens": 5},
    ])
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    base = parse_case(raw)
    base_memory = Memory(base, MemoryPolicy(derive_workspace=False))
    capacity = int(base_memory.weights[0] + 12288)
    parsed = replace(base, machine=replace(
        base.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in base.machine.dies)))
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    for request in ("L0", "L1", "L2"):
        assert memory.resize(0, request, 32)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
    assert scheduler.hold_state_consumer(
        "L0", "reader", storage="gqa_kv", ranges=((0, 32),),
        generation=7, event="reader:event")
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
    parsed = _draft_case(trace=(), output=1)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    assert scheduler.schedule(0) is not None
    assert scheduler.hold_state_consumer("r", "reader")
    scheduler.retire_state("r", force=True)
    assert (0, "r") not in scheduler.memory.owners
    assert scheduler.memory.pending_async_consumers(0, "r") == ()
    assert scheduler.feedback(0)["pending_retired_requests"] == []


def test_preemption_keeps_an_unfinished_reader_and_blocks_growth_until_release():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 10},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 10},
    ])
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False},
                          max_num_batched_tokens=1)
    base = parse_case(raw)
    base_memory = Memory(base, MemoryPolicy(derive_workspace=False))
    capacity = int(base_memory.weights[0] + 1024)
    parsed = replace(base, machine=replace(
        base.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in base.machine.dies)))
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, "a", 4)
    assert memory.resize(0, "b", 4)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
    for state in scheduler.states:
        state.computed = 4
        state.started = True
        state.admission_ms = 0.0
        state.location = "decode"
    assert memory.hold_for_async_consumer(0, "b", "reader")
    assert scheduler.schedule(0) is None
    assert scheduler.last_reason == "waiting_for_release"
    assert scheduler.states[0].started
    assert scheduler.states[1].started is False
    assert scheduler.states[1].waiting_for_release is True
    assert scheduler.states[1].reason is None
    assert memory.owners[(0, "a")] == (512.0,)
    assert memory.owners[(0, "b")] == (512.0,)
    assert memory.live == [[1024.0]]
    # A's 640 B growth cannot be admitted while B's real 512 B is held.
    assert scheduler.states[0].computed == 4
    assert scheduler.release_state_consumer("b", "reader") is True
    assert (0, "b") not in memory.owners
    plan = scheduler.schedule(1.0)
    assert plan is not None
    assert scheduler.emissions(plan) == {"a": 1}
    assert scheduler.states[0].computed == 4
    assert memory.live == [[640.0]]


def test_async_reader_holds_cross_two_service_iterations_and_last_use_release():
    parsed = _draft_case(trace=(), output=2)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    first = scheduler.schedule(0)
    assert first is not None
    scheduler.commit(1.0, {"r": 1.0})
    assert scheduler.hold_state_consumer("r", "pd-reader") is True
    second = scheduler.schedule(1.0)
    assert second is not None
    scheduler.commit(2.0, {"r": 2.0})
    assert scheduler.states[0].finish_ms == 2.0
    assert scheduler.memory.is_retired(0, "r")
    assert (0, "r") in scheduler.memory.owners
    assert scheduler.feedback(1.0)["pending_retired_requests"] == ["r"]
    assert scheduler.release_state_consumer("r", "pd-reader") is True
    assert (0, "r") not in scheduler.memory.owners


def test_t08_twelve_kib_grants_are_atomic_and_the_fourth_waits():
    parsed, memory = _memory_with_free_budget(case(output=1), 12288)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
    assert scheduler.grant_state([("a", 4096.0), ("b", 4096.0),
                                  ("c", 4096.0)]) is True
    assert scheduler.feedback(0)["capacity"]["granted_bytes_by_stage"] == [12288.0]
    assert scheduler.grant_state([("d", 4096.0)]) is False
    assert scheduler.feedback(0)["capacity"]["granted_bytes_by_stage"] == [12288.0]
    assert scheduler.release_state_grants(("a",)) == pytest.approx(4096.0)
    assert scheduler.grant_state([("d", 4096.0)]) is True
    assert scheduler.feedback(0)["capacity"]["granted_bytes_by_stage"] == [12288.0]


def test_scheduler_consumes_t08_union_grants_and_async_holds():
    parsed = _draft_case(trace=(), output=2)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    plan = scheduler.schedule(0)
    assert plan is not None
    assert scheduler.grant_state([("g1", 1024.0), ("g2", 2048.0)]) is True
    feedback = scheduler.feedback(0)
    assert feedback["capacity"]["granted_bytes_by_stage"] == [3072.0]
    assert scheduler.grant_state([("dup", 1.0), ("dup", 1.0)]) is False
    assert scheduler.release_state_grants(("g1", "g2")) == pytest.approx(3072.0)

    assert scheduler.retain_shared_state("shared-prefix", [0.0], version=7)
    assert scheduler.release_shared_state("shared-prefix")
    assert scheduler.hold_state_consumer(
        "r", "reader-1", storage="gqa_kv", ranges=((0, 1),),
        generation=7, event="reader:event")
    binding = scheduler.feedback(0)["requests"][0]["async_consumer_bindings"]
    assert binding["reader-1"] == {
        "event": "reader:event", "generation": 7, "ranges": ((0, 1),),
        "storage": "gqa_kv"}
    scheduler.retire_state("r")
    assert scheduler.memory.is_retired(0, "r")
    assert scheduler.memory.owners[(0, "r")][0] > 0
    assert scheduler.feedback(0)["pending_retired_requests"] == ["r"]
    assert scheduler.release_state_consumer("r", "reader-1")
    assert (0, "r") not in scheduler.memory.owners


def test_commit_batch_is_idempotent_for_the_published_prefix():
    parsed = _draft_case(trace=(), output=2)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    plan = scheduler.schedule(0)
    assert plan is not None
    producer = SimpleNamespace(time=1.0)
    step = SimpleNamespace(completion=producer,
                           commits={"r": SimpleNamespace(time=1.0)})
    scheduler.commit_batch(step)
    version = scheduler.states[0].state_version
    generated = scheduler.states[0].generated
    scheduler.commit_batch(step)
    assert scheduler.states[0].state_version == version
    assert scheduler.states[0].generated == generated


def test_commit_batch_uses_actual_commit_events():
    parsed = _draft_case(trace=(), output=2)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    plan = scheduler.schedule(0)
    assert plan is not None and scheduler.emissions(plan) == {"r": 1}
    step = SimpleNamespace(
        completion=SimpleNamespace(time=4.5),
        commits={"r": SimpleNamespace(time=4.0)},
    )
    scheduler.commit_batch(step)
    assert scheduler.states[0].first_token_ms == 4.0
    assert scheduler.states[0].state_version == 1


def test_scheduler_drives_real_execution_without_waiting_for_every_delivery():
    raw = case(output=3, requests=[
        {"id": "r0", "prompt_tokens": 4, "output_tokens": 3},
        {"id": "r1", "prompt_tokens": 4, "output_tokens": 3},
    ])
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582",
                      "num_devices": 1}
    raw["runtime"]["serving_io"] = {
        "host_prepare_ms": .2, "host_result_ms": .2,
        "h2d": {"bytes_per_s": 8000, "bytes_per_token": 8, "latency_us": 50},
        "d2h": {"bytes_per_s": 4000, "bytes_per_token": 4, "latency_us": 50},
        "client_delivery_ms": .5,
    }
    case_spec = parse_case(raw)
    memory = Memory(case_spec)
    execution = Execution(case_spec, CostModel(case_spec),
                          include_serving_io=True)
    frontend = Frontend(attention_policy=AttentionPolicy(
        **case_spec.runtime.attention_policy))
    scheduler = Scheduler(case_spec, list(case_spec.workload.requests), memory)
    engine = execution.engine
    engine.now = 0.0
    batches, decisions = 0, []
    while not scheduler.done:
        plan = scheduler.schedule(engine.now)
        if plan is None:
            feedback = scheduler.feedback(engine.now, include_done=False)
            if feedback["admission"]["waiting_for_release"]:
                tails = tuple(execution.serving_io.client_tails.values())
                if tails:
                    engine.advance(until=tails)
                    continue
            arrival = scheduler.next_arrival(engine.now)
            if arrival is not None:
                engine.advance(horizon=arrival)
                engine.now = arrival
                continue
            break
        emitted = scheduler.emissions(plan)
        program = frontend.program(
            plan, case_spec.model, scheduler.next_runtime(plan),
            case_spec.machine, assembly="service")
        step = execution.add(program.work, 0, engine.now,
                             emitted_tokens=emitted)
        # The next batch is ready at device/step completion; client delivery
        # may still be in flight and must not serialize the loop.
        engine.advance(until=(step.completion,), first=False)
        scheduler.commit_batch(step)
        batches += 1
        decisions.append(scheduler.feedback(engine.now, include_done=False))
    engine.advance()
    assert batches == 3
    assert [state.state_version for state in scheduler.states] == [3, 3]
    assert all(not decision["in_flight_requests"] for decision in decisions)
    assert all(decision["requests"] for decision in decisions[:-1])
    assert decisions[-1]["requests"] == []
    assert scheduler.done
    assert execution.summary()["live_tasks"] == 0
    assert execution.summary()["live_programs"] == 0


def test_dp_schedulers_share_one_memory_ledger():
    raw = case(dp=2, requests=[
        {"id": "r0", "prompt_tokens": 4, "output_tokens": 3},
        {"id": "r1", "prompt_tokens": 4, "output_tokens": 3},
    ])
    parsed = parse_case(raw)
    memory = Memory(parsed)
    schedulers = [
        Scheduler(parsed, [parsed.workload.requests[index]], memory, replica=index)
        for index in range(2)
    ]
    assert schedulers[0].memory is memory is schedulers[1].memory
    plans = [scheduler.schedule(0) for scheduler in schedulers]
    assert all(plan is not None for plan in plans)
    assert memory.live[0][0] > 0 and memory.live[1][0] > 0
    for replica, scheduler in enumerate(schedulers):
        feedback = scheduler.feedback(0)
        assert feedback["capacity"]["cache_bytes_by_stage"][0] > 0
        assert set(feedback["capacity"]["owner_bytes_by_request"]) == {
            f"r{replica}"}
    schedulers[0].commit(1.0, {"r0": 1.0})
    assert schedulers[1].in_flight and schedulers[1].states[0].pending_context is not None
    assert schedulers[1].feedback(1.0)["requests"][0]["status"] == "in_flight"


def test_scheduler_rejects_a_second_inflight_batch_and_feedback_omits_done():
    parsed = _draft_case(trace=(), output=2)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    plan = scheduler.schedule(0)
    assert plan is not None
    with pytest.raises(Exception, match="scheduler_batch_already_in_flight"):
        scheduler.schedule(0)
    scheduler.commit(1.0, {"r": 1.0})
    assert scheduler.schedule(1.0) is not None
    scheduler.commit(2.0, {"r": 2.0})
    assert scheduler.states[0].finish_ms == 2.0
    assert scheduler.feedback(2.0, include_done=True)["requests"][0]["status"] == "done"
    decision = scheduler.feedback(2.0, include_done=False)
    assert decision["requests"] == []
    assert decision["capacity"]["owner_bytes_by_request"] == {}


def test_observed_service_regime_does_not_select_an_algorithm():
    raw = case(output=1)
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582",
                      "num_devices": 1}
    raw["runtime"]["kernels"] = {
        "service_regimes": [{"id": "observed", "mode": "memory",
                             "phases": ["decode"]}],
    }
    parsed = parse_case(raw)
    effective, report = attention_defaults(parsed, "decode", tokens=1)
    decision = next(row for row in report["selections"]
                    if row["family"] == "gqa_decode")
    assert decision["selection"] == "default_installed"
    recipe = effective.runtime.kernels["recipes"]["flash_attention_decode"]
    assert recipe["kind"] == "staged_gqa_decode"


def test_observed_service_regime_does_not_preserve_kda_algorithm():
    raw = compose_case("kimi-k3", hardware="Ascend950DT_9582", tp=8, dp=1,
                       ep=1, num_devices=8, layers=1)
    raw["runtime"]["kernels"] = {
        "service_regimes": [{"id": "observed", "mode": "memory",
                             "phases": ["decode"]}],
    }
    parsed = parse_case(raw)
    effective, report = attention_defaults(parsed, "decode", tokens=1)
    decision = next(row for row in report["selections"]
                    if row["family"] == "kda_decode")
    assert decision["selection"] == "default_installed"
    kda = effective.runtime.kernels["lowerings"]["kda"]
    assert (kda if isinstance(kda, str) else kda["implementation"]).startswith(
        "vllm_ascend_recurrent_kda_")


def test_quantized_gqa_uses_the_declared_fattention_plan_and_preserves_override():
    from dataclasses import replace
    from mindie_sim.catalog.quant import quant_preset

    raw = case(output=1)
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582",
                      "num_devices": 1}
    parsed = parse_case(raw)
    quant = replace(quant_preset("w8a8_dynamic"), kv_dtype="int8")
    parsed = replace(parsed, model=replace(parsed.model, quant=quant))
    effective, report = attention_defaults(parsed, "decode", tokens=1)
    decision = next(row for row in report["selections"]
                    if row["family"] == "gqa_decode")
    assert decision["selection"] == "default_installed"
    recipe = effective.runtime.kernels["recipes"]["flash_attention_decode"]
    assert recipe["kind"] == "staged_gqa_decode"
    explicit, explicit_report = attention_defaults(
        replace(parsed, runtime=replace(parsed.runtime, kernels={
            "recipes": {"flash_attention_decode": {"kind": "staged_gqa_decode",
                                                   "kv_splits": 2}}})),
        "decode", tokens=1)
    explicit_decision = next(row for row in explicit_report["selections"]
                             if row["family"] == "gqa_decode")
    assert explicit_decision["reason"] == "explicit_implementation_preserved"
    assert explicit.runtime.kernels["recipes"]["flash_attention_decode"]["kv_splits"] == 2


def test_a5_production_entry_commits_the_scheduled_state():
    # A5 hardware production predict on the small GQA+MTP fixture.  This keeps
    # the test independent of the concurrently changing GDN/MoE engine while
    # still exercising the normal compose/parse/predict path, serving I/O,
    # speculative acceptance and committed-state feedback.
    a5 = parse_case({**case(),
                     "machine": {"backend": "ascend",
                                 "sku": "Ascend950DT_9582",
                                 "num_devices": 1}})
    base = draft_case()
    case_spec = replace(
        base, machine=a5.machine,
        runtime=replace(base.runtime, serving_io={
            "host_prepare_ms": .2, "host_result_ms": .2,
            "h2d": {"bytes_per_s": 8000, "bytes_per_token": 8,
                    "latency_us": 50},
            "d2h": {"bytes_per_s": 4000, "bytes_per_token": 4,
                    "latency_us": 50},
            "client_delivery_ms": .5,
        }))
    result = predict(case_spec, trace_limit=1, event_limit=120)

    assert result["status"] == "modeled"
    assert result["execution_status"] == "completed"
    assert result["execution"]["live_tasks"] == 0
    for row in result["requests"]:
        assert row["generated_tokens"] == 6
        assert row["status"] == "served"
        assert row["state_version"] > 0
        assert row["committed_tokens"] == row["generated_tokens"]
        assert row["visible_context_tokens"] == (
            row["prompt_tokens"] + row["generated_tokens"] - 1)
        assert row["pending_context_tokens"] is None
        assert row["reserved_tokens"] == 0
        assert row["speculative"]["accepted_tokens"] >= 1
        assert row["tpot_ms"] == pytest.approx(
            (row["last_token_abs_ms"] - row["first_token_abs_ms"])
            / (row["output_tokens"] - 1))
        delivery = row["boundaries"]["client_delivery"]
        assert delivery["generated_tokens"] == row["generated_tokens"]
        assert delivery["e2e_ms"] > row["e2e_ms"]


def _preempted_reader_fixture():
    raw = case(requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 10},
        {"id": "b", "prompt_tokens": 4, "output_tokens": 10},
    ])
    raw["runtime"].update(max_num_batched_tokens=4, max_num_seqs=2)
    parsed, memory = _memory_with_free_budget(raw, 1024)
    assert memory.resize(0, "a", 4)
    assert memory.resize(0, "b", 4)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
    for state in scheduler.states:
        state.computed = 4
        state.generated = 1
        state.started = True
        state.admission_ms = 0.0
        state.location = "decode"
    assert memory.hold_for_async_consumer(0, "b", "reader")
    assert scheduler.schedule(0) is None
    assert scheduler.last_reason == "waiting_for_release"
    assert scheduler.states[1].waiting_for_release is True
    return scheduler, memory


def _after_last_read_recompute_then_decode(release):
    scheduler, memory = _preempted_reader_fixture()
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


def test_direct_memory_last_read_reauths_the_same_request_as_the_wrapper():
    _after_last_read_recompute_then_decode(
        lambda scheduler, memory: memory.release_async_consumer(
            0, "b", "reader"))
    _after_last_read_recompute_then_decode(
        lambda scheduler, memory: scheduler.release_state_consumer(
            "b", "reader"))


def test_permanent_minimum_span_reports_failure_and_keeps_shared_reference():
    raw = case(requests=[{"id": "b", "prompt_tokens": 9,
                          "output_tokens": 10}])
    raw["runtime"]["enable_chunked_prefill"] = False
    parsed, memory = _memory_with_free_budget(raw, 1024)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
    assert scheduler.retain_shared_state("shared", [128.0], version=7)
    state = scheduler.states[0]
    before = (state.generated, state.state_version,
              dict(memory.shared_pages))
    assert scheduler.schedule(0) is None
    assert scheduler.last_reason == "capacity_or_budget_exhausted"
    assert state.reason == "insufficient_cache_or_token_budget"
    assert (state.generated, state.state_version) == before[:2]
    assert memory.shared_reserved[0][0] == 128.0
    assert (0, "shared") in memory.shared_pages
    assert memory.shared_pages[(0, "shared")]["owners"] == {"shared": 1}
    assert dict(memory.shared_pages) == before[2]
    feedback = scheduler.feedback(0)
    assert feedback["admission"]["capacity_or_budget_exhausted"] is True
    assert feedback["admission"]["waiting_for_release"] is False


def test_stale_batch_callback_cannot_publish_the_next_in_flight_batch():
    parsed = _draft_case(trace=(), output=3)
    scheduler = Scheduler(parsed, list(parsed.workload.requests), Memory(parsed))
    plan0 = scheduler.schedule(0)
    assert plan0 is not None
    step0 = SimpleNamespace(
        iteration=plan0.iteration,
        completion=SimpleNamespace(time=1.0),
        commits={"r": SimpleNamespace(time=1.0)},
    )
    assert scheduler.commit_batch(step0) is True

    plan1 = scheduler.schedule(1.0)
    assert plan1 is not None
    state = scheduler.states[0]
    current = (state.state_version, state.generated, state.pending_context,
               state.reserved_tokens)
    # A repeated callback from the already published i0 must not commit i1.
    assert scheduler.commit_batch(step0) is False
    assert (state.state_version, state.generated, state.pending_context,
            state.reserved_tokens) == current

    real = SimpleNamespace(
        iteration=plan1.iteration,
        completion=SimpleNamespace(time=2.0),
        commits={"r": SimpleNamespace(time=2.0)},
    )
    assert scheduler.commit_batch(real) is True
    published = (state.state_version, state.generated)
    assert published[0] == current[0] + 1
    assert scheduler.commit_batch(real) is False
    assert (state.state_version, state.generated) == published
