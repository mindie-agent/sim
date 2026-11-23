"""Serving core: scheduler state, cohort/parallel synthesis, SLA and DBO laws.

[Role]
  Establish the scheduled-batch state contract and the multi-parallelism,
  speculation, MoE, DBO/PD, SLA and exact-target invariants of the serving
  entry: versioned request state, reservation/commit visibility, independent
  physical owners, conserved work and exact analytical boundaries.

[Invariants]
  * A committed state version is visible only after its producing batch;
    pending context is never read as committed, and stale callbacks cannot
    publish a later in-flight batch.
  * A cloned replica carries its own executable/owner; a draft block prices the
    block it executes; a solver never selects a batch its capacity rejects.
  * DBO is refused explicitly with its declared unmodeled error until in-op DBO
    lands; every other graph conserves work, rows and link bytes exactly.

[Timeouts & Locks]
  Tier 0: bounded in-memory graphs and one production predict; no device I/O.

[Side-Effect Policy]
  Local cases and ledgers only; writes pytest temporary state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import pytest
from mindie_sim import compose_case, predict, predict_pd
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction import efficiency as efficiency_module
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    _lower_and_run_graph, analyze_network_efficiency, solve_sla_sizing,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.efficiency.unmodeled_gate import assert_report_supported
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.workload import RequestView
from _support.serving_cases import (
    TinyMemory, bind_pd_pool, by_prefix, by_recipe, capture_lowered_work, case,
    commit_step, deepseek_128, dispatch_tp_rows, held_decode_scheduler, linear_work,
    memory_with_free_budget, pd_case, pipeline_draft_case, pipeline_draft_transfers, pipeline_draft_workload,
    production_draft_case, replica_schedulers, routed_prefill_case, scheduler_for,
    run_scheduled_service, speculative_case, sp_mtp_case,
)
pytestmark = pytest.mark.tier0
def test_scheduler_commits_versioned_state_from_integer_acceptance():
    scheduler = scheduler_for(speculative_case())
    state = scheduler.states[0]

    plan = scheduler.schedule(0)
    assert plan is not None and scheduler.last_reason == "scheduled"
    prefill_view = plan.requests[0]
    assert (prefill_view.phase, prefill_view.q_len) == ("prefill", 4)
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
    scheduler = scheduler_for(speculative_case(trace=(), output=2))
    plan = scheduler.schedule(0)
    assert plan is not None
    active = plan.requests[0]
    padding = RequestView("serving1:padding", 4, 0, False, "decode", padding=True, draft_verify=False)
    assert scheduler.emissions(replace(plan, requests=(active, padding))) == {"r": 1}
    assert scheduler.emissions(replace(plan, requests=(padding,))) == {}
    before = [state.state_version for state in scheduler.states]
    scheduler.commit(1.0, {"r": 1.0})
    assert [state.state_version for state in scheduler.states] == [value + 1 for value in before]
    assert all(state.finish_ms is None for state in scheduler.states)
    with pytest.raises(Exception, match="scheduler_emission_request_not_registered"):
        scheduler.emissions(replace(plan, requests=( replace(active, request_id="ghost"),)))
def test_chunked_prefill_commits_only_at_the_sampling_boundary():
    raw = case(prompt=8, output=2)
    raw["runtime"].update(enable_chunked_prefill=True, max_num_batched_tokens=4)
    scheduler = scheduler_for(parse_case(raw))
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
    parsed = parse_case(case(prompt=3, output=3))
    memory = TinyMemory(capacity=6, rejected=(), reject_once=(5,))
    scheduler = scheduler_for(parsed, memory)
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
    parsed = speculative_case(trace=(), output=2)
    memory = TinyMemory(capacity=2, rejected=())
    scheduler = scheduler_for(parsed, memory)

    assert scheduler.schedule(0) is None
    assert scheduler.last_reason == "capacity_or_budget_exhausted"
    feedback = scheduler.feedback(0)
    assert feedback["admission"]["capacity_or_budget_exhausted"] is True
    assert feedback["admission"]["infeasible_requests"] == ["r"]
    assert feedback["requests"][0]["reason"] == "insufficient_cache_or_token_budget"
    assert feedback["capacity"]["kv_capacity_bytes_by_stage"] == [2.0]
    assert feedback["capacity"]["headroom_bytes_by_stage"] == [2.0]
def test_non_speculative_decode_cannot_commit_two_ungenerated_tokens():
    scheduler = scheduler_for(speculative_case(trace=(), output=2))
    scheduler.schedule(0)
    scheduler.commit(1.0, {"r": 1.0})
    plan = scheduler.schedule(1.0)
    assert plan is not None and plan.requests[0].q_len == 1
    bad = replace(plan, requests=(replace(plan.requests[0], q_len=2),))
    with pytest.raises(Exception, match="non_speculative_decode_forward_must_commit_one_token"):
        scheduler.emissions(bad)
    with pytest.raises(Exception, match="non_speculative_decode_forward_must_commit_one_token"):
        scheduler.by_id["r"].commit(bad.requests[0], 2.0)

def test_transient_copy_grant_waits_for_actual_release_then_admits():
    parsed, memory = memory_with_free_budget(
        case(prompt=1, output=1), 12288)
    # The copy grant occupies all free headroom until its real release at t=1.
    scheduler = scheduler_for(parsed, memory)
    assert scheduler.grant_state([("copy", 12288.0)]) is True
    assert scheduler.schedule(0) is None
    assert scheduler.last_reason == "waiting_for_release"
    assert scheduler.states[0].reason is None
    assert scheduler.states[0].generated == 0
    assert scheduler.states[0].state_version == 0
    assert memory.granted[0][0] == 12288.0
    assert scheduler.release_state_grants(("copy",)) == pytest.approx(12288.0)
    scheduler.now = 1.0
    plan = scheduler.schedule(1.0)
    assert plan is not None and scheduler.emissions(plan) == {"r": 1}
    assert scheduler.last_reason == "scheduled"
    assert scheduler.states[0].generated == 0
    scheduler.commit(1.0, {"r": 1.0})
    assert scheduler.states[0].generated == 1
def test_preemption_keeps_an_unfinished_reader_and_blocks_growth_until_release():
    scheduler, memory = held_decode_scheduler(prompt=1, max_batch_tokens=1, weights_only=True)
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
def test_twelve_kib_grants_are_atomic_and_the_fourth_waits():
    parsed, memory = memory_with_free_budget(case(output=1), 12288)
    scheduler = scheduler_for(parsed, memory)
    assert scheduler.grant_state([("a", 4096.0), ("b", 4096.0),
                                  ("c", 4096.0)]) is True
    assert scheduler.feedback(0)["capacity"]["granted_bytes_by_stage"] == [12288.0]
    assert scheduler.grant_state([("d", 4096.0)]) is False
    assert scheduler.feedback(0)["capacity"]["granted_bytes_by_stage"] == [12288.0]
    assert scheduler.release_state_grants(("a",)) == pytest.approx(4096.0)
    assert scheduler.grant_state([("d", 4096.0)]) is True
    assert scheduler.feedback(0)["capacity"]["granted_bytes_by_stage"] == [12288.0]
def test_scheduler_consumes_union_grants_and_async_holds():
    scheduler = scheduler_for(speculative_case(trace=(), output=2))
    assert scheduler.schedule(0) is not None
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

def test_commit_batch_is_idempotent_and_uses_actual_commit_events():
    scheduler = scheduler_for(speculative_case(trace=(), output=2))
    plan = scheduler.schedule(0)
    assert plan is not None and scheduler.emissions(plan) == {"r": 1}
    step = commit_step(4.0, completion_ms=4.5)
    assert scheduler.commit_batch(step) is True
    assert scheduler.states[0].first_token_ms == 4.0
    assert scheduler.states[0].state_version == 1
    version, generated = scheduler.states[0].state_version, scheduler.states[0].generated
    scheduler.commit_batch(step)
    assert (scheduler.states[0].state_version, scheduler.states[0].generated) == (version, generated)
def test_scheduler_drives_real_execution_without_waiting_for_every_delivery():
    batches, decisions, scheduler, execution = run_scheduled_service()
    assert batches == 3
    assert [state.state_version for state in scheduler.states] == [3, 3]
    assert all(not decision["in_flight_requests"] for decision in decisions)
    assert all(decision["requests"] for decision in decisions[:-1])
    assert decisions[-1]["requests"] == []
    assert scheduler.done
    assert execution.summary()["live_tasks"] == 0
    assert execution.summary()["live_programs"] == 0
def test_dp_schedulers_share_one_memory_ledger():
    parsed = parse_case(case(dp=2, requests=[ {"id": "r0", "prompt_tokens": 4, "output_tokens": 3}, {"id": "r1", "prompt_tokens": 4, "output_tokens": 3}]))
    memory = Memory(parsed)
    schedulers = replica_schedulers(parsed, memory)
    assert schedulers[0].memory is memory is schedulers[1].memory
    plans = [scheduler.schedule(0) for scheduler in schedulers]
    assert all(plan is not None for plan in plans)
    assert memory.live[0][0] > 0 and memory.live[1][0] > 0
    for replica, scheduler in enumerate(schedulers):
        feedback = scheduler.feedback(0)
        assert feedback["capacity"]["cache_bytes_by_stage"][0] > 0
        assert set(feedback["capacity"]["owner_bytes_by_request"]) == {f"r{replica}"}
    schedulers[0].commit(1.0, {"r0": 1.0})
    assert schedulers[1].in_flight and schedulers[1].states[0].pending_context is not None
    assert schedulers[1].feedback(1.0)["requests"][0]["status"] == "in_flight"
def test_scheduler_rejects_a_second_inflight_batch_and_feedback_omits_done():
    scheduler = scheduler_for(speculative_case(trace=(), output=2))
    assert scheduler.schedule(0) is not None
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
def test_production_entry_commits_the_scheduled_state():
    result = predict(production_draft_case(), trace_limit=1, event_limit=120)
    assert result["status"] == "modeled" and result["execution_status"] == "completed"
    assert result["execution"]["live_tasks"] == 0
    assert [row["request_id"] for row in result["requests"]] == ["r"]
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
def test_permanent_minimum_span_reports_failure_and_keeps_shared_reference():
    raw = case(requests=[{"id": "b", "prompt_tokens": 9, "output_tokens": 10}])
    raw["runtime"]["enable_chunked_prefill"] = False
    parsed, memory = memory_with_free_budget(raw, 1024)
    scheduler = scheduler_for(parsed, memory)
    assert scheduler.retain_shared_state("shared", [128.0], version=7)
    state = scheduler.states[0]
    before = (state.generated, state.state_version, deepcopy(memory.shared_pages))
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
    scheduler = scheduler_for(speculative_case(trace=(), output=3))
    plan0 = scheduler.schedule(0)
    assert plan0 is not None
    step0 = commit_step(1.0, iteration=plan0.iteration)
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

    real = commit_step(2.0, iteration=plan1.iteration)
    assert scheduler.commit_batch(real) is True
    published = (state.state_version, state.generated)
    assert published[0] == current[0] + 1
    assert scheduler.commit_batch(real) is False
    assert (state.state_version, state.generated) == published

def test_run_graph_false_lowers_many_ranks_without_executing_the_cohort():
    parsed = parse_case(deepseek_128())
    graphs, _effective, report = _lower_and_run_graph( parsed, phase="decode", tokens=6, context=4096, sequences=1, scope="model", run_graph=False)
    assert report is None and len(graphs) == 64
    assert len(graphs.program(0).work.ops) > 0
    raw = deepseek_128()
    raw["runtime"]["metadata"]["moe_comm_mode"] = "dynamic"
    graphs, _, report = _lower_and_run_graph(
        parse_case(raw), phase="decode", tokens=6, context=4096, sequences=1,
        scope="body", run_graph=False)
    assert report is None and len(graphs) == 64
    assert any(op.family == "mc2_protocol" and (dict(op.metadata).get("device_kernel_group") or {}).get("kernel") for op in graphs[0])
def test_attached_program_uses_the_last_local_tp_replica_work():
    raw = compose_case("glm-5.3-flash", hardware="a5", tp=1, dp=32, ep=32, num_devices=32, layers=4, o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, mtp_proposal_tokens=5, prompt_tokens=4096, output_tokens=8)
    graphs = target_graphs(parse_case(raw), phase="decode", tokens=6, context=4096, sequences=1, scope="model")
    graph_ids = {op.op_id for op in graphs[0]}
    program_ids = {op.op_id for op in graphs.program(0).work.ops}
    assert "logits:local_tp:gather" in graph_ids
    assert {"logits:local_tp:gather", "logits:local_tp:gemm"} <= program_ids

def test_cohort_reuse_rebinds_nested_commit_request_identities():
    raw = compose_case("glm-5.3-flash", hardware="a5", tp=1, dp=2, ep=2, num_devices=2, layers=4, recover_ssm=True, mtp_proposal_tokens=3, mtp_acceptance_length=3.5, homogeneous_graph_reuse="require", prompt_tokens=8, output_tokens=4)
    graphs = target_graphs(parse_case(raw), phase="decode", tokens=4, context=0, sequences=2, scope="model")
    commits = [op for op in graphs[1] if op.recipe == "recover_ssm_commit"]
    assert commits
    for op in commits:
        assert op.requests == ("analysis1:0", "analysis1:1")
        committed = dict(op.metadata)["committed_tokens"]
        assert tuple(request for request, _ in committed) == op.requests
def test_physical_key_groups_identical_fused_layer_occurrences():
    raw = compose_case("deepseek-v4-pro", hardware="a5", tp=8, ep=8, num_devices=8, layers=61, prompt_tokens=8, output_tokens=2)
    graphs = target_graphs(parse_case(raw), phase="decode", tokens=1, context=4096, sequences=1, scope="model")
    assert len(by_recipe(graphs[0], "indexer_quant_cache")) == 30
    rows = [row for row in operator_costs(parse_case(raw), graphs)["operators"] if row["operator"].get("family") == "indexer_quant_cache"]
    assert len(rows) == 1 and rows[0]["occurrences"] == 30
def test_independent_bf16_draft_strips_target_packed_prolog():
    quant = quant_preset("w8a8").to_dict() | {"kv_dtype": "int8"}
    parsed = parse_case(compose_case("kimi-k3.1", hardware="a5", tp=4, ep=4, num_devices=4, layers=4, quant=quant, mtp_proposal_tokens=2, prompt_tokens=64, output_tokens=8))
    decode = target_graphs(parsed, phase="decode", tokens=3, context=0, sequences=1, scope="model")
    prefill = target_graphs(parsed, phase="prefill", tokens=8, context=0, sequences=1, scope="model", draft_prepare=True)
    assert any(op.phase == "draft" for op in decode[0])
    assert any(op.op_id.startswith("draft_prefill.") for op in prefill[0])
def test_cyclic_nextn_executes_one_declared_block_per_proposal():
    graphs = target_graphs(parse_case(compose_case( "step-5-preview", hardware="a5", tp=4, ep=4, num_devices=4, layers=4, mtp_proposal_tokens=3, prompt_tokens=64, output_tokens=8)), phase="decode", tokens=4, context=0, sequences=1, scope="model")
    drafts = [op for op in graphs[0] if op.op_id.startswith("draft") and op.family == "attention"]
    assert {op.op_id.split(".", 1)[0] for op in drafts} == {"draft0", "draft1", "draft2"}
    assert len(drafts) == 3

def test_pp_proposal_transfer_sums_local_proposals_from_last_active_step():
    raw = pipeline_draft_case()
    parsed = parse_case(raw)
    transfers = pipeline_draft_transfers(parsed)
    assert len(transfers) == 1
    transfer = transfers[0]
    assert (transfer.family, transfer.recipe) == ("pp_p2p", "pp_p2p")
    assert transfer.link_bytes == 6 * 8
    assert transfer.local_shape == (6,)
    assert transfer.stage == 1
    graphs = target_graphs(parsed, phase="decode", tokens=6, context=4, sequences=2, scope="model")
    assert graph_costs(parsed, graphs)["duration_ms"] > 0.0
    raw["workload"] = pipeline_draft_workload()
    serving = predict(raw, max_model_steps=2)
    assert serving["execution"]["iterations"] == 2
def test_pcp_draft_prefill_partitions_query_rows_and_keeps_global_gather():
    graphs = target_graphs(parse_case(compose_case( "glm-5.3-flash", hardware="a5", tp=2, dp=1, ep=2, pcp=2, num_devices=4, layers=4, mtp_proposal_tokens=5, prompt_tokens=16, output_tokens=2)), phase="prefill", tokens=16, context=0, sequences=1, scope="model", draft_prepare=True)
    draft = by_prefix(graphs[0], "draft_prefill.")
    qkv = by_recipe(draft, "qkv_a", layer=0)
    gather = by_recipe(draft, "pcp_kv_gather", layer=0)
    assert qkv and qkv[0].local_shape[0] == 8
    assert gather and gather[0].link_bytes == 4160
def test_dspark_scatter_charges_the_owning_position_block():
    graphs = target_graphs(parse_case(compose_case( "kimi-k3.1", hardware="a5", tp=8, dp=1, ep=8, dcp=8, num_devices=8, layers=4, mtp_proposal_tokens=5, prompt_tokens=8, output_tokens=2)), phase="decode", tokens=6, context=4, sequences=1, scope="model")
    scatters = by_recipe(graphs[0], "draft_context_kv_scatter")
    assert scatters
    for op in scatters:
        metadata = dict(op.metadata)
        assert (metadata["dcp_degree"], metadata["dcp_row_shard"]) == (8, 6)
def test_dynamic_moe_group_selects_one_protocol_and_prices_the_cohort():
    raw = compose_case("glm-5.3-flash", hardware="a5", tp=1, dp=32, ep=32, num_devices=32, layers=4, o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, moe_comm_mode="dynamic", max_num_seqs=2, prompt_tokens=8, output_tokens=4)
    raw["runtime"]["graph_capture_keys"] = [12]
    parsed, sequences = parse_case(raw), [1] * 31 + [2]
    graphs = target_graphs(parsed, phase="prefill", tokens=7, context=0, sequences=sequences, scope="body")
    assert {any("device_kernel_group" in dict(op.metadata) for op in graphs[replica]) for replica in graphs} == {False}
    assert all(by_recipe(graphs[replica], "expert_dispatch") for replica in graphs)
def test_sp_mtp_preserves_captured_proposal_rows_in_serving_and_analysis(monkeypatch):
    parsed = parse_case(sp_mtp_case(dp=16))
    graphs = target_graphs(parsed, phase="decode", tokens=6, context=4, sequences=1, scope="model")
    tp_rows, total = dispatch_tp_rows(graphs[0])
    assert tp_rows == (2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert total == pytest.approx(1536.0)

    raw = sp_mtp_case(dp=2)
    raw["workload"] = {"requests": [
        {"id": f"r{i}", "prompt_tokens": 4, "output_tokens": 8, "arrival_ms": 0.0}
        for i in range(2)
    ]}
    captured = capture_lowered_work(raw, monkeypatch)
    decode_ops = [work.ops for work in captured if any(
        op.op_id == "s0.l1.dispatch" and op.phase == "decode" for op in work.ops)]
    assert decode_ops
    served_rows, served_total = dispatch_tp_rows(decode_ops[0])
    assert served_rows == (2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert served_total == pytest.approx(2 * 6 * 16.0)
def test_alltoallv_receive_rows_follow_destination_assignments():
    raw = routed_prefill_case()
    graphs = target_graphs(parse_case(raw), phase="prefill", tokens=6, context=0, sequences=[1, 2], scope="body")
    dispatch = by_recipe(graphs[0], "expert_dispatch")
    experts = by_recipe(graphs[0], "grouped_gate_up")
    assert dispatch and dispatch[0].local_shape[0] == 12
    assert dict(dispatch[0].metadata)["payload_rows"] == 36
    assert experts and experts[0].local_shape[0] == 36
    assert dict(by_recipe(graphs[1], "expert_dispatch")[0].metadata)["payload_rows"] == 0
def test_redundant_expert_slot_binding_uses_the_replica_owner():
    graphs = target_graphs(parse_case(compose_case( "glm-5.3-flash", hardware="a5", tp=1, dp=32, ep=32, num_devices=32, layers=4, o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, redundant_experts_per_rank=1, prompt_tokens=8, output_tokens=2)), phase="decode", tokens=1, context=0, sequences=1, scope="body")
    def gate(replica):
        return next(op for op in graphs[replica] if op.recipe == "grouped_gate_up" and op.layer == 3)
    assert {replica: dict(gate(replica).metadata)["ep_rank"] for replica in (0, 2)} == {0: 0, 2: 2}
    assert (dict(gate(0).metadata)["routed_weight_read_bytes"] != dict(gate(2).metadata)["routed_weight_read_bytes"])
def test_local_tp_quantizer_and_gemm_derive_the_same_scale_geometry():
    graphs = target_graphs(parse_case(compose_case( "glm-5.3-flash", hardware="a5", tp=1, dp=8, ep=8, num_devices=8, layers=4, o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, quant="w8a8_mxfp8", prompt_tokens=8, output_tokens=2)), phase="decode", tokens=6, context=0, sequences=1, scope="body")
    quantizers = {op.op_id: dict(op.metadata).get("activation_scale_shape") for op in graphs[0] if op.layer == 3 and op.family == "quantize"}
    gemms = {op.op_id: dict(op.metadata).get("activation_scale_shape") for op in graphs[0] if op.layer == 3 and op.op_id.endswith(":local_tp:gemm")}
    assert gemms
    for op_id, shape in gemms.items():
        assert quantizers[op_id.removesuffix(":local_tp:gemm") + ".quantize"] == shape
def test_dbo_join_and_microbatches_refuse_the_unmodeled_stage():
    for sequences in ([2, 3], [2, 1]):
        raw = compose_case("qwen3-30b-a3b", hardware="a5", tp=1, dp=2, ep=2, num_devices=2, layers=1, prompt_tokens=8, output_tokens=2, max_num_seqs=4, enable_dbo=True)
        with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
            target_graphs(parse_case(raw), phase="decode", tokens=1, context=32, sequences=sequences, scope="body")
    raw = compose_case("qwen3-30b-a3b", hardware="a5", tp=1, dp=2, ep=2, num_devices=2, layers=1, enable_dbo=False, exclude_optimizations=("cann_megamoe_fusion",), prompt_tokens=8, output_tokens=2, max_num_seqs=2)
    graphs = target_graphs(parse_case(raw), phase="decode", tokens=1, context=32, sequences=2, scope="body")
    assert len(tuple(op for op in graphs[0] if op.recipe in {"grouped_gate_up", "grouped_down"})) >= 2
    raw = compose_case("qwen3-32b", hardware="a5", tp=1, ep=1, num_devices=1, layers=1, enable_dbo=True, max_num_batched_tokens=128, max_num_seqs=2, prompt_tokens=64, output_tokens=2)
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        target_graphs(parse_case(raw), phase="decode", tokens=1, context=64, sequences=2, scope="body")
def test_dbo_qualifies_shared_expert_stream_per_partition():
    raw = compose_case("deepseek-v4-pro", hardware="a5", tp=1, dp=1, ep=1, num_devices=1, layers=1, enable_dbo=False, multistream_overlap_shared_expert=True, prompt_tokens=8, output_tokens=2, max_num_seqs=2)
    graphs = target_graphs(parse_case(raw), phase="decode", tokens=1, context=32, sequences=2, scope="body")
    streams = {op.stream for op in graphs[0] if "shared" in op.stream}
    assert "shared_stream" in streams
    assert not any(stream.startswith("dbo") for stream in streams)
    raw["runtime"].update(enable_dbo=True)
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        target_graphs(parse_case(raw), phase="decode", tokens=1, context=32, sequences=2, scope="body")
def test_pd_handoff_intersects_token_spans_before_charging_rows():
    raw = pd_case([{"id": "r", "prompt_tokens": 4, "output_tokens": 3}])
    raw["model"].update(layers=1, hidden_size=64, num_heads=4, num_kv_heads=1, head_dim=16, intermediate_size=64, vocab_size=128)
    bind_pd_pool(raw, "prefill", (0, 1), tp=2, cp=2)
    bind_pd_pool(raw, "decode", (2,), tp=1, cp=1)
    raw["connector"] = {"link": {"name": "handoff", "capacity_bytes_per_ms": 64, "latency_us": 0, "semantics": "per_pair"}, "chunk_bytes": 64}
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    transfers = result["execution"]["pd_handoffs"][0]["transfers"]
    assert {(row["source_rank"], tuple(row["source_token_span"])) for row in transfers} == {(0, (0, 2)), (1, (2, 4))}
    assert all(row["payload_bytes"] == pytest.approx(128.0) for row in transfers)
def test_sla_sizing_requires_current_and_lifecycle_capacity_windows():
    raw = case(prompt=4, output=2)
    raw["machine"]["device"]["memory_bytes"] = 208000
    raw["runtime"].update(max_num_seqs=8, async_scheduling=False, enable_chunked_prefill=True)
    result = solve_sla_sizing(raw, phase="decode", context=4, output_budget=2, sla_ms=1000, sensitivity="baseline")["scenarios"][0]
    assert (result["b_feasible"], result["b_sla"], result["b_vram_max"], result["b_vram_max_joint"]) == (7, 8, 7, 7)
    assert result["hbm_bound"] is True
def test_sla_sizing_rejects_unchunkable_prefill_before_pricing(monkeypatch):
    raw = case(prompt=4, output=2)
    raw["runtime"].update(max_num_batched_tokens=16, enable_chunked_prefill=False)
    def fail_pricing(*_args, **_kwargs):
        raise AssertionError("unchunkable geometry was priced")
    monkeypatch.setattr(efficiency_module, "target_graphs", fail_pricing)
    result = solve_sla_sizing(raw, phase="prefill", tokens_per_seq=17, context=0, output_budget=2, sla_ms=1000, sensitivity="baseline")["scenarios"][0]
    assert (result["b_feasible"], result["b_sla"], result["domain_max_b"]) == (0, 0, 0)
    assert result["domain_limit_reason"] == ( "prompt_exceeds_batch_budget_without_chunking")
def test_exact_target_preserves_occurrence_dag_and_never_exceeds_one():
    raw = case(prompt=4, output=2)
    raw["model"]["layers"] = 1
    baseline = analyze_network_efficiency(raw, phase="decode", tokens=1, context=32, sequences=1, scope="model")
    assert len(baseline["critical_path_nodes"]) == 15
    assert baseline["critical_path_roofline_ms"] == pytest.approx(0.033888)
def test_unmodeled_gate_resolves_layer_override_and_flat_timeline():
    raw = case(prompt=4, output=2)
    raw["model"]["layer_overrides"] = {"0": {"attention": "unmodeled"}}
    with pytest.raises(UnmodeledCapabilityError):
        predict(raw, lower=linear_work)
    with pytest.raises(UnmodeledCapabilityError):
        assert_report_supported( {"timeline": [{"op_id": "a", "family": "attention", "recipe": "unmodeled_attention_core", "layer": 0}]}, entry="analyze_network_efficiency")
def test_dspark_body_does_not_inherit_target_cache_writer():
    quant = quant_preset("w8a8_mxfp8").to_dict() | {"kv_dtype": "int8"}
    raw = compose_case("kimi-k3.1", hardware="a5", tp=8, dp=1, ep=8, dcp=8, num_devices=8, layers=4, quant=quant, mtp_proposal_tokens=5, prompt_tokens=8, output_tokens=2)
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[ "native_fusion"] = {"implementation": "declared_native_fusion", "options": {"roles": ["mla_prolog"]}}
    parsed = parse_case(raw)
    assert parsed.runtime.kernels["lowerings"]["mla_prolog"]["options"]["cache_layout"]
    graph = target_graphs(parsed, phase="decode", tokens=6, context=4, sequences=1, scope="model")[0]
    body = [op for op in graph if op.phase == "draft" and dict(op.metadata).get("fusion_role") == "mla_prolog"]
    context = by_recipe(graph, "draft_context_mla_prolog_v3")
    assert not body and len(context) == 5
def test_native_moe_inverse_indices_are_charged_once_on_pack_and_unpack():
    graph = target_graphs(parse_case(compose_case( "glm-5.3-flash", hardware="a5", tp=8, dp=1, ep=8, num_devices=8, layers=4, prompt_tokens=64, output_tokens=2, max_num_batched_tokens=256, exclude_optimizations=("cann_megamoe_fusion",))), phase="decode", tokens=64, context=0, sequences=1, scope="body")[0]
    pack = next(op for op in graph if op.recipe == "moe_init_routing_v3")
    unpack = next(op for op in graph if op.recipe == "route_unpack")
    assignments = int(pack.local_shape[0]) * int(pack.local_shape[1])
    index_bytes = assignments * 4
    assert assignments == 64 and index_bytes == 256
    assert dict(pack.metadata)["inverse_index_bytes"] == index_bytes
    assert dict(pack.metadata)["moe_init_routing"]["write_bytes"] == pytest.approx(pack.mte3_bytes)
    assert pack.mte3_bytes == pytest.approx(270336 + index_bytes)
    assert pack.hbm_bytes == pytest.approx(pack.mte2_bytes + pack.mte3_bytes)
    assert dict(unpack.metadata)["inverse_index_bytes"] == index_bytes
    assert unpack.mte2_bytes == pytest.approx(524544 + index_bytes)
def test_alltoallv_receive_buffers_follow_incoming_uniform_assignments():
    graphs = target_graphs(parse_case(compose_case( "qwen3-30b-a3b", hardware="a5", tp=1, dp=2, ep=2, num_devices=2, layers=1, exclude_optimizations=("cann_megamoe_fusion",), prompt_tokens=8, output_tokens=2, max_num_seqs=4)), phase="decode", tokens=1, context=32, sequences=[2, 3], scope="body")
    for replica, outgoing in ((0, 16), (1, 24)):
        dispatch = next(op for op in graphs[replica] if op.recipe == "expert_dispatch")
        metadata = dict(dispatch.metadata)
        assert dispatch.local_shape[0] == outgoing
        assert (metadata["output_shape"], metadata["capacity_rows"], metadata["payload_rows"]) == ((20, 2048), 20, 20)
        assert metadata["capacity_rows"] * metadata["payload_row_bytes"] == pytest.approx(40960.0)
        experts = [op for op in graphs[replica] if op.recipe in {"grouped_gate_up", "grouped_down"}]
        assert [op.local_shape[0] for op in experts] == [20, 20]
def test_local_tp_packed_views_use_the_parent_row_stride():
    graph = target_graphs(parse_case(compose_case( "glm-5.3-flash", hardware="a5", tp=1, dp=8, ep=8, num_devices=8, layers=4, o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, quant="w8a8_mxfp8", prompt_tokens=8, output_tokens=2)), phase="decode", tokens=6, context=0, sequences=1, scope="body")[0]
    packed = next(op for op in graph if op.op_id.endswith(".linear_qkvbfg_a:local_tp:gemm"))
    slices = {item["role"]: item for item in dict(packed.metadata)["packed_projection_slices"]}
    assert packed.local_shape[1] == 3336
    q, k = slices["q"], slices["k"]
    assert (q["shape"][1], q["strides"], k["offset_elements"]) == (1024, (3336, 1), 1024)
