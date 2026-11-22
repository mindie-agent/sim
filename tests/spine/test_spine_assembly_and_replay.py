"""Model and serving assembly, and causal catalog execution.

[Role] Bind model, framework and replay steps to real events on one runtime.
[Invariants] Conserve service, state, allocations and commits across occurrences.
[Timeouts & Locks] Virtual time, bounded cursor windows and no external locks.
[Side-Effect Policy] Read a packaged example and construct local runtimes.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from _support.serving_cases import case, linear_work
from mindie_sim import predict
from mindie_sim.prediction import runtime, scheduling
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import assemble, compose_case
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Engine, Event, ExecutionError, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend, SERVICE_FAMILIES, _ports_for_work
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.program import Stage
from mindie_sim.prediction.operators.recurrent.recover_ssm_commit import bind_recover_ssm_commits, is_recover_ssm_intent
from mindie_sim.prediction.operators.recurrent.recover_ssm_work import emit_commit_op
from mindie_sim.prediction.runtime.arbiter import fair_rates
from mindie_sim.prediction.runtime.contracts import (
    AllocationSpec, BindingFrame, EdgeSpec, FiniteProgramSpec, HostProgramSpec, ProgramCatalog, ReleaseSpec, StageClass, StageSpec, WorkTerm,
)
from mindie_sim.prediction.runtime.host import HostCursor
from mindie_sim.prediction.runtime.periodic import IntervalSummary
from mindie_sim.prediction.runtime.state import RuntimeState
from mindie_sim.prediction.efficiency.step_metrics import StepSummary
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork, OpSpec, StateEffect
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

EXAMPLE_CASE = Path(__file__).resolve().parents[2] / "examples/cases/qwen35-a5.json"

def _fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected

def _case_with_io(*requests):
    raw = case(requests=list(requests) or None)
    raw["runtime"]["serving_io"] = {
        "host_prepare_ms": 0.5, "host_result_ms": 0.5, "client_delivery_ms": 2,
        "h2d": {"bytes_per_s": 8000, "bytes_per_token": 8, "latency_us": 250},
        "d2h": {"bytes_per_s": 4000, "bytes_per_token": 4, "latency_us": 250},
    }
    return raw

def _request(identifier, prompt, output):
    return {"id": identifier, "prompt_tokens": prompt, "output_tokens": output}

def _programs(raw, view):
    parsed, frontend, plan = parse_case(raw), Frontend(), IterationPlan((view,))
    return tuple(frontend.program(plan, parsed.model, parsed.runtime, parsed.machine, assembly=scope)
                 for scope in ("model", "service"))

def _op(name, family, recipe, requests=("r",), phase="decode", **kwargs):
    return OpSpec(name, family, phase, 0, -1, requests, (1,), (1,), (1,), (1,), recipe=recipe, **kwargs)

def test_model_scope_preserves_service_forward_work():
    model, service = _programs(json.loads(EXAMPLE_CASE.read_text()), RequestView("r0", 1, 8, True, "decode"))
    model_ops, service_ops = ({op.op_id: op for op in item.work.ops} for item in (model, service))
    assert model_ops.keys() < service_ops.keys()
    fields = ("flops", "hbm_bytes", "vector_ops", "link_bytes")
    for identifier, op in model_ops.items():
        other = service_ops[identifier]
        assert tuple(getattr(op, field) for field in fields) == tuple(getattr(other, field) for field in fields)
    assert not ({op.family for op in model_ops.values()} & SERVICE_FAMILIES)
    assert {"sampling", "token_commit"} <= {op.family for op in service_ops.values()}
    assert model.layer_scope == "per_model_layer" and not model.ports.state_out
    assert any(port.name == "state.token_committed" for port in service.ports.state_out)
    assert model.ports.tensor_in == () and not model.triggers
    assert {trigger[0] for trigger in service.triggers} == {"sampling", "token_commit"}

def test_logits_port_uses_declared_output_precision():
    parsed = parse_case(case())
    quant = QuantSpec("int8-lm", weight_format="int8", activation_format="int8", expert_weight_format="int8",
                      scale_format="fp32", native_format="int8", output_format="bf16", exclusions=())
    plan = IterationPlan((RequestView("r", 1, 4, True, "decode"),))
    program = Frontend().program(plan, replace(parsed.model, quant=quant), parsed.runtime,
                                 parsed.machine, assembly="model")
    logits = next(op for op in program.work.ops if op.family == "logits")
    assert logits.dtype == "int8"
    _fields(dict(logits.metadata), compute_format="int8", output_format="bf16")
    logits_port = next(port for port in program.ports.tensor_out if port.name == "output.logits")
    assert logits_port.dtype == "bf16" and program.ports.tensor_in == ()
    assert program.to_dict()["unbound_tensor_ports"] == []

def test_unknown_tensor_boundary_stays_explicitly_unbound():
    root = _op("custom_root", "vector", "custom", dtype="")
    plan = IterationPlan((RequestView("r", 1, 0, True, "decode"),))
    ports = _ports_for_work(IterationWork((root,), plan, 1, 1, 1, None, "decode", "test"))
    assert [port.dtype for port in ports.tensor_in] == [""]

def test_generic_graph_executes_service_nodes_in_dependency_order():
    raw = case()
    model, service = _programs(raw, RequestView("r", 1, 4, True, "decode", draft_prepare=False))
    report = graph_costs(raw, service.work.ops, isolated=True)
    assert {"sampling", "token_commit"} <= {row["family"] for row in report["timeline"]}
    assert report["duration_ms"] > 0
    pure = graph_costs(raw, model.work.ops)
    assert not ({row["family"] for row in pure["timeline"]} & SERVICE_FAMILIES)
    ops = (_op("sample", "sampling", "sample_logits", vector_ops=8, dtype="fp32"),
           _op("commit", "token_commit", "token_commit", deps=("sample",)),
           _op("accept", "acceptance", "speculative_acceptance", vector_ops=4, dtype="fp32", deps=("commit",)))
    assert [row["family"] for row in graph_costs(raw, ops)["timeline"]] == ["sampling", "token_commit", "acceptance"]

def _request_events(result):
    return result["execution"]["step_summary"]["total"]["request_events"]

def test_service_assembly_feedback_and_host_io_conserve_work():
    raw = _case_with_io(_request("r0", 4, 4), _request("r1", 3, 4))
    result = assemble(raw, scope="service", iterations=3, tokens=1, diagnostic_window=True)
    assert result["same_context"] is True
    _fields(result, independent_model_invocations=0, iterations_executed=3)
    _fields(result["triggers"], model_invocation=3, batch_prepare=3, request_commit_callbacks=6,
            token_commit_nodes=3, sampling=3, client_delivery_batches=3, client_order_events=6)
    first, *following_steps = rows = result["steps"]
    assert first["submitted_basis"] == "actual_step_host_submissions"
    first_host, last_host = first["host_submit_first"], first["host_submit_last"]
    assert first_host is not None and first["host_submit_event_count"] > 0
    assert first["submitted_ms"] == first_host["submit_ms"] >= first["host_prepare_done_ms"]
    assert first["submitted_last_ms"] == last_host["submit_ms"] >= first["submitted_first_ms"]
    assert first["submitted_ms"] > first["release_ms"]
    assert all("host_submit" in record["event"] for record in (first_host, last_host))
    assert max(first["delivery_ms"].values()) > first["step_ready_ms"] > first["device_done_ms"]
    for current, following in zip(rows, following_steps):
        assert following["release_ms"] == pytest.approx(max(current["delivery_ms"].values()))
    for request in result["requests"]:
        commits = [row["commit_ms"][request["request_id"]] for row in rows]
        _fields(request, generated_tokens=3, first_commit_ms=pytest.approx(commits[0]),
                last_commit_ms=pytest.approx(commits[-1]), ttft_ms=pytest.approx(commits[0] - request["arrival_ms"]),
                tpot_ms=pytest.approx((commits[-1] - commits[0]) / 2))
        assert request["output_tps"] > 0
    _fields(result["metrics"], output_tokens=6, commit_events=6, delivery_events=6,
            client_propagation_events=3, fallback_completion_events=0)
    _fields(result["framework_occurrences"], client_delivery=3, client_order=6)
    _fields(result["execution"], live_tasks=0, live_programs=0)
    serviced = result["execution"]["serviced"]
    assert serviced["scheduler@r0"] > 0
    assert (serviced["h2d@r0"], serviced["d2h@r0"]) == (3 * 2 * 8, 3 * (1 + 1) * 4)
    _fields(_request_events(result), commits=6, fallback_completions=0)
    assert first["declared_commit_ms"] and first["fallback_commit_ms"] == {}

def test_decode_token_budget_expands_into_single_token_steps():
    raw = _case_with_io(_request("r0", 4, 6), _request("r1", 3, 6))
    result = assemble(raw, scope="service", iterations=2, tokens=2, diagnostic_window=True)
    _fields(result, iterations_requested=2, decode_tokens_per_iteration=2,
            steps_requested=4, iterations_executed=4, steps_executed=4)
    _fields(result["triggers"], model_invocation=4, request_commit_callbacks=8)
    assert result["state_contract"]["decode_step"] == "one_token_per_active_request"
    for offset, row in enumerate(result["steps"]):
        _fields(row, context_tokens={"r0": 4 + offset, "r1": 3 + offset}, valid_tokens=2,
                emitted_tokens={"r0": 1, "r1": 1}, submitted_basis="actual_step_host_submissions")
        assert len(row["commit_ms"]) == 2
    for request in result["requests"]:
        assert request["generated_tokens"] == 4 and request["tpot_ms"] > 0
        assert request["last_commit_ms"] > request["first_commit_ms"]
    assert tuple(result["execution"]["serviced"][key] for key in ("h2d@r0", "d2h@r0")) == (4 * 2 * 8, 4 * 2 * 4)

def test_replay_enqueue_and_batch_propagation_bind_real_events():
    raw = _case_with_io(_request("r0", 2, 2), _request("r1", 2, 2))
    config = raw["runtime"]
    config["serving_io"]["client_delivery_ms"] = 1
    for transfer in ("h2d", "d2h"):
        del config["serving_io"][transfer]["latency_us"]
    config.update(graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full",
                  submission={"graph_replay_ms": 0.25})
    result = assemble(raw, scope="service", iterations=1, diagnostic_window=True)
    step = result["steps"][0]
    _fields(result["triggers"], graph_replay_submissions=1, client_delivery_batches=1, client_order_events=2)
    _fields(step, graph_replay_submissions=1, eager_submissions=0, submitted_basis="actual_step_host_submissions")
    assert step["release_ms"] < step["host_prepare_done_ms"] <= step["submitted_ms"] < min(step["commit_ms"].values())
    assert step["client_propagation_ms"] is not None
    assert step["client_propagation_ms"] <= min(step["delivery_ms"].values()) and len(step["delivery_ms"]) == 2
    assert result["execution"]["submission"]["graph_replays"] == 1

def test_bulk_token_budget_matches_individual_service_steps():
    raw = _case_with_io(_request("r0", 4, 2))
    bulk, stepped = (assemble(raw, scope="service", iterations=iterations, tokens=tokens, diagnostic_window=True)
                     for iterations, tokens in ((1, 2), (2, 1)))
    for result in (bulk, stepped):
        assert result["steps_executed"] == 2
        _fields(result["triggers"], model_invocation=2, request_commit_callbacks=2)
        assert result["execution"]["serviced"]["d2h@r0"] == 2 * 4
        for step in result["steps"]:
            _fields(step, valid_tokens=1, emitted_tokens={"r0": 1})
    assert [step["context_tokens"] for step in bulk["steps"]] == [step["context_tokens"] for step in stepped["steps"]]
    assert bulk["requests"][0]["generated_tokens"] == 2 and bulk["requests"][0]["tpot_ms"] > 0
    assert bulk["requests"][0]["tpot_ms"] == pytest.approx(stepped["requests"][0]["tpot_ms"])

@pytest.mark.parametrize("phase,options,modes,contexts,computed", [
    ("prefill", {"iterations": 2}, ["prefill", "decode"], [0, 4], [4, 5]),
    ("decode", {"iterations": 1, "tokens": 2, "context": 4}, ["decode", "decode"], [4, 5], [5, 6]),
])
def test_service_context_advances_by_the_written_kv(phase, options, modes, contexts, computed):
    raw = case(requests=[_request("r", 4, 2)])
    result = assemble(raw, scope="service", phase=phase, diagnostic_window=True, **options)
    assert [row["mode"] for row in result["steps"]] == modes
    for field, expected in (("context_tokens", contexts), ("computed_kv_tokens_after", computed)):
        assert [row[field]["r"] for row in result["steps"]] == expected
    if phase == "prefill":
        assert result["requests"][0]["generated_tokens"] == 2
    else:
        assert result["requests"][0]["tpot_ms"] > 0

def test_framework_scope_accounts_for_host_work_and_no_model_launches():
    raw = _case_with_io(_request("r0", 1, 2), _request("r1", 1, 2))
    result = assemble(raw, scope="framework", iterations=2, emitted_tokens={"r0": 1, "r1": 1})
    assert result["model_fragment"] is None
    _fields(result["triggers"], model_invocation=0, batch_prepare=2, request_commit_callbacks=0)
    _fields(result["execution"]["work"], flops=0, vector_ops=0)
    assert all(result["execution"]["serviced"][key] > 0 for key in ("scheduler@r0", "h2d@r0", "d2h@r0"))
    for row in result["steps"]:
        assert row["submitted_ms"] is None and row["submitted_basis"] == "no_engine_launch"
        assert row["device_done_ms"] > 0
    fragments = {fragment["name"]: fragment for fragment in result["framework_fragments"]}
    for fragment in fragments.values():
        assert fragment["layer"] is None
        assert all(port["direction"] in {"in", "out"} for group in fragment["ports"].values() for port in group)
    for name, trigger, occurrences in (("host_prepare", "per_batch", 2), ("client_delivery", "per_batch_with_emissions", 2),
                                       ("client_order", "per_request_per_batch", 4)):
        _fields(fragments[name], trigger=trigger, occurrences=occurrences)
    _fields(result["metrics"], client_propagation_events=2, delivery_events=4, fallback_completion_events=4)
    _fields(_request_events(result), commits=0, fallback_completions=4)
    second = result["steps"][1]
    assert second["declared_commit_ms"] == {}
    assert set(second["fallback_commit_ms"]) == set(second["active_requests"])

def test_framework_delivers_only_to_active_requests_after_short_request_finishes():
    raw = _case_with_io(_request("short", 1, 1), _request("long", 2, 3))
    result = assemble(raw, scope="framework", iterations=2, emitted_tokens={"short": 1, "long": 1})
    first, second = result["steps"]
    assert first["active_requests"] == ["long", "short"]
    _fields(second, active_requests=["long"], emitted_tokens={"long": 1})
    assert second["delivery_ms"].keys() == {"long"}
    _fields(result["triggers"], client_delivery_batches=2, client_order_events=3)
    assert result["requests"][0]["generated_tokens"] == 1

@pytest.mark.parametrize("declared", [True, False])
def test_kernel_group_commit_classification_uses_declared_event_maps(declared):
    commit = Event(1, "r0:i0:token_commit:kernel:7:complete", time=4.0)
    done = Event(2, "r0:i0:step_ready", time=5.0)
    step = SimpleNamespace(
        replica=0, iteration=0, mode="decode", replay_scope="none", graph_bucket=None,
        implementation_selections=(), completion=done, start_ms=0.0, valid_tokens=1,
        padded_tokens=1, operators=2, device_complete=done, declared_commits={"r": commit} if declared else {},
        fallback_commits={} if declared else {"r": commit}, commits={"r": commit}, host_results={}, client_deliveries={},
    )
    summary = StepSummary()
    summary.record(step)
    _fields(summary.to_dict()["total"]["request_events"], commits=int(declared),
            fallback_completions=int(not declared), unclassified_commits=0)

def test_execution_distinguishes_declared_and_implicit_commits():
    parsed = parse_case(case())
    plan = IterationPlan(tuple(RequestView(key, 1, 4, True, "decode") for key in ("r0", "r1")))
    work = IterationWork((_op("commit", "token_commit", "token_commit", ("r0",)),), plan, 1, 2, 2, None, "decode", "test")
    execution = Execution(parsed, CostModel(parsed), include_serving_io=False, include_submission=False)
    step = execution.add(work, 0, 0)
    execution.engine.advance(until=(step.completion,), first=False)
    assert set(step.declared_commits) == {"r0"} and set(step.fallback_commits) == {"r1"}
    assert set(step.commits) == {"r0", "r1"}
    public = step.to_dict()
    assert public["declared_commit_ms"].keys() == {"r0"} and public["fallback_commit_ms"].keys() == {"r1"}
    _fields(execution.summary()["step_summary"]["total"]["request_events"],
            commits=1, fallback_completions=1, unclassified_commits=0, merged_commits=2)

def test_model_assembly_omits_serving_io_and_host_submission():
    result = assemble(_case_with_io(), scope="model", iterations=1)
    assert result["model_fragment"]["service_families_present"] == [] and result["framework_fragments"] == []
    _fields(result["triggers"], request_commit_callbacks=0, batch_prepare=0)
    serviced = result["execution"]["serviced"]
    assert all(serviced.get(key, 0) == 0 for key in ("h2d@r0", "d2h@r0", "scheduler@r0"))
    first = result["steps"][0]
    _fields(first, submitted_ms=None, submitted_basis="submission_not_included", declared_commit_ms={})
    assert first["fallback_commit_ms"].keys() == {"r"}
    _fields(_request_events(result), commits=0, fallback_completions=1, host_results=0)
    _fields(result["metrics"], commit_events=0, fallback_completion_events=1)

@pytest.mark.parametrize("scope,error", [
    ("everything", "assembly_scope_unknown"), ("framework", "framework_assembly_requires_declared_serving_io"),
])
def test_assembly_rejects_invalid_scope_or_missing_framework_io(scope, error):
    with pytest.raises(ModelingError, match=error):
        assemble(case(), scope=scope)

def test_multiple_replicas_use_cohort_assembly():
    assert assemble(case(dp=2), scope="service")["assemble_backend"] == "cohort_driver"

def _emit_commit(view):
    builder = Builder("s0.l0", SimpleNamespace(index=0, scope="target"), 0, (view,),
                      SimpleNamespace(compute_dtype="bf16", activation_dtype="bf16"), "decode")
    identifier = emit_commit_op(builder, name="recover_ssm_commit", deps=("scan",), views=(view,),
                                heads=2, dk=4, dv=8, state_dtype="fp32")
    return identifier, builder.ops

def test_deferred_intent_is_uncharged_and_binds_at_acceptance():
    view = RequestView("r0", 6, 128, True, "decode", draft_verify=True, draft_advanced_tokens=3)
    identifier, ops = _emit_commit(view)
    assert identifier == "s0.l0.recover_ssm_commit"
    intent = ops[-1]
    assert is_recover_ssm_intent(intent) and intent.requests == ("r0",)
    assert (intent.flops, intent.vector_ops, intent.hbm_bytes, intent.mte2_bytes, intent.mte3_bytes) == (0, 0, 0, 0, 0)
    _fields(dict(intent.metadata)["recover_ssm_geometry"], heads=2, dk=4, dv=8, scope="target", layer=0)
    bound, = bind_recover_ssm_commits((intent,), views=(view,), acceptance_events={"r0": "accept"},
                                      state_bindings=None, runtime=None)
    assert bound.deps == ("scan", "accept") and bound.vector_ops == pytest.approx(4 * 3 * 2 * 4 * 8)
    state = 2 * 4 * 8 * 4
    record_bytes = 2 * (8 * 4 + 2 * 4 * 2)
    assert bound.mte2_bytes == pytest.approx(state + 3 * record_bytes)
    assert bound.mte3_bytes == pytest.approx(state) and bound.hbm_bytes == pytest.approx(bound.mte2_bytes + state)
    _fields(dict(bound.metadata), recover_ssm_stage="commit", recover_ssm_pending_acceptance=False,
            committed_tokens=(("r0", 3),))

def test_prefill_multitoken_query_does_not_emit_or_bind_a_commit():
    view = RequestView("p0", 6, 0, True, "prefill", prompt_tokens=6)
    assert _emit_commit(view)[0] is None
    intent = OpSpec("intent", "elementwise", "prefill", 0, 0, ("p0",), (1,), (1,), (1,), (1,),
                    metadata=(("recover_ssm_intent", True), ("state_owner", "layer0"), ("heads", 2), ("dk", 4), ("dv", 8)))
    assert bind_recover_ssm_commits((intent,), views=(view,), acceptance_events={},
                                    state_bindings=None, runtime=None) == (intent,)

def test_recurrent_commit_is_bound_only_between_acceptance_and_service_commit():
    parsed = parse_case(compose_case("glm-5.3-flash", "a5", tp=1, dp=1, ep=1, o_proj_tp=1, linear_proj_tp=1,
                                     lm_head_tp=1, layers=2, concurrency=1, recover_ssm=True,
                                     redundant_experts_per_rank=0, homogeneous_graph_reuse="off"))
    body = target_graphs(parsed, phase="decode", tokens=6, context=128, sequences=1, scope="body")[0]
    assert not any(is_recover_ssm_intent(op) for op in body)
    view = RequestView("r0", 1, 128, True, "decode", draft_verify=True,
                       draft_proposal_tokens=5, draft_advanced_tokens=3)
    ops = Frontend().program(IterationPlan((view,)), parsed.model, parsed.runtime,
                             parsed.machine, assembly="service").work.ops
    assert not any(is_recover_ssm_intent(op) for op in ops)
    commits = [op for op in ops if dict(op.metadata).get("recover_ssm_stage") == "commit"]
    assert len(commits) == 2
    ids = [op.op_id for op in ops]
    for op in commits:
        assert ids.index("accept") < ids.index(op.op_id) < ids.index("token_commit") and "accept" in op.deps
    assert set(next(op for op in ops if op.family == "token_commit").deps) - {"accept"} == {op.op_id for op in commits}

def test_speculative_rejections_preserve_integer_commits_and_real_gaps():
    raw = case(output=6)
    raw["runtime"]["metadata"] = {"draft_mode": "mtp", "draft_proposal_tokens": 2,
                                  "draft_accept_trace": [0, 2], "draft_accept_tokens": 1}
    row = predict(raw, lower=linear_work)["requests"][0]
    _fields(row, generated_tokens=6, iterations=4, finish_ms=11, itl={"count": 5, "mean_ms": 7 / 5, "max_ms": 3},
            speculative={"proposed_tokens": 4, "accepted_tokens": 2, "rolled_back_tokens": 2})

def test_pipeline_microbatches_produce_finite_fill_and_drain():
    raw = case(pp=2, requests=[_request(key, 1, 1) for key in ("a", "b")])

    def staged(plan, model, runtime_spec, machine):
        tokens = plan.valid_tokens
        shape = (tokens,)
        requests = tuple(view.request_id for view in plan.requests)
        ops = tuple(OpSpec(f"stage{i}", "vector", "prefill", i, i, requests, shape, shape, shape, shape,
                           flops=tokens * 1e6, vector_ops=tokens * 1e6, deps=("stage0",) if i else (),
                           dtype="bf16") for i in range(2))
        return IterationWork(ops, plan, 2, tokens, tokens, None, "prefill", "two_stage")

    serial = predict(raw, lower=staged)
    raw["runtime"]["pipeline_microbatches"] = 2
    pipeline = predict(raw, lower=staged)
    assert (serial["metrics"]["makespan_ms"], pipeline["metrics"]["makespan_ms"]) == (4, 3)
    assert [row["first_token_abs_ms"] for row in pipeline["requests"]] == [2, 3]
    assert pipeline["execution"]["work"] == serial["execution"]["work"]


def _stage(identifier, name, amount=8.0, role="cpu", **options):
    return StageSpec(local_id=identifier, label=name, work=(WorkTerm(physical_resource_role=role, issued=amount),),
                     **options)

def _catalog(*stages, **options):
    return ProgramCatalog(catalog_id="catalog", implementation_id="spine", stages=stages, **options)

def _submit(engine, catalog, name, **kwargs):
    return engine.submit_catalog(catalog, BindingFrame(occurrence=engine.new_occurrence()), name=name, **kwargs)

def _drain(engine, submission):
    engine.advance(until=(submission.done,))
    assert submission.done.done

def test_engine_identity_and_causal_service_conservation():
    engine = Engine([Resource("cpu", 4.0)])
    assert isinstance(engine, RuntimeState) and runtime.Engine is Engine and runtime.Event is Event
    _fields(engine.snapshot(), uid=engine.uid, now_ms=0.0)
    assert Engine([Resource("cpu", 8.0)]).uid != engine.uid
    first = engine.submit("first", {"cpu": 8.0})
    second = engine.submit("second", {"cpu": 4.0}, after=(first,))
    engine.advance(until=(second,))
    assert second.started >= first.time
    assert engine.now == pytest.approx((8.0 + 4.0) / 4.0)
    assert engine.serviced["cpu"] == sum(engine.summary()["serviced"].values()) == pytest.approx(8.0 + 4.0)

def test_fifo_is_nonpreemptive_and_priority_only_orders_admission():
    engine = Engine([Resource("wire", 1.0, policy="fifo")])
    low = engine.submit("low", {"wire": 2.0}, priority=0)
    high = engine.submit("high", {"wire": 1.0}, priority=9)
    engine.advance(until=(high,))
    assert low.done and high.done and low.time < high.time

@pytest.mark.parametrize("facade", [False, True])
def test_catalog_ports_and_receipts_belong_to_the_submitting_runtime(facade):
    if facade:
        parsed = parse_case(json.loads(EXAMPLE_CASE.read_text()))
        submitter = Execution(parsed, CostModel(parsed), include_submission=False)
        engine = submitter.engine
        role = next(name for name, resource in engine.resources.items() if resource.policy == "fair" and resource.rate > 0)
        stages = (_stage(1, "op", 4.0, role, output_ports=("x",)),)
    else:
        submitter = engine = Engine([Resource("cpu", 8.0)])
        role = "cpu"
        stages = (_stage(1, "a", output_ports=("x",)), _stage(2, "b", relative_after=()))
    occurrence = engine.new_occurrence()
    binding = BindingFrame(occurrence=occurrence, resources={role: role} if facade else {})
    result = submitter.submit_catalog(_catalog(*stages), binding, name="ports")
    if facade:
        assert result.occurrence == occurrence
        assert {result.start.sequence, result.done.sequence} <= engine._tasks.keys()
    _drain(engine, result)
    assert result.ports["x"].ready.done
    receipt = engine.snapshot_submission(result)
    if facade:
        assert submitter.engine is engine and receipt["metrics"]["completed"] == 1
        assert receipt["metrics"]["service"][role] == pytest.approx(4.0)
    else:
        assert result.start.done and "serviced" not in receipt
        assert receipt["metrics"]["completed"] >= 2
        assert engine.serviced["cpu"] == receipt["metrics"]["service"]["cpu"] == pytest.approx(2 * 8.0)

def _finite(total, slots, stages, index_of, **options):
    return _catalog(finite=FiniteProgramSpec(total=total, slots=slots,
                    classes=(StageClass(class_id=0, stages=stages),), index_of=index_of, **options))

@pytest.mark.parametrize("setup_drain", [False, True])
def test_finite_cursor_counts_tiles_and_honors_setup_drain_minimum(setup_drain):
    engine = Engine([Resource("cpu", 8.0)])
    if setup_drain:
        catalog = _finite(1, 1, (_stage(1, "t"),), lambda i: (0, 1, 0, 0, 0),
                          setup=(_stage(90, "setup"),), drain=(_stage(91, "drain"),), minimum_ms=5.0)
    else:
        catalog = _finite(3, 2, (_stage(1, "tile"),), lambda i: (0, i + 1, i, 0, i // 2))
    _drain(engine, _submit(engine, catalog, "finite"))
    assert engine.serviced["cpu"] == pytest.approx(3 * 8.0)
    assert engine.now == pytest.approx(5.0 if setup_drain else 3 * 8.0 / 8.0)

def test_finite_cursor_publishes_only_the_declared_live_window():
    engine = Engine([Resource("cpu", 8.0)])
    total, slots, calls = 50_000, 4, {"n": 0}

    def coordinates(index):
        calls["n"] += 1
        if not 0 <= index < total:
            raise IndexError(index)
        return (0, index + 1, index, index, index % slots)

    submission = _submit(engine, _finite(total, slots, (_stage(1, "tile", 2.0),), coordinates), "finite")
    assert calls["n"] <= 2 * slots + 8 and len(engine._tasks) <= 2 * slots + 2
    _drain(engine, submission)
    assert (engine.serviced["cpu"], engine.now) == pytest.approx((2.0 * total, 2.0 * total / 8.0))
    summary = engine.summary()
    assert summary["live_tasks"] == 0 and summary["peak_live_tasks"] <= 2 * slots + 2 and not engine._promises

@pytest.mark.parametrize("nested_count", [1, 2], ids=["flat", "finite"])
def test_nested_catalog_inherits_parent_readiness(nested_count):
    engine = Engine([Resource("cpu", 8.0)])
    ready = engine.submit("external:ready", latency=10.0)
    outer = StageSpec(local_id=1, label="outer", nested_catalog=_catalog(_stage(1, "inner")))
    catalog = (_catalog(outer) if nested_count == 1
               else _finite(nested_count, 1, (outer,), lambda i: (0, i + 1, i, i, 0)))
    _drain(engine, _submit(engine, catalog, "nested", after=(ready,)))
    assert (engine.now, engine.serviced["cpu"]) == pytest.approx((10.0 + nested_count, nested_count * 8.0))

def _allocation_stages(amount):
    grant = AllocationSpec(symbol=0, label="buf", pool_role="l1", amount=amount, units=(0,))
    return (_stage(1, "load", acquire=(grant,)),
            _stage(2, "free", relative_after=(EdgeSpec(source_node=1),), release=(ReleaseSpec(symbol=0),)))

def test_finite_release_frees_only_the_grant_of_its_panel():
    total = 3
    engine = Engine([Resource("cpu", 8.0)], pools={"l1": 16}, pool_units={"l1": 2})
    catalog = _finite(total, 2, _allocation_stages(16.0), lambda i: (0, i + 1, i, i, i % 2))
    _drain(engine, _submit(engine, catalog, "alloc"))
    assert engine.allocated == {"l1": 0.0} and engine.peak_allocated == {"l1": 16.0}
    acquire, release = ([row["name"] for row in engine.allocation_trace if row["action"] == action]
                        for action in ("acquire", "release"))
    assert len(acquire) == len(release) == len(set(acquire)) == total and set(acquire) == set(release)

def test_allocation_evidence_is_independent_of_verbose_trace():
    summaries = []
    for limit in (64, 0):
        engine = Engine([Resource("cpu", 8.0)], pools={"l1": 8}, trace_limit=limit)
        _drain(engine, _submit(engine, _catalog(*_allocation_stages(8.0)), "trace"))
        summary = engine.summary()
        for key in ("retained_trace_events", "trace_dropped"):
            summary.pop(key)
        summaries.append(summary)
    assert summaries[0] == summaries[1]
    _fields(summaries[0], retained_allocation_events=2, allocation_events=2)

@pytest.mark.parametrize("commit", [False, True], ids=["live", "committed"])
def test_catalog_state_effects_share_the_fragment_runtime_ledger(commit):
    engine = Engine([Resource("cpu", 8.0)])
    key = "kv" if commit else "kv_cache:layer:0"
    ledger = engine.ensure_state_ledger() if commit else None
    stages = [_stage(1, "write", state_effects=(StateEffect(key, "write", 1),))]
    if commit:
        stages.append(_stage(2, "commit", relative_after=(EdgeSpec(source_node=1),), state_effects=(StateEffect(key, "commit", 1),)))
    _drain(engine, _submit(engine, _catalog(*stages), "state"))
    assert engine.ensure_state_ledger().snapshot() == {
        "committed": {key: 1} if commit else {}, "live": {key: (1,)}}
    if commit:
        assert engine.state_ledger is ledger and engine.state_versions == {key: [1]}
        conductor = scheduling.FragmentConductor(engine)
        assert conductor.state is ledger and conductor.state.check_read(key, 1, at=engine.now) == 1

def test_fragment_rejects_a_state_ledger_owned_by_another_runtime():
    first, second = (Engine([Resource("cpu", 1.0)]) for _ in range(2))
    ledger = first.ensure_state_ledger({"kv": 1})
    assert scheduling.FragmentConductor(first, state=ledger).state is ledger
    with pytest.raises(ValueError, match="state_ledger_runtime_conflict"):
        scheduling.FragmentConductor(second, state=ledger)
    assert second.state_ledger is None and second.ensure_state_ledger({"kv": 1}) is not ledger

def test_catalog_validation_publishes_nothing_on_failure():
    engine = Engine([Resource("cpu", 1.0)])
    before = engine.snapshot()
    with pytest.raises(ExecutionError, match="resource_missing:missing"):
        _submit(engine, _catalog(_stage(1, "bad", 1.0, "missing")), "bad")
    assert before["live_tasks"] == engine.snapshot()["live_tasks"] == 0
    assert engine.snapshot()["next_sequence"] == before["next_sequence"]

def test_host_fifo_progress_is_independent_of_device_and_bounded():
    cursor = HostCursor(fifo_role="host", occurrence_uid="u", queue_capacity=1)
    assert cursor.can_submit()
    cursor.note_submit()
    assert not cursor.can_submit()
    with pytest.raises(ValueError, match="host_fifo_queue_full"):
        cursor.note_submit()
    cursor.note_host_done(0)
    assert cursor.can_submit() and cursor.host_completed == 1
    engine = Engine([Resource("host", 1.0, policy="fifo"), Resource("cpu", 8.0)])
    catalog = _catalog(_stage(1, "dev"), host_spec=HostProgramSpec(fifo_role="host", submit_work_ms=2.0,
                                                                 queue_capacity=1))
    submission = _submit(engine, catalog, "h")
    with pytest.raises(ExecutionError, match="host_fifo_queue_full:host"):
        _submit(engine, catalog, "h2")
    _drain(engine, submission)
    assert engine.now == pytest.approx(2.0 + 8.0 / 8.0)
    assert (engine.serviced["host"], engine.serviced["cpu"]) == pytest.approx((2.0, 8.0))
    assert engine.host_fifos["host"].host_completed == 1
    _drain(engine, _submit(engine, catalog, "h3"))

def test_binding_frame_snapshots_nested_mappings():
    geometry = {"shape": (2, 3)}
    frame = BindingFrame(occurrence=Engine([Resource("cpu", 1.0)]).new_occurrence(),
                         geometry=geometry, coordinates={"unit": 1})
    geometry["shape"] = (99, 99)
    assert frame.geometry["shape"] == (2, 3)
    with pytest.raises(TypeError, match="does not support item assignment"):
        frame.coordinates["unit"] = 2

@pytest.mark.parametrize("rate,caps,expected", [
    (10.0, {1: 3.0, 2: 100.0}, {1: 3.0, 2: 7.0}),
    (9.0, {1: 100.0, 2: 100.0, 3: 100.0}, {1: 3.0, 2: 3.0, 3: 3.0}),
])
def test_capped_max_min_fairness(rate, caps, expected):
    assert fair_rates(rate, caps) == pytest.approx(expected)

def test_periodic_summary_peaks_do_not_multiply():
    first = IntervalSummary(dt=2.0, completed=2, service={"cpu": 4.0}, peak_allocated={"hbm": 8.0})
    second = IntervalSummary(dt=1.0, completed=1, service={"cpu": 2.0}, peak_allocated={"hbm": 5.0})
    combined, scaled = first.plus(second), first.scaled(3)
    assert (combined.dt, combined.completed, combined.service["cpu"]) == (3.0, 3, 6.0)
    assert combined.peak_allocated["hbm"] == scaled.peak_allocated["hbm"] == 8.0
    assert scaled.service["cpu"] == 3 * 4.0

def test_prepared_work_uses_the_public_submission_api():
    from unittest.mock import Mock
    engine = Mock(spec=("prepare_stage", "submit_prepared"))
    engine.configure_mock(**{"prepare_stage.return_value": "PREPARED", "submit_prepared.return_value": "EVENT"})
    stage = Stage("compute", {"cube": 1.0}, caps={"cube": 4.0})
    prepared = stage.prepare(engine, "task", tuple)
    event = stage.submit(engine, "task", (), tuple, "owner", prepared=prepared)
    assert prepared == "PREPARED" and event == "EVENT"
    assert engine.submit_prepared.call_args.args[0] == "PREPARED"
    assert engine.prepare_stage.call_args.kwargs["caps"] == {"cube": 4.0}
