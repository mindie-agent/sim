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
from mindie_sim.compose import compose_case
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

def test_service_feedback_and_host_io_conserve_work():
    raw = _case_with_io(_request("r0", 4, 3), _request("r1", 3, 3))
    result = predict(raw, window={"r0": {"computed": 4}, "r1": {"computed": 3}}, trace_limit=3)
    steps = result["traces"]
    occurrences = result["framework_occurrences"]
    assert len(steps) == len(occurrences) == 3
    for step, io in zip(steps, occurrences):
        submits = [host["start_ms"] for host in step["host_submits"]]
        assert step["start_ms"] < io["host_prepare_ms"] <= min(submits) <= max(submits)
        assert step["fallback_commit_ms"] == {}
        assert step["declared_commit_ms"].keys() == io["deliveries_ms"].keys() == {"r0", "r1"}
        assert min(io["deliveries_ms"].values()) > step["end_ms"] > step["device_end_ms"]
    for previous, following in zip(steps, steps[1:]):
        assert following["start_ms"] == pytest.approx(previous["end_ms"])
    for request in result["requests"]:
        commits = [step["declared_commit_ms"][request["request_id"]] for step in steps]
        _fields(request, generated_tokens=3, first_token_abs_ms=pytest.approx(commits[0]),
                last_token_abs_ms=pytest.approx(commits[-1]), ttft_ms=pytest.approx(commits[0] - request["arrival_ms"]),
                tpot_ms=pytest.approx((commits[-1] - commits[0]) / 2))
    assert result["metrics"]["output_tps"] > 0
    _fields(result["metrics"], output_tokens=6)
    _fields(result["execution"], live_tasks=0, live_programs=0)
    serviced = result["execution"]["serviced"]
    assert serviced["scheduler@r0"] > 0
    assert (serviced["h2d@r0"], serviced["d2h@r0"]) == (3 * 2 * 8, 3 * 2 * 4)
    _fields(_request_events(result), commits=6, fallback_completions=0, host_results=6, client_deliveries=6)

def test_decode_token_budget_expands_into_single_token_steps():
    raw = _case_with_io(_request("r0", 4, 6), _request("r1", 3, 6))
    result = predict(raw, window={"r0": {"computed": 4}, "r1": {"computed": 3}},
                     token_quotas={"r0": 4, "r1": 4}, trace_limit=4)
    assert len(result["traces"]) == 4
    for step in result["traces"]:
        _fields(step, mode="decode", valid_tokens=2)
        assert step["declared_commit_ms"].keys() == {"r0", "r1"}
    for request in result["requests"]:
        _fields(request, generated_tokens=4, visible_context_tokens=request["prompt_tokens"] + 4)
        assert request["last_token_abs_ms"] > request["first_token_abs_ms"]
    assert tuple(result["execution"]["serviced"][key] for key in ("h2d@r0", "d2h@r0")) == (4 * 2 * 8, 4 * 2 * 4)

def test_replay_enqueue_and_batch_propagation_bind_real_events():
    raw = _case_with_io(_request("r0", 2, 2), _request("r1", 2, 2))
    config = raw["runtime"]
    config["serving_io"]["client_delivery_ms"] = 1
    for transfer in ("h2d", "d2h"):
        del config["serving_io"][transfer]["latency_us"]
    config.update(graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full",
                  submission={"graph_replay_ms": 0.25})
    result = predict(raw, window={"r0": {"computed": 2}, "r1": {"computed": 2}}, max_model_steps=1)
    step, = result["traces"]
    io, = result["framework_occurrences"]
    host, = step["host_submits"]
    _fields(result["execution"]["submission"], graph_replays=1, eager_invocations=0)
    assert step["start_ms"] < io["host_prepare_ms"] <= host["start_ms"] < min(step["declared_commit_ms"].values())
    assert io["deliveries_ms"].keys() == {"r0", "r1"}
    assert io["client_propagation_ms"] <= min(io["deliveries_ms"].values())

def test_decode_quota_and_step_limit_execute_the_same_work():
    raw = _case_with_io(_request("r0", 4, 2))
    quota, limited = (predict(raw, window={"r0": {"computed": 4}}, **limit)
                      for limit in ({"token_quotas": {"r0": 2}}, {"max_model_steps": 2}))
    for result in (quota, limited):
        assert len(result["traces"]) == 2
        for step in result["traces"]:
            _fields(step, mode="decode", valid_tokens=1)
            assert step["declared_commit_ms"].keys() == {"r0"}
        assert result["execution"]["serviced"]["d2h@r0"] == 2 * 4
        row, = result["requests"]
        assert row["generated_tokens"] == 2 and row["tpot_ms"] > 0
    assert quota["requests"] == limited["requests"]

@pytest.mark.parametrize("window,modes,computed", [
    (None, ["prefill", "decode"], [4, 5]),
    ({"r": {"computed": 4}}, ["decode", "decode"], [5, 6]),
])
def test_service_context_advances_by_the_written_kv(window, modes, computed):
    raw = case(requests=[_request("r", 4, 2)])
    for steps, expected in enumerate(computed, start=1):
        result = predict(raw, window=window, max_model_steps=steps)
        assert [row["mode"] for row in result["traces"]] == modes[:steps]
        _fields(result["requests"][0], visible_context_tokens=expected, generated_tokens=steps)

def test_framework_io_accounts_for_host_work_and_no_model_launches():
    parsed = parse_case(_case_with_io(_request("r0", 1, 2), _request("r1", 1, 2)))
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True)
    fragments = {fragment.name: fragment for fragment in execution.serving_io.fragments()}
    for fragment in fragments.values():
        assert fragment.layer is None
        assert all(port.direction == "in" for port in fragment.ports.event_in)
        assert all(port.direction == "out" for port in fragment.ports.event_out)
    for name, trigger in (("host_prepare", "per_batch"), ("client_delivery", "per_batch_with_emissions"),
                          ("client_order", "per_request_per_batch")):
        assert fragments[name].trigger == trigger
    for iteration in range(2):
        plan = IterationPlan(tuple(RequestView(key, 1, 1 + iteration, True, "decode")
                                   for key in ("r0", "r1")), iteration=iteration)
        work = IterationWork((), plan, 1, 2, 2, None, "decode", "framework_only")
        step = execution.add(work, 0, execution.engine.now, emitted_tokens={"r0": 1, "r1": 1})
        execution.engine.advance()
        assert not step.host_submits and not step.declared_commits
        assert step.fallback_commits.keys() == step.client_deliveries.keys() == {"r0", "r1"}
    summary = execution.summary()
    _fields(summary["work"], flops=0, vector_ops=0)
    assert all(summary["serviced"][key] > 0 for key in ("scheduler@r0", "h2d@r0", "d2h@r0"))
    _fields(summary["step_summary"]["total"]["request_events"], commits=0, client_deliveries=4, fallback_completions=4)

def test_framework_delivers_only_to_active_requests_after_short_request_finishes():
    raw = _case_with_io(_request("short", 1, 1), _request("long", 2, 3))
    result = predict(raw, window={"short": {"computed": 1}, "long": {"computed": 2}}, max_model_steps=2)
    first, second = result["framework_occurrences"]
    assert first["deliveries_ms"].keys() == {"long", "short"}
    assert second["deliveries_ms"].keys() == {"long"}
    _fields(_request_events(result), client_deliveries=3)
    assert {r["request_id"]: r["generated_tokens"] for r in result["requests"]} == {"short": 1, "long": 2}

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

def test_model_forward_omits_serving_io_and_host_submission():
    result = predict(_case_with_io(), scope="model", phase="decode", tokens=1, context=4)
    assert not ({row["family"] for row in result["timeline"]} & SERVICE_FAMILIES)
    _fields(result["execution"]["submission"], included=False, eager_invocations=0, graph_replays=0)
    serviced = result["execution"]["serviced"]
    assert all(serviced.get(key, 0) == 0 for key in ("h2d@r0", "d2h@r0", "scheduler@r0"))
    _fields(_request_events(result), commits=0, fallback_completions=1, host_results=0)

def test_prediction_rejects_invalid_scope():
    with pytest.raises(ModelingError, match="invalid_prediction_scope"):
        predict(case(), scope="everything")

def test_multiple_replicas_drain_their_own_requests():
    result = predict(case(dp=2, requests=[_request("a", 1, 1), _request("b", 1, 1)]))
    assert result["execution"]["cohort_stop"] == "drained"
    assert {(row["replica"], tuple(row["declared_commit_ms"])) for row in result["traces"]} == {(0, ("a",)), (1, ("b",))}

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
                                      state_bindings=None,
                                      runtime=SimpleNamespace(metadata=(("recover_ssm_readiness", "shipping"),)),
                                      machine=SimpleNamespace(arch_family="A5"))
    # Upstream-exact KDA reverse accumulation (GA-09, VA/ops/triton/kda/recoverssm.py:593-615,788): 2*(u+1)*E.
    assert bound.deps == ("scan", "accept") and bound.vector_ops == pytest.approx(2 * (3 + 1) * 2 * 4 * 8)
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
                                    state_bindings=None, runtime=None,
                                    machine=SimpleNamespace(arch_family="A5")) == (intent,)

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
    plans = [op for op in ops if op.recipe == "kda_prepare_commit_plan"]
    convs = [op for op in ops if op.recipe == "kda_compact_conv_state"]
    commits = [op for op in ops if dict(op.metadata).get("recover_ssm_stage") == "commit"]
    pubs = [op for op in ops if dict(op.metadata).get("recover_ssm_stage") == "state_publish"]
    assert len(plans) == 1 and len(convs) == 1 and len(commits) == 1 and len(pubs) == 2
    assert dict(commits[0].metadata)["num_layers"] == 2
    ids = [op.op_id for op in ops]
    for op in plans + convs + commits + pubs:
        assert ids.index("accept") < ids.index(op.op_id) < ids.index("token_commit")
    assert "accept" in plans[0].deps and plans[0].op_id in commits[0].deps and convs[0].op_id in commits[0].deps
    assert all(commits[0].op_id in op.deps for op in pubs)
    assert set(next(op for op in ops if op.family == "token_commit").deps) - {"accept"} == {op.op_id for op in pubs}

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
    # minimum_ms anchors after setup completes (matching PipelineProgram and PipelineTemplate).
    assert engine.now == pytest.approx((8.0 / 8.0) + 5.0 if setup_drain else 3 * 8.0 / 8.0)

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
