"""Shared serving cases, scheduler doubles, and service-regime builders.

[Role] Centralize prefill-decode (PD) cases, study specifications,
  scheduler doubles, and service-regime fixtures.
[Invariants] Cases and ledgers are fresh for each caller. PD pools own their
  machine and runtime declarations.
[Timeouts & Locks] In-memory setup and bounded graph execution; no locks.
[Side-Effect Policy] Local cases and ledgers, with a scoped lowering spy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from mindie_sim import compose_case, graph_costs, predict
from mindie_sim.prediction.execution import dispatch
from mindie_sim.prediction.api import run_cohort
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.service_regime_domains import compile_service_regime_domains
from mindie_sim.prediction.service_regime_domains.services import service_identity
from mindie_sim.prediction.frontend.speculation import draft_program
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import RequestView

from _support.collective_cases import declared, isolated_calls, joint_case
from _support.hardware import hardware_case
from _support.kernel_invocation import copy_op
from _support.memory_cases import case, draft_case, linear_work

__all__ = (
    "STUDY_PROBES",
    "bind_pd_pool",
    "case",
    "linear_work",
    "pd_case",
    "pd_model_case",
)

STUDY_PROBES: list[dict[str, Any]] = [
    {"id": "prefill", "phase": "prefill", "tokens": 4},
    {"id": "decode", "phase": "decode", "tokens": 1, "context": 16},
]


def pd_case(requests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Two-pool disaggregated PD raw case with a 128 KB/s zero-latency connector."""
    raw = case(prompt=3, output=3, requests=requests)
    raw["pools"] = {
        "prefill": {"machine": deepcopy(raw["machine"]), "runtime": deepcopy(raw["runtime"])},
        "decode": {"machine": deepcopy(raw["machine"]), "runtime": deepcopy(raw["runtime"])},
    }
    raw["connector"] = {"bytes_per_s": 128000, "latency_us": 0}
    return raw


def pd_model_case(tokens: int, num_kv_heads: int) -> dict[str, Any]:
    """Single-layer disaggregated PD case with 64 B/ms per-pair connector chunks."""
    raw = pd_case([{"id": "r", "prompt_tokens": tokens, "output_tokens": 3}])
    raw["model"].update(
        layers=1,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        head_dim=16,
        intermediate_size=64,
        vocab_size=128,
    )
    raw["connector"] = {
        "link": {
            "name": "handoff",
            "capacity_bytes_per_ms": 64,
            "latency_us": 0,
            "semantics": "per_pair",
        },
        "chunk_bytes": 64,
    }
    return raw


def bind_pd_pool(
    raw: dict[str, Any],
    pool: str,
    ranks: tuple[int, ...] | list[int],
    tp: int,
    cp: int,
) -> None:
    """Bind physical rank ids and TP/DCP parallelism on one pool of ``raw``."""
    machine = raw["pools"][pool]["machine"]
    machine["num_devices"] = len(ranks)
    machine["devices"] = [{"rank": rank} for rank in ranks]
    machine["links"][0]["members"] = list(ranks)
    raw["pools"][pool]["runtime"].update(
        tp=tp, pp=1, cp=cp, dcp=cp, cp_plan="dcp" if cp > 1 else "none"
    )


class TinyMemory:
    """Capacity double with explicit lookahead reservation targets."""
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

def speculative_case(trace=(0, 2), output=4, prompt=4):
    raw = case(requests=[{"id": "r", "prompt_tokens": prompt, "output_tokens": output}])
    raw["runtime"]["metadata"] = { "draft_mode": "mtp", "draft_proposal_tokens": 3, "draft_accept_trace": list(trace), "draft_accept_tokens": 1}
    return parse_case(raw)

def scheduler_for(parsed, memory=None):
    return Scheduler(parsed, list(parsed.workload.requests),
                     Memory(parsed) if memory is None else memory)

def memory_with_free_budget(raw, extra_bytes, *, weights_only=False):
    raw["runtime"].update(
        kv_block_size=1, gpu_memory_utilization=1.0, memory={"derive_workspace": False})
    case_spec = parse_case(raw)
    policy = MemoryPolicy(derive_workspace=False)
    ledger = Memory(case_spec, policy)
    baseline = ledger.weights[0] if weights_only else ledger.fixed_per_rank[0]
    capacity = int(baseline + extra_bytes)
    dies = tuple(replace(die, memory_bytes=capacity) for die in case_spec.machine.dies)
    parsed = replace(case_spec, machine=replace(case_spec.machine, dies=dies))
    memory = Memory(parsed, policy)
    assert memory.capacity[0][0] == pytest.approx(extra_bytes)
    return parsed, memory

def deepseek_128():
    return compose_case("deepseek-v4-pro", hardware="a5", tp=2, dp=64, ep=128, num_devices=128, layers=4, mtp_proposal_tokens=5, prompt_tokens=4096, output_tokens=8)

def by_recipe(ops, recipe, *, layer=None):
    return tuple(op for op in ops if op.recipe == recipe and (layer is None or op.layer == layer))

def by_prefix(ops, prefix):
    return tuple(op for op in ops if op.op_id.startswith(prefix))


def single_device_machine():
    return {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}


def serving_io():
    return {
        "host_prepare_ms": .2,
        "host_result_ms": .2,
        "h2d": {"bytes_per_s": 8000, "bytes_per_token": 8, "latency_us": 50},
        "d2h": {"bytes_per_s": 4000, "bytes_per_token": 4, "latency_us": 50},
        "client_delivery_ms": .5,
    }


def production_draft_case():
    machine = parse_case({**case(), "machine": single_device_machine()}).machine
    base = draft_case()
    return replace(base, machine=machine, runtime=replace(base.runtime, serving_io=serving_io()))


SUPPORT = {"q": {"min": 1, "max": 200}}
WINDOW = {"q": {"min": 1, "max": 200}, "kv_len": {"min": 1, "max": 200}}
WINDOW_COORDINATES = {"q": {"path": "semantic_shape.0"},
                      "kv_len": {"path": "semantic_shape.1"}}


def install(raw, rule):
    raw.setdefault("runtime", {}).setdefault("kernels", {})["service_regimes"] = [rule]
    return raw

def response(us, **kwargs):
    return {"id": "native-response", "match": {"kernel": "NativeDispatch"},
            "response": {"fixed_us": us}, **kwargs}

def clean(result):
    execution = result["execution"]
    assert execution["live_tasks"] == execution["live_programs"] == 0
    assert not any(execution["live_allocated"].values())

def affine(fixed_us=8000.0):
    return {"kind": "affine", "fixed_us": fixed_us}

def curve(outside="analytic", knots=((1, 8000.0), (200, 10000.0))):
    return {"kind": "piecewise_linear", "coordinate": "q", "outside": outside,
            "knots": [list(pair) for pair in knots]}

def rule(rid, low, high, *, group="copy-domain", support=None, response=None,
         outside="analytic", coordinates=None, extra_domain=None, execution=None,
         match=None):
    raw = {"id": rid, "match": match or {"recipe": "copy"},
           "coordinates": coordinates or {"q": {"path": "semantic_shape.0"}},
           "domain": {"q": {"min": low, "max": high}, **(extra_domain or {})},
           "response": response or affine(),
           "continuity": {"group": group, "support": support or SUPPORT,
                          "outside": outside}}
    if execution is not None:
        raw["execution"] = execution
    return raw

def rect(rid, qlo, qhi, klo, khi, **changes):
    return rule(rid, qlo, qhi, support=changes.pop("support", WINDOW),
                coordinates=WINDOW_COORDINATES,
                extra_domain={"kv_len": {"min": klo, "max": khi}}, **changes)

def compiled(*declarations):
    return compile_service_regime_domains(list(declarations))

def engine(declarations):
    raw = hardware_case()
    raw["runtime"] = {"kernels": {"service_regimes": list(declarations)}}
    return raw

def select(raw, shape):
    result = graph_costs(raw, (replace(copy_op(4000), semantic_shape=shape),), isolated=False)
    selected = [row["id"] for row in result["parameters"]["service_regimes"]
                if row["queries"].get("observable_response")]
    assert len(selected) == 1, result["parameters"]["service_regimes"]
    return selected[0], result

def rejects(code, *declarations):
    with pytest.raises(ModelingError, match=code):
        compiled(*declarations)

def source_call():
    raw = declared("AICPU_TS", "mesh_oneshot")
    ops = isolated_calls(raw)[0]
    group_id = dict(ops[0].metadata)["hccl_source_group"]
    selected = tuple(op for op in ops
                     if dict(op.metadata)["hccl_source_group"] == group_id)
    keys = {op.op_id for op in selected}
    selected = tuple(replace(op, deps=tuple(d for d in op.deps if d in keys))
                     for op in selected)
    metadata = dict(selected[0].metadata)
    identity = service_identity(parse_case(raw), selected[0])
    raw["runtime"]["kernels"]["service_regimes"] = [{
        "id": "captured-native-communication-domain",
        "match": {"kernel": metadata["device_kernel_group"]["kernel"],
                  "transport": identity["transport"]},
        "response": {"fixed_us": 5000},
        "provenance": {"kind": "causal_control",
                       "boundary": "one_native_collective_invocation"}}]
    return raw, selected

def declared_rule(rule, raw=None):
    return install(deepcopy(raw or hardware_case()), rule)

def response_rule(**changes):
    return {"id": "observed-copy", "match": {"family": "copy"},
            "coordinates": {"bytes": {"path": "work.hbm"}},
            "domain": {"bytes": {"min": 1000, "max": 4000}},
            "response": {"fixed_us": 1000, "terms": [{
                "coordinate": "bytes", "us_per_unit": 2,
                "scale_with": {"resource": "hbm", "reference_rate": 1e6}}]},
            "provenance": {"kind": "synthetic_causal_control"}, **changes}

def geometry_case(tp):
    raw = case(tp=tp)
    raw["model"].update(hidden_size=128, num_heads=16, num_kv_heads=8,
                        intermediate_size=256, vocab_size=256)
    raw["runtime"].update(vllm_impl="declared-test-runtime", vllm_version="fixture-v1")
    device = raw["machine"]["device"]
    device["rates"]["matrix_service"] = 1e9
    device["work_units"] = {"matrix_service": "busy_ns"}
    device["resource_units"] = {"matrix_service": 1}
    return raw

def lowered(raw, recipe, rows, *, context=4096):
    operation = next(op for op in target_operators(
        raw, phase="decode" if rows == 1 else "prefill", tokens=rows, context=context)
        if op.recipe == recipe)
    return replace(operation, deps=())

def rule_for(raw, operation, response):
    identity = service_identity(parse_case(raw), operation)
    return {"id": "geometry-response", "match": {
        **{key: identity[key] for key in (
            "runtime", "revision", "format", "dtype", "layout", "implementation",
            "replay_scope")},
        "hardware": identity["hardware"]["key"], "recipe": operation.recipe},
        "coordinates": {"m": {"path": "local_shape.0"}, "n": {"path": "local_shape.1"},
                        "k": {"path": "local_shape.2"}, "flops": {"path": "work.cube"},
                        "memory_bytes": {"path": "work.hbm"}},
        "response": response,
        "execution": {"kind": "joint_native", "resource": "matrix_service",
                      "reference_rate": 1e9, "native_kernel": "DeclaredPhysicalProgram"},
        "provenance": {"basis": "declared_supply_control_not_hardware_calibration"}}

def bind(raw, rule):
    raw["runtime"]["kernels"] = {"service_regimes": [deepcopy(rule)]}
    return raw

def roofline():
    return {"kind": "roofline", "fixed_us": 3, "services": [
        {"coordinate": "flops", "resource": "cube", "dtype": "bf16", "utilization": .5},
        {"coordinate": "memory_bytes", "resource": "hbm", "utilization": .25}]}

def curve_case(*, outside="analytic", joint=True):
    raw = joint_case()
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    rule.update(match={"recipe": "copy", "dtype": "bf16",
                       "metadata.protocol": "retained"},
                coordinates={"bytes": {"path": "work.hbm"},
                             "context": {"path": "metadata.context_tokens"}},
                domain={"bytes": {"min": 500, "max": 8000},
                        "context": {"min": 2048, "max": 8192}},
                response={"kind": "piecewise_linear", "coordinate": "bytes",
                          "knots": [[1000, 4000], [2000, 6000], [4000, 9000]],
                          "outside": outside})
    if not joint:
        rule.pop("execution", None)
    return raw

def operation(amount, *, context=4096):
    return copy_op(amount, metadata=(("context_tokens", context),
                                     ("protocol", "retained")))


def pipeline_case():
    raw = case(pp=2, requests=[
        {"id": "a", "prompt_tokens": 1, "output_tokens": 2},
        {"id": "b", "prompt_tokens": 1, "output_tokens": 2}])
    raw["runtime"]["pipeline_microbatches"] = 2
    return raw

def binding_lower(plan, model, runtime, machine, dispatch_binding=None):
    ops = []
    for view in plan.requests:
        record = {"request": "template", "direction": "read", "storage": "kv:L0",
                  "bytes": 16, "span": (0, 16), "request_slot": 0,
                  "state_version": 0, "reader": "template", "template": True}
        shape = (1,)
        ops.append(OpSpec(f"op:{view.request_id}", "vector", view.phase, 0, 0,
                          (view.request_id,), shape, shape, shape, shape, flops=1e6,
                          vector_ops=1e6, metadata=(("state_accesses", (record,)),)))
    tokens = sum(view.q_len for view in plan.requests)
    return IterationWork(tuple(ops), plan, 1, tokens, tokens, None, "body",
                         "test_binding")

def capture_binding(parsed, lower, versions):
    memory = Memory(parsed)
    schedulers = replica_schedulers(parsed, memory)
    for scheduler in schedulers:
        for state in scheduler.states:
            state.state_version = int(versions[state.request.request_id])
    seen = []
    def capture(op, replica, recipe, ready, complete):
        seen.append((replica, op.op_id, dict(op.metadata)))
    execution = serving_execution(parsed, memory, include_io=False, on_operator=capture)
    run = run_cohort(parsed, execution, schedulers, memory, lower, trace_limit=0)
    return run, seen

def pp2_binding_case(dp):
    raw = case(dp=dp, pp=2, requests=[
        {"id": request, "prompt_tokens": 1, "output_tokens": 1}
        for request in ("a", "b", "c", "d")])
    raw["runtime"]["pipeline_microbatches"] = 2
    return raw


def replica_schedulers(parsed, memory):
    if parsed.runtime.parallel.dp == 1:
        return [scheduler_for(parsed, memory)]

    groups = [[] for _ in range(parsed.runtime.parallel.dp)]
    for index, request in enumerate(sorted(
            parsed.workload.requests, key=lambda r: (r.arrival_ms, r.request_id))):
        groups[index % parsed.runtime.parallel.dp].append(request)
    return [Scheduler(parsed, group, memory, replica)
            for replica, group in enumerate(groups)]


def commit_step(committed_ms, *, completion_ms=None, iteration=None):
    step = SimpleNamespace(
        completion=SimpleNamespace(time=committed_ms if completion_ms is None else completion_ms),
        commits={"r": SimpleNamespace(time=committed_ms)},
    )
    if iteration is not None:
        step.iteration = iteration
    return step

def held_decode_scheduler(*, prompt, max_batch_tokens, generated=0, weights_only=False):
    raw = case(requests=[
        {"id": request, "prompt_tokens": prompt, "output_tokens": 10}
        for request in ("a", "b")
    ])
    raw["runtime"].update(max_num_batched_tokens=max_batch_tokens, max_num_seqs=2)
    parsed, memory = memory_with_free_budget(raw, 1024, weights_only=weights_only)
    scheduler = scheduler_for(parsed, memory)
    for state in scheduler.states:
        assert memory.resize(0, state.request.request_id, 4)
        state.computed, state.generated, state.started = 4, generated, True
        state.admission_ms, state.location = 0.0, "decode"
    assert memory.hold_for_async_consumer(0, "b", "reader")
    return scheduler, memory

def run_scheduled_service():
    raw = case(output=3, requests=[
        {"id": request, "prompt_tokens": 4, "output_tokens": 3}
        for request in ("r0", "r1")
    ])
    raw["machine"] = single_device_machine()
    raw["runtime"]["serving_io"] = serving_io()
    case_spec = parse_case(raw)
    execution = Execution(case_spec, CostModel(case_spec), include_serving_io=True)
    frontend = Frontend(attention_policy=AttentionPolicy(**case_spec.runtime.attention_policy))
    scheduler = scheduler_for(case_spec)
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
        program = frontend.program(plan, case_spec.model, scheduler.next_runtime(plan),
                                   case_spec.machine, assembly="service")
        step = execution.add(program.work, 0, engine.now, emitted_tokens=emitted)
        engine.advance(until=(step.completion,), first=False)
        scheduler.commit_batch(step)
        batches += 1
        decisions.append(scheduler.feedback(engine.now, include_done=False))
    engine.advance()
    return batches, decisions, scheduler, execution


def pipeline_draft_case():
    return compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=1, ep=1, pp=2, num_devices=2,
        pp_layer_partition=(1, 3), layers=4, graph_capture_mode="none",
        max_num_batched_tokens=64, mtp_proposal_tokens=5, prompt_tokens=4, output_tokens=6,
    )


def pipeline_draft_workload():
    return {"requests": [
        {"id": "r0", "prompt_tokens": 4, "output_tokens": 2, "arrival_ms": 0.0},
        {"id": "r1", "prompt_tokens": 4, "output_tokens": 6, "arrival_ms": 0.0},
    ]}


def pipeline_draft_transfers(parsed):
    views = (
        RequestView("short", 1, 0, False, "decode", draft_proposal_tokens=1, draft_verify=True),
        RequestView("long", 1, 0, False, "decode", draft_proposal_tokens=5, draft_verify=True),
    )
    regions, _ = draft_program(
        Frontend(), parsed.model, parsed.runtime, views, "iteration_input",
        prefill=False, machine=parsed.machine,
    )
    return [op for region in regions for op in region.ops if op.op_id == "draft_proposal_transfer"]


def sp_mtp_case(dp):
    tp = 8
    raw = compose_case(
        "kimi-k3.1", hardware="a5", dp=dp, tp=tp, dcp=tp, pcp=1, ep=dp * tp, pp=1,
        num_devices=dp * tp, sp=True, recover_ssm=True, layers=4, mtp_proposal_tokens=5,
        moe_comm_mode="dynamic", graph_capture_mode="full", prompt_tokens=4, output_tokens=8,
    )
    raw["runtime"]["graph_capture_keys"] = [16]
    return raw


def dispatch_tp_rows(ops):
    operation = next(op for op in ops if op.op_id == "s0.l1.dispatch")
    metadata = dict(operation.metadata)
    packet = float(metadata["peer_traffic_row_format"]["forward_bytes"])
    matrix = metadata["mc2_phase"]["route_matrix_bytes"]
    return (tuple(round(sum(row) / packet / 16.0, 6) for row in matrix[:8]),
            round(sum(sum(row) for row in matrix) / packet, 6))


def capture_lowered_work(raw, monkeypatch):
    captured = []
    lower = dispatch._lower_with_binding

    def capture(*args, **kwargs):
        work = lower(*args, **kwargs)
        captured.append(work)
        return work

    with monkeypatch.context() as patch:
        patch.setattr(dispatch, "_lower_with_binding", capture)
        predict(raw, max_model_steps=2)
    return captured


def routed_prefill_case():
    raw = case(dp=2, tp=1, ep=2, prompt=6, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=8, top_k=2, hidden_size=32,
        moe_intermediate_size=32, activation_dtype="bf16",
    )
    raw["runtime"].update(max_num_batched_tokens=32, routing={"realizations": [{
        "scope": "body", "layer": 0, "phase": "prefill", "cohort": True,
        "topk_ids": [[0, 1] for _ in range(18)],
        "expert_placement": [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]],
        "source_owners": [0] * 6 + [1] * 12,
    }]})
    return raw


def slow_second_request(plan, model, runtime, machine):
    tokens = sum(view.q_len for view in plan.requests)
    scale = 20.0 if any(view.request_id == "b" for view in plan.requests) else 1.0
    shape = (tokens,)
    operation = OpSpec("work", "vector", "body", 0, 0,
                       tuple(view.request_id for view in plan.requests),
                       shape, shape, shape, shape, flops=tokens * 1e6 * scale,
                       vector_ops=tokens * 1e6 * scale, dtype="bf16")
    return IterationWork((operation,), plan, 1, tokens, tokens, None, "body",
                         "test_slow")


def serving_execution(parsed, memory, *, include_io, bind_memory=True, on_operator=None):
    execution = Execution(
        parsed, CostModel(parsed), include_serving_io=include_io,
        memory=memory if bind_memory else None, memory_policy=memory.policy,
        on_operator=on_operator,
    )
    execution.engine.now = 0
    return execution


def grant_bytes_for_headroom(memory, headroom):
    used = (memory.live[0][0] + memory.staging_live[0] + memory.granted[0][0]
            + memory.shared_reserved[0][0] + memory.selected_reserved[0][0])
    return max(0.0, memory.capacity[0][0] - used - headroom)
