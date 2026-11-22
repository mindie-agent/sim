"""Prediction contracts, serving oracles, and physical graph equivalence.

[Role] Exercise prediction, analysis and capacity through the public spine.
[Invariants] Conserve work, request completion and topology through reuse and deployment.
[Timeouts & Locks] Bounded in-memory execution and isolated evaluation caches.
[Side-Effect Policy] Construct local cases and use scoped monkeypatches for instrumentation.
"""
from __future__ import annotations

import json
from dataclasses import replace
from functools import wraps

import pytest

import mindie_sim.api as sdk_api
import mindie_sim.prediction.body_defaults as body_defaults_mod
import mindie_sim.prediction.operator_analysis as op_analysis_mod
import mindie_sim.structure as structure_mod
from _support.builders import dense_case, dense_layer
from _support.serving_cases import case, linear_work, pd_case
from mindie_sim import Case, predict
from mindie_sim.catalog.expert_slots import expert_slot_plan
from mindie_sim.compose import compose_case
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.cohort_reuse import prepare_cohort_graphs, prove_homogeneous_cohort, rebind_replica_graph
from mindie_sim.prediction.cost import CostLedger, CostModel
from mindie_sim.prediction.efficiency import analyze_network_efficiency, capacity_window
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.evaluation import BoundedCache, EvaluationContext, ExecutionBindings, FrozenHardwareView
from mindie_sim.prediction.operators.moe_routing.expert_slot_work import bind_expert_slot_work, physical_slot_work
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.graph_cache import SemanticGraphCache, semantic_graph_key
from mindie_sim.prediction.layer_runs import LayerRunProgram, homogeneous_layer_runs, layer_templates, run_summary
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.physical_key import physical_id, physical_key, physical_signature
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.operators.recurrent.recurrent import serial_context_parallel
from mindie_sim.prediction.results import StepRequest, StructureRequest
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

def _assert_fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected

def _evaluation_case():
    return parse_case(compose_case(
        "qwen3-32b", "a3", tp=2, layers=4, prompt_tokens=16, output_tokens=2, concurrency=1))

@pytest.fixture
def evaluation_calls(monkeypatch):
    counts = dict.fromkeys(("cost", "bindings", "device_dict", "body_defaults"), 0)

    def counted(function, key):
        @wraps(function)
        def wrapped(*args, **kwargs):
            counts[key] += 1
            return function(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(CostModel, "__init__", counted(CostModel.__init__, "cost"))
    create = counted(ExecutionBindings.create.__func__, "bindings")
    monkeypatch.setattr(ExecutionBindings, "create", classmethod(create))
    monkeypatch.setattr(Device, "to_dict", counted(Device.to_dict, "device_dict"))
    effective = counted(body_defaults_mod.effective_body_declaration, "body_defaults")
    for module in (body_defaults_mod, op_analysis_mod):
        monkeypatch.setattr(module, "effective_body_declaration", effective)
    return counts

def test_network_efficiency_uses_one_evaluation_context(evaluation_calls):
    case = _evaluation_case()
    report = analyze_network_efficiency(case, phase="decode", tokens=1, context=16, sequences=1, scope="body")
    assert report["operators"]
    _assert_fields(evaluation_calls, cost=1, bindings=1)
    assert evaluation_calls["device_dict"] <= len(case.machine.dies)

def test_evaluation_local_caches_are_bounded():
    cache = BoundedCache(limit=3)
    for index in range(6):
        cache.put(index, index)
    assert len(cache) == 3
    assert cache.evictions == 3 and cache.get(5) == 5
    assert cache.hits == 1 and cache.get(99) is None and cache.misses == 1

def test_frozen_hardware_counterfactual_changes_signature():
    case = _evaluation_case()
    first = FrozenHardwareView.from_machine(case.machine)
    die, *other_dies = case.machine.dies
    rates = tuple((name, float(value) / 2 if name == "hbm" else value) for name, value in die.rates)
    changed = replace(case.machine, dies=(replace(die, rates=rates), *other_dies))
    assert first.digest() != FrozenHardwareView.from_machine(changed).digest()

def test_isolated_rows_do_not_leak_into_main_basis():
    from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
    case = _evaluation_case()
    context = EvaluationContext.create(case)
    ops = target_graphs(case, phase="decode", tokens=1, context=16, sequences=1, scope="body")[0][:8]
    report = operator_costs(case, tuple(ops), evaluation=context)
    counts = context.cost.summary()["operator_occurrences"]
    assert counts.get("analytic_work", 0) == 0
    assert not any(counts.values()) and report["operator_occurrences"] > 0
    for row in report["operators"]:
        assert row["basis"]["operator_occurrences"].get("analytic_work", 0) >= 1
    assert CostLedger().counts["analytic_work"] == 0

def _composed_case():
    model = {
        "name": "compact-moe", "layers": 2, "hidden_size": 32,
        "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "intermediate_size": 64, "vocab_size": 128,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    }
    hardware = {"sku": "Ascend950DT_9582", "num_devices": 4, "ranks_per_node": 4}
    return compose_case(
        model, hardware, tp=1, dp=4, ep=4, num_devices=4, ranks_per_node=4,
        prompt_tokens=16, output_tokens=2, concurrency=2, prefix_tokens=8,
        o_proj_tp=4, linear_proj_tp=4, lm_head_tp=4, redundant_experts_per_rank=1, recover_ssm=True,
        prefix_cache_mode="shared", prefix_read_mode="auto",
        collective_startup_mode="log2", matrix_compute_policy="auto", homogeneous_graph_reuse="auto",
    )

def test_compose_parse_predict_capacity_agree_without_new_engine():
    case = parse_case(_composed_case())
    parallel = case.runtime.parallel
    assert (parallel.o_proj_tp, parallel.redundant_experts_per_rank) == (4, 1)
    assert case.runtime.recover_ssm is True
    encoded = json.dumps(case.to_dict())
    assert Case.from_mapping(json.loads(encoded)).to_dict() == case.to_dict()
    def lower(plan, model, runtime, machine, **bindings):
        return linear_work(plan, model, runtime, machine)

    result = predict(case, lower=lower)
    assert result["status"] in {"modeled", "infeasible"}
    assert result["execution"]["work"]["flops"] > 0
    memory = Memory(case)
    window = capacity_window(case, phase="decode", context=32, sequences=2, memory=memory)
    assert window["schema"] == "mindie-sim.capacity-window/1"
    assert min(window["stages"][0][name] for name in ("peak_bytes", "supply_bytes")) > 0
    assert window["fits"] in {True, False}

def _replica_op(op_id="layer@r0", flops=100.0, *, deps=()):
    return OpSpec(op_id, "gemm", "decode", 0, 0, ("r0",), (8,), (8,), (8,), (8,),
                  flops=flops, hbm_bytes=flops, deps=deps)

def test_homogeneous_proof_and_asymmetric_cohort():
    case = dense_case(dp=2)
    plan = (_replica_op("a@r0", 100.0), _replica_op("b@r0", 200.0, deps=("a@r0",)))
    proof = prove_homogeneous_cohort(case, {0: plan, 1: plan})
    assert proof.homogeneous is True and proof.template_reused is True
    asymmetric = (_replica_op("a@r0", 100.0), _replica_op("b@r0", 999.0, deps=("a@r0",)))
    failed = prove_homogeneous_cohort(case, {0: plan, 1: asymmetric})
    assert failed.homogeneous is False
    assert "asymmetric_replica_graph" in failed.reasons
    prepared = prepare_cohort_graphs(case, {0: plan, 1: asymmetric}, reuse="auto")
    assert prepared.template_reused is False and prepared.fallback_reason
    with pytest.raises(ModelingError, match="homogeneous_graph_reuse_proof_required:asymmetric_replica_graph"):
        prepare_cohort_graphs(case, {0: plan, 1: asymmetric}, reuse="require")

def test_template_reuse_rebinds_rank_and_dependency_identity():
    case = dense_case(dp=2)
    template = (_replica_op("a@r0"), _replica_op("b@r0", deps=("a@r0",)))
    rebound = rebind_replica_graph(
        template, replica=1, placement={0: 1}, request_binding={"r0": "r1"}, state_bindings={})
    assert [item.op_id for item in rebound] == ["a@r1", "b@r1"]
    assert (rebound[1].deps, rebound[0].requests) == (("a@r1",), ("r1",))
    prepared = prepare_cohort_graphs(case, {0: template, 1: template}, reuse="auto")
    assert prepared.template_reused is True and set(prepared.graphs) == {0, 1}
    assert prepared.graphs[0] == template
    assert prepared.graphs[1][0].op_id == "a@r1"

def test_graph_cache_is_a_bounded_lru_and_skips_oversize_entries():
    cache = SemanticGraphCache(max_bytes=4096, max_entries=2)
    key = lambda name: semantic_graph_key(dense_case(), None, scope=name, selection_signature=())
    expected = ("a" * 100,)
    for value in (expected, ("miss",)):
        assert cache.get_or_build(key("one"), lambda: value) == expected
    assert (cache.hits, cache.misses) == (1, 1)
    cache.get_or_build(key("two"), lambda: ("b" * 100,))
    cache.get_or_build(key("three"), lambda: ("c" * 100,))
    assert cache.entries == 2 and cache.evictions >= 1
    oversized = cache.get_or_build(key("huge"), lambda: ("x" * 10000,))
    assert oversized == ("x" * 10000,)
    assert cache.oversized == 1
    cache.clear()
    assert cache.stats()["entries"] == 0

def test_semantic_key_invalidates_on_frontend_structure():
    case = dense_case()
    base = semantic_graph_key(case, None, scope="body")
    runtime = replace(case.runtime, parallel=replace(case.runtime.parallel, linear_proj_tp=2))
    changed = semantic_graph_key(replace(case, runtime=runtime), None, scope="body")
    assert base != changed
    assert base == semantic_graph_key(case, None, scope="body")

def _physical_graph(model="qwen3-32b", *, tp=1, layers=4, **options):
    case = parse_case(compose_case(model, "a3", tp=tp, layers=layers, prompt_tokens=16,
                                   output_tokens=2, concurrency=1, **options))
    return target_graphs(case, phase="decode", tokens=1, context=16, sequences=1, scope="body")[0]

def test_corresponding_layers_share_one_physical_identity():
    ops = _physical_graph()
    by_recipe = {}
    for op in ops:
        key = op.recipe, op.local_shape, dict(op.metadata).get("structure_block"), op.op_id.rsplit(".", 1)[-1]
        by_recipe.setdefault(key, []).append(op)
    repeated = [group for group in by_recipe.values() if len({op.layer for op in group}) > 1]
    assert repeated, "expected a repeated layer template"
    for group in repeated:
        corresponding = [op for op in group if op.layer > 0]
        assert len(corresponding) > 1
        for identity in (physical_signature, physical_key, physical_id):
            assert len({identity(op) for op in corresponding}) == 1

def test_shape_dtype_bytes_state_and_local_tp_change_identity():
    norm = next(op for op in _physical_graph() if op.family in {"rmsnorm", "rmsnorm_quant", "add_rmsnorm"}
                and len(op.local_shape) >= 2)
    updates = (
        {"local_shape": (int(norm.local_shape[0]) + 1, *norm.local_shape[1:])},
        {"dtype": "fp32"}, {"hbm_bytes": float(norm.hbm_bytes) + 1},
        *({"metadata": (*norm.metadata, (field, value))}
          for field, value in (("state_write_bytes", 4096), ("local_tp_width", 8))),
    )
    for changes in updates:
        assert physical_signature(replace(norm, **changes)) != physical_signature(norm)

def _rotate(value, stage, layer, target_layer):
    if isinstance(value, str):
        return value.replace(f"s{stage}.l{layer}.", f"s{stage}.l{target_layer}.")
    if isinstance(value, dict):
        return {key: item + target_layer - layer
                if key in {"layer", "layer_index"} and type(item) is int
                else _rotate(item, stage, layer, target_layer)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_rotate(item, stage, layer, target_layer) for item in value)
    return value

def test_reference_roles_and_aliasing_survive_normalization():
    key = "structure_subgraph_memberships"
    first = next(op for op in _physical_graph() if op.layer > 0 and dict(op.metadata).get(key))
    metadata = dict(first.metadata)
    assert any(f"s{first.stage}.l{first.layer}." in value for value in metadata[key])
    renamed = {**metadata, key: tuple(value[:-1] + "other_role" for value in metadata[key])}
    rotated = _rotate(metadata, first.stage, first.layer, first.layer + 1)
    before = physical_signature(first)
    assert physical_signature(replace(first, metadata=tuple(renamed.items()))) != before
    rebound = replace(first, op_id="other", layer=first.layer + 1, metadata=tuple(rotated.items()))
    assert physical_signature(rebound) == before

def test_collective_participant_change_breaks_identity():
    ops = _physical_graph(tp=2)
    collective = next(op for op in ops if op.link_bytes > 0)
    metadata = {**dict(collective.metadata), "group_ranks": (0, 1, 2, 3)}
    changed = replace(collective, metadata=tuple(metadata.items()))
    assert physical_signature(changed) != physical_signature(collective)

class _DelayProgram:
    def __init__(self, latency_ms):
        self.latency_ms = float(latency_ms)

    def start(self, engine, name, ready, *, owner=""):
        return engine.submit(name, {"host": self.latency_ms}, after=(ready,), owner=owner)

def test_homogeneous_dense_layers_fold_into_one_run():
    # Use raw declarations so residual-norm fusion does not cut the dense run.
    ops = _physical_graph(layers=8, va_recommended=False)
    runs = homogeneous_layer_runs(ops)
    assert runs
    largest = max(runs, key=lambda run: run.count)
    assert largest.count >= 2
    templates = layer_templates(ops)
    for run in runs:
        assert templates[run.start:run.start + run.count]
    summary = run_summary(runs)
    assert summary["basis"] == "structural_layer_template_reuse_not_duration_multiplication"
    assert summary["folded_layers"] >= 2

def test_hybrid_attention_schedule_cuts_runs():
    ops = _physical_graph("qwen3.6-35b-a3b", layers=8)
    runs = homogeneous_layer_runs(ops)
    # A GDN/KDA layer is never folded with a full-attention layer.
    for run in runs:
        attentions = {
            int(op.layer) for op in ops
            if int(op.layer) in run.layers and op.family in {"attention", "recurrent_scan"}
        }
        assert len(attentions) <= 1

class _SyntheticOp:
    """Minimal physical operator declaration for run identification."""

    def __init__(self, layer):
        self.layer = int(layer)
        self.stage = 0
        self.stream = "main"
        self.dtype = "bf16"
        self.metadata = (("structure_block", "attention"),
                         ("structure_role", "residual"), ("payload_producer", f"s0.l{layer}.residual"))

    def to_dict(self):
        return {"op_id": f"s0.l{self.layer}.residual", "deps": (),
                "stage": 0, "layer": self.layer, "requests": (),
                "stream": "main", "notes": (), "family": "elementwise",
                "recipe": "residual_add", "dtype": self.dtype,
                "local_shape": (1, 16), "metadata": dict(self.metadata)}

def test_run_program_sequences_occurrences_without_multiplying_time():
    from mindie_sim.prediction.runtime.engine import Engine
    from mindie_sim.prediction.runtime.engine_types import Resource

    engine = Engine([Resource("host", 1.0, "ms", "fifo")])
    run = homogeneous_layer_runs([_SyntheticOp(index) for index in range(3)])[0]
    programs = tuple(_DelayProgram(2.0) for _ in run.layers)
    ready = engine.submit("ready", release=0.0)
    completion = LayerRunProgram(run=run, occurrences=programs).start(engine, "run", ready)
    engine.advance(until=(completion,))
    assert completion.time == pytest.approx(2.0 * len(run.layers))
    with pytest.raises(ModelingError, match="layer_run_time_skip_requires_engine_certification"):
        LayerRunProgram(run=run, occurrences=programs, skip_authorized=True)

def test_serial_kda_context_plan_preserves_pcp():
    with pytest.raises(ModelingError, match="pcp_requires_pcp_plan"):
        ParallelSpec(tp=2, pcp=2, cp_plan="none")
    spec = serial_context_parallel(ParallelSpec(tp=2, pcp=2, cp_plan="pcp"))
    assert (spec.cp, spec.dcp, spec.cp_plan, spec.pcp) == (1, 1, "pcp", 2)
    flat = serial_context_parallel(ParallelSpec(tp=2))
    assert (flat.cp, flat.dcp, flat.cp_plan) == (1, 1, "none")

def test_redundant_slots_increase_active_weight_reads_at_saturation():
    def gate_reads(redundant):
        case = parse_case(compose_case(
            model="glm-5.3-flash", hardware="a5", tp=1, dp=8, ep=8,
            o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, layers=4, concurrency=16, recover_ssm=True,
            redundant_experts_per_rank=redundant))
        graphs = target_graphs(case, phase="decode", tokens=1, context=128, sequences=16, scope="body")
        gate = [op for op in graphs[0] if dict(op.metadata).get("expert_slot_weight_kind") == "gate_up"]
        return sum(op.mte2_bytes for op in gate)

    assert gate_reads(1) > gate_reads(0)

def test_slot_bytes_replace_only_the_declared_weight_read():
    layer = dense_layer(0, ffn="moe", num_experts=8, top_k=2,
                        moe_intermediate_size=32, num_shared_experts=0)
    plan = expert_slot_plan(None, layer, ParallelSpec(ep=2))
    work = physical_slot_work(
        slot_plan=plan, logical_inclusions={expert: 2 / 8 for expert in range(8)},
        global_tokens=10 ** 9, packed_weight_bytes=128)
    assert work.slot_weight_bytes
    op = OpSpec("grouped", "grouped_gemm", "decode", 0, 0, (), (1,), (1,),
                (1,), (1,), hbm_bytes=1000.0, mte2_bytes=700.0,
                metadata=(("ep_rank", 0), ("routed_weight_read_bytes", 200.0)))
    bound = bind_expert_slot_work((op,), slot_work=work)[0]
    active = work.weight_read_bytes(0)
    assert bound.mte2_bytes == pytest.approx(700.0 - 200.0 + active)
    assert bound.hbm_bytes == pytest.approx(1000.0 - 200.0 + active)
    assert dict(bound.metadata)["packed_weight_bytes"] == pytest.approx(active)

def test_require_cohort_refuses_asymmetry_and_rebinds_requests():
    case = dense_case(dp=2)
    template = (_replica_op("a@r0", deps=("x@r0",)), _replica_op("b@r0", deps=("a@r0",)))
    replica = tuple(replace(op, op_id=op.op_id.replace("@r0", "@r1"), requests=("r1",),
                            deps=tuple(dep.replace("@r0", "@r1") for dep in op.deps))
                    for op in template)
    prepared = prepare_cohort_graphs(case, {0: template, 1: replica}, reuse="auto")
    assert prepared.graphs[1][0].requests == ("r1",)
    assert prepared.graphs[1][1].deps == ("a@r1",)
    asymmetric = (replica[0], replace(replica[1], flops=999))
    with pytest.raises(ModelingError, match="homogeneous_graph_reuse_proof_required:asymmetric_replica_graph"):
        prepare_cohort_graphs(case, {0: template, 1: asymmetric}, reuse="require")

@pytest.mark.parametrize("dp", (2, 4))
def test_local_tp_groups_span_dp_replicas_without_participant_mismatch(dp):
    width = 2
    case = parse_case(compose_case(
        model="longcat-flash-lite", hardware="a5", tp=1, dp=dp, ep=dp,
        o_proj_tp=width, linear_proj_tp=width, lm_head_tp=width, layers=1,
        concurrency=1, recover_ssm=True, redundant_experts_per_rank=0, homogeneous_graph_reuse="require"))
    graphs = target_graphs(case, phase="decode", tokens=1, context=128, sequences=1, scope="body")
    for replica, graph in graphs.items():
        base = replica // width * width
        expected = tuple(range(base, base + width))
        groups = {dict(op.metadata)["group_ranks"] for op in graph if "group_ranks" in dict(op.metadata)}
        assert groups == {expected}
    report = graph_costs(
        case, graphs, external_inputs={"analysis_input": 0},
        include_submission=False, isolated=False, compact=True)
    assert report["duration_ms"] > 0

def test_metrics_follow_commits_not_step_count_or_padding():
    result = predict(case(), lower=linear_work)
    row = result["requests"][0]
    _assert_fields(row, ttft_ms=4, tpot_ms=1, finish_ms=6)
    assert result["metrics"]["output_tps"] == 500
    assert row["itl"] == {"count": 2, "mean_ms": 1, "max_ms": 1}

def test_arrival_during_work_waits_for_the_next_batch_and_keeps_queue_time():
    value = case(requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 2, "arrival_ms": 0},
        {"id": "b", "prompt_tokens": 2, "output_tokens": 1, "arrival_ms": 2},
    ])
    value["runtime"]["max_num_batched_tokens"] = 4
    rows = predict(value, lower=linear_work)["requests"]
    assert [(r["first_token_abs_ms"], r["finish_ms"]) for r in rows] == [(4, 7), (7, 7)]
    assert [r["ttft_ms"] for r in rows] == [4, 5]

def test_independent_data_replicas_overlap_without_double_counting_elapsed_time():
    value = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 4, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 4, "output_tokens": 1},
    ])
    result = predict(value, lower=linear_work)
    assert result["metrics"]["makespan_ms"] == 4
    assert result["metrics"]["output_tps"] == 500
    assert result["execution"]["work"]["flops"] == 8e6

def test_closed_loop_arrivals_are_released_by_predicted_completion():
    value = case(dp=2, requests=[
        {"id": "a", "prompt_tokens": 2, "output_tokens": 1, "chain_id": 1},
        {"id": "b", "prompt_tokens": 3, "output_tokens": 1, "chain_id": 1, "gap_ms": 5},
    ])
    rows = predict(value, lower=linear_work)["requests"]
    assert rows[0]["finish_ms"] == 2
    _assert_fields(rows[1], arrival_ms=7, first_token_abs_ms=10, ttft_ms=3)

def test_capacity_failure_stays_in_slo_denominator():
    value = case()
    value["slo"] = {"ttft_ms": 100}
    result = predict(value, memory_policy=MemoryPolicy(workspace_bytes=2 * 1024 ** 3))
    assert result["status"] == "infeasible"
    assert result["requests"][0]["generated_tokens"] == 0
    assert result["metrics"]["slo"]["passing_fraction"] == 0
    assert result["memory"]["status"] == "oom"

def test_shared_interconnect_charges_every_sender_on_the_same_wire():
    parsed = parse_case(case(tp=2))
    topology = Topology(parsed.machine, parsed.runtime.parallel)
    shape = (1,)
    op = OpSpec("reduce", "allreduce", "decode", 0, 0, (), shape, shape, shape, shape, link_bytes=100)
    demand, latency = topology.demand(op, 0)
    engine = Engine(topology.resources())
    event = engine.submit("reduce", demand, latency=latency)
    engine.advance(until=(event,))
    assert demand == {"link:fabric": 200}
    assert event.time == .2

def test_model_label_and_measurement_availability_do_not_change_the_program():
    parsed = parse_case(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    lower = Frontend()
    first = lower(plan, parsed.model, parsed.runtime, parsed.machine)
    renamed = replace(parsed.model, name="unseen-architecture-label")
    second = lower(plan, renamed, parsed.runtime, parsed.machine)
    assert first.ops == second.ops
    without = predict(parsed)
    with PrimitiveStore(":memory:") as parameters:
        with_store = predict(parsed, primitives=parameters)
    assert without["metrics"] == with_store["metrics"]
    assert without["execution"]["serviced"] == with_store["execution"]["serviced"]

def test_ep_cohort_includes_idle_replicas_and_finishes_all_collectives():
    value = case(dp=2, ep=2, output=2)
    value["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    result = predict(value, event_limit=500)
    assert result["requests"][0]["status"] == "served"
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    collectives = [event for event in result["events"] if event["owner"] == "expert_group"]
    assert collectives
    assert all(event["end_ms"] >= event["start_ms"] for event in collectives)

def test_generic_gpu_and_split_copy_engine_use_the_same_serving_program():
    gpu = case(output=2)
    accelerator = case(output=2)
    accelerator["machine"]["architecture"] = "new-accelerator"
    # Even a one-of-four tail must supply the full shared memory rate.
    accelerator["machine"]["device"]["rates"].update(read=4e9, write=4e9)
    a, b = predict(gpu), predict(accelerator)
    assert a["metrics"]["output_tps"] == pytest.approx(b["metrics"]["output_tps"])
    assert a["metrics"]["ttft_ms"]["mean"] == pytest.approx(b["metrics"]["ttft_ms"]["mean"])
    assert a["execution"]["work"] == b["execution"]["work"]
    assert not any(key.startswith("mte") for key in a["execution"]["serviced"])
    assert any(key.startswith("mte") for key in b["execution"]["serviced"])
    accelerator["machine"]["device"]["rates"].update(read=2e9, write=2e9)
    limited = predict(accelerator)
    assert limited["metrics"]["output_tps"] < b["metrics"]["output_tps"]
    assert limited["execution"]["work"] == b["execution"]["work"]

PD_ROW_BYTES = 2 * 2 * 2 * 8 * 2  # layers, K/V planes, heads, head dimension, BF16 bytes

def _pd_pair():
    return pd_case([{"id": key, "prompt_tokens": 3, "output_tokens": 3} for key in ("a", "b")])

def test_pd_handoff_delays_decode_and_conserves_bytes():
    result = predict(pd_case(), lower=linear_work)
    row = result["requests"][0]
    assert row["ttft_ms"] == 3
    assert result["execution"]["handoff_bytes"] == 3 * PD_ROW_BYTES
    assert row["finish_ms"] == 8
    assert row["tpot_ms"] == 2.5
    assert all(v == 0 for pool in result["memory"].values()
               for stage in pool["live_cache_bytes_per_replica_stage"] for v in stage)

def test_pd_pools_overlap_and_destination_admission_applies_backpressure():
    value = _pd_pair()
    value["pools"]["prefill"]["runtime"]["max_num_batched_tokens"] = 3
    value["pools"]["decode"]["runtime"]["max_num_seqs"] = 1
    result = predict(value, lower=linear_work)
    assert [(r["first_token_abs_ms"], r["finish_ms"]) for r in result["requests"]] == [(3, 8), (6, 13)]
    assert result["execution"]["serviced"]["pd_link"] == 2 * 3 * PD_ROW_BYTES
    assert result["metrics"]["output_tps"] == pytest.approx(6000 / 13)

def test_async_scheduler_preparation_overlaps_actual_device_work():
    raw = case(prompt=4, output=3)
    raw["runtime"]["submission"] = {"scheduler_ms": .5}
    synchronous = predict(raw, lower=linear_work)
    raw["runtime"]["async_scheduling"] = True
    asynchronous = predict(raw, lower=linear_work)
    assert synchronous["metrics"]["makespan_ms"] == 7.5
    assert asynchronous["metrics"]["makespan_ms"] == 6.5
    assert asynchronous["execution"]["work"] == synchronous["execution"]["work"]
    assert asynchronous["execution"]["live_tasks"] == asynchronous["execution"]["live_programs"] == 0

def test_source_cache_pressure_waits_for_inflight_handoff_before_failing_admission():
    value = _pd_pair()
    pool = value["pools"]["prefill"]
    pool["runtime"].update(kv_block_size=1, max_num_batched_tokens=3,
                           gpu_memory_utilization=1, memory={"derive_workspace": False})
    parsed = Case.from_mapping({**pool, "model": value["model"], "workload": value["workload"]})
    pool["machine"]["device"]["memory_bytes"] = int(Memory(parsed).weights[0] + 4 * PD_ROW_BYTES)
    result = predict(value, lower=linear_work)
    assert result["status"] == "modeled"
    assert [(r["first_token_abs_ms"], r["finish_ms"]) for r in result["requests"]] == [(3, 8), (9, 14)]

def test_pd_handoff_services_both_memories_even_on_an_uncontended_fast_link():
    value = pd_case()
    value["connector"]["bytes_per_s"] = 1e12
    for pool in value["pools"].values():
        pool["machine"]["device"]["rates"]["memory"] = 128000
    result = predict(value, lower=linear_work)
    assert result["requests"][0]["finish_ms"] == 8
    for pool in ("prefill", "decode"):
        assert result["execution"]["serviced"][f"{pool}:memory@r0:s0"] == 3 * PD_ROW_BYTES

def _disable_homogeneous_lowering(monkeypatch):
    """Force the direct body path so the homogeneous fast path can be compared."""
    monkeypatch.setattr(Frontend, "_try_lower_homogeneous", lambda self, **kwargs: None)

def test_dbo_partitioning_is_never_bypassed_by_homogeneous_lowering(monkeypatch):
    from mindie_sim.public.errors import UnmodeledCapabilityError

    case = compose_case(
        "qwen3-32b", hardware="a5", tp=1, ep=1, num_devices=1, layers=4,
        enable_dbo=True, prompt_tokens=64, output_tokens=2, max_num_seqs=2)
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        analyze_network_efficiency(case, phase="decode", tokens=1, context=64, sequences=2)
    _disable_homogeneous_lowering(monkeypatch)
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        analyze_network_efficiency(case, phase="decode", tokens=1, context=64, sequences=2)

def test_homogeneous_attention_uses_selected_machine_capacity(monkeypatch):
    case = compose_case(
        "qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=1, layers=4,
        prompt_tokens=512, output_tokens=2)
    analysis = {"phase": "prefill", "tokens": 512, "context": 30000, "sequences": 1}
    optimized = analyze_network_efficiency(case, **analysis)
    _disable_homogeneous_lowering(monkeypatch)
    direct = analyze_network_efficiency(case, **analysis)
    for field in ("total_hbm_bytes", "graph_duration_ms"):
        assert optimized[field] == pytest.approx(direct[field])

@pytest.mark.parametrize("isolated", [False, True])
def test_spine_constructs_once_and_agrees_with_structure_graph_identity(evaluation_calls, isolated):
    case = sdk_api.compose_case("qwen3-32b", hardware="Ascend910B2",
                                 options={"dp": 1, "tp": 1, "pp": 1, "layers": 2})
    parameters = {"phase": "decode", "tokens": 1, "context": 128, "sequences": 1, "scope": "body"}
    request = StepRequest(**parameters)
    options = {"replay": sdk_api.ReplayOptions(isolated=True)} if isolated else {}
    result = sdk_api.evaluate_step(case, request, **options)
    _assert_fields(evaluation_calls, cost=1, bindings=1, body_defaults=1)
    if isolated:
        assert any(row.isolated_ms is not None and row.isolated_ms > 0 for row in result.operators)
    else:
        assert result.span_ms > 0 and result.operators and result.wave_breakdown is None
        public = sdk_api.export_structure(case, StructureRequest(step=request))
        internal = structure_mod.export_structure(case, **parameters)
        assert result.graph_id == public.graph_id == internal["graph_id"]

def test_prediction_body_uses_one_evaluation_context(evaluation_calls):
    case = sdk_api.compose_case("qwen3-32b", hardware="Ascend910_9362",
                                 options={"dp": 1, "tp": 2, "pp": 1, "layers": 1})
    report = predict(case, scope="body", phase="decode", tokens=1, context=16)
    duration = report.get("duration_ms") or report.get("body_ms") or report.get("step_ms")
    assert duration is not None and float(duration) > 0
    _assert_fields(evaluation_calls, cost=1, bindings=1)
