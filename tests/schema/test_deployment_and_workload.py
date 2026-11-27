"""Deployment declarations, projection placement, and workload execution.

[Role] Verify that input policies bind the declared resource, projection,
  stack, and workload behavior from parsing through execution.
[Invariants] Resource controls preserve work, tensor placement preserves
  resident bytes, and speculative plans preserve valid-token ledgers.
[Timeouts & Locks] Deterministic local studies and bounded layer graphs.
[Side-Effect Policy] Read retained JSON and write offline manifests in tmp_path.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from _support.builders import dense_layer, dense_model, hardware, linear_model, quant, view
from _support.hardware import hardware_case
from _support.kernel_invocation import op
from _support.memory_cases import SPARSE_OFFLOAD_SKU, case as memory_case
from mindie_sim import compose_case, load_runtime_stack_manifest, operator_costs
from mindie_sim import parse_case, predict, study, target_graphs
from mindie_sim.catalog.expert_slots import expert_slot_plan, routed_expert_weight_terms
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.projection_placement import projection_placement, projection_slices
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes
from mindie_sim.catalog.weights import body_head_terms, layer_weight_terms
from mindie_sim.cli.options import add_runtime_arguments, case_overrides
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.execution.dispatch import cohorts
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.prediction.operators.moe_routing.expert_slot_work import bind_expert_slot_work, physical_slot_work
from mindie_sim.prediction.sharding.local_tp import lower_local_projection_tp
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.projections import kda_projections
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import RequestSpec, WorkloadSpec
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.feature_mechanisms import RUNTIME_STACKS
from mindie_sim.schema.runtime_features import normalize_case_runtime, normalize_runtime_features, split_routing_features
from scripts.audit_stack_prs import main as audit_stack_main
from tools.deployment_generalization import compare_sweep, plan_scheme, scaled_case

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
from _support.axes import DUAL_DIE_SKU
DEPLOYMENT_SKUS = (DUAL_DIE_SKU, SPARSE_OFFLOAD_SKU)
DRAM_SERVICE_SCALE = 1 / .8316
BF16 = {"weight_format": "bf16", "activation_format": "bf16"}
INT8 = {"weight_format": "int8", "activation_format": "int8"}
FP32 = {"weight_format": "fp32", "activation_format": "fp32"}
RESOURCE_CASES = [(axis, pd) for axis in ("aic", "aiv", "hbm", "network") for pd in (False, True)] + [("collective", False)]

@pytest.mark.parametrize("axis,pd", RESOURCE_CASES)
def test_resource_scaling_changes_bound_service_with_fixed_work(axis, pd):
    raw = memory_case() | {"machine": dict(backend="ascend", sku=SPARSE_OFFLOAD_SKU, num_devices=1)}
    if pd:
        raw["pools"] = {side: deepcopy({key: raw[key] for key in ("machine", "runtime")}) for side in ("prefill", "decode")}
        raw["connector"] = dict(bytes_per_s=128000, latency_us=0)
    if axis == "collective":
        raw["machine"].update(num_devices=4, ranks_per_node=2)
        raw["runtime"]["tp"] = 4
    family, work = {
        "aic": ("gemm", {"flops": 1e12}), "aiv": ("vector", {"vector_ops": 1e12}),
        "hbm": ("vector", {"hbm_bytes": 1e10}), "network": ("vector", {"flops": 1}),
        "collective": ("allreduce", {"link_bytes": 25e6, "metadata": (("group_axis", "tp"),)}),
    }[axis]
    def lower(plan, model, runtime, machine):
        tokens = plan.valid_tokens
        shape = (1, 1, tokens) if axis != "collective" else (tokens,)
        node = OpSpec("control", family, "body", 0, 0, tuple(v.request_id for v in plan.requests),
                      shape, shape, shape, shape, dtype="bf16", **work)
        return IterationWork((node,), plan, 1, tokens, tokens, None, "body", "resource_control")
    rate = "network" if axis == "collective" else axis
    reports = [predict(scaled_case(raw, rate, scale), lower=lower) for scale in (.5, 1, 2)]
    idle_network = axis == "network" and not pd
    plateau = axis in {"hbm", "collective"} or idle_network
    assert compare_sweep(reports)["status"] == ("plateau" if plateau else "strict_response")
    for report in reports:
        ledger = report["execution"]
        assert report["requests"][0]["status"] == "served"
        assert ledger["live_tasks"] == ledger["live_programs"] == 0
        if pd:
            assert ledger["handoff_bytes"] == 512
        else:
            assert ledger["work"] == reports[1]["execution"]["work"]
        if axis == "collective":
            assert any(v > 0 for k, v in ledger["serviced"].items() if "d2d_inter_node" in k)

@pytest.mark.parametrize("sku", DEPLOYMENT_SKUS)
@pytest.mark.parametrize("axis", ("network", "hbm", "aic", "aiv"))
def test_resource_axes_scale_independent_supply_and_service_modes(sku, axis):
    raw = memory_case() | {"machine": dict(backend="ascend", sku=sku, num_devices=16, ranks_per_node=8)}
    before, after = [Hardware.from_mapping(case["machine"]) for case in (raw, scaled_case(raw, axis, .5))]
    a, b = before.dies[0], after.dies[0]
    assert (a.memory_bytes, a.resource_units) == (b.memory_bytes, b.resource_units)
    base, scaled = dict(a.rates), dict(b.rates)
    assert dict(b.service_modes) == {
        mode: (resource, rate * (scaled[resource] / base[resource]))
        for mode, (resource, rate) in a.service_modes}
    assert scaled["hbm"] == base["hbm"] * (.5 if axis == "hbm" else 1)
    if axis == "network":
        assert a == b
        for first, last in zip(before.links, after.links):
            factor = .5 if first.channel == "d2d_inter_node" else 1
            assert last.capacity_bytes_per_ms == first.capacity_bytes_per_ms * factor

def test_dense_capacity_plan_keeps_expert_degree_and_cross_node_pipeline():
    case = parse_case(plan_scheme("qwen38", "a3", "multi", "colocated")["case"])
    assert case.model.layer_plans == load_model("qwen3.8-27b").layer_plans
    assert case.runtime.parallel.ep == 1 < case.runtime.parallel.pp
    assert Memory(case).resize(0, "r0", 130)

def ascend_case(device):
    return hardware_case() | {"machine": dict(backend="ascend", sku=SPARSE_OFFLOAD_SKU, device=device)}

def test_study_canonicalizes_hardware_overrides_before_merging():
    raw = ascend_case({"hbm_bw_bytes_per_s": 4e12})
    raw["model"].update(hidden_size=2048, num_heads=16, num_kv_heads=4, head_dim=128, intermediate_size=4096)
    variants = ({}, {"rates": {"hbm": 2e12}}, {"hbm_bw_bytes_per_s": 2e12}, {"rates": {"memory": 2e12}})
    result = study([{"id": str(i), "case": raw, "overrides": {"machine": {"device": device}}}
                    for i, device in enumerate(variants)],
                   [{"id": "decode", "phase": "decode", "tokens": 1, "context": 16}])
    base, *overrides = result["evaluations"]
    for row in overrides:
        assert row["operators"]["work"] == base["operators"]["work"]
        assert row["duration_ms"] == overrides[0]["duration_ms"] > base["duration_ms"]
    from mindie_sim.prediction.evaluation import EvaluationContext
    for declaration, duration in zip(variants[:2], (DRAM_SERVICE_SCALE, 2)):
        case_case = parse_case(ascend_case({"hbm_bw_bytes_per_s": 4e12} | declaration))
        row, = operator_costs(
            case_case, (op("read", memory=4e9),),
            evaluation=EvaluationContext.create(case_case, primitive_table=False),
        )["operators"]
        assert row["serviced"]["memory@r0:s0"] >= 4e9 and row["duration_ms"] == pytest.approx(duration)

def test_device_overrides_drive_native_compute_transfer_and_memory_service():
    from mindie_sim.prediction.evaluation import EvaluationContext
    raw = ascend_case({"cube_flops_per_s": 1e12, "mte1_bw_bytes_per_s": 2e12, "aic_count": 8, "l1_bytes": 65536})
    nominal = EvaluationContext.create(parse_case(raw), primitive_table=False)
    transfer = replace(op("inner"), family="transfer", metadata=(("resource_work", {"mte1": 4e9}),))
    for node, duration in ((op("compute", flops=1e9), 1), (transfer, 2)):
        assert operator_costs(raw, (node,), evaluation=nominal)["operator_sum_ms"] == duration
    raw["machine"].update(hbm_gib=7, num_devices=2, devices=[
        {"hbm_bw_bytes_per_s": 1e12, "cube_flops_per_s": 1e12, "hbm_gib": 4},
        {"rates": {"memory": 2e12, "hbm": 3e12}, "compute_peaks": {"bf16": 2e12}},
    ])
    raw["runtime"] = {"dp": 2}
    machine = Hardware.from_mapping(raw["machine"])
    assert [die.memory_bytes for die in machine.dies] == [4 * 1024**3, 7 * 1024**3]
    assert [die.peak("bf16") for die in machine.dies] == [1e12, 2e12]
    assert Hardware.from_mapping(machine.to_dict()) == machine
    for replica, duration in enumerate((3, 1)):
        assert operator_costs(raw, (op("read", memory=3e9),), replica=replica)["operator_sum_ms"] == pytest.approx(duration)

@pytest.mark.parametrize("role,category,shard,degree", [
    ("attention_output", "attn.o", "in_tp", 4), ("linear_input", "kda.q", "out_tp", 2), ("logits", "lm_head", "out_tp", 2),
])
def test_local_projection_residency_scales_by_its_declared_width(role, category, shard, degree):
    parallel = ParallelSpec(o_proj_tp=4, linear_proj_tp=2, lm_head_tp=2)
    term = WeightTerm("projection", 128, 256, category, shard, placement_role=role, quant=quant())
    assert term_bytes(term, quant(), parallel) == 128 * 256 * 2 // degree

@pytest.mark.parametrize("category,first,last,shape,payloads", [
    ("kda.q", "allgather", "alltoall", (32, 16, 128), (128, 16)),
    ("mla.o", "alltoall", "reduce_scatter", (32, 64, 32), (32, 64)),
])
def test_local_projection_graph_preserves_work_and_exchange_bytes(category, first, last, shape, payloads):
    node = gemm_op(op_id="matrix", deps=("x",), stage=0, layer=0, requests=("r",),
                   m=8, n=64, k=128, quant=quant(), category=category, phase="decode")

    def lower(ops, width=4, views=None):
        runtime = Software(parallel=ParallelSpec(dp=4, ep=4, o_proj_tp=width, linear_proj_tp=width))
        return lower_local_projection_tp(ops, model=None, runtime=runtime, machine=hardware(4),
                                         replica=0, cohort_views=views or {}, scope="target")

    gather, matrix, restore = steps = lower((node,))
    assert (gather.family, restore.family, matrix.local_shape) == (first, last, shape)
    assert (gather.link_bytes, restore.link_bytes) == tuple(3 * 8 * width * 2 for width in payloads)
    assert matrix.flops == 2 * 8 * 64 * 128
    assert (matrix.deps, restore.deps) == ((gather.op_id,), (matrix.op_id,))
    assert restore.op_id == node.op_id and lower(steps) == steps and lower((node,), 1) == (node,)
    if category == "kda.q":
        views = {i: (view(str(i), rows),) if rows else () for i, rows in enumerate((8, 4, 0, 2))}
        uneven = lower((node,), views=views)
        assert uneven[1].local_shape == (14, 16, 128)
        assert (uneven[0].link_bytes, uneven[2].link_bytes) == (gather.link_bytes, restore.link_bytes)
    with pytest.raises(ModelingError, match="local_tp_source_placement_unsupported"):
        lower((replace(node, metadata=(*node.metadata, ("implementation", "unknown_source"))),))

def test_local_tp_preserves_groups_scale_tails_and_tied_weight_ownership():
    p = ParallelSpec(o_proj_tp=4, linear_proj_tp=2, lm_head_tp=2)
    group = WeightTerm("wo_a", 128, 32, "mla.o_a", "group_out", groups=8,
                       quant=quant(), placement_role="attention_output")
    partner = replace(group, name="wo_b", shard="group_in", rows=32, cols=128)
    assert term_bytes(group, quant(), p) == term_bytes(partner, quant(), p) == 32 * 32 * 2
    with pytest.raises(ModelingError, match="local_tp_group_geometry_indivisible"):
        term_bytes(replace(group, rows=96, groups=6), quant(), p)
    kda = linear_model("kda", layers=1)
    packed = quant(weight_format="mxfp4", activation_format="fp8_e4m3", weight_block=(32, 32))
    place = projection_placement(kda, kda.layer_plans[0], p, role="linear_input", node_ranks=4)
    terms = [term for term in place.weight_terms if term.name in {"kda_q", "kda_k"}]
    first, second = projection_slices(terms, place, packed)
    assert (first.byte_offset, second.physical_extent) == (0, 32)
    term = replace(terms[0], quant=packed)
    assert term_bytes(term, packed, p) * 2 == term_bytes(term, packed, ParallelSpec())
    tied = dense_model(layers=1, tie_word_embeddings=True)
    row = projection_placement(tied, tied.layer_plans[0], p, role="attention_output", node_ranks=4)
    assert (row.physical_rows, row.physical_cols) == (32, 16)
    assert {term.name for term in body_head_terms(tied)}.isdisjoint({"lm_head"})

EXPERT_LAYER = dense_layer(0, ffn="moe", num_experts=8, top_k=2, moe_intermediate_size=32)
@pytest.mark.parametrize("ep,etp,extra", [(4, 1, 1), (2, 4, 1)])
def test_redundant_slots_conserve_logical_parameters_and_placed_bytes(ep, etp, extra):
    parallel = ParallelSpec(tp=ep * etp, ep=ep, expert_tp=etp, redundant_experts_per_rank=extra)
    plan = expert_slot_plan(None, EXPERT_LAYER, parallel)
    per_rank = 8 // ep
    assert plan.base_slots_by_rank == (per_rank,) * ep
    native = WeightTerm("expert_gate", 64, 32, "routed_expert", "expert_gate_up", count=8, quant=quant())
    base, copied = routed_expert_weight_terms((native,), plan, ep_rank=ep - 1)
    slot_bytes = (64 // etp) * 32 * 2
    assert base.logical_parameters == 8 * 64 * 32
    assert (term_bytes(base, quant(), parallel), term_bytes(copied, quant(), parallel)) == (
        per_rank * slot_bytes, extra * slot_bytes)

@pytest.mark.parametrize("ep,extra,routing,reason", [
    (4, 7, None, "exceed_distinct_local_experts"), (1, 1, None, "exceed_distinct_local_experts"),
    (4, 1, {"expert_slot_policy": "explicit", "expert_slot_map": {0: [[0, 1], [2, 3], [4, 5], [6, 7]]}},
     "redundant_count_mismatch"),
])
def test_expert_slot_declarations_require_distinct_and_complete_owners(ep, extra, routing, reason):
    with pytest.raises(ModelingError, match=reason):
        expert_slot_plan(None, EXPERT_LAYER, ParallelSpec(ep=ep, redundant_experts_per_rank=extra),
                         routing=routing)

@pytest.mark.parametrize("realization", (None, {0: 5, 1: 3, 2: 7, 3: 1}))
def test_active_expert_slots_conserve_routed_rows_and_charge_resident_reads(realization):
    plan = expert_slot_plan(None, EXPERT_LAYER, ParallelSpec(ep=4, redundant_experts_per_rank=1))
    tokens = 16 if realization else 1000
    work = physical_slot_work(slot_plan=plan, global_tokens=tokens, routing_realization=realization,
                              logical_inclusions={} if realization else dict.fromkeys(range(8), .25),
                              packed_weight_bytes=128)
    assert sum(work.expected_rows_by_rank) == pytest.approx(tokens if realization else 2 * tokens)
    node = replace(op("grouped", memory=1000), family="grouped_gemm",
                   metadata=(("ep_rank", 0), ("routed_weight_read_bytes", 250.0)))
    bound, = bind_expert_slot_work((node,), slot_work=work)
    assert bound.hbm_bytes == pytest.approx(750 + work.weight_read_bytes(0))

def stack_case(model, **overrides):
    return compose_case(model, **(dict(hardware=SPARSE_OFFLOAD_SKU, tp=8, ep=8, layers=4) | overrides))

@pytest.mark.parametrize("width,block", [(1, None), (8, None), (16, None), (8, 64), (8, 1536)])
def test_paired_allocator_resolves_defaults_and_honors_explicit_page_size(width, block):
    case = parse_case(stack_case("kimi-k3.1", hardware="a3", tp=width, ep=width, layers=8, block_size=block))
    assert case.runtime.kv_block_size == block if block else case.runtime.kv_block_size >= 544
    if block == 64:
        with pytest.raises(ModelingError, match="paired_sfa_block_too_small_for_recurrent_page"):
            Memory(case)
    else:
        memory = Memory(case)
        assert width == 1 or memory.capacity[0][0] > 0

@pytest.mark.parametrize("model,phase,level,ep,path,expected", [
    ("qwen3.5-397b-a17b", "decode", "shipping", 8, ("metadata", "additional_config", "enable_fused_mc2"), 2),
    ("qwen3.8-flash-next", "prefill", "proposed", 8, ("metadata", "text_gated_residual_min_rows"), 513),
    ("qwen3.8-flash-next", "decode", "proposed", 8, ("metadata", "text_gated_residual"), None),
    ("qwen3.8-27b", "prefill", "shipping", 1, ("attention_policy", "recurrent_chunk"), 64),
    ("qwen3.8-27b", "prefill", "proposed", 1, ("attention_policy", "recurrent_chunk"), 32),
    ("qwen3.8-2.4t-a95b", "prefill", "proposed", 8, ("kernels", "lowerings", "gdn_prefill"), None),
])
def test_stack_bindings_respect_phase_and_operand_geometry(model, phase, level, ep, path, expected):
    value = stack_case(model, ep=ep, layers=2, feature_track=level, _stack_phase=phase)["runtime"]
    for key in path:
        value = (value or {}).get(key)
    assert value == expected

def test_stack_manifest_support_roundtrip_and_resolution(tmp_path):
    repo = "jd-opensource/xllm"
    models = ("deepseek-v3.2", "qwen3.8-flash-next", "deepseek-v4.1-flash")
    prov = {"repository": repo, "revision": "0" * 40, "relative_path": "xllm/pass.py", "source_sha256": "a" * 64}
    path = tmp_path / "stack.json"
    path.write_text(json.dumps({
        "schema": "mindie-sim.stack-manifest/1", "stack_id": "custom-stack", "name": "custom-stack",
        "manifest_revision": "main", "repository": repo,
        "supported_domains": ["llm", "vlm"],
        "supported_tracks": ["eager", "shipping", "proposed", "target"],
        "supported_models": [models[0]], "unmerged_pr_models": [models[1]],
        "bindings": [
            {"mechanism_id": "aclgraph_decode_capture", "state": "merged", "ref": repo + "@main", "lowering_pass_id": "aclgraph_decode_capture", "provenance": prov},
            {"mechanism_id": "gmm_swiglu_quant_fusion", "state": "unmerged_pr", "ref": repo + "#888", "lowering_pass_id": "gmm_swiglu_quant_fusion", "provenance": prov},
        ],
    }), encoding="utf-8")
    profile = load_runtime_stack_manifest(path)
    assert profile == load_runtime_stack_manifest(profile.to_dict())
    for minimum, model in enumerate(models):
        for index, level in enumerate(("shipping", "proposed", "target")):
            options = dict(runtime_stack=path, feature_track=level)
            if index < minimum:
                with pytest.raises(ModelingError, match=f"model_not_supported_by_runtime_stack:custom-stack:{model}:{level}"):
                    stack_case(model, **options)
            else:
                report = stack_case(model, **options)["runtime"]["metadata"]["feature_track_report"]
                assert (report["stack"], report["feature_track"]) == (profile.name, level)
    for level, expected in (("shipping", False), ("proposed", True)):
        runtime = stack_case(models[0], runtime_stack=path, feature_track=level)["runtime"]
        active = runtime["metadata"]["feature_track_report"]["active_mechanisms"]
        assert any(binding["id"] == "gmm_swiglu_quant_fusion" and binding["ref"] == repo + "#888"
                   for binding in active) is expected

def test_builtin_stack_inventory_and_offline_binding_status(tmp_path, capsys):
    assert set(RUNTIME_STACKS) == {"vllm-ascend", "raw_eager"}
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({
        "open_prs": [{"number": 99999, "title": "[Performance] Fuse speculative router", "state": "OPEN"}],
        "tracked_prs": {"21999": {"number": 21999, "state": "MERGED"}},
    }))
    assert audit_stack_main(["--offline-fixture", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert {item["pr_number"] for item in report["ready_to_promote_to_merged"]} == {21999}
    assert {item["pr_number"] for item in report["uncataloged_performance_candidates"]} == {99999}

@pytest.mark.parametrize("recommended", (True, False))
def test_stack_exclusions_remove_bindings_before_graph_lowering(recommended):
    model, exclusion = ("kimi-k3.1", "attn_res_prefix_fusion") if recommended else ("deepseek-v3", "mla_prolog_fusion")
    case = parse_case(stack_case(model, va_recommended=recommended, graph_capture_mode="none",
                                 exclude_optimizations=(exclusion,)))
    assert "attention_residual" not in case.runtime.kernels.get("lowerings", {})
    if recommended:
        ops = target_graphs(case, phase="decode", tokens=1, context=8, scope="model")[0]
        assert not any("attn_res" in op.op_id for op in ops)
    else:
        assert not case.runtime.kernels and case.runtime.graph_capture_mode == "none"

@pytest.mark.parametrize("proposals,acceptance,quota,requests,expected,contexts", [
    (5, 6, 6, ("r",), [6], [4]), (5, 3.5, 8, ("r",), [6, 5, 2], [4, 7, 10]),
    (2, 1.5, 3, ("r0", "r1"), [2, 2, 2, 2, 1], None),
])
def test_speculative_model_and_service_share_the_valid_token_ledger(
        proposals, acceptance, quota, requests, expected, contexts):
    output = 3 if len(requests) > 1 else 8
    raw = stack_case("glm-5.3-flash", tp=1, dp=1, ep=1, raw_eager=True, prompt_tokens=4,
                     graph_capture_mode="none", output_tokens=output,
                     mtp_proposal_tokens=proposals, mtp_acceptance_length=acceptance,
                     **({"max_num_batched_tokens": 2} if len(requests) > 1 else {}))
    case = parse_case(raw)
    case = replace(case, workload=replace(case.workload, requests=tuple(
        RequestSpec(key, 0, 4, output) for key in requests)))
    plans = []

    class RecordingScheduler(Scheduler):
        def next_runtime(self, plan):
            runtime = super().next_runtime(plan)
            plans.append((plan, runtime))
            return runtime

    service = predict(case, window={key: {"computed": 4} for key in requests},
                      token_quotas=dict.fromkeys(requests, quota), scheduler_factory=RecordingScheduler,
                      trace_limit=len(expected))
    frontend = Frontend()
    model_tokens = [frontend(plan, case.model, runtime, case.machine, assembly="model").valid_tokens
                    for plan, runtime in plans]
    assert model_tokens == [step["valid_tokens"] for step in service["traces"]] == expected
    if contexts is not None:
        assert [plan.requests[0].context for plan, _ in plans] == contexts

@pytest.mark.parametrize("staged", (False, True))
def test_expert_rank_table_groups_follow_declared_pipeline_and_context_membership(staged):
    if staged:
        raw = stack_case("deepseek-v4-pro", hardware="a3", tp=4, dp=2, pp=2, pcp=2, ep=16,
                         prompt_tokens=128, output_tokens=8, sp=True, va_recommended=True, quant="w4a8")
        table = (tuple(range(8)) + tuple(range(16, 24)), tuple(range(8, 16)) + tuple(range(24, 32)))
    else:
        raw = stack_case("qwen3-30b-a3b", tp=1, dp=4, ep=2, pcp=2, num_devices=8, layers=1,
                         enable_dbo=False, exclude_optimizations=("cann_megamoe_fusion",),
                         prompt_tokens=8, output_tokens=2, max_num_seqs=2)
        table = ((3, 2), (1, 0), (7, 6), (5, 4))
        raw["runtime"]["parallel"]["rank_table"] = table
    case = parse_case(raw)
    parallel = case.runtime.parallel
    topology = Topology(case.machine, parallel)
    assert parallel.rank_table == table
    members = [(replica, stage, group) for stage, group in enumerate(table) for replica in range(parallel.dp)]
    if not staged:
        assert cohorts(parallel, case.machine) == [(0,), (1,), (2,), (3,)]
        members = [(3, 0, (7, 6))]
        graphs = target_graphs(case, phase="prefill", tokens=8, context=0, sequences=2,
                               selected_replica=3, physical_rank=7, scope="body")
        assert [op.local_shape[0] for op in graphs[3] if op.recipe == "grouped_gate_up"] == [64]
    for replica, stage, group in members:
        assert topology.groups("ep", replica, stage) == (group,)

PROJECTION_CAPTURE = json.loads((ROOT / "tests/fixtures/verification/kimi-mixed-projection-contract.json").read_text())
PROJECTION_PHASE = dict(phase="decode", tokens=6, sequences=4, context=2048, scope="body")

def projection_case(overrides=None):
    return parse_case({
        "model": {"catalog_key": "glm-5.3-flash", "quantization": "w8a8", "layers": 4,
                  **({"quant": replace(load_model("glm-5.3-flash").quant,
                                      category_overrides=overrides).to_dict()} if overrides else {})},
        "machine": {"backend": "ascend", "sku": DUAL_DIE_SKU, "num_devices": 16},
        "runtime": {"tp": 16, "ep": 16, "kernels": {"programs": {"matmul": {"kind": "steady_matmul"}}}},
        "workload": {"requests": [{"prompt_tokens": 1, "output_tokens": 1}]},
    })

def layer_graph(case, index, **overrides):
    return {op.recipe: op for op in target_graphs(case, **(PROJECTION_PHASE | overrides))[0] if op.layer == index}

@pytest.mark.parametrize("capture", PROJECTION_CAPTURE["captures"], ids=lambda row: f"tp{row['tp']}")
def test_mixed_projection_retains_quantized_inputs_packed_views_and_residency(capture):
    tp, batch = capture["tp"], capture["batch_slots"]
    raw = json.loads((ROOT / capture["case"]).read_text())
    case, heads, hidden = parse_case(raw), 96 // tp, 7168
    graph = target_graphs(case, **(PROJECTION_PHASE | dict(tokens=1, sequences=batch,
                          context=126000, resource_fusion=False)))[0]
    bfg, qkv = (next(op for op in graph if op.layer == 0 and op.recipe == recipe)
                for recipe in ("linear_bfg", "linear_qkv"))
    for node, width, dtype in ((bfg, 257, "bf16"), (qkv, 384, "int8")):
        assert (node.local_shape, node.dtype) == ((batch, heads * width, hidden), dtype)
    producer = next(op for op in graph if op.op_id == qkv.op_id + ".quantize")
    assert qkv.deps == (producer.op_id, bfg.op_id) and producer.mte3_bytes == batch * (hidden + 4)
    slices = dict(bfg.metadata)["packed_projection_slices"]
    assert [row["offset_elements"] for row in slices] == [0, heads, heads * 129]
    terms = layer_weight_terms(case.model.layer_plans[0], case.model)
    derived = next(term for term in terms if term.name == "kda_composed_f")
    assert term_bytes(derived, case.model.quant, case.runtime.parallel, source=True) == 0
    assert term_bytes(derived, case.model.quant, case.runtime.parallel) == heads * 128 * hidden * 2
    if tp == 8:
        raw["runtime"]["tensor_layouts"] = {"kda_recurrent": {"gate": "contiguous"},
                                          "linear_output_norm_gate": {"gate": "contiguous"}}
        copied = layer_graph(parse_case(raw), 0, tokens=1, sequences=3, resource_fusion=False)
        for name, offset in (("kda_decay_contiguous", heads), ("kda_output_gate_contiguous", heads * 129)):
            copy = copied[name]
            assert copy.mte2_bytes == copy.mte3_bytes == 3 * heads * 128 * 2
            assert dict(copy.metadata)["source_layout"]["offset"] == offset
        raw["model"]["quant"]["category_overrides"]["kda.q"] = BF16
        with pytest.raises(ValueError, match="qkv_group_requires_one_operand"):
            layer_graph(parse_case(raw), 0, tokens=1, sequences=2, resource_fusion=False)

@pytest.mark.parametrize("mode", ("packed", "split", "unequal", "direct"))
def test_kda_projection_packing_respects_shapes_precision_and_input_traffic(mode):
    case = projection_case({"kda": INT8, "kda.q": BF16} if mode == "split" else None)
    dk = 64 if mode in {"unequal", "direct"} else 128
    overrides = dict(packed_input_projection=False, gate_rank=0, projection_dtype="") if mode == "direct" else {}
    plan = replace(case.model.layer_plans[0], linear_key_head_dim=dk, linear_head_dim=128, **overrides)

    def lower(plan):
        builder = Builder("kda", plan, 0, [], case.model, "decode")
        kda_projections(builder, plan=plan, tokens=24, hidden=4096, key_width=4 * dk,
                        value_width=512, heads=4, dk=dk, deps="analysis_input")
        return builder.ops

    ops = lower(plan)
    by_recipe = {op.recipe: op for op in ops}
    if mode == "packed":
        assert [op.local_shape for op in ops] == [(24, 1796, 4096), (24, 512, 128), (24, 512, 128)]
        slices = dict(ops[0].metadata)["packed_projection_slices"]
        assert [row["offset_elements"] for row in slices] == [0, 512, 1024, 1536, 1540, 1668]
        separate = lower(replace(plan, packed_input_projection=False))
        assert sum(op.flops for op in ops) == sum(op.flops for op in separate)
        assert sum(op.mte2_bytes for op in separate) - sum(op.mte2_bytes for op in ops) == 5 * 24 * 4096 * 2
    elif mode == "split":
        assert by_recipe["linear_q"].dtype == "bf16"
        packed = next(op for op in ops if op.dtype == "int8")
        assert [row["role"] for row in dict(packed.metadata)["packed_projection_slices"]] == ["k", "v", "b", "f_a", "g_a"]
    else:
        assert by_recipe["linear_decay_b"].local_shape == (24, 256, 64)
        gate = by_recipe["linear_gate" if mode == "direct" else "linear_gate_b"]
        assert gate.local_shape == (24, 512, 4096 if mode == "direct" else 128)

@pytest.mark.parametrize("mode", ("packed", "mixed", "source", "fp32_index", "bf16_index"))
def test_mla_projection_dependencies_and_weight_placement_follow_declared_formats(mode):
    override = {"mixed": {"mla.q_a": INT8}, "fp32_index": {"indexer.weights": FP32},
                "bf16_index": {"indexer.weights": BF16}}.get(mode)
    case = projection_case(override)
    if mode == "source":
        plans = tuple(
            replace(
                p, qk_rope_head_dim=64, qk_nope_head_dim=192,
                index_kpool=1, index_kpool_compress=False,
            ) if p.attention == "mla" and p.qk_rope_head_dim == 0 else p
            for p in case.model.layer_plans
        )
        kernels = case.runtime.kernels | {"lowerings": {"mla_prolog": "vllm_ascend_mla_prolog_v3"}}
        case = replace(case, model=replace(case.model, layer_plans=plans),
                       runtime=replace(case.runtime, kernels=kernels))
    ops = layer_graph(case, 3)
    if mode in {"mixed", "source"}:
        assert "qkv_a" not in ops and ops["kv_a"].dtype == ("bf16" if mode == "mixed" else "int8")
        assert ops["q_a"].dtype == "int8"
    else:
        assert ops["qkv_a"].local_shape == (24, 2048, 4096)
        assert ops["q_norm"].deps == ops["kv_norm"].deps == (ops["qkv_a"].op_id,)
    if mode.endswith("_index"):
        joint = mode == "fp32_index"
        matrix = ops["index_k_weights" if joint else "index_weights"]
        names = {"indexer_wk_prepared_fp32", "indexer_weights_prepared_fp32"} if joint else {"indexer_weights"}
        assert (matrix.dtype, matrix.local_shape) == ("fp32" if joint else "bf16", (24, 160 if joint else 32, 4096))
        assert dict(matrix.metadata)["packed_weight_bytes"] == sum(
            term_bytes(term, case.model.quant, case.runtime.parallel)
            for term in layer_weight_terms(case.model.layer_plans[3], case.model) if term.name in names)

@pytest.mark.parametrize("layout", (None, "contiguous", "mixed"))
def test_projection_consumers_materialize_only_required_packed_views(layout):
    case = projection_case({"kda": INT8, "kda.q": BF16} if layout == "mixed" else None)
    selections = {"short_convolution": {"input": "contiguous"}, "kda_recurrent": {"beta": "contiguous"}}
    ops = layer_graph(replace(case, runtime=replace(case.runtime, tensor_layouts=selections if layout else {})),
                      0, tokens=4)
    for name, width, offset, expected in (
        ("convolution_input_contiguous", 1536, 0, layout == "contiguous"),
        ("kda_beta_contiguous", 4, 1536, layout is not None),
    ):
        assert (name in ops) == expected
        if expected:
            node = ops[name]
            assert node.local_shape == (16, width) and node.mte2_bytes == node.mte3_bytes == 16 * width * 2
            if layout == "contiguous":
                assert node.deps == (ops["linear_qkvbfg_a"].op_id,)
                assert dict(node.metadata)["slice_offsets"] == (0, offset)

def test_cli_runtime_aliases_roundtrip_and_preserve_omitted_declarations():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser, command="compose")
    assert case_overrides(parser.parse_args([])) == {}
    flags = "--o-proj-tp 2 --kda-proj-tp 4 --recover-ssm --memory-gib 1"
    assert case_overrides(parser.parse_args(flags.split())) == dict(
        o_proj_tp=2, linear_proj_tp=4, recover_ssm=True, memory_bytes=2**30)
    with pytest.raises(ModelingError, match="runtime_feature_conflict:linear_proj_tp"):
        case_overrides(parser.parse_args((flags + " --linear-proj-tp 3").split()))
    canonical = normalize_runtime_features({"parallel": {"kda_proj_tp": 4}, "routing": {
        "policy": "balanced", "redundant_experts_per_rank": 2}, "collective_startup_mode": "direct",
        "prefix_cache_mode": "SHARED", "prefix_read_mode": "per_sequence", "homogeneous_graph_reuse": "require"})
    assert canonical["collective_startup_mode"] == "concurrent" and canonical["prefix_cache_mode"] == "shared"
    software = Software.from_mapping(canonical)
    assert Software.from_mapping(software.to_dict()) == software
    assert canonical["parallel"] == {"linear_proj_tp": 4, "redundant_experts_per_rank": 2}
    assert normalize_case_runtime({"software": {"recover_ssm": True}})["runtime"]["recover_ssm"] is True
    assert split_routing_features({"policy": "balanced", "expert_slot_policy": "explicit"}) == (
        {"policy": "balanced"}, {"expert_slot_policy": "explicit"})

@pytest.mark.parametrize("raw,error", [
    ({"runtime": {"collective_startup_mode": "broadcast"}}, "runtime_feature_unsupported"),
    ({"runtime": {"recover_ssm": "true"}}, "runtime_recover_ssm_invalid"),
    ({"software": {"max_num_seqs": 4}}, "case_software_alias_unknown"),
    ({"runtime": {"parallel": {"linear_proj_tp": 2, "kda_proj_tp": 4}}}, "runtime_feature_conflict:linear_proj_tp"),
    ({"software": {"recover_ssm": True}, "runtime": {"recover_ssm": False}}, "runtime_feature_conflict"),
    ({"runtime": {"parallel": {"redundant_experts_per_rank": 1},
                  "routing": {"redundant_experts_per_rank": 2}}}, "runtime_feature_conflict"),
] + [({"runtime": {"parallel": {"o_proj_tp": bad}}}, "o_proj_tp") for bad in (True, -1, 1.5, "4", None)])
def test_runtime_features_reject_unsupported_or_conflicting_values(raw, error):
    with pytest.raises(ModelingError, match=error):
        normalize_case_runtime(raw)

REQUEST = RequestSpec("ok", 0, 128, 1)
INTEGER_FIELDS = ("prompt_tokens", "output_tokens", "prefix_tokens", "priority", "text_prompt_tokens",
                  "media_tokens", "chain_id")
REQUEST_FIELDS = {
    **dict.fromkeys(INTEGER_FIELDS, (-1, True, False, 1.0, 1.5, "1", None, [], {}, float("nan"), float("inf"))),
    **dict.fromkeys(("arrival_ms", "gap_ms"), (-1, -.1, True, False, "0", None, [], {}, float("nan"), float("inf"))),
    "request_id": ([], {}, None, 1, True, "", " \t\n", b"ok"),
    "media_encoded": (0, 1, "true", None, [], {}), "prefix_cache_key": (),
}

@pytest.mark.parametrize("field", REQUEST_FIELDS)
def test_invalid_request_values_rejected_at_each_construction_boundary(field):
    giant = 10**5000
    boundaries = {"prompt_tokens": (0,), "output_tokens": (0,), "prefix_tokens": (129,),
                  "text_prompt_tokens": (127, 129), "media_tokens": (129,)}
    for bad in (*REQUEST_FIELDS[field], *boundaries.get(field, ()), giant, [giant], {"bad": giant}):
        fields = asdict(REQUEST) | {field: bad}
        forged = object.__new__(RequestSpec)
        for key, value in fields.items():
            object.__setattr__(forged, key, value)
        for construct in (lambda: RequestSpec(**fields), lambda: replace(REQUEST, **{field: bad}),
                          lambda: RequestSpec.from_mapping(REQUEST.to_dict() | {field: bad}, 0),
                          lambda: WorkloadSpec((forged,))):
            with pytest.raises(ModelingError):
                construct()

@pytest.mark.parametrize("key,bad", [
    ("requests", value) for value in ((), [], {}, set(), "", b"", None, (123,), [None])
] + [("synthetic", value) for value in (1, "yes", 0, None, (), [])] + [
    ("metadata", value) for value in (None, 1, True, {}, set(), "ab", b"ab", (None,), ((),),
                                      (("key",),), (("a", "b", "c"),), (("a", []),), ((1, "b"),))
])
def test_workload_rejects_invalid_containers_and_flags(key, bad):
    with pytest.raises(ModelingError):
        WorkloadSpec(**({"requests": (REQUEST,)} | {key: bad}))
    with pytest.raises(ModelingError):
        replace(WorkloadSpec((REQUEST,)), **{key: bad})

@pytest.mark.parametrize("container", (tuple, list))
def test_workload_copies_metadata_and_roundtrips_request_boundaries(container):
    row = replace(REQUEST, prefix_tokens=128, text_prompt_tokens=96, media_tokens=32, media_encoded=True)
    pair = ["source", "fixture"]
    workload = WorkloadSpec((row,), synthetic=True, metadata=container([pair]))
    pair[1] = "changed"
    assert workload.metadata == (("source", "fixture"),)
    assert WorkloadSpec.from_mapping(workload.to_dict()) == workload
    duplicated = (row, replace(row, arrival_ms=1))
    with pytest.raises(ModelingError, match="workload.request_id_duplicate"):
        WorkloadSpec(duplicated)
    with pytest.raises(ModelingError):
        replace(workload, metadata=(entry for entry in (pair,)))


def test_shipping_baseline_enables_quant_capture_mlapo_and_arch_disjoint_collectives():
    from _support.axes import REPRESENTATIVE_SKUS
    from mindie_sim.catalog.quant import resolve_auto_va_quant

    expected = {
        "a5": ("CCU_MS", "ub_direct", None, "fp8_e4m3"),
        "a3": ("AIV", "hccs_sw", True, "int8"),
        "a2": ("AIV", "hccs_8p_roce", True, "int8"),
    }
    for family, (exp_mode, fabric, expect_nz, expect_kv) in expected.items():
        case_dict = compose_case(
            "deepseek-v3.2", REPRESENTATIVE_SKUS[family],
            tp=1, dp=8, ep=8, layers=4, feature_track="shipping",
        )
        rt = case_dict["runtime"]
        meta = rt["metadata"]
        policy = meta["execution_policy"]
        assert rt["graph_capture_mode"] == "full"
        assert rt["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
        assert policy["mlapo"] is True
        assert meta["va_env"]["VLLM_ASCEND_ENABLE_MLAPO"] == "1"
        assert "HCCL_INTRA_PCIE_ENABLE" not in meta["va_env"]
        assert meta["va_env"]["HCCL_OP_EXPANSION_MODE"] == exp_mode
        assert meta["shipping_collective_policy"]["fabric"] == fabric
        assert rt["kernels"]["lowerings"]["mla_prolog"] == "vllm_ascend_mla_prolog_v3"
        assert meta.get("additional_config", {}).get("enable_weight_nz_layout") is expect_nz
        if family != "a5":
            assert "CCU" not in json.dumps(meta)
        _, kv_dt, _ = resolve_auto_va_quant(
            load_model("deepseek-v3.2"), REPRESENTATIVE_SKUS[family], context_tokens=128,
        )
        assert kv_dt == expect_kv

    _, kpool_kv, kpool_idx = resolve_auto_va_quant(
        load_model("glm-5.3-flash"), REPRESENTATIVE_SKUS["a3"], context_tokens=128,
    )
    assert kpool_kv == "int8" and kpool_idx == "bf16"

    prefill = compose_case(
        "deepseek-v3.2", REPRESENTATIVE_SKUS["a5"],
        tp=1, dp=8, ep=8, layers=4, role="pd_prefill", ttft_slo_ms=5000.0, feature_track="shipping",
    )
    assert prefill["runtime"]["metadata"]["execution_policy"]["mlapo"] is False
    assert "VLLM_ASCEND_ENABLE_MLAPO" not in prefill["runtime"]["metadata"]["va_env"]

    non_mla = compose_case(
        "qwen3.5-397b-a17b", REPRESENTATIVE_SKUS["a5"], tp=8, ep=8, layers=2, feature_track="shipping",
    )
    assert non_mla["runtime"]["metadata"]["execution_policy"]["mlapo"] is False
    assert "VLLM_ASCEND_ENABLE_MLAPO" not in non_mla["runtime"]["metadata"]["va_env"]


def test_moe_route_policy_enforces_fused_collective_mutex_and_shared_fusion_guards():
    from _support.axes import REPRESENTATIVE_SKUS

    eligible = compose_case(
        "deepseek-v3.2", REPRESENTATIVE_SKUS["a5"],
        tp=1, dp=8, ep=8, layers=4, feature_track="shipping",
        enable_fused_mc2=3, multistream_overlap_shared_expert=True,
    )
    rt = eligible["runtime"]
    assert rt["metadata"]["additional_config"]["enable_fused_mc2"] == 3
    assert not rt.get("multistream_overlap_shared_expert", False)
    assert rt["metadata"]["moe_route"]["megamoe_shared_expert_fusion"] is True
    assert rt["metadata"]["moe_route"]["mutex_disabled_mso"] is True

    gated = compose_case(
        "qwen3.5-397b-a17b", REPRESENTATIVE_SKUS["a5"],
        tp=8, ep=8, layers=2, feature_track="shipping",
        multistream_overlap_shared_expert=True,
    )
    assert gated["runtime"]["metadata"]["additional_config"]["enable_fused_mc2"] == 2
    assert not gated["runtime"].get("multistream_overlap_shared_expert", False)

    kimi_merged = compose_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"],
        tp=8, ep=8, layers=4, role="pd_decode", feature_track="shipping",
    )
    assert kimi_merged["runtime"]["metadata"]["additional_config"]["enable_fused_mc2"] == 0
    assert kimi_merged["runtime"].get("multistream_overlap_shared_expert") is True

    for family in ("a3", "a2"):
        with pytest.raises(ModelingError, match="enable_fused_mc2_level3_requires_a5"):
            compose_case(
                "deepseek-v3.2", REPRESENTATIVE_SKUS[family],
                tp=1, dp=8, ep=8, layers=4, enable_fused_mc2=3,
            )

    from mindie_sim.model_catalog import CONFIG_DIR
    latent_cfg = json.loads((CONFIG_DIR / "deepseek-v3.2.json").read_text())
    latent_cfg["routed_expert_hidden_size"] = latent_cfg["hidden_size"]
    with pytest.raises(ModelingError, match="enable_fused_mc2_level3_unsupported_latent_moe"):
        compose_case(
            latent_cfg, REPRESENTATIVE_SKUS["a5"],
            tp=1, dp=8, ep=8, layers=4, enable_fused_mc2=3,
        )


def test_architecture_isolation_rejects_cross_generation_presets_and_knobs():
    from _support.axes import REPRESENTATIVE_SKUS
    from mindie_sim.catalog.npu_checkpoints import normalize_hw_family
    from mindie_sim.stacks.vllm_ascend.speculative import build_va_serving_rules

    assert normalize_hw_family("Arch20") == "a2"
    assert normalize_hw_family("Arch22") == "a3"
    assert normalize_hw_family("Arch35") == "a5"
    assert normalize_hw_family("H100") == "other"
    assert normalize_hw_family("") == "other"

    for family, bad_preset in (("a5", "a3_superpod"), ("a3", "a5_pod"), ("a2", "a5_pod")):
        with pytest.raises(ModelingError, match="ascend_topology_preset_incompatible"):
            compose_case(
                "deepseek-v3.2", REPRESENTATIVE_SKUS[family],
                tp=1, dp=8, ep=8, layers=4, topology_preset=bad_preset,
            )
    with pytest.raises(ModelingError, match="a2_superpod_unsupported"):
        compose_case(
            "deepseek-v3.2", REPRESENTATIVE_SKUS["a2"],
            tp=1, dp=8, ep=8, layers=4, superpod_size=384,
        )

    for family in ("a2", "a3", "a5"):
        rules = build_va_serving_rules(
            "deepseek-v3.2", hw_family=REPRESENTATIVE_SKUS[family],
            tp=8, dp=1, ep=8, pp=1, enable_weight_nz_layout=True,
        )
        blob = f"{rules.collective_guidance} {rules.hw_isa_guidance}"
        assert "AIV/CCU" not in blob and "CCU/AIV" not in blob
        if family == "a5":
            assert "weight_nz_mode" not in rules.additional_config
            assert "enable_weight_nz_layout" not in rules.additional_config
            assert any("weight_nz_mode_not_applicable" in w for w in rules.warnings)
            assert "CCU" in blob and "UB-Direct" in blob
        else:
            assert rules.additional_config.get("weight_nz_mode") == 1
            assert "CCU" not in blob
            if family == "a2":
                assert "SuperPoD" not in blob and "UB-Direct" not in blob
