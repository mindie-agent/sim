"""Issue #36 Track 3 wiring and runtime regressions.

[Role]
  Exercise the RecoverSSM deferred-commit boundary, redundant expert slot
  reads, homogeneous cohort reuse and exposed-breakdown reconstruction through
  the same public producers used by serving and analysis.

[Invariants]
  * A pure body/model assembly never resolves or retains a post-acceptance
    commit; a service assembly publishes exactly one commit per state owner
    after acceptance and before token commit.
  * PCP remains declared when a serial KDA recurrence collapses cp/dcp.
  * Redundant physical slots increase active routed-weight reads only through
    real slot bytes, never through logical assignments.
  * A "require" cohort refuses asymmetric replicas, while a proved template
    rebinds requests and ranks without dropping traffic.
  * Exposed attribution either conserves the executed device-step boundary or
    reports it unavailable; no clock is rescaled.

[Timeouts & Locks]
  Pure in-memory construction; no I/O, sleeps, device access or locks.

[Side-Effect Policy]
  Builds fresh immutable cases/operators only; no global mutation.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim.catalog.expert_slots import expert_slot_plan
from mindie_sim.compose import compose_case
from mindie_sim.prediction.cohort_reuse import prepare_cohort_graphs
from mindie_sim.prediction.expert_slot_work import (
    bind_expert_slot_work, physical_slot_work,
)
from mindie_sim.prediction.exposed_program import _role_times
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.graph_analysis import _exposed_breakdown, graph_costs
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.operator_analysis import (
    target_graphs, target_operators,
)
from mindie_sim.prediction.recover_ssm_commit import (
    bind_recover_ssm_commits, is_recover_ssm_intent,
)
from mindie_sim.prediction.recover_ssm_work import emit_commit_op
from mindie_sim.prediction.recurrent import serial_context_parallel
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from issue34_helpers import dense_case, dense_layer, hardware


def glm_case(*, layers=4, dp=8, ep=8, sequences=1, reuse="off",
             redundant=1):
    return parse_case(compose_case(
        model="glm-5.3-flash", hardware="a5", tp=1, dp=dp, ep=ep,
        o_proj_tp=max(dp, 1) if dp > 1 else 1,
        linear_proj_tp=max(dp, 1) if dp > 1 else 1,
        lm_head_tp=max(dp, 1) if dp > 1 else 1, layers=layers,
        concurrency=sequences, recover_ssm=True,
        redundant_experts_per_rank=redundant, homogeneous_graph_reuse=reuse,
    ))


def spec_view(q_len=6, advanced=3):
    return RequestView("r0", q_len, 128, True, "decode",
                       draft_verify=True, draft_advanced_tokens=advanced)


def proposal_view(proposals=5, advanced=3):
    return RequestView("r0", 1, 128, True, "decode", draft_verify=True,
                       draft_proposal_tokens=proposals,
                       draft_advanced_tokens=advanced)


class Plan:
    index = 0
    scope = "target"


class TinyModel:
    compute_dtype = "bf16"
    activation_dtype = "bf16"


def test_serial_kda_context_plan_preserves_pcp():
    with pytest.raises(ModelingError, match="pcp_requires_pcp_plan"):
        ParallelSpec(tp=2, pcp=2, cp_plan="none")
    spec = serial_context_parallel(ParallelSpec(tp=2, pcp=2, cp_plan="pcp"))
    assert (spec.cp, spec.dcp, spec.cp_plan, spec.pcp) == (1, 1, "pcp", 2)
    flat = serial_context_parallel(ParallelSpec(tp=2))
    assert (flat.cp, flat.dcp, flat.cp_plan) == (1, 1, "none")


def test_deferred_intent_is_uncharged_and_binds_at_acceptance():
    views = (spec_view(),)
    builder = Builder("s0.l0", Plan(), 0, views, TinyModel(), "decode")
    op_id = emit_commit_op(
        builder, name="recover_ssm_commit", deps=("scan",), views=views,
        heads=2, dk=4, dv=8, state_dtype="fp32")
    assert op_id == "s0.l0.recover_ssm_commit"
    intent = builder.ops[-1]
    assert is_recover_ssm_intent(intent)
    assert intent.requests == ("r0",)
    assert (intent.flops, intent.vector_ops, intent.hbm_bytes,
            intent.mte2_bytes, intent.mte3_bytes) == (0, 0, 0, 0, 0)
    geometry = dict(intent.metadata)["recover_ssm_geometry"]
    assert geometry["heads"] == 2 and geometry["dk"] == 4 and geometry["dv"] == 8
    assert geometry["scope"] == "target" and geometry["layer"] == 0
    bound = bind_recover_ssm_commits(
        (intent,), views=views, acceptance_events={"r0": "accept"},
        state_bindings=None, runtime=None)[0]
    assert bound.deps == ("scan", "accept")
    assert bound.vector_ops == pytest.approx(4 * 3 * 2 * 4 * 8)
    state = 2 * 4 * 8 * 4
    record_bytes = 2 * (8 * 4 + 2 * 4 * 2)
    assert bound.mte2_bytes == pytest.approx(state + 3 * record_bytes)
    assert bound.mte3_bytes == pytest.approx(state)
    assert bound.hbm_bytes == pytest.approx(bound.mte2_bytes + state)
    metadata = dict(bound.metadata)
    assert metadata["recover_ssm_stage"] == "commit"
    assert metadata["recover_ssm_pending_acceptance"] is False
    assert metadata["committed_tokens"] == (("r0", 3),)


def test_prefill_multitoken_query_is_never_a_commit():
    prefill = RequestView("p0", 6, 0, True, "prefill", prompt_tokens=6)
    builder = Builder("s0.l0", Plan(), 0, (prefill,), TinyModel(), "decode")
    assert emit_commit_op(
        builder, name="recover_ssm_commit", deps=("scan",), views=(prefill,),
        heads=2, dk=4, dv=8, state_dtype="fp32") is None
    intent = OpSpec("intent", "elementwise", "prefill", 0, 0, ("p0",),
                    (1,), (1,), (1,), (1,), metadata=(
                        ("recover_ssm_intent", True), ("state_owner", "layer0"),
                        ("heads", 2), ("dk", 4), ("dv", 8)))
    assert bind_recover_ssm_commits(
        (intent,), views=(prefill,), acceptance_events={},
        state_bindings=None, runtime=None) == (intent,)


def test_body_tokens_six_has_no_deferred_intent():
    case = glm_case(layers=2, dp=1, ep=1, sequences=1, redundant=0)
    graphs = target_graphs(case, phase="decode", tokens=6, context=128,
                           sequences=1, scope="body")
    assert all(not is_recover_ssm_intent(op) for op in graphs[0])


def test_service_assembly_places_commit_between_accept_and_token_commit():
    case = glm_case(layers=2, dp=1, ep=1, sequences=1, redundant=0)
    plan = IterationPlan((proposal_view(),))
    program = Frontend().program(
        plan, case.model, case.runtime, case.machine, assembly="service")
    ops = program.work.ops
    ids = [op.op_id for op in ops]
    assert not any(is_recover_ssm_intent(op) for op in ops)
    commits = [op for op in ops
               if dict(op.metadata).get("recover_ssm_stage") == "commit"]
    assert len(commits) == 2
    accept_index = ids.index("accept")
    token_index = ids.index("token_commit")
    commit_indexes = [ids.index(op.op_id) for op in commits]
    assert all(accept_index < index < token_index for index in commit_indexes)
    commit_ids = {op.op_id for op in commits}
    assert set(next(op for op in ops if op.family == "token_commit").deps) - {
        "accept"} == commit_ids
    assert all("accept" in op.deps for op in commits)


def test_redundant_slots_increase_active_weight_reads_at_saturation():
    def gate_reads(redundant):
        case = parse_case(compose_case(
            model="glm-5.3-flash", hardware="a5", tp=1, dp=8, ep=8,
            o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8, layers=4,
            concurrency=16, recover_ssm=True,
            redundant_experts_per_rank=redundant))
        graphs = target_graphs(case, phase="decode", tokens=1, context=128,
                               sequences=16, scope="body")
        gate = [op for op in graphs[0]
                if dict(op.metadata).get("expert_slot_weight_kind") == "gate_up"]
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
    template = (OpSpec("a@r0", "gemm", "decode", 0, 0, ("r0",), (8,), (8,),
                       (8,), (8,), flops=100.0, hbm_bytes=100.0,
                       deps=("x@r0",)),
                OpSpec("b@r0", "gemm", "decode", 0, 0, ("r0",), (8,), (8,),
                       (8,), (8,), flops=100.0, hbm_bytes=100.0,
                       deps=("a@r0",)))
    replica = tuple(
        OpSpec(op.op_id.replace("@r0", "@r1"), op.family, op.phase, op.stage,
               op.layer, ("r1",), op.semantic_shape, op.local_shape,
               op.padded_shape, op.launch_shape, flops=op.flops,
               hbm_bytes=op.hbm_bytes, deps=tuple(
                   dep.replace("@r0", "@r1") for dep in op.deps))
        for op in template)
    prepared = prepare_cohort_graphs(
        case, {0: template, 1: replica}, reuse="auto")
    assert prepared.graphs[1][0].requests == ("r1",)
    assert prepared.graphs[1][1].deps == ("a@r1",)
    asymmetric = (replica[0], OpSpec(
        replica[1].op_id, "gemm", "decode", 0, 0, ("r1",), (8,), (8,), (8,),
        (8,), flops=999.0, hbm_bytes=100.0, deps=("a@r1",)))
    with pytest.raises(ModelingError, match="proof_required"):
        prepare_cohort_graphs(
            case, {0: template, 1: asymmetric}, reuse="require")


@pytest.mark.parametrize("dp", (2, 4))
def test_local_tp_groups_span_dp_replicas_without_participant_mismatch(dp):
    width = 2
    case = parse_case(compose_case(
        model="longcat-flash-lite", hardware="a5", tp=1, dp=dp, ep=dp,
        o_proj_tp=width, linear_proj_tp=width, lm_head_tp=width, layers=1,
        concurrency=1, recover_ssm=True, redundant_experts_per_rank=0,
        homogeneous_graph_reuse="require"))
    graphs = target_graphs(case, phase="decode", tokens=1, context=128,
                           sequences=1, scope="body")
    for replica, graph in graphs.items():
        base = replica // width * width
        expected = tuple(range(base, base + width))
        groups = {dict(op.metadata)["group_ranks"]
                  for op in graph if "group_ranks" in dict(op.metadata)}
        assert groups == {expected}
    report = graph_costs(
        case, graphs, external_inputs={"analysis_input": 0},
        include_submission=False, isolated=False, compact=True)
    assert report["duration_ms"] > 0


def record(replica, family, start, end, *, link=0.0, flops=0.0, vector=0.0,
           hbm=0.0):
    op = SimpleNamespace(
        family=family, link_bytes=link, flops=flops, vector_ops=vector,
        hbm_bytes=hbm, mte2_bytes=0.0, mte3_bytes=0.0, metadata=())
    return {"replica": replica, "op": op,
            "ready": SimpleNamespace(time=start),
            "complete": SimpleNamespace(time=end)}


def test_graph_costs_publishes_conserving_device_step_breakdown():
    case = replace(dense_case(), machine=hardware(
        1, rates=(("hbm", 1.0e12), ("vector", 1.0e12))))
    ops = target_operators(case, phase="decode", tokens=1, context=8,
                           sequences=1, scope="body")
    report = graph_costs(
        case, {0: tuple(ops)}, external_inputs={"analysis_input": 0},
        include_submission=False, isolated=False)
    assert report["exposed_breakdown_basis"]["available"] is True
    assert report["exposed_breakdown_basis"]["boundary"] == "device_step"
    assert sum(report["exposed_breakdown_ms"].values()) == pytest.approx(
        report["duration_ms"])


def test_exposed_breakdown_conserves_or_declares_unreconciled():
    records = (record(0, "gemm", 0.0, 5.0, flops=10.0),
               record(0, "alltoall", 5.0, 8.0, link=100.0))
    result = _exposed_breakdown(records, 8.0, 0.0, 0)
    assert result["exposed_breakdown_ms"]["cube"] == pytest.approx(5.0)
    assert result["exposed_breakdown_ms"]["communication"] == pytest.approx(3.0)
    assert sum(result["exposed_breakdown_ms"].values()) == pytest.approx(8.0)
    unavailable = _exposed_breakdown(records, 9.0, 0.0, 0)
    assert unavailable["exposed_breakdown_ms"] is None
    assert unavailable["exposed_breakdown_basis"]["reason"] == (
        "unreconciled_execution_boundary")


def test_scoped_resource_names_map_to_semantic_roles():
    times = _role_times(
        {"memory@r0:s0": 100.0, "matrix@r0:s0": 200.0,
         "vector@r0:s0": 50.0, "link:fabric": 10.0},
        {"memory@r0:s0": 100.0, "matrix@r0:s0": 100.0,
         "vector@r0:s0": 100.0, "link:fabric": 100.0})
    assert times == {"cube_ms": 2.0, "vector_ms": 0.5,
                     "memory_ms": 1.0, "communication_ms": 0.1}
