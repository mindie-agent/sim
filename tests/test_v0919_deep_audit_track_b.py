"""v0.9.19 Track B deep-audit regressions.

[Role]
  Pin the multi-parallelism, speculation, DBO/PD, solver and 128-rank speed
  findings at their physical boundary: a cloned replica must carry its own
  executable/owner, a draft block must price the block it executes, a join
  must wait for every microbatch, and a solver must not select a batch its
  own capacity window rejects.

[Invariants]
  Each case asserts an independent work, dependency, owner or geometry
  invariant rather than a source string. Performance tests bound the
  redundant-lowering cost; the remaining tests use two to four layers and
  at most 128 declared ranks.

[Timeouts & Locks]
  Bounded in-memory graphs; no device context, network, sleep or file I/O.

[Side-Effect Policy]
  Pure immutable cases; writes only pytest's temporary state.
"""
from __future__ import annotations

import time

import pytest

from mindie_sim import compose_case
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction import efficiency as efficiency_module
from mindie_sim.prediction.efficiency import (
    _lower_and_run_graph,
    analyze_network_efficiency,
    evaluate_target_efficiency,
    solve_sla_sizing,
)
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import (
    operator_costs,
    target_graphs,
)
from mindie_sim.prediction.speculation import draft_program
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.unmodeled_gate import assert_report_supported
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.workload import RequestView
from test_prediction_deployment import pd_case
from test_prediction_system import case as plain_case
from test_prediction_system import linear_work


def _graph_ops(graphs, replica):
    return tuple(graphs[replica])


def _by_recipe(ops, recipe, *, layer=None):
    return tuple(op for op in ops
                 if op.recipe == recipe and (layer is None or op.layer == layer))


def _by_prefix(ops, prefix):
    return tuple(op for op in ops if op.op_id.startswith(prefix))


def _deepseek_128():
    return compose_case(
        "deepseek-v4-pro", hardware="a5", tp=2, dp=64, ep=128,
        num_devices=128, layers=4, mtp_proposal_tokens=5,
        prompt_tokens=4096, output_tokens=8)


# --------------------------------------------------------------------------- #
# 1. 128-rank lowering speed, attached-program parity, cohort and physical key.
# --------------------------------------------------------------------------- #
def test_run_graph_false_lowers_128_ranks_without_executing_the_cohort():
    parsed = parse_case(_deepseek_128())
    started = time.perf_counter()
    graphs, _effective, report = _lower_and_run_graph(
        parsed, phase="decode", tokens=6, context=4096, sequences=1,
        scope="model", run_graph=False)
    elapsed = time.perf_counter() - started

    assert report is None
    assert len(graphs) == 64
    assert elapsed < 15.0
    assert len(graphs.program(0).work.ops) > 0


def test_explicit_run_graph_false_is_not_overridden_by_mc2_kernel_groups():
    raw = _deepseek_128()
    raw["runtime"]["metadata"]["moe_comm_mode"] = "dynamic"
    parsed = parse_case(raw)
    graphs, _effective, report = _lower_and_run_graph(
        parsed, phase="decode", tokens=6, context=4096, sequences=1,
        scope="body", run_graph=False)

    assert report is None
    assert any("device_kernel_group" in dict(op.metadata) for op in graphs[0])
    assert len(graphs) == 64


def test_attached_program_uses_the_final_local_tp_replica_work():
    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=32, ep=32,
        num_devices=32, layers=4, o_proj_tp=8, linear_proj_tp=8,
        lm_head_tp=8, mtp_proposal_tokens=5, prompt_tokens=4096,
        output_tokens=8)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=6, context=4096,
        sequences=1, scope="model")

    graph_ids = {op.op_id for op in graphs[0]}
    program_ids = {op.op_id for op in graphs.program(0).work.ops}
    logits = [op for op in _by_recipe(graphs[0], "vocab_logits")
              if op.op_id == "logits:local_tp:gemm"]
    assert logits
    assert "logits:local_tp:gather" in graph_ids
    assert {"logits:local_tp:gather", "logits:local_tp:gemm"} <= program_ids
    assert graphs.program(0).ports.tensor_out


def test_cohort_reuse_rebinds_nested_commit_request_identities():
    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=2, ep=2,
        num_devices=2, layers=4, recover_ssm=True,
        mtp_proposal_tokens=3, mtp_acceptance_length=3.5,
        homogeneous_graph_reuse="require", prompt_tokens=8, output_tokens=4)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=4, context=0, sequences=2,
        scope="model")

    commits = [op for op in graphs[1] if op.recipe == "recover_ssm_commit"]
    assert commits
    for op in commits:
        assert all(request.startswith("analysis1:") for request in op.requests)
        committed = dict(op.metadata)["committed_tokens"]
        assert tuple(request for request, _count in committed) == op.requests


def test_physical_key_groups_identical_fused_layer_occurrences():
    raw = compose_case(
        "deepseek-v4-pro", hardware="a5", tp=8, ep=8, num_devices=8,
        layers=61, prompt_tokens=8, output_tokens=2)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=1, context=4096,
        sequences=1, scope="model")
    assert len(_by_recipe(graphs[0], "indexer_quant_cache")) == 30

    report = operator_costs(parse_case(raw), graphs)
    rows = [row for row in report["operators"]
            if row["operator"].get("family") == "indexer_quant_cache"]
    assert len(rows) == 1
    assert rows[0]["occurrences"] == 30


# --------------------------------------------------------------------------- #
# 2. Speculation: independent BF16 draft, cyclic NextN, PP ids, PCP and DCP.
# --------------------------------------------------------------------------- #
def test_independent_bf16_draft_strips_target_packed_prolog():
    quant = quant_preset("w8a8").to_dict() | {"kv_dtype": "int8"}
    raw = compose_case(
        "kimi-k3.1", hardware="a5", tp=4, ep=4, num_devices=4, layers=4,
        quant=quant, mtp_proposal_tokens=2, prompt_tokens=64, output_tokens=8)
    parsed = parse_case(raw)

    decode = target_graphs(
        parsed, phase="decode", tokens=3, context=0, sequences=1, scope="model")
    prefill = target_graphs(
        parsed, phase="prefill", tokens=8, context=0, sequences=1,
        scope="model", draft_prepare=True)

    assert any(op.phase == "draft" for op in decode[0])
    assert any(op.op_id.startswith("draft_prefill.") for op in prefill[0])


def test_cyclic_nextn_executes_one_declared_block_per_proposal():
    raw = compose_case(
        "step-5-preview", hardware="a5", tp=4, ep=4, num_devices=4,
        layers=4, mtp_proposal_tokens=3, prompt_tokens=64, output_tokens=8)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=4, context=0, sequences=1,
        scope="model")

    drafts = [op for op in graphs[0] if op.op_id.startswith("draft")
              and op.family == "attention"]
    per_step = {op.op_id.split(".", 1)[0] for op in drafts}
    assert per_step == {"draft0", "draft1", "draft2"}
    assert len(drafts) == 3


def test_pp_proposal_transfer_sums_local_proposals_from_last_active_step():
    from mindie_sim.prediction.api import predict
    from mindie_sim.prediction.graph_analysis import graph_costs

    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=1, ep=1, pp=2,
        num_devices=2, pp_layer_partition=(1, 3), layers=4,
        graph_capture_mode="none", max_num_batched_tokens=64,
        mtp_proposal_tokens=5, prompt_tokens=4, output_tokens=6)
    parsed = parse_case(raw)
    views = (
        RequestView("short", 1, 0, False, "decode",
                    draft_proposal_tokens=1, draft_verify=True),
        RequestView("long", 1, 0, False, "decode",
                    draft_proposal_tokens=5, draft_verify=True),
    )
    regions, _previous = draft_program(
        Frontend(), parsed.model, parsed.runtime, views, "iteration_input",
        prefill=False, machine=parsed.machine)

    transfers = [op for region in regions for op in region.ops
                 if op.op_id == "draft_proposal_transfer"]
    assert len(transfers) == 1
    transfer = transfers[0]
    assert transfer.family == "pp_p2p"
    assert transfer.recipe == "pp_p2p"
    assert transfer.link_bytes == 6 * 8
    assert transfer.local_shape == (6,)
    assert transfer.stage == 1

    graphs = target_graphs(
        parsed, phase="decode", tokens=6, context=4, sequences=2, scope="model")
    assert graph_costs(parsed, graphs)["duration_ms"] > 0.0

    raw["workload"] = {"requests": [
        {"id": "r0", "prompt_tokens": 4, "output_tokens": 2, "arrival_ms": 0.0},
        {"id": "r1", "prompt_tokens": 4, "output_tokens": 6, "arrival_ms": 0.0},
    ]}
    serving = predict(raw, max_model_steps=2)
    assert serving["execution"]["iterations"] == 2


def test_pcp_draft_prefill_partitions_query_rows_and_keeps_global_gather():
    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=2, dp=1, ep=2, pcp=2,
        num_devices=4, layers=4, mtp_proposal_tokens=5,
        prompt_tokens=16, output_tokens=2)
    graphs = target_graphs(
        parse_case(raw), phase="prefill", tokens=16, context=0, sequences=1,
        scope="model", draft_prepare=True)
    draft = _by_prefix(graphs[0], "draft_prefill.")
    qkv = _by_recipe(draft, "qkv_a", layer=0)
    gather = _by_recipe(draft, "pcp_kv_gather", layer=0)

    assert qkv and qkv[0].local_shape[0] == 8
    assert gather and gather[0].link_bytes == 8192
    assert max(op.local_shape[0] for op in draft
               if op.family == "gemm" and len(op.local_shape) >= 2
               and op.local_shape[0] > 0 and op.local_shape[0] < 16) <= 8


def test_dspark_scatter_charges_the_owning_position_block():
    raw = compose_case(
        "kimi-k3.1", hardware="a5", tp=8, dp=1, ep=8, dcp=8,
        num_devices=8, layers=4, mtp_proposal_tokens=5,
        prompt_tokens=8, output_tokens=2)
    parsed = parse_case(raw)
    graphs = target_graphs(
        parsed, phase="decode", tokens=6, context=4, sequences=1,
        scope="model")
    scatters = _by_recipe(graphs[0], "draft_context_kv_scatter")
    assert scatters
    for op in scatters:
        metadata = dict(op.metadata)
        assert metadata["dcp_degree"] == 8
        assert metadata["dcp_row_shard"] == 6
        assert op.mte3_bytes == 6 * 1152


# --------------------------------------------------------------------------- #
# 3. Multi-parallelism, MoE routing, DBO, P/D and KDA.
# --------------------------------------------------------------------------- #
def test_dynamic_moe_group_selects_one_protocol_and_prices_the_cohort():
    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=32, ep=32,
        num_devices=32, layers=4, o_proj_tp=8, linear_proj_tp=8,
        lm_head_tp=8, moe_comm_mode="dynamic", max_num_seqs=2,
        prompt_tokens=8, output_tokens=4)
    raw["runtime"]["graph_capture_keys"] = [12]
    parsed = parse_case(raw)
    graphs = target_graphs(
        parsed, phase="prefill", tokens=7, context=0,
        sequences=[1] * 31 + [2], scope="body")

    protocols = {
        any("device_kernel_group" in dict(op.metadata) for op in graphs[replica])
        for replica in graphs
    }
    assert protocols == {False}
    assert all(_by_recipe(graphs[replica], "expert_dispatch")
               for replica in graphs)
    assert operator_costs(parsed, graphs)["operator_occurrences"] > 0


def test_sp_mtp_preserves_captured_proposal_rows_in_serving_and_analysis():
    from mindie_sim.prediction.api import predict

    raw = compose_case(
        "kimi-k3.1", hardware="a5", dp=16, tp=8, dcp=8, pcp=1, ep=128, pp=1,
        num_devices=128, sp=True, recover_ssm=True, layers=4,
        mtp_proposal_tokens=5, moe_comm_mode="dynamic",
        graph_capture_mode="full", prompt_tokens=4, output_tokens=8)
    raw["runtime"]["graph_capture_keys"] = [16]
    parsed = parse_case(raw)
    graphs = target_graphs(
        parsed, phase="decode", tokens=6, context=4, sequences=1, scope="model")

    def dispatch_tp_rows(ops):
        op = next(item for item in ops if item.op_id == "s0.l1.dispatch")
        meta = dict(op.metadata)
        pkt = float(meta["peer_traffic_row_format"]["forward_bytes"])
        matrix = meta["mc2_phase"]["route_matrix_bytes"]
        tp_rows = tuple(round(sum(row) / pkt / 16.0, 6) for row in matrix[:8])
        total = round(sum(sum(row) for row in matrix) / pkt, 6)
        return tp_rows, total

    tp_rows, total_assignments = dispatch_tp_rows(graphs[0])
    assert tp_rows == (2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert total_assignments == pytest.approx(1536.0)

    captured_works = []
    from mindie_sim.prediction import dispatch as dispatch_module
    orig_lower = dispatch_module._lower_with_binding

    def spy_lower(*args, **kwargs):
        work = orig_lower(*args, **kwargs)
        captured_works.append(work)
        return work

    serving_raw = compose_case(
        "kimi-k3.1", hardware="a5", dp=2, tp=8, dcp=8, pcp=1, ep=16, pp=1,
        num_devices=16, sp=True, recover_ssm=True, layers=4,
        mtp_proposal_tokens=5, moe_comm_mode="dynamic",
        graph_capture_mode="full", prompt_tokens=4, output_tokens=8)
    serving_raw["runtime"]["graph_capture_keys"] = [16]
    serving_raw["workload"] = {"requests": [
        {"id": f"r{i}", "prompt_tokens": 4, "output_tokens": 8, "arrival_ms": 0.0}
        for i in range(2)
    ]}
    dispatch_module._lower_with_binding = spy_lower
    try:
        predict(serving_raw, max_model_steps=2)
    finally:
        dispatch_module._lower_with_binding = orig_lower
    decode_ops = [w.ops for w in captured_works
                  if any(op.op_id == "s0.l1.dispatch" and op.phase == "decode"
                         for op in w.ops)]
    assert decode_ops
    srv_tp_rows, srv_total = dispatch_tp_rows(decode_ops[0])
    assert srv_tp_rows == (2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert srv_total == pytest.approx(2 * 6 * 16.0)


def test_alltoallv_receive_rows_follow_destination_assignments():
    raw = plain_case(dp=2, tp=1, ep=2, prompt=6, output=1)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=8, top_k=2, hidden_size=32,
        moe_intermediate_size=32, activation_dtype="bf16")
    raw["runtime"].update(max_num_batched_tokens=32)
    placement = [[0, 0], [0, 1], [0, 2], [0, 3],
                 [1, 0], [1, 1], [1, 2], [1, 3]]
    raw["runtime"]["routing"] = {"realizations": [{
        "scope": "body", "layer": 0, "phase": "prefill", "cohort": True,
        "topk_ids": [[0, 1] for _ in range(18)],
        "expert_placement": placement,
        "source_owners": [0] * 6 + [1] * 12,
    }]}
    graphs = target_graphs(
        parse_case(raw), phase="prefill", tokens=6, context=0,
        sequences=[1, 2], scope="body")

    dispatch = _by_recipe(graphs[0], "expert_dispatch")
    experts = _by_recipe(graphs[0], "grouped_gate_up")
    assert dispatch and dispatch[0].local_shape[0] == 12
    assert dict(dispatch[0].metadata)["payload_rows"] == 36
    assert experts and experts[0].local_shape[0] == 36
    assert dict(_by_recipe(graphs[1], "expert_dispatch")[0].metadata)[
        "payload_rows"] == 0


def test_redundant_expert_slot_binding_uses_the_replica_owner():
    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=32, ep=32,
        num_devices=32, layers=4, o_proj_tp=8, linear_proj_tp=8,
        lm_head_tp=8, redundant_experts_per_rank=1,
        prompt_tokens=8, output_tokens=2)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=1, context=0,
        sequences=1, scope="body")

    def gate(replica):
        return next(op for op in graphs[replica]
                    if op.recipe == "grouped_gate_up" and op.layer == 3)

    owners = {replica: dict(gate(replica).metadata)["ep_rank"]
              for replica in (0, 2)}
    assert owners == {0: 0, 2: 2}
    assert (dict(gate(0).metadata)["routed_weight_read_bytes"]
            != dict(gate(2).metadata)["routed_weight_read_bytes"])


def test_local_tp_quantizer_and_gemm_derive_the_same_scale_geometry():
    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, dp=8, ep=8, num_devices=8,
        layers=4, o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8,
        quant="w8a8_mxfp8", prompt_tokens=8, output_tokens=2)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=6, context=0, sequences=1,
        scope="body")
    quantizers = {op.op_id: dict(op.metadata).get("activation_scale_shape")
                  for op in graphs[0] if op.layer == 3
                  and op.family == "quantize"}
    gemms = {op.op_id: dict(op.metadata).get("activation_scale_shape")
             for op in graphs[0] if op.layer == 3
             and op.op_id.endswith(":local_tp:gemm")}

    assert gemms
    for op_id, shape in gemms.items():
        producer = op_id.removesuffix(":local_tp:gemm") + ".quantize"
        assert quantizers[producer] == shape


def test_dbo_join_waits_for_every_partition_sink():
    raw = compose_case(
        "qwen3-32b", hardware="a5", tp=1, ep=1, num_devices=1, layers=1,
        enable_dbo=True, max_num_batched_tokens=128, max_num_seqs=2,
        prompt_tokens=64, output_tokens=2)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=1, context=64,
        sequences=2, scope="body")
    joins = [op for op in graphs[0] if op.op_id.endswith("dbo.join")]
    assert len(joins) == 1
    assert joins[0].deps == (
        "dbo0.s0.l0.ffn.residual", "dbo1.s0.l0.ffn.residual")


def test_dbo_moe_partition_reduces_the_global_row_ledger_without_doubling():
    def grouped(enable_dbo):
        raw = compose_case(
            "qwen3-30b-a3b", hardware="a5", tp=1, dp=2, ep=2,
            num_devices=2, layers=1, enable_dbo=enable_dbo,
            prompt_tokens=8, output_tokens=2, max_num_seqs=2)
        graphs = target_graphs(
            parse_case(raw), phase="decode", tokens=1, context=32,
            sequences=2, scope="body")
        return tuple(op for op in graphs[0]
                     if op.recipe in {"grouped_gate_up", "grouped_down"})

    direct = grouped(False)
    partitioned = grouped(True)
    assert len(partitioned) == 2 * len(direct)
    assert sum(op.flops for op in partitioned) == pytest.approx(
        sum(op.flops for op in direct))


def test_dbo_qualifies_shared_expert_stream_per_partition():
    raw = compose_case(
        "deepseek-v4-pro", hardware="a5", tp=1, dp=1, ep=1,
        num_devices=1, layers=1, enable_dbo=True,
        multistream_overlap_shared_expert=True,
        prompt_tokens=8, output_tokens=2, max_num_seqs=2)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=1, context=32,
        sequences=2, scope="body")
    streams = {op.stream for op in graphs[0] if "shared" in op.stream}
    assert {"dbo0:shared_stream", "dbo1:shared_stream"} <= streams


def test_pd_handoff_intersects_token_spans_before_charging_rows():
    raw = pd_case([{"id": "r", "prompt_tokens": 4, "output_tokens": 3}])
    raw["model"].update(
        layers=1, hidden_size=64, num_heads=4, num_kv_heads=1,
        head_dim=16, intermediate_size=64, vocab_size=128)
    for pool, ranks, tp, cp in (("prefill", (0, 1), 2, 2),
                                ("decode", (2,), 1, 1)):
        machine = raw["pools"][pool]["machine"]
        machine["num_devices"] = len(ranks)
        machine["devices"] = [{"rank": rank} for rank in ranks]
        machine["links"][0]["members"] = list(ranks)
        raw["pools"][pool]["runtime"].update(
            tp=tp, pp=1, cp=cp, dcp=cp,
            cp_plan="dcp" if cp > 1 else "none")
    raw["connector"] = {
        "link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                 "latency_us": 0, "semantics": "per_pair"},
        "chunk_bytes": 64}

    from mindie_sim import predict_pd
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    transfers = result["execution"]["pd_handoffs"][0]["transfers"]
    assert {(row["source_rank"], tuple(row["source_token_span"]))
            for row in transfers} == {(0, (0, 2)), (1, (2, 4))}
    assert all(row["payload_bytes"] == pytest.approx(128.0)
               for row in transfers)
    assert sum(row["payload_bytes"] for row in transfers) == pytest.approx(256.0)


def test_kda_respects_raw_eager_and_explicit_analytical_policy():
    raw = compose_case(
        "glm-5.3-flash", hardware="a5", tp=1, ep=1, num_devices=1,
        layers=4, raw_eager=True, prompt_tokens=8, output_tokens=2)
    graphs = target_graphs(
        parse_case(raw), phase="decode", tokens=1, context=0,
        sequences=1, scope="model")
    scans = _by_prefix(graphs[0], "s0.l0.recurrent")
    assert scans and all("vllm_ascend" not in op.recipe for op in scans)

    raw["runtime"]["kernels"] = {"lowerings": {"kda": {
        "cases": [{
            "domain": {"phases": ["decode"],
                       "min_query_tokens": {"min": 2},
                       "max_query_tokens": {"max": 8}},
            "binding": "vllm_ascend_recurrent_kda_arch35"}],
        "otherwise": "analytical"}}}
    analytical = target_graphs(
        parse_case(raw), phase="decode", tokens=1, context=0,
        sequences=1, scope="model")
    native = target_graphs(
        parse_case(raw), phase="decode", tokens=3, context=0,
        sequences=1, scope="model")
    assert all("vllm_ascend" not in op.recipe
               for op in _by_prefix(analytical[0], "s0.l0.recurrent"))
    assert all("vllm_ascend" in op.recipe
               for op in _by_prefix(native[0], "s0.l0.recurrent"))


# --------------------------------------------------------------------------- #
# 4. SLA, 4D target solver and the unmodeled gate.
# --------------------------------------------------------------------------- #
def _sla_case():
    raw = plain_case(prompt=4, output=2)
    raw["machine"]["device"]["memory_bytes"] = 208000
    raw["runtime"].update(max_num_seqs=8, async_scheduling=False,
                          enable_chunked_prefill=True)
    return raw


def test_sla_sizing_requires_current_and_lifecycle_capacity_windows():
    result = solve_sla_sizing(
        _sla_case(), phase="decode", context=4, output_budget=2,
        sla_ms=1000, sensitivity="baseline")["scenarios"][0]

    assert result["b_feasible"] == 7
    assert result["b_sla"] == 8
    assert result["b_vram_max"] == 7
    assert result["b_vram_max_joint"] == 7
    assert result["hbm_bound"] is True
    assert result["joint_capacity"]["fits"] is True
    assert result["capacity_window"]["fits"] is True
    assert result["capacity_window"]["headroom_bytes"] == pytest.approx(14912.0)


def test_sla_sizing_rejects_unchunkable_prefill_before_pricing(monkeypatch):
    raw = plain_case(prompt=4, output=2)
    raw["runtime"].update(max_num_batched_tokens=16,
                          enable_chunked_prefill=False)

    def fail_pricing(*_args, **_kwargs):
        raise AssertionError("unchunkable geometry was priced")

    monkeypatch.setattr(efficiency_module, "target_graphs", fail_pricing)
    result = solve_sla_sizing(
        raw, phase="prefill", tokens_per_seq=17, context=0,
        output_budget=2, sla_ms=1000,
        sensitivity="baseline")["scenarios"][0]

    assert result["b_feasible"] == 0
    assert result["b_sla"] == 0
    assert result["domain_max_b"] == 0
    assert result["domain_limit_reason"] == (
        "prompt_exceeds_batch_budget_without_chunking")


def test_exact_target_preserves_occurrence_dag_and_never_exceeds_one():
    raw = plain_case(prompt=4, output=2)
    raw["model"]["layers"] = 1
    baseline = analyze_network_efficiency(
        raw, phase="decode", tokens=1, context=32, sequences=1, scope="model")
    target = evaluate_target_efficiency(
        raw, phase="decode", tokens=1, context=32, sequences=1,
        scope="model", mode="exact_target")
    unit = evaluate_target_efficiency(
        raw, phase="decode", tokens=1, context=32, sequences=1,
        scope="model", mode="exact_target",
        target_mbu=1.0, target_cube_mfu=1.0,
        target_vec_mfu=1.0, target_link_util=1.0)

    assert len(baseline["critical_path_nodes"]) == 15
    assert baseline["critical_path_roofline_ms"] == pytest.approx(0.033888)
    assert target["critical_path_nodes"] == baseline["critical_path_nodes"]
    assert target["target_network_roofline_efficiency"] <= 1.0
    assert target["target_step_duration_ms"] >= target["physical_ideal_ms"]
    assert unit["target_step_duration_ms"] == pytest.approx(0.033888)
    assert unit["target_network_roofline_efficiency"] == pytest.approx(1.0)


def test_unmodeled_gate_resolves_layer_override_and_flat_timeline():
    raw = plain_case(prompt=4, output=2)
    raw["model"]["layer_overrides"] = {"0": {"attention": "unmodeled"}}
    from mindie_sim.prediction import predict
    with pytest.raises(UnmodeledCapabilityError):
        predict(raw, lower=linear_work)

    with pytest.raises(UnmodeledCapabilityError):
        assert_report_supported(
            {"timeline": [{"op_id": "a", "family": "attention",
                           "recipe": "unmodeled_attention_core",
                           "layer": 0}]},
            entry="analyze_network_efficiency")
