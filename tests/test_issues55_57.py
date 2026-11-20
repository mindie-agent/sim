"""Behavior and conservation regressions for issues 55--57."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.draft import draft_quant
from mindie_sim.compose import compose_case
from mindie_sim.prediction.batch_prefix_attention import apply_sparse_prefix_reads
from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import ParallelSpec
from test_communication_topology import exchange
from test_joint_collective_execution import cohort_case
from _support.hardware import A5, op, sku_case


def glm(**options):
    return compose_case(
        "glm-5.3-flash", hardware="a5-pod-dt", quant="w8a8_mxfp8",
        tp=1, layers=4, prompt_tokens=4096, output_tokens=16,
        concurrency=8, graph_capture_mode="none", **options)


def test_glm_mxfp8_body_and_draft_roles():
    case = parse_case(glm())
    quant = case.model.quant
    for category in ("mla", "routed_expert", "shared_expert", "mtp"):
        assert quant.format_for(category) == "fp8_e4m3"
    for category in ("kda", "mla_absorb", "indexer", "indexer_fp8", "dspark_input"):
        assert quant.format_for(category) == "bf16"
    draft = draft_quant(case.model, case.model.auxiliary.mtp)
    assert draft.expert_weight_format == "fp8_e4m3"
    assert draft.format_for("mla_absorb") == "bf16"
    from mindie_sim.catalog.draft_weights import _draft_attention_terms
    terms = _draft_attention_terms(case.model.auxiliary.mtp, draft)
    assert next(term for term in terms if term.name == "draft_kv_b").category == "mla_absorb"
    operators = target_graphs(case, scope="body", phase="decode",
                              tokens=1, context=64)[0]
    projections = {op.recipe: dict(op.metadata) for op in operators if op.layer == 3}
    for recipe in ("qkv_a", "q_b", "output", "grouped_gate_up", "grouped_down"):
        assert projections[recipe]["weight_format"] == "fp8_e4m3"
        assert projections[recipe]["activation_format"] == "fp8_e4m3"


def test_fp8_cache_prolog_keeps_native_operand_contract():
    from mindie_sim.prediction.family_mla_prolog import validate
    case = parse_case(glm())
    quant = replace(case.model.quant, weight_format="bf16", activation_format="bf16",
                    kv_dtype="fp8_e4m3", category_overrides=())
    validate(plan=case.model.layer_plans[3], parallel=case.runtime.parallel,
             model=case.model, quant=quant, lowering_passes=case.runtime.lowering_passes)


def test_model_verify_acceptance_precedes_all_recover_commits():
    raw = glm(mtp_proposal_tokens=5, mtp_acceptance_length=3.5, recover_ssm=True)
    ops = target_graphs(raw, scope="model", phase="decode", tokens=6,
                        context=128, sequences=2)[0]
    acceptance = next(op for op in ops if op.family == "acceptance")
    commits = [op for op in ops if op.recipe == "recover_ssm_commit"]
    assert len(commits) == 3
    state_bytes = 64 * 128 * 128 * 4
    for commit in commits:
        assert acceptance.op_id in commit.deps
        assert tuple(n for _, n in dict(commit.metadata)["committed_tokens"]) == (3, 4)
        assert commit.mte3_bytes == 2 * state_bytes
        assert commit.hbm_bytes == commit.mte2_bytes + commit.mte3_bytes
    ordinary = target_graphs(raw, scope="model", phase="decode", tokens=1,
                             context=128, sequences=2)[0]
    assert not any(op.recipe == "recover_ssm_commit" for op in ordinary)
    # A probe below the declared verification width keeps the ordinary target
    # rows (2 per sequence) instead of inheriting the MTP proposal rows.
    probe = target_graphs(raw, scope="model", phase="decode", tokens=2,
                          context=128, sequences=2)[0]
    assert not any(op.recipe == "recover_ssm_commit" for op in probe)
    assert not any(op.family == "acceptance" for op in probe)
    rows = {dict(op.metadata)["graph_active_query_rows"] for op in probe
            if "graph_active_query_rows" in dict(op.metadata)}
    assert rows == {4}


def test_model_program_speculative_cohort_routes_all_verification_rows():
    from mindie_sim.prediction.operator_analysis import model_program

    raw = compose_case(
        "qwen3.5-35b-a3b", hardware="a5-pod-dt", quant="w8a8",
        tp=1, dp=2, ep=2, layers=2, prompt_tokens=64, output_tokens=8,
        concurrency=1, graph_capture_mode="none", mtp_proposal_tokens=3,
        mtp_acceptance_length=2.0,
    )
    program = model_program(parse_case(raw), phase="decode", tokens=4,
                            context=128, sequences=1, selected_replica=0)
    ops = program.work.ops
    dispatch = next(op for op in ops if op.recipe == "expert_dispatch")
    gate = next(op for op in ops if op.recipe == "grouped_gate_up")
    verification_rows = 2 * (1 + 3)
    assert dispatch.local_shape[0] == verification_rows
    assert gate.local_shape[0] == verification_rows


def test_c8_sparse_batch_reuses_external_bytes_preserving_gathers():
    shared = glm(prefix_cache_mode="shared", prefix_read_mode="batch_shared",
                 prefix_cache_key="shared")
    shared["model"]["quant"]["kv_dtype"] = "fp8_e4m3"
    for request in shared["workload"]["requests"]:
        request["prefix_tokens"] = 4096
    private = deepcopy(shared)
    private["runtime"]["prefix_cache_mode"] = "private"

    def attention(raw):
        return [op for op in target_graphs(
            raw, scope="body", phase="decode", tokens=6, context=4096,
            sequences=8)[0] if op.family == "attention"]

    batched, control = attention(shared), attention(private)
    assert len(batched) == len(control) == 1
    a, b = batched[0], control[0]
    assert a.mte2_bytes == b.mte2_bytes and a.flops == b.flops
    assert a.hbm_bytes < b.hbm_bytes
    meta = dict(a.metadata)
    assert meta["indexed_attention_geometry"]["sequences"] == ((6, 4096),) * 8
    # NoPE C8 resolves an FP8 MLA checkpoint to the 128-element block row:
    # 512 B payload + four FP16 scales per 512-element latent row.
    from mindie_sim.catalog.cache_format import block_quantized_row_bytes
    row_bytes = block_quantized_row_bytes(512)
    assert row_bytes == 520
    assert meta["prefix_read_hbm_bytes"] == (4096 + 8 * 6) * row_bytes
    assert meta["attention_tensor_view"]["query_shape"] == (48, 64, 512)
    assert meta["attention_traffic_plan"]["hbm_read_bytes"] == meta["prefix_read_hbm_bytes"]
    assert meta["attention_traffic_plan"]["private_refill_bytes"] == meta["prefix_read_l2_core_bytes"]
    report = operator_costs(shared, batched)
    assert report["operators"][0]["operator"]["hbm_bytes"] == a.hbm_bytes
    eager = deepcopy(shared)
    eager["runtime"]["metadata"]["raw_eager"] = True
    assert len(attention(eager)) == 8


def test_sparse_unknown_selection_capacity_bound():
    view = SimpleNamespace(q_len=1, context=100, kv_len=100,
                           prefix_tokens=100, prefix_cache_key="p", padding=False)
    runtime = SimpleNamespace(prefix_cache_mode="shared",
                              prefix_read_mode="batch_shared", metadata={})
    original = replace(op("kv", "kv"), hbm_bytes=2000, mte2_bytes=2000)
    # Two unknown 100-row selections share a 100-row candidate plane.
    for capacity, expected in ((0, 2000), (500, 1500), (1000, 1000)):
        result = apply_sparse_prefix_reads(
            original, views=(view, view), row_bytes=10, runtime=runtime,
            hierarchy=SimpleNamespace(effective_l2_bytes=capacity),
            kv_hbm_bytes=2000, kv_mte2_bytes=2000)
        assert result.hbm_bytes == expected
        assert result.mte2_bytes == 2000


def test_a5_streaming_and_multipass_share_sustainable_dram():
    for sku, (_, _, _, _, peak) in A5.items():
        case = sku_case(sku)
        device = case.machine.dies[0]
        modes = dict(device.service_modes)
        for role in ("mte2_aic_dram", "mte2_aiv_dram"):
            assert modes[role][1] == pytest.approx(0.8316 * peak)
        for family in ("elementwise", "rmsnorm", "quantize"):
            operator = replace(
                op("stream", "stream"), family=family, dtype="fp32",
                hbm_bytes=2e9, mte2_bytes=1e9, mte3_bytes=1e9, vector_ops=1)
            result = graph_costs(case, (operator,), isolated=False)
            mbu = operator.hbm_bytes / (result["duration_ms"] / 1000) / peak
            assert 0.82 < mbu <= 0.8316
            assert result["execution"]["work"]["hbm_bytes"] == operator.hbm_bytes


def test_a5_pod_ports_protocols_and_explicit_legacy():
    machine = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950DT_9582",
        "topology_preset": "a5_pod", "num_devices": 128})
    assert dict(machine.collective_protocols) == {
        "alltoall": "concurrent", "alltoallv": "concurrent"}
    intra = [link for link in machine.links if link.name.startswith("d2d_intra_node")]
    assert intra and all(link.semantics == "per_port" for link in intra)
    parallel = ParallelSpec(tp=1, dp=128, ep=128)
    operator = exchange(1024, axis="ep")
    for mode, rounds in (("topology", 1), ("legacy", 127)):
        topology = Topology(machine, parallel, collective_startup_mode=mode)
        assert topology.startup_descriptor(operator, 128).startup_rounds == rounds


def test_concurrent_efficiency_without_graph_uses_complete_cohort():
    raw = cohort_case(4)
    raw["runtime"]["kernels"] = {}
    raw["runtime"]["collective_startup_mode"] = "concurrent"
    report = analyze_network_efficiency(
        raw, scope="body", phase="decode", tokens=1, context=16,
        sequences=1, run_graph=False)
    collective = [row for row in report["operators"]
                  if row["family"] in {"alltoall", "alltoallv"}]
    assert collective and all(row["program"] == "StartupWaves" for row in collective)
    graphs = target_graphs(raw, scope="body", phase="decode", tokens=1, context=16)
    assert all("steps" not in dict(op.metadata) for op in graphs[0]
               if op.recipe in {"expert_dispatch", "expert_combine"})


def test_topology_defaults_execute_direct_peers():
    raw = cohort_case(4)
    raw["runtime"]["kernels"] = {}
    raw["runtime"]["collective_startup_mode"] = "topology"
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582",
                      "topology_preset": "a5_pod", "num_devices": 4}
    graphs = target_graphs(raw, scope="body", phase="decode", tokens=1, context=16)
    report = operator_costs(raw, graphs)
    rows = [row for row in report["operators"]
            if row["operator"]["recipe"] in {"expert_dispatch", "expert_combine"}]
    assert rows and all(row["program"] == "DirectPeerCollective" for row in rows)
    assert all(row["duration_ms"] < 0.003 for row in rows)
