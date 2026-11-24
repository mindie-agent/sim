"""Live & causal NPU subgraph parity and ProbedAbiRegistry verification tests.

[Role] Verify causal 3-layer trace correlation (strict unique
  ``(rank_id, stream_id, flow_id)`` and ACLGraph node-table joins), streaming
  ``trace_view.json`` ingestion, non-tautological Case -> PredictionSpine ->
  PreparedStep parity gating, strict 1:1 occurrence/task-type/completion-order
  checks, honest offline inventory auditing, and sealed CANN + non-CANN
  ``ProbedAbiRegistry`` legal + neighboring-illegal admission checks.
[Invariants]
  * Marked ``pytest.mark.live_npu`` per ``tests/TAXONOMY.toml``.
  * Zero NPU device allocation; deterministic ``tmp_path`` roundtrips with zero
    ``pytest.skip``.
[Timeouts & Locks] Fast in-memory & tmp_path checks (< 5 s); no locks.
[Side-Effect Policy] Read-only except pytest ``tmp_path``; zero host pollution.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_dir, default_abi_registry
from mindie_sim.observations.subgraph_parity import (
    ObservedKernelNode,
    ObservedRankDAG,
    PredictedKernelNode,
    SubgraphParityChecker,
    classify_runtime_errors,
    correlate_trace_view_events,
    iter_trace_view_events,
    load_offline_a3_capture_dags,
)
from mindie_sim.schema.abi import AbiQueryKey, OperandPortSig
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.live_npu

_SAMPLE_SKU_A3 = "Ascend910_9392"
_SAMPLE_SKU_A5 = "Ascend950DT_9582"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample_trace_events() -> list[dict]:
    return [
        {"ph": "s", "cat": "HostToDevice", "id": "f1", "ts": 100.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnConstantPadNd", "cpu_op": "aten::constant_pad_nd"}},
        {"ph": "s", "cat": "HostToDevice", "id": "f2", "ts": 110.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnConstantPadNd", "cpu_op": "aten::constant_pad_nd"}},
        {"ph": "s", "cat": "HostToDevice", "id": "f3", "ts": 120.0, "tid": "1",
         "args": {"stream_id": "comm0", "cann_api": "HcclReduceScatter", "cpu_op": "vllm::reduce_scatter"}},
        {"ph": "X", "cat": "kernel", "name": "MemSet", "ts": 101.0, "dur": 2.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE", "flow_id": "f1", "Input Shapes": "4,7168"}},
        {"ph": "X", "cat": "kernel", "name": "PadV3", "ts": 104.0, "dur": 3.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE", "flow_id": "f2", "Input Shapes": "4,7168"}},
        {"ph": "X", "cat": "kernel", "name": "hcom_reduceScatter_", "ts": 122.0, "dur": 25.0, "tid": "1",
         "args": {
             "stream_id": "comm0", "stream_lane": "comm", "Task Type": "COMMUNICATION",
             "flow_id": "f3", "collective_group": "tp_group", "comm_ranks": [0, 1, 2, 3],
         }},
    ]


def _port(
    role: str, dtype: str, shape: tuple[int, ...] = (16, 128),
    layout: str = "ND", quant_mode: str = "dense", scale_dtype: str = "none",
) -> OperandPortSig:
    return OperandPortSig(
        role=role, dtype=dtype, layout=layout, transposed=False, shape=shape,
        quant_mode=quant_mode, group_shape=(1,), scale_dtype=scale_dtype, scale_shape=(),
    )


def _abi_query(
    arch: str, abi_id: str, motif: str,
    operands: tuple[OperandPortSig, ...], outputs: tuple[OperandPortSig, ...],
    *, world_size: int = 1, placement: str = "local", activation: str = "none",
) -> AbiQueryKey:
    sku = "ascend950dt_9582" if arch == "A5" else "ascend910_9392"
    return AbiQueryKey(
        soc_arch=arch, sku_id=sku,  # type: ignore[arg-type]
        abi_id=abi_id, graph_motif=motif, model_scope="target", draft_quant_mode="none",
        pr_track="shipping", operands=operands, outputs=outputs,
        rank_placement=placement, world_size=world_size,  # type: ignore[arg-type]
        comm_group_id="local" if world_size == 1 else "tp",
        comm_domain_width=world_size, comm_dtype="none" if world_size == 1 else "bf16",
        reduction_kind="none", activation_kind=activation, activation_params=(),
        has_bias=False, has_active_mask=False, dispatch_quant_mode=-1,
        shared_experts_per_rank=0, top_k=0, moe_expert_num=0, qk_rope_head_dim=64,
        mla_variant="none", sfa_dcp_active=False, pcp_size=1, kv_cache_dtype="bf16",
        has_kpool_indexer=False, dcp_size=1, is_act_and_mul=False, enabled_stack_passes=frozenset(),
    )


def test_strict_flow_stream_join_and_zero_proximity_bypass() -> None:
    # Cross-stream flow ("s" on stream 0, kernel on stream 1) MUST NOT be labeled exact.
    cross_stream = [
        {"ph": "s", "cat": "HostToDevice", "id": "f1", "ts": 10.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnMatmul"}},
        {"ph": "X", "cat": "kernel", "name": "MatMulV3", "ts": 11.0, "dur": 5.0, "tid": "1",
         "args": {"stream_id": "1", "Task Type": "AI_CORE", "flow_id": "f1"}},
    ]
    cross_dag = correlate_trace_view_events(cross_stream, rank_id=0, phase="decode", exec_mode="eager")
    assert cross_dag.exact_causal_attribution_rate == 0.0
    assert cross_dag.nodes[0].attribution_kind != "exact_flow_id"

    # 19 exact + 1 proximity-only (95% rate) MUST STILL FAIL causal parity!
    events: list[dict] = []
    preds: list[PredictedKernelNode] = []
    for i in range(19):
        t = float(i * 10 + 10)
        events.append({"ph": "s", "cat": "HostToDevice", "id": f"e{i}", "ts": t, "tid": "0",
                       "args": {"stream_id": "0", "cann_api": "aclnnMatmul"}})
        events.append({"ph": "X", "cat": "kernel", "name": "MatMulV3", "ts": t + 1.0, "dur": 2.0, "tid": "0",
                       "args": {"stream_id": "0", "Task Type": "AI_CORE", "flow_id": f"e{i}"}})
        preds.append(PredictedKernelNode(
            node_id=f"mm_{i}", rank_id=0, stream_lane="main", task_type="AI_CORE",
            api_or_symbol="aclnnMatmul", expected_kernels=("MatMulV3",),
        ))
    events.append({"ph": "X", "cat": "cpu_op", "name": "aten::mul", "ts": 300.0, "dur": 1.0, "tid": "0"})
    events.append({"ph": "X", "cat": "kernel", "name": "Mul", "ts": 301.0, "dur": 1.0, "tid": "0",
                   "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE"}})
    dag = correlate_trace_view_events(events, rank_id=0, phase="decode", exec_mode="eager")
    assert dag.exact_causal_attribution_rate == pytest.approx(0.95)
    assert len(dag.ambiguous_matches) == 1
    rep = SubgraphParityChecker().compare(
        signature_id="prox_fail", hardware_sku=_SAMPLE_SKU_A5, device_arch="A5",
        config_payload={}, observed_dags=(dag,), predicted_nodes=preds,
    )
    assert not rep.passed and not rep.causal_parity_certified
    assert any(d.code == "UNATTRIBUTED_KERNEL" for d in rep.launch_diffs)


def test_one_to_one_occurrence_task_type_completion_order_and_per_rank_checks() -> None:
    checker = SubgraphParityChecker()
    # 1. One AI_CORE kernel MUST NOT satisfy two (or even one) AI_CPU predictions.
    single_aicore_dag = correlate_trace_view_events([
        {"ph": "s", "cat": "HostToDevice", "id": "f1", "ts": 10.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnSparseFlashMlaMetadata"}},
        {"ph": "X", "cat": "kernel", "name": "SparseFlashMlaMetadata", "ts": 11.0, "dur": 4.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_CORE", "flow_id": "f1"}},
    ], rank_id=0)
    two_aicpu_preds = [
        PredictedKernelNode(node_id="m1", rank_id=0, stream_lane="main", task_type="AI_CPU",
                            api_or_symbol="SparseFlashMlaMetadata", expected_kernels=("SparseFlashMlaMetadata",),
                            is_ai_cpu_metadata=True),
        PredictedKernelNode(node_id="m2", rank_id=0, stream_lane="main", task_type="AI_CPU",
                            api_or_symbol="SparseFlashMlaMetadata", expected_kernels=("SparseFlashMlaMetadata",),
                            is_ai_cpu_metadata=True),
    ]
    rep_tt = checker.compare(
        signature_id="tt_mismatch", hardware_sku=_SAMPLE_SKU_A5, device_arch="A5",
        config_payload={}, observed_dags=(single_aicore_dag,), predicted_nodes=two_aicpu_preds,
    )
    assert not rep_tt.passed
    assert sum(1 for d in rep_tt.launch_diffs if d.code == "MISSING_AI_CPU_METADATA") == 2

    # 2. One AI_CPU occurrence MUST NOT satisfy two AI_CPU predictions (1:1 consumption).
    one_aicpu_plus_npu = correlate_trace_view_events([
        {"ph": "s", "cat": "HostToDevice", "id": "f1", "ts": 10.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnSparseFlashMlaMetadata"}},
        {"ph": "X", "cat": "kernel", "name": "SparseFlashMlaMetadata", "ts": 11.0, "dur": 4.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_CPU", "flow_id": "f1"}},
        {"ph": "s", "cat": "HostToDevice", "id": "f2", "ts": 20.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnMatmul"}},
        {"ph": "X", "cat": "kernel", "name": "MatMulV3", "ts": 21.0, "dur": 4.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_CORE", "flow_id": "f2"}},
    ], rank_id=0)
    rep_reuse = checker.compare(
        signature_id="no_reuse", hardware_sku=_SAMPLE_SKU_A5, device_arch="A5",
        config_payload={}, observed_dags=(one_aicpu_plus_npu,), predicted_nodes=two_aicpu_preds,
    )
    assert not rep_reuse.passed
    assert sum(1 for d in rep_reuse.launch_diffs if d.code == "MISSING_AI_CPU_METADATA") == 1

    # 3. Predecessor ending AFTER consumer starts MUST fail DEPENDENCY_ORDER_MISMATCH.
    overlap_dag = correlate_trace_view_events([
        {"ph": "s", "cat": "HostToDevice", "id": "f1", "ts": 10.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnMatmul"}},
        {"ph": "X", "cat": "kernel", "name": "MatMulV3", "ts": 10.0, "dur": 20.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_CORE", "flow_id": "f1"}},
        {"ph": "s", "cat": "HostToDevice", "id": "f2", "ts": 15.0, "tid": "1",
         "args": {"stream_id": "1", "cann_api": "aclnnSwiGlu"}},
        {"ph": "X", "cat": "kernel", "name": "SwiGlu", "ts": 15.0, "dur": 5.0, "tid": "1",
         "args": {"stream_id": "1", "Task Type": "AI_VECTOR_CORE", "flow_id": "f2"}},
    ], rank_id=0)
    rep_overlap = checker.compare(
        signature_id="dep_overlap", hardware_sku=_SAMPLE_SKU_A5, device_arch="A5",
        config_payload={}, observed_dags=(overlap_dag,),
        predicted_nodes=[
            PredictedKernelNode(node_id="a", rank_id=0, stream_lane="main", task_type="AI_CORE",
                                api_or_symbol="aclnnMatmul", expected_kernels=("MatMulV3",)),
            PredictedKernelNode(node_id="b", rank_id=0, stream_lane="main", task_type="AI_VECTOR_CORE",
                                api_or_symbol="aclnnSwiGlu", expected_kernels=("SwiGlu",), predecessors=("a",)),
        ],
    )
    assert not rep_overlap.passed
    assert any(d.code == "DEPENDENCY_ORDER_MISMATCH" for d in rep_overlap.dag_order_diffs)

    # 4. Empty rank alongside an active rank MUST FAIL, and empty predicted_nodes MUST FAIL.
    active_r0 = correlate_trace_view_events(_sample_trace_events(), rank_id=0)
    empty_r1 = ObservedRankDAG(
        rank_id=1, phase="decode", exec_mode="eager", nodes=(),
        exact_causal_attribution_rate=0.0, ambiguous_matches=(),
        unattributed_device_kernels=(), npu_device_kernel_count=0, ai_cpu_kernel_count=0,
    )
    rep_empty_r1 = checker.compare(
        signature_id="empty_r1", hardware_sku=_SAMPLE_SKU_A3, device_arch="A3",
        config_payload={}, observed_dags=(active_r0, empty_r1),
        predicted_nodes=[PredictedKernelNode(node_id="pad", rank_id=0, stream_lane="main",
                                             task_type="AI_VECTOR_CORE", api_or_symbol="aclnnConstantPadNd")],
    )
    assert not rep_empty_r1.passed
    assert any(d.rank_id == 1 and d.code == "MISSING_OR_EXTRA_LAUNCH" for d in rep_empty_r1.launch_diffs)

    rep_empty_pred = checker.compare(
        signature_id="empty_pred", hardware_sku=_SAMPLE_SKU_A3, device_arch="A3",
        config_payload={}, observed_dags=(active_r0,), predicted_nodes=(),
    )
    assert not rep_empty_pred.passed
    # Verify full structured ParityDiff dict serialization (P2-1).
    serialized = rep_empty_pred.to_dict()
    assert serialized["launch_diffs"][0]["code"] == "MISSING_OR_EXTRA_LAUNCH"
    assert "node_id" in serialized["launch_diffs"][0]


def test_error_classification_exact_and_case_driven_live_gate(tmp_path: Path) -> None:
    missing_so, missing_py = classify_runtime_errors((
        "OSError: libopapi.so: undefined symbol: aclnnVllmAscendQuantLightningIndexerV2",
        "ModuleNotFoundError: No module named 'ops'",
    ))
    assert missing_so == ("aclnnVllmAscendQuantLightningIndexerV2",)
    assert missing_py == ("ops",)

    # Streaming trace_view.json ingestion + independent Case -> PredictionSpine -> PreparedStep gate.
    from tools.live_subgraph_parity_gate import build_predicted_nodes_from_case, evaluate_live_dir

    case_path = _REPO_ROOT / "examples" / "cases" / "balanced-experts-control.json"
    prepared, preds = build_predicted_nodes_from_case(case_path, phase="decode", tokens=4, context=1024, sequences=4)
    assert prepared.graph_id and len(preds) > 0

    # Feed an unrelated AI_CORE trace against Case-lowered predictions -> MUST FAIL!
    live_dir = tmp_path / "live_run"
    live_dir.mkdir()
    unrelated_trace = live_dir / "trace_view.json"
    unrelated_trace.write_text(json.dumps({"traceEvents": [
        {"ph": "s", "cat": "HostToDevice", "id": "u1", "ts": 10.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnUnrelatedKernel"}},
        {"ph": "X", "cat": "kernel", "name": "UnrelatedKernel", "ts": 12.0, "dur": 3.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_CORE", "flow_id": "u1"}},
    ]}), encoding="utf-8")
    assert len(list(iter_trace_view_events(unrelated_trace))) == 2
    res_fail = evaluate_live_dir(live_dir, device_arch="A3", hardware_sku=_SAMPLE_SKU_A3, case=case_path,
                                 phase="decode", tokens=4, context=1024, sequences=4)
    assert res_fail["passed"] is False and res_fail["causal_parity_certified"] is False
    with pytest.raises(ModelingError, match="live_parity_requires_independent_predictions"):
        evaluate_live_dir(live_dir, device_arch="A3", hardware_sku=_SAMPLE_SKU_A3)

    # Matching 2-rank trace built strictly from Case-lowered predictions passes causal_dag_parity;
    # reversing two dependent kernels fails DEPENDENCY_ORDER_MISMATCH; duplicate rank trace fails!
    match_dir = tmp_path / "live_match"
    for r in sorted({p.rank_id for p in preds}):
        rdir = match_dir / f"rank{r}"
        rdir.mkdir(parents=True)
        evs: list[dict] = []
        for idx, pnode in enumerate(p for p in preds if p.rank_id == r):
            t = float((idx + 1) * 10)
            stream = "comm0" if pnode.stream_lane == "comm" else "0"
            tts = pnode.expected_task_types if len(pnode.expected_task_types) == len(pnode.expected_kernels) else (pnode.task_type,) * len(pnode.expected_kernels)
            for k_idx, (kname, tt) in enumerate(zip(pnode.expected_kernels, tts)):
                fid = f"r{r}_n{idx}_k{k_idx}"
                evs.append({"ph": "s", "cat": "HostToDevice", "id": fid, "ts": t, "tid": stream,
                            "args": {"stream_id": stream, "cann_api": pnode.api_or_symbol, "vllm_scope": pnode.vllm_scope}})
                evs.append({"ph": "X", "cat": "kernel", "name": kname, "ts": t + k_idx * 2.0, "dur": 1.0,
                            "tid": stream, "args": {"stream_id": stream, "stream_lane": pnode.stream_lane,
                                                    "Task Type": tt, "flow_id": fid,
                                                    "collective_group": pnode.collective_group,
                                                    "comm_ranks": list(pnode.comm_ranks),
                                                    "Input Shapes": pnode.input_shapes,
                                                    "vllm_scope": pnode.vllm_scope}})
        (rdir / "trace_view.json").write_text(json.dumps({"traceEvents": evs}), encoding="utf-8")
    res_ok = evaluate_live_dir(match_dir, device_arch="A3", hardware_sku=_SAMPLE_SKU_A3, case=case_path,
                               phase="decode", tokens=4, context=1024, sequences=4)
    assert res_ok["passed"] is True and res_ok["causal_parity_certified"] is True

    # Duplicate rank trace in live_dir MUST raise ModelingError!
    dup_dir = tmp_path / "dup_rank_dir"
    shutil.copytree(match_dir, dup_dir)
    (dup_dir / "dup" / "rank0").mkdir(parents=True)
    shutil.copy2(dup_dir / "rank0" / "trace_view.json", dup_dir / "dup" / "rank0" / "trace_view.json")
    with pytest.raises(ModelingError, match="duplicate_rank_trace_detected"):
        evaluate_live_dir(dup_dir, device_arch="A3", hardware_sku=_SAMPLE_SKU_A3, case=case_path,
                          phase="decode", tokens=4, context=1024, sequences=4)

    # Swap timestamps of norm1 (idx 0) and qkv (idx 1) on rank 0 -> MUST FAIL DEPENDENCY_ORDER_MISMATCH!
    r0_path = match_dir / "rank0" / "trace_view.json"
    r0_data = json.loads(r0_path.read_text(encoding="utf-8"))
    k_evs = [e for e in r0_data["traceEvents"] if e["ph"] == "X"]
    k_evs[0]["ts"], k_evs[1]["ts"] = k_evs[1]["ts"], k_evs[0]["ts"]
    r0_path.write_text(json.dumps(r0_data), encoding="utf-8")
    res_misordered = evaluate_live_dir(match_dir, device_arch="A3", hardware_sku=_SAMPLE_SKU_A3, case=case_path,
                                       phase="decode", tokens=4, context=1024, sequences=4)
    assert res_misordered["passed"] is False
    assert any(d["code"] == "DEPENDENCY_ORDER_MISMATCH" for d in res_misordered["dag_order_diffs"])


def test_offline_bundle_non_tautological_pad_removal_fails(tmp_path: Path) -> None:
    from tools.live_subgraph_parity_gate import build_offline_expected_nodes, evaluate_offline_bundle

    # Derive expected 1-step inventory strictly from real compose_case -> spine.lower (ZERO toy specs!).
    expected = build_offline_expected_nodes("gemma4_26b_a4b_swa_moe", device_arch="A3")
    assert expected is not None and len(expected) >= 5
    by_type: dict[str, int] = {}
    sigs: list[dict[str, object]] = []
    for pnode in expected:
        if pnode.min_count <= 0:
            continue
        for kname in pnode.expected_kernels:
            by_type[kname] = by_type.get(kname, 0) + pnode.min_count
            sigs.append({"signature": f"{kname}|{pnode.api_or_symbol}|{pnode.task_type}",
                         "count": pnode.min_count, "total_dur_us": 5.0})
    a3_ok = {
        "environment": {"vllm_ascend_commit": "0c99d3fc5"},
        "captures": {
            "gemma4_26b_a4b_swa_moe": {
                "config": {"aclgraph": False},
                "real_npu_kernels": sum(by_type.values()),
                "task_type_counts": {"MIX_AIC": 4, "MIX_AIV": 4, "AI_VECTOR_CORE": 8},
                "representative_decode_step": {"by_type": by_type},
                "top_op_signatures": sigs,
            },
        },
    }
    a3_file = tmp_path / "a3_ok.json"
    a3_file.write_text(json.dumps(a3_ok), encoding="utf-8")
    _, dags, _ = load_offline_a3_capture_dags(a3_file)
    assert dags["gemma4_26b_a4b_swa_moe"].is_offline_summary is True
    assert dags["gemma4_26b_a4b_swa_moe"].exact_causal_attribution_rate == 0.0
    ok_sum = evaluate_offline_bundle(a3_path=a3_file, a5_path=None)
    assert ok_sum["all_passed"] is True and ok_sum["causal_parity_certified"] is False

    # Removing ANY multi-kernel or single-kernel member (e.g. PadV3 or first lowered kernel) MUST FAIL!
    target_k = "PadV3" if "PadV3" in by_type else next(iter(by_type))
    a3_bad = json.loads(json.dumps(a3_ok))
    a3_bad["captures"]["gemma4_26b_a4b_swa_moe"]["representative_decode_step"]["by_type"].pop(target_k)
    a3_bad["captures"]["gemma4_26b_a4b_swa_moe"]["top_op_signatures"] = [
        s for s in a3_bad["captures"]["gemma4_26b_a4b_swa_moe"]["top_op_signatures"]
        if not s["signature"].startswith(f"{target_k}|")
    ]
    a3_bad_file = tmp_path / "a3_bad.json"
    a3_bad_file.write_text(json.dumps(a3_bad), encoding="utf-8")
    bad_sum = evaluate_offline_bundle(a3_path=a3_bad_file, a5_path=None)
    assert bad_sum["all_passed"] is False and len(bad_sum["failed"]) == 1

    # Non-catalog architecture (seed_20) is honestly reported as unmodeled_catalog_model, NEVER fake-passed!
    a5_unmodeled = tmp_path / "a5_seed20.json"
    a5_unmodeled.write_text(json.dumps({
        "vllm_ascend_commit": "c12397a08",
        "captures": {"seed_20": {"real_npu_kernels": 1, "decode_op_sequence_summary": [
            {"kernel": "MatMulV3", "core": "AI_CORE", "count": 1},
        ]}},
    }), encoding="utf-8")
    u_sum = evaluate_offline_bundle(a3_path=None, a5_path=a5_unmodeled)
    assert u_sum["unmodeled_catalog_signatures"] == 1 and u_sum["passed_signatures"] == 0


def test_probed_abi_registry_strict_joint_io_shapes_and_manifest_seal(tmp_path: Path) -> None:
    reg = default_abi_registry()
    # 1. Invented symbol with real motif -> rejected_unprobed.
    v_fake = reg.verify(_abi_query(
        "A5", "NoSuchShippingSymbol", "npu_rms_norm",
        (_port("self", "bf16"), _port("gamma", "bf16")), (_port("y", "bf16"), _port("rstd", "fp32")),
    ))
    assert v_fake.status == "rejected_abi" and "rejected_unprobed" in v_fake.rejection_reason
    # 2. AllGatherMatmulV2: legal (16,512)x(512,64)->(32,64) at ws=2 passes; wrong output/rank/transpose/K/dtype fail.
    assert reg.verify(_abi_query(
        "A5", "aclnnAllGatherMatmulV2", "allgather_matmul",
        (_port("x1", "bf16", (16, 512)), _port("x2", "bf16", (512, 64))), (_port("y", "bf16", (32, 64)),),
        world_size=2, placement="ub_ccu_intramachine",
    )).status == "admitted_shipping"
    for bad_in, bad_out in (
        ((_port("x1", "bf16", (16, 512)), _port("x2", "bf16", (512, 64))), (_port("y", "bf16", (16, 64)),)),
        ((_port("x1", "bf16", (16, 512)), _port("x2", "bf16", (512, 64))), (_port("y", "bf16", (987, 64)),)),
        ((_port("x1", "bf16", (512,)), _port("x2", "bf16", (13, 64))), (_port("y", "bf16", (32, 64)),)),
        ((_port("x1", "bf16", (2, 16, 512)), _port("x2", "bf16", (512, 64))), (_port("y", "bf16", (32, 64)),)),
        ((OperandPortSig("x1", "bf16", "ND", True, (16, 512), "dense", (1,), "none", ()),
          _port("x2", "bf16", (512, 64))), (_port("y", "bf16", (32, 64)),)),
        ((_port("x1", "bf16", (16, 512)), _port("x2", "bf16", (13, 64))), (_port("y", "bf16", (32, 64)),)),
        ((_port("x1", "bf16", (16, 512)), _port("x2", "bf16", (512, 64))), (_port("y", "fp16", (32, 64)),)),
        ((_port("x1", "bf16", ()), _port("x2", "bf16", (512, 64))), (_port("y", "bf16", ()),)),
        ((_port("x1", "bf16", (16777216, 512)), _port("x2", "bf16", (512, 64))), (_port("y", "bf16", (33554432, 64)),)),
    ):
        assert reg.verify(_abi_query(
            "A5", "aclnnAllGatherMatmulV2", "allgather_matmul", bad_in, bad_out,
            world_size=2, placement="ub_ccu_intramachine",
        )).status == "rejected_abi"
    # A5 allows N=0 empty tensor on AGMM, while A3 rejects N=0; A5 32 GiB allowed on ub_aicpu_supernode.
    assert reg.verify(_abi_query(
        "A5", "aclnnAllGatherMatmulV2", "allgather_matmul",
        (_port("x1", "bf16", (16, 512)), _port("x2", "bf16", (512, 0))), (_port("y", "bf16", (32, 0)),),
        world_size=2, placement="ub_ccu_intramachine",
    )).status == "admitted_shipping"
    assert reg.verify(_abi_query(
        "A3", "aclnnAllGatherMatmulV2", "allgather_matmul",
        (_port("x1", "bf16", (16, 512)), _port("x2", "bf16", (512, 0))), (_port("y", "bf16", (32, 0)),),
        world_size=2, placement="hccs_supernode_a3",
    )).status == "rejected_abi"
    # 3. Grounded 4-backend callable checks: CANN, torch_npu, vllm_ascend_csrc, Triton.
    assert reg.verify(_abi_query(
        "A5", "aclnnTextGatedResidual", "text_gated_residual",
        (_port("x", "bf16", (2, 16, 128)),), (_port("mixed", "bf16", (2, 16, 128)),),
    )).status == "rejected_abi"
    assert reg.verify(_abi_query(
        "A5", "torch_npu.npu_moe_gating_top_k", "moe_gating_top_k",
        (_port("x", "bf16", (4, 128)),), (_port("y", "bf16", (4, 8)), _port("expert_idx", "int32", (4, 8)), _port("out", "fp32", (4, 128))),
    )).status == "admitted_shipping"
    assert reg.verify(_abi_query(
        "A5", "torch_npu.npu_moe_gating_top_k", "moe_gating_top_k",
        (_port("x", "bf16", (4, 128)), _port("bias", "bf16", (4, 128))),
        (_port("y", "bf16", (4, 8)), _port("expert_idx", "int32", (4, 8)), _port("out", "fp32", (4, 128))),
    )).status == "rejected_abi"
    assert reg.verify(_abi_query(
        "A5", "SparseFlashMla", "sparse_flash_mla",
        (_port("query", "fp16"), _port("kv_cache", "fp16"), _port("topk_indices", "int32", quant_mode="none")),
        (_port("out", "fp16"),),
    )).status == "rejected_abi"
    ar_in = tuple(_port(r, "bf16") for r in ("prefix_sum", "block_residual", "proj_weight", "norm_weight", "addend", "output_norm"))
    ar_out = tuple(_port(r, "bf16") for r in ("hidden_states", "prefix_out", "materialized"))
    assert reg.verify(_abi_query("A3", "AttnResFwd", "attn_res_prefix", ar_in, ar_out)).status == "admitted_shipping"
    assert reg.verify(_abi_query(
        "A5", "_prepare_local_sort_inputs", "sharded_sort",
        (_port("indices", "int32", quant_mode="none"), _port("scores", "fp32")),
        (_port("keys", "fp32"), _port("values", "fp32")),
    )).status == "admitted_shipping"
    assert reg.verify(_abi_query(
        "A5", "_prepare_local_sort_inputs", "sharded_sort",
        (_port("indices", "int32", quant_mode="none"), _port("scores", "fp32")),
        (_port("keys", "fp32"), _port("values", "int32")),
    )).status == "rejected_abi"
    # 4. Removing sealed_sha256 from abi_schema_manifest.json MUST fail; single-switch publication keeps active+rollback.
    from tools.probe_shipping_abi import publish_sealed_dir, verify_sealed_dir

    abi_copy = tmp_path / "abi"
    shutil.copytree(default_abi_dir(), abi_copy)
    snaps = {a: reg.get_snapshot(a) for a in ("A2", "A3", "A5")}
    publish_sealed_dir(abi_copy, snaps)
    assert verify_sealed_dir(abi_copy)["A5"]["total"] == len(snaps["A5"].entries)
    manifest_p = abi_copy / "abi_schema_manifest.json"
    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    manifest_data.pop("sealed_sha256")
    manifest_p.write_text(json.dumps(manifest_data), encoding="utf-8")
    with pytest.raises(ModelingError, match="abi_manifest_sealed_sha256_missing"):
        verify_sealed_dir(abi_copy)
    with pytest.raises(ModelingError, match="abi_manifest_sealed_sha256_missing"):
        ProbedAbiRegistry.from_directory(abi_copy)


def test_causal_parity_hardening_wrong_name_order_chrome_flow_and_truncated_json(tmp_path: Path) -> None:
    from mindie_sim.observations.subgraph_parity import evaluate_trace_view_stream
    from tools.live_subgraph_parity_gate import evaluate_live_dir

    reg = default_abi_registry()
    checker = SubgraphParityChecker(reg)
    pad_pred = (PredictedKernelNode(
        node_id="sp.pad", rank_id=0, stream_lane="main", task_type="MIX_AIV",
        api_or_symbol="aclnnConstantPadNd", expected_kernels=("MemSet", "PadV3"),
        expected_task_types=("AI_VECTOR_CORE", "MIX_AIV"), vllm_scope="l0.sp_pad", input_shapes="4,7168",
    ),)
    # Reversed, prefix-bypass, extra launch, overlapping subkernels, wrong scope, stripped shape, unknown extra ALL fail.
    for ev_spec in (
        (("PadV3", 5.0, "MIX_AIV", "l0.sp_pad", "4,7168"), ("MemSet", 15.0, "AI_VECTOR_CORE", "l0.sp_pad", "4,7168")),
        (("MemSet", 5.0, "AI_VECTOR_CORE", "l0.sp_pad", "4,7168"), ("PadV3Incompatible", 15.0, "MIX_AIV", "l0.sp_pad", "4,7168")),
        (("MemSet", 5.0, "AI_VECTOR_CORE", "l0.sp_pad", "4,7168"), ("PadV3", 15.0, "MIX_AIV", "l0.sp_pad", "4,7168"), ("ExtraJunk", 25.0, "UNKNOWN", "l0.sp_pad", "")),
        (("MemSet", 5.0, "AI_VECTOR_CORE", "l0.sp_pad", "4,7168"), ("PadV3", 5.2, "MIX_AIV", "l0.sp_pad", "4,7168")),
        (("MemSet", 5.0, "AI_VECTOR_CORE", "wrong.scope", "4,7168"), ("PadV3", 15.0, "MIX_AIV", "wrong.scope", "4,7168")),
        (("MemSet", 5.0, "AI_VECTOR_CORE", "l0.sp_pad", ""), ("PadV3", 15.0, "MIX_AIV", "l0.sp_pad", "")),
    ):
        evs: list[dict] = []
        for idx, (kname, ts, tt, sc, shp) in enumerate(ev_spec):
            fid = f"f{idx}"
            evs.extend((
                {"ph": "s", "cat": "HostToDevice", "id": fid, "ts": ts - 1.0, "tid": "0",
                 "args": {"stream_id": "0", "cann_api": "aclnnConstantPadNd", "vllm_scope": sc}},
                {"ph": "f", "cat": "HostToDevice", "id": fid, "ts": ts, "tid": "0", "args": {"stream_id": "0"}},
                {"ph": "X", "cat": "kernel", "name": kname, "ts": ts, "dur": 1.0, "tid": "0",
                 "args": {"stream_id": "0", "Task Type": tt, "vllm_scope": sc, "Input Shapes": shp}},
            ))
        dag = correlate_trace_view_events(evs, rank_id=0, phase="decode", exec_mode="eager")
        assert dag.exact_causal_attribution_rate == 1.0
        assert not checker.compare(
            signature_id="neg", hardware_sku=_SAMPLE_SKU_A3, device_arch="A3",
            config_payload={}, predicted_nodes=pad_pred, observed_dags=(dag,),
        ).passed
    # Host after device, Chrome 'f' ts mismatch (99 vs 20), & eager ACLGraph ID reuse MUST NOT get exact attribution.
    bad_order = correlate_trace_view_events([
        {"ph": "s", "cat": "HostToDevice", "id": "late", "ts": 50.0, "tid": "0", "args": {"stream_id": "0", "cann_api": "aclnnAdd"}},
        {"ph": "X", "cat": "kernel", "name": "Add", "ts": 10.0, "dur": 1.0, "tid": "0", "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE", "flow_id": "late"}},
        {"ph": "s", "cat": "HostToDevice", "id": "tsmis", "ts": 15.0, "tid": "0", "args": {"stream_id": "0", "cann_api": "aclnnAdd"}},
        {"ph": "f", "cat": "HostToDevice", "id": "tsmis", "ts": 99.0, "tid": "0", "args": {"stream_id": "0"}},
        {"ph": "X", "cat": "kernel", "name": "Add", "ts": 20.0, "dur": 1.0, "tid": "0", "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE"}},
        {"ph": "X", "cat": "kernel", "name": "Add", "ts": 30.0, "dur": 1.0, "tid": "1", "args": {"stream_id": "1", "Task Type": "AI_VECTOR_CORE", "graph_id": "g1"}},
    ], rank_id=0, phase="decode", exec_mode="eager")
    assert bad_order.exact_causal_attribution_rate == 0.0
    # 512 launches followed by late-listed flow events -> 100% 2-pass streaming attribution!
    late_trace = tmp_path / "late_flows.json"
    late_evs = [
        {"ph": "X", "cat": "kernel", "name": "Add", "ts": float(i * 5 + 2), "dur": 1.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE", "flow_id": f"lf{i}"}}
        for i in range(512)
    ] + [
        {"ph": "s", "cat": "HostToDevice", "id": f"lf{i}", "ts": float(i * 5 + 1), "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnAdd"}}
        for i in range(512)
    ]
    late_trace.write_text(json.dumps({"traceEvents": late_evs}), encoding="utf-8")
    late_preds = tuple(
        PredictedKernelNode(f"add_{i}", 0, "main", "AI_VECTOR_CORE", "aclnnAdd", ("Add",))
        for i in range(512)
    )
    late_out = evaluate_trace_view_stream(
        late_trace, late_preds, rank_id=0, phase="decode", exec_mode="eager",
        device_arch="A3", registry=default_abi_registry(),
    )
    assert late_out.exact_causal_attribution_rate == 1.0
    assert not (late_out.launch_diffs or late_out.exp_diffs or late_out.dag_diffs)
    # Truncated JSON, invalid skipped JSON, & nested array before empty traceEvents.
    trunc = tmp_path / "trunc.json"
    trunc.write_text('{"traceEvents":[{"ph":"X","name":"Add"}', encoding="utf-8")
    with pytest.raises(ModelingError, match="trace_view_json_truncated"):
        list(iter_trace_view_events(trunc))
    bad_skip = tmp_path / "bad_skip.json"
    bad_skip.write_text('{"metadata":[INVALID],"traceEvents":[]}', encoding="utf-8")
    with pytest.raises(ModelingError):
        list(iter_trace_view_events(bad_skip))
    nested = tmp_path / "nested.json"
    nested.write_text('{"other":[{"ph":"X","name":"Add"}],"traceEvents":[]}', encoding="utf-8")
    assert list(iter_trace_view_events(nested)) == []
    r17_dir = tmp_path / "rank17"
    r17_dir.mkdir()
    (r17_dir / "trace_view.json").write_text(json.dumps({"traceEvents": [
        {"ph": "s", "cat": "HostToDevice", "id": "f1", "ts": 1.0, "tid": "0", "args": {"stream_id": "0", "cann_api": "aclnnAdd"}},
        {"ph": "X", "cat": "kernel", "name": "Add", "ts": 2.0, "dur": 1.0, "tid": "0", "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE", "flow_id": "f1"}},
    ]}), encoding="utf-8")
    res17 = evaluate_live_dir(
        r17_dir, device_arch="A3", hardware_sku=_SAMPLE_SKU_A3,
        predicted_nodes=(PredictedKernelNode("n0", 17, "main", "AI_VECTOR_CORE", "aclnnAdd", ("Add",)),),
    )
    assert res17["world_ranks"] == [17] and res17["causal_parity_certified"] is True


def test_exhaustive_bijection_perturbation_every_kernel_and_edge_fails(tmp_path: Path) -> None:
    """Mathematical proof (#706): deleting/duplicating/corrupting ANY launch across ALL ranks or inverting ANY edge fails."""
    from mindie_sim.catalog.ops import vector_op
    from mindie_sim.compose import compose_case
    from tools.live_subgraph_parity_gate import _resolve_op_kernels, build_predicted_nodes_from_case, evaluate_live_dir

    reg = default_abi_registry()
    checker = SubgraphParityChecker(reg)

    # 1. Zero-whitelist multi-kernel CANN expansion check: dropping ANY kernel MUST fail!
    for arch, sym, expected_seq in (
        ("A3", "aclnnConstantPadNd", ("MemSet", "PadV3")),
        ("A5", "aclnnChunkKdaFwdV2", ("ChunkKdaFwdPrepare", "ChunkFwdH", "ChunkKdaFwdFinalize")),
        ("A5", "aclnnChunkGatedDeltaRuleFwd", ("ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO")),
        ("A5", "aclnnClamp", ("Slice", "ClipByValueV2", "ViewCopy")),
        ("A5", "aclnnTransposeBatchMatMul", ("Slice", "TransposeBatchMatMul")),
        ("A5", "aclnnIndex", ("IndexCheck", "Index")),
    ):
        op = vector_op(op_id="m", deps=(), stage=0, layer=0, requests=(), elements=4, ops_per_element=1,
                       dtype="bf16", family="elementwise", phase="decode", recipe="m", shape=(1, 4),
                       metadata=(("aclnn_symbol", sym),))
        _, resolved = _resolve_op_kernels(op, meta=dict(op.metadata), device_arch=arch, registry=reg)
        assert resolved == expected_seq
        pred = (PredictedKernelNode("m", 0, "main", "AI_VECTOR_CORE", sym, resolved),)
        for drop_idx in range(len(resolved)):
            partial = [k for j, k in enumerate(resolved) if j != drop_idx]
            evs = [ev for j, k in enumerate(partial) for ev in (
                {"ph": "s", "cat": "HostToDevice", "id": f"f{j}", "ts": float(j * 10 + 1), "tid": "0",
                 "args": {"stream_id": "0", "cann_api": sym}},
                {"ph": "X", "cat": "kernel", "name": k, "ts": float(j * 10 + 2), "dur": 1.0, "tid": "0",
                 "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE", "flow_id": f"f{j}"}},
            )]
            dag = correlate_trace_view_events(evs, rank_id=0, phase="decode", exec_mode="eager")
            rep = checker.compare(signature_id=f"drop:{sym}:{drop_idx}", hardware_sku=_SAMPLE_SKU_A5,
                                  device_arch=arch, config_payload={}, predicted_nodes=pred, observed_dags=(dag,))
            assert rep.passed is False and any(d.code == "EXPANSION_MISMATCH" for d in rep.expansion_diffs)

    # 2. Exhaustive for i in range(N) bijection + exact edge-inversion across ALL ranks + evaluate_live_dir!
    corrupt_tt = {"AI_CORE": "AI_VECTOR_CORE", "MIX_AIC": "AI_CORE", "AI_VECTOR_CORE": "AI_CORE",
                  "MIX_AIV": "AI_CORE", "COMMUNICATION": "AI_VECTOR_CORE", "AI_CPU": "AI_VECTOR_CORE"}
    for model_id, sku, arch, kw, tok in (
        ("glm-5.3-flash", _SAMPLE_SKU_A5, "A5", dict(tp=1, dp=1, ep=1, layers=4, quant="bf16"), 2),
        ("kimi-k3.1", _SAMPLE_SKU_A5, "A5", dict(tp=2, dp=1, ep=2, dcp=2, layers=4, quant="bf16"), 3),
    ):
        case_dict = compose_case(model_id, sku, prompt_tokens=128, output_tokens=8, **kw)
        _, all_preds = build_predicted_nodes_from_case(
            case_dict, phase="decode", tokens=tok, context=128, sequences=tok, device_arch=arch, registry=reg,
        )
        ranks = sorted({p.rank_id for p in all_preds})
        assert (len(ranks) == 2 if model_id == "kimi-k3.1" else len(ranks) == 1)
        m_dir = tmp_path / f"bij_{model_id}"
        rank_launches: dict[int, list[tuple[dict, dict, str]]] = {}
        for r in ranks:
            r_preds = tuple(p for p in all_preds if p.rank_id == r)
            assert any(len(p.expected_kernels) > 1 for p in r_preds)
            if model_id == "glm-5.3-flash":
                assert any(p.is_ai_cpu_metadata for p in r_preds)
            launches: list[tuple[dict, dict, str]] = []
            cursor_ts = 10.0
            for p_idx, pnode in enumerate(r_preds):
                stream = "comm0" if pnode.stream_lane == "comm" else "0"
                tts = pnode.expected_task_types if len(pnode.expected_task_types) == len(pnode.expected_kernels) else (pnode.task_type,) * len(pnode.expected_kernels)
                for k_idx, (kname, tt) in enumerate(zip(pnode.expected_kernels, tts)):
                    fid = f"r{r}_n{p_idx}_k{k_idx}"
                    s_ev = {"ph": "s", "cat": "HostToDevice", "id": fid, "ts": cursor_ts - 0.5, "tid": stream,
                            "args": {"stream_id": stream, "cann_api": pnode.api_or_symbol, "vllm_scope": pnode.vllm_scope}}
                    x_ev = {"ph": "X", "cat": "kernel", "name": kname, "ts": cursor_ts, "dur": 1.0, "tid": stream,
                            "args": {"stream_id": stream, "stream_lane": pnode.stream_lane, "Task Type": tt,
                                     "flow_id": fid, "collective_group": pnode.collective_group,
                                     "comm_ranks": list(pnode.comm_ranks), "Input Shapes": pnode.input_shapes,
                                     "vllm_scope": pnode.vllm_scope}}
                    launches.append((s_ev, x_ev, pnode.node_id))
                    cursor_ts += 5.0
            rank_launches[r] = launches
            rdir = m_dir / f"rank{r}"
            rdir.mkdir(parents=True)
            (rdir / "trace_view.json").write_text(json.dumps({"traceEvents": [e for s, x, _ in launches for e in (s, x)]}), encoding="utf-8")
        # Baseline through real production evaluate_live_dir (disk JSON -> streaming -> checker)!
        live_res = evaluate_live_dir(m_dir, device_arch=arch, hardware_sku=sku, predicted_nodes=all_preds, registry=reg)
        assert live_res["passed"] is True and live_res["causal_parity_certified"] is True
        for r in ranks:
            r_preds = tuple(p for p in all_preds if p.rank_id == r)
            launches = rank_launches[r]
            base_pairs = [(s, x) for s, x, _ in launches]
            tail_ts = launches[-1][1]["ts"] + 5.0

            def _eval(pairs: list[tuple[dict, dict]], rank: int = r, rp: tuple = r_preds):
                dag = correlate_trace_view_events([e for p in pairs for e in p], rank_id=rank, phase="decode", exec_mode="eager")
                return checker.compare(signature_id=f"{model_id}:r{rank}", hardware_sku=sku, device_arch=arch,
                                       config_payload={"m": model_id}, predicted_nodes=rp, observed_dags=(dag,))

            assert _eval(base_pairs).passed and len(launches) >= 80
            for i in range(len(launches)):
                assert not _eval(base_pairs[:i] + base_pairs[i + 1:]).passed, f"del r={r} i={i}"
                dup_s = {**launches[i][0], "id": f"dup_{r}_{i}", "ts": tail_ts - 0.5}
                dup_x = {**launches[i][1], "ts": tail_ts, "args": {**launches[i][1]["args"], "flow_id": f"dup_{r}_{i}"}}
                assert not _eval(base_pairs + [(dup_s, dup_x)]).passed, f"dup r={r} i={i}"
                assert not _eval(base_pairs[:i] + [(launches[i][0], {**launches[i][1], "name": "CorruptedKernel"})] + base_pairs[i + 1:]).passed
                bad_tt = corrupt_tt.get(launches[i][1]["args"]["Task Type"], "UNKNOWN")
                assert not _eval(base_pairs[:i] + [(launches[i][0], {**launches[i][1], "args": {**launches[i][1]["args"], "Task Type": bad_tt}})] + base_pairs[i + 1:]).passed
            by_nid: dict[str, list[int]] = {}
            for idx, (_, _, nid) in enumerate(launches):
                by_nid.setdefault(nid, []).append(idx)
            edges = [(u, pnode.node_id) for pnode in r_preds for u in pnode.predecessors if u in by_nid]
            assert len(edges) >= 50
            for u_id, v_id in edges:
                u_min_ts = min(launches[k][1]["ts"] for k in by_nid[u_id])
                v_idxs = by_nid[v_id]
                inv_pairs = list(base_pairs)
                step = min(0.25, 3.0 / max(1, len(v_idxs)))
                k_dur = step * 0.4
                for off, k in enumerate(v_idxs):
                    new_ts = u_min_ts - (len(v_idxs) - off) * step
                    inv_pairs[k] = (
                        {**launches[k][0], "ts": new_ts - k_dur * 0.25},
                        {**launches[k][1], "ts": new_ts, "dur": k_dur},
                    )
                rep_inv = _eval(inv_pairs)
                assert not rep_inv.passed and any(
                    d.code == "DEPENDENCY_ORDER_MISMATCH" and d.node_id == v_id and u_id in d.detail
                    for d in rep_inv.dag_order_diffs
                )
