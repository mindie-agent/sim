"""Unified target step receipt, packed-FP4 Cube precision parity, and link/HBM bounds.

[Role] Verify hardware-resolved packed-FP4 matrix precision (W4A4/W4A8/W4A16),
  prepared OpSpec persistence across EvaluationContext and dual_bound,
  hardware peak link supply on single-node vs multi-node topologies, and
  monotonic 4-stage ``solve_target_step_receipt`` execution with zero L* inversions.
[Invariants] Every assertion derives directly from declared hardware rates,
  compute peaks, and canonical Engine schedules with zero model-name branches.
[Timeouts & Locks] Tier 0 bounded in-memory simulation; no live NPU or I/O.
[Side-Effect Policy] Pure read-only simulation over composed/parsed cases.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim import target_graphs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.quant import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.optimization.dual_bound import build_dual_inputs_from_scenario, certify_lower_bound
from mindie_sim.optimization.ladder import solve_target_step_receipt
from mindie_sim.prediction.efficiency.peaks import extract_hardware_peaks, parse_case
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.execution.layer_runs import execute_native_cohort_graph
from mindie_sim.prediction.operators.matrix.matrix_precision import (
    FP4_ACTIVATION_FORMATS,
    execution_compute_peaks,
    resolve_case_matrix_precision,
    resolve_matrix_precision,
)
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.results import RankOps, StepRequest
from mindie_sim.schema.operators import OpSpec
from tests._support.axes import GEN3_SKU, GEN5_DT_SKU

pytestmark = pytest.mark.tier0

PREFILL_TOKENS = 1024
COMPACT_LAYERS = 2
TP_QUAD = 4
TP_OCT = 8
TP_DUAL_NODE = 16
FP4_TO_BF16_RATIO = 4.0
FP8_TO_BF16_RATIO = 2.0
PER_PAIR_SINGLE_LINK_GBPS = 47.5
BYTES_PER_GB = 1_000_000_000.0
BYTES_PER_MB = 1_000_000.0
MS_PER_SECOND = 1000.0
SYNTHETIC_ROCE_BPMS = 50.0 * BYTES_PER_MB


def _packed_fp4_gemm(activation_format: str, *, m: int = 1024, n: int = 4096, k: int = 4096) -> OpSpec:
    q = QuantSpec(
        "mixed",
        weight_format="mxfp4",
        activation_format=activation_format,
        weight_block=(1, 32),
        scale_format="fp8_e8m0",
        scale_bytes=1,
    )
    return gemm_op(
        op_id=f"s0.l0.mx_{activation_format}",
        deps=("x",),
        stage=0,
        layer=0,
        requests=("r0",),
        m=m,
        n=n,
        k=k,
        quant=q,
        category="projection",
        phase="prefill",
    )


def test_packed_fp4_matrix_precision_resolves_w4a4_w4a8_and_w4a16() -> None:
    """Resolve W4A4 -> mxfp4 (4x BF16), W4A8 -> fp8_e4m3 (2x BF16), W4A16 -> bf16 (1x)."""
    assert {"fp4", "fp4_e2m1", "mxfp4"} <= FP4_ACTIVATION_FORMATS
    case_gen5 = parse_case(compose_case("qwen3-32b", GEN5_DT_SKU, tp=1, dp=1, layers=1))
    case_gen3 = parse_case(compose_case("qwen3-32b", GEN3_SKU, tp=1, dp=1, layers=1))
    dies_gen5 = case_gen5.machine.dies[:1]
    dies_gen3 = case_gen3.machine.dies[:1]
    bf16_peak = float(dies_gen5[0].peak("bf16"))

    for act_fmt in ("fp4", "fp4_e2m1", "mxfp4"):
        op_w4a4 = _packed_fp4_gemm(act_fmt)
        res_w4a4 = resolve_matrix_precision(op_w4a4, devices=dies_gen5, policy="auto", recipe=None)
        assert res_w4a4.dtype == "mxfp4"
        assert res_w4a4.matrix_flops == (("mxfp4", op_w4a4.flops),)
        assert resolve_matrix_precision(res_w4a4, devices=dies_gen5, policy="auto", recipe=None) == res_w4a4
        peaks = execution_compute_peaks(res_w4a4, devices=dies_gen5, base_peaks={"bf16": bf16_peak})
        assert peaks["mxfp4"] == pytest.approx(FP4_TO_BF16_RATIO * bf16_peak)

        res_gen3 = resolve_matrix_precision(op_w4a4, devices=dies_gen3, policy="auto", recipe=None)
        assert res_gen3.dtype == "bf16"
        assert dict(res_gen3.metadata)["precision_resolution_reason"] == "fp4_unavailable"

    op_w4a8 = _packed_fp4_gemm("fp8_e4m3")
    res_w4a8 = resolve_matrix_precision(op_w4a8, devices=dies_gen5, policy="auto", recipe=None)
    assert res_w4a8.dtype == "fp8_e4m3"
    assert res_w4a8.matrix_flops == (("fp8_e4m3", op_w4a8.flops),)
    peaks_w4a8 = execution_compute_peaks(res_w4a8, devices=dies_gen5, base_peaks={"bf16": bf16_peak})
    assert peaks_w4a8["fp8_e4m3"] == pytest.approx(FP8_TO_BF16_RATIO * bf16_peak)

    op_w4a16 = _packed_fp4_gemm("bf16")
    res_w4a16 = resolve_matrix_precision(op_w4a16, devices=dies_gen5, policy="auto", recipe=None)
    assert res_w4a16.dtype == "bf16"
    assert res_w4a16 == op_w4a16

    mixed_dies = (dies_gen5[0], replace(dies_gen3[0], rank=1))
    for raw_dt, act_fmt, expected_reason in (
        ("fp8_e4m3", "fp8_e4m3", "fp8_unavailable"),
        ("fp8_e5m2", "fp8_e5m2", "fp8_unavailable"),
        ("mxfp4", "mxfp4", "fp4_unavailable"),
    ):
        base_op = _packed_fp4_gemm(act_fmt)
        meta = dict(base_op.metadata)
        meta["compute_format"] = raw_dt
        meta["compute_format_origin"] = "catalog_default"
        cat_op = replace(base_op, dtype=raw_dt, metadata=tuple(sorted(meta.items())))
        for target_dies in (dies_gen3, mixed_dies):
            fell_back = resolve_matrix_precision(cat_op, devices=target_dies, policy="auto", recipe=None)
            assert fell_back.dtype == "bf16"
            assert fell_back.matrix_flops == ()
            assert dict(fell_back.metadata)["precision_resolution_reason"] == expected_reason
            assert dict(fell_back.metadata)["compute_format"] == "bf16"
        exp_meta = dict(meta)
        exp_meta["compute_format_origin"] = "explicit"
        exp_op = replace(base_op, dtype=raw_dt, metadata=tuple(sorted(exp_meta.items())))
        kept = resolve_matrix_precision(exp_op, devices=dies_gen3, policy="auto", recipe=None)
        assert kept.dtype == raw_dt
        assert dict(kept.metadata)["precision_resolution_reason"] == "explicit_compute_identity"


def test_prepared_opspec_persists_across_context_dual_bound_and_prefill_models() -> None:
    """Persist resolved precision onto EvaluationContext.prepared_ops, dual_bound, and target receipt."""
    raw_w4a4 = compose_case(
        "gemma-4-31b", GEN5_DT_SKU, tp=1, dp=1, layers=1,
        prompt_tokens=PREFILL_TOKENS, output_tokens=2, quant="w4a4_mxfp4",
    )
    case_w4a4 = parse_case(raw_w4a4)
    ops_w4a4 = target_graphs(case_w4a4, scope="body", phase="prefill", tokens=PREFILL_TOKENS, context=0, sequences=1)[0]
    book = RecipeBook.from_mapping(case_w4a4.runtime.kernels)
    ev = EvaluationContext.create(case_w4a4, book)
    execute_native_cohort_graph(
        case_w4a4, {0: ops_w4a4}, recipes=book, external_inputs={"analysis_input": 0.0},
        isolated=False, compact=False, evaluation=ev, timeline_replicas=(0,),
    )
    prepared_map = getattr(ev, "prepared_ops", {})
    assert prepared_map
    qkv_prep = prepared_map.get((0, "s0.l0.qkv"))
    assert qkv_prep is not None
    assert qkv_prep.dtype == "mxfp4"
    assert qkv_prep.matrix_flops == (("mxfp4", qkv_prep.flops),)

    # dual_bound divides W4A4 FLOPs by native mxfp4 rate (~1730 TFLOPS), not BF16 (~432.5 TFLOPS).
    op_w4a4 = _packed_fp4_gemm("fp4_e2m1")
    scen = SimpleNamespace(
        case=case_w4a4,
        machine=case_w4a4.machine,
        atomic_ops=(RankOps(replica=0, ops=(op_w4a4,)),),
        summaries=(),
        aliases={},
    )
    prog, dag = build_dual_inputs_from_scenario(scen, primitive_table=None, rate_mode="engine_ceiling")[:2]
    cert, _ = certify_lower_bound(prog, dag)
    die0 = case_w4a4.machine.dies[0]
    mxfp4_peak = float(die0.peak("mxfp4"))
    bf16_peak = float(die0.peak("bf16"))
    simt_peak = float(dict(die0.rates).get("simt", dict(die0.rates).get("vector", 0.0)))
    expected_mxfp4_ns = (float(op_w4a4.flops) / (mxfp4_peak + simt_peak)) * 1_000_000_000.0
    expected_bf16_ns = (float(op_w4a4.flops) / (bf16_peak + simt_peak)) * 1_000_000_000.0
    assert cert.l_star_lean_ns == pytest.approx(expected_mxfp4_ns, rel=1e-3)
    assert cert.l_star_lean_ns < expected_bf16_ns * 0.5

    # Verify resolve_case_matrix_precision & monotonic solve_target_step_receipt with zero L* inversions.
    req = StepRequest(phase="prefill", tokens=PREFILL_TOKENS, context=0, sequences=1, scope="body")
    for model_name in ("gemma-4-31b", "qwen3.8-flash-next"):
        raw_case = compose_case(
            model_name, GEN5_DT_SKU, tp=1, ep=1, dp=1, layers=COMPACT_LAYERS,
            prompt_tokens=PREFILL_TOKENS, output_tokens=2, quant="mxfp4", feature_track="merged",
        )
        parsed = parse_case(raw_case)
        body_ops = target_graphs(parsed, scope="body", phase="prefill", tokens=PREFILL_TOKENS, context=0, sequences=1)[0]
        fp4_ops = [op for op in body_ops if dict(op.metadata).get("weight_format") == "mxfp4"]
        assert fp4_ops
        for op in fp4_ops:
            resolved = resolve_case_matrix_precision(op, parsed)
            assert resolved.dtype == "fp8_e4m3"
            assert resolved.matrix_flops == (("fp8_e4m3", op.flops),)
        receipt = solve_target_step_receipt(parsed, req, stack_track="target", max_trials=2)
        spans = [st.span_ns for st in receipt.stages]
        assert all(cur <= prev for prev, cur in zip(spans, spans[1:]))
        assert all(0 < st.l_star_ns <= st.span_ns for st in receipt.stages if st.status == "committed")


def test_multinode_link_peaks_and_target_receipt_monotonicity() -> None:
    """Verify per_pair vs cross-node link peak scaling and monotonic target receipts with L* <= span."""
    raw_tp8 = compose_case("qwen3-32b", GEN5_DT_SKU, tp=TP_OCT, dp=1, layers=1)
    raw_tp16 = compose_case("qwen3-32b", GEN5_DT_SKU, tp=TP_DUAL_NODE, dp=1, layers=1)
    pk_tp8 = extract_hardware_peaks(parse_case(raw_tp8), stage=0)
    pk_tp16 = extract_hardware_peaks(parse_case(raw_tp16), stage=0)
    assert pk_tp8["link_peak_bytes_per_s"] == pytest.approx(
        PER_PAIR_SINGLE_LINK_GBPS * BYTES_PER_GB * float(TP_OCT - 1), rel=1e-6,
    )
    assert pk_tp16["link_peak_bytes_per_s"] == pytest.approx(SYNTHETIC_ROCE_BPMS * MS_PER_SECOND, rel=1e-6)

    req_pre = StepRequest(phase="prefill", tokens=PREFILL_TOKENS // TP_QUAD, context=0, sequences=1, scope="body")
    req_dec = StepRequest(phase="decode", tokens=1, context=PREFILL_TOKENS // 2, sequences=1, scope="body")
    for model_name, tp_ep, layer_cnt in (("qwen3-235b-a22b", TP_QUAD, COMPACT_LAYERS), ("deepseek-v3.2", TP_OCT, 4)):
        raw_case = compose_case(
            model_name, GEN5_DT_SKU, tp=tp_ep, ep=tp_ep, dp=1, layers=layer_cnt,
            prompt_tokens=PREFILL_TOKENS, output_tokens=2,
        )
        for req in (req_pre, req_dec):
            receipt = solve_target_step_receipt(raw_case, req, stack_track="target", max_trials=2)
            spans = [st.span_ns for st in receipt.stages]
            assert all(cur <= prev for prev, cur in zip(spans, spans[1:]))
            assert all(0 < st.l_star_ns <= st.span_ns for st in receipt.stages if st.status == "committed")


def test_hoffman_cp_circulation_noncausal_swa_and_dcp_legality() -> None:
    """Verify Hoffman exact-lcm circulation, noncausal SWA span, and structural DCP legality."""
    from mindie_sim.catalog.context_parallel import (
        legal_vllm_ascend_dcp_values,
        round_sparse_dcp_matrix,
        split_causal_pairs,
        validate_vllm_ascend_dcp_codomain,
    )
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.parallel import ParallelSpec

    # Noncausal SWA (window > 0): exact upstream min(context + q_len, window + q_len).
    swa_hi = split_causal_pairs(context=100, q_len=4, cp=2, interleave=1, window=16, noncausal=True)
    assert swa_hi.total == pytest.approx(4.0 * 20.0)
    swa_lo = split_causal_pairs(context=8, q_len=4, cp=2, interleave=1, window=16, noncausal=True)
    assert swa_lo.total == pytest.approx(4.0 * 12.0)

    # Exact Hoffman min-cost circulation preserves row/column totals on [0.4,0.6] & [1/3,2/3].
    for weights, col_tgt in (
        (((0.4, 0.6), (0.6, 0.4)), None),
        (((1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0)), (1, 1)),
    ):
        mat = round_sparse_dcp_matrix(weights, (1, 1), ((1, 1), (1, 1)), column_targets=col_tgt)
        assert [sum(row) for row in mat] == [1, 1]
        assert [sum(row[c] for row in mat) for c in range(2)] == [1, 1]
        assert all(isinstance(v, int) and v >= 0 for row in mat for v in row)

    # Infeasible column_targets fail fast instead of silently breaking conservation.
    with pytest.raises(ModelingError, match="dcp_column_targets_infeasible"):
        round_sparse_dcp_matrix(((0.5, 0.5),), (1,), ((1, 1),), column_targets=(2, 0))

    # Structural DCP legality: KPool (glm-5.3-flash) and DS-V4.1 restrict DCP to {1}.
    for model_id in ("glm-5.3-flash", "deepseek-v4.1-flash"):
        raw = compose_case(model_id, GEN5_DT_SKU, tp=8, dp=1, ep=1, layers=2)
        parsed = parse_case(raw)
        assert legal_vllm_ascend_dcp_values(parsed.model, tp=8) == (1,)
        ok, reason = validate_vllm_ascend_dcp_codomain(parsed.model, ParallelSpec(tp=8, cp=2))
        assert ok is False and reason is not None


def test_exposed_observer_clipped_grant_live_retention_and_eviction_fail_closed() -> None:
    """Verify clipped grant proportional scaling (500/500), live predecessor retention, and fail-closed eviction."""
    from mindie_sim.prediction.runtime.engine import Engine, Resource
    from mindie_sim.prediction.runtime.exposed_observer import (
        ExposedObserver,
        ObservedInterval,
        ObservedServiceGrant,
        StageDescriptor,
        attribute_stream0_intervals,
    )
    from mindie_sim.schema.common import ModelingError

    res = attribute_stream0_intervals(
        [
            ObservedInterval("opA", "compute", 0, 0, 1000),
            ObservedInterval("opB", "compute", 0, 0, 1000),
        ],
        start_ns=0,
        terminal_ns=1000,
        grants=[
            ObservedServiceGrant("opA", "cube", 0, 2000, 2.0),
            ObservedServiceGrant("opB", "cube", 0, 1000, 1.0),
        ],
    )
    assert res.exposed_ns_by_op == {"opA": 500, "opB": 500}
    assert res.conservation_delta_ns == 0

    # Active predecessor with refcount > 0 is retained across unrelated submits until child resolves.
    obs_live = ExposedObserver(max_events=3)
    obs_live.begin_scope("live", 0.0)
    obs_live.on_submit("old0", declaration=StageDescriptor("cube", cube_ms=1.0))
    obs_live.on_submit("parent", declaration=StageDescriptor("cube", cube_ms=5.0))
    obs_live.on_submit("child", after=("parent",), declaration=StageDescriptor("vector", vector_ms=2.0))
    obs_live.on_submit("unrelated", declaration=StageDescriptor("cube", cube_ms=1.0))
    assert "parent" in obs_live.scope_events("live") and "child" in obs_live.scope_events("live")
    child_clock = obs_live.on_resolve("child", after=("parent",), moment_ms=7.0)
    assert child_clock.all_ms == pytest.approx(7.0)

    # Evicted unreferenced event fails closed on both dependent submit and late resolve.
    obs_ev = ExposedObserver(max_events=2)
    obs_ev.begin_scope("ev", 0.0)
    for idx in range(5):
        obs_ev.on_submit(f"u{idx}", declaration=StageDescriptor("cube", cube_ms=1.0))
    assert len(obs_ev.scope_events("ev")) <= 2
    with pytest.raises(ModelingError, match="exposed_observer_evicted_predecessor:u0"):
        obs_ev.on_submit("late_dep", after=("u0",), declaration=StageDescriptor("cube", cube_ms=1.0))
    with pytest.raises(ModelingError, match="exposed_observer_evicted_predecessor:u0"):
        obs_ev.on_resolve("u0", moment_ms=1.0)
    with pytest.raises(ModelingError, match="exposed_observer_evicted_predecessor:u0"):
        obs_ev.release("u0")

    # Exact Fraction sweep-line attribution is cut-invariant and conserves 2**53+3 ns exactly.
    for span, cuts in ((10, 1), (10, 10), (2**53 + 3, 1)):
        ivs = [ObservedInterval(k, f"s{j}", j, 0, span) for j, k in enumerate("abx")]
        grants = [ObservedServiceGrant(k, "hbm", 0, span, float(span)) for k in "ab"]
        for j in range(cuts):
            a_ns, b_ns = j * span // cuts, (j + 1) * span // cuts
            grants.append(ObservedServiceGrant("x", "aux", a_ns, b_ns, float(b_ns - a_ns)))
        cut_res = attribute_stream0_intervals(ivs, start_ns=0, terminal_ns=span, grants=grants)
        assert cut_res.conservation_delta_ns == 0
        if span == 10:
            assert (cut_res.exposed_ns_by_op["a"], cut_res.exposed_ns_by_op["b"]) == (5, 5)

    eng = Engine([Resource("cube", 1.0)], backend="python", trace_limit=1)
    e1 = eng.submit("t1", {"cube": 2.0})
    e2 = eng.submit("t2", {"cube": 3.0}, after=(e1,))
    eng.advance(until=(e2,))
    assert e2.done and e2.time == pytest.approx(5.0)
    assert eng.now == pytest.approx(5.0)


def test_cohort_reciprocity_rejects_asymmetric_expected_bytes_and_replica_wire_bytes() -> None:
    """Reject non-reciprocal expected-byte matrices and cross-replica wire_bytes asymmetry."""
    from dataclasses import replace
    from mindie_sim.api import compose_case as compose_spec
    from mindie_sim.prediction.efficiency.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.operators.mc2.mc2_program import native_program_from_contract
    from mindie_sim.prediction.program import NativeProgram, RecipeContext
    from mindie_sim.prediction.runtime.dp_cohort_dedup import is_symmetric_native_program
    from mindie_sim.schema.hardware_links import Link

    ctx = RecipeContext(object(), {}, {}, {"hbm": 1.0, "vector": 1.0, "link": 1.0}, 2)
    asym_row = (1536.0, 512.0)
    raw_asym = dict(
        kind="dispatch", width=32, dtype="bf16", source_assignments=4, received_assignments=6,
        experts=2, ep=2, source_tokens=4, top_k=1,
        route_matrix_bytes=(asym_row, asym_row), payload_basis="expected",
    )
    prog_asym = native_program_from_contract(raw_asym, ctx)
    assert is_symmetric_native_program(NativeProgram(prog_asym, prog_asym.work), 2) is False

    grp = tuple(range(4))
    spec = compose_spec("deepseek-v3.2", "a5", options=dict(tp=1, dp=4, ep=4, layers=4, ranks_per_node=2))
    pair_links = tuple(
        Link(str(i), 1.0, (i, i + 1), semantics="per_pair", scope="node", transaction_bytes=512 * (1 + i // 2))
        for i in (0, 2)
    )
    spec = replace(spec, machine=replace(
        spec.machine, links=pair_links + (Link("inter", 1.0, grp, semantics="per_pair", scope="cross_node"),),
    ))
    exe = Execution(spec, CostModel(spec))
    mat = tuple(tuple(int((i ^ 1) == j) for j in grp) for i in grp)
    raw_eq = dict(
        kind="dispatch", width=32, dtype="bf16", source_assignments=1, received_assignments=1,
        experts=4, ep=4, source_tokens=1, top_k=1, route_matrix=mat, payload_basis="exact",
    )
    prog_eq = native_program_from_contract(raw_eq, ctx)
    target = dict(group=grp, sources=(0,), route=exe.topology.route_message)
    ok_rec, _, _ = exe._verify_canonical_cohort_reciprocity(
        NativeProgram(prog_eq, prog_eq.work), target, grp, 0, is_symmetric_native_program,
    )
    assert ok_rec is False
