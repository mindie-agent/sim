"""Unified target scheduler, packed-FP4 Cube precision parity, and directional link/HBM physics.

[Role] Verify hardware-resolved packed-FP4 matrix precision (W4A4/W4A8/W4A16),
  prepared OpSpec persistence across EvaluationContext / dual_bound / causal_tail,
  per_pair & multi-tier bottleneck link supply, directional HBM bus + mixed mte2
  conservation, and strict native-baseline non-regression in solve_target_graph.
[Invariants] Every assertion derives directly from declared hardware rates,
  compute peaks, and Engine schedules with zero model-name branches or magic literals.
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
from mindie_sim.export.bundle import StepRequest, build_export_bundle
from mindie_sim.optimization import TargetOpNode, solve_target_graph
from mindie_sim.optimization.causal_tail import (
    _active_comm_width,
    _directional_hbm_bus_params,
    _effective_die_supply,
    _node_physical_demand_and_placement,
    _resolve_die_link_supply_bps,
    schedule_target_graph_on_engine,
    weighted_step_span_us,
)
from mindie_sim.optimization.dual_bound import build_dual_inputs_from_scenario, certify_lower_bound
from mindie_sim.optimization.kernel_synthesis import _extract_hardware_supply
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
from mindie_sim.prediction.results import OpOccurrence, RankOps
from mindie_sim.primitives.loader import load_primitives
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.operators import OpSpec
from tests._support.axes import GEN3_SKU, GEN5_DT_SKU

pytestmark = pytest.mark.tier0

PREFILL_TOKENS = 1024
COMPACT_LAYERS = 2
GEMMA_COMPACT_LAYERS = 6
FLASH_NEXT_LAYERS = 4
MOE_COMPACT_LAYERS = 5
TP_QUAD = 4
TP_OCT = 8
TP_DUAL_NODE = 16
TP_DUAL = 2
FP4_TO_BF16_RATIO = 4.0
FP8_TO_BF16_RATIO = 2.0
PER_PAIR_SINGLE_LINK_GBPS = 47.5
BYTES_PER_GB = 1_000_000_000.0
BYTES_PER_MB = 1_000_000.0
HBM_READ_DOM_BYTES = 100.0 * BYTES_PER_MB
HBM_WRITE_TINY_BYTES = 0.4 * BYTES_PER_MB
MIXED_MTE2_BYTES = 600.0 * BYTES_PER_MB
MIXED_CUBE_FLOPS = 80.0 * BYTES_PER_GB
MIXED_VEC_OPS = 50.0 * BYTES_PER_MB
MS_PER_SECOND = 1000.0
US_PER_MS = 1000.0
L_STAR_PARITY_TOL_US = 1e-2
NON_REGRESSION_TOL_US = 1e-3
SYNTHETIC_SIO_BPMS = 270.0 * BYTES_PER_MB
SYNTHETIC_HCCS_BPMS = 56.0 * BYTES_PER_MB
SYNTHETIC_ROCE_BPMS = 50.0 * BYTES_PER_MB
SYNTHETIC_BASE_DUR_US = 40.0
SYNTHETIC_LSTAR_US = 20.0
SYNTHETIC_LINK_BYTES = 20.0 * BYTES_PER_MB


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

        # Gen3 lacks native FP4 -> stays on BF16 with explicit reason.
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

    # Catalog-default FP8/FP4 dtypes fall back cleanly to bf16 on dies without native support,
    # preserving precision_resolution_reason while leaving explicit compute identity untouched.
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
    """Persist resolved precision onto EvaluationContext.prepared_ops, dual_bound, and target solver."""
    # 1. EvaluationContext.prepared_ops + dual_bound parity for W4A4 and W4A8.
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

    # 2. dual_bound divides W4A4 FLOPs by native mxfp4 rate (~1730 TFLOPS), not BF16 (~432.5 TFLOPS).
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

    # 3. Zero superluminal dur_us < l_star_us inversions in After on gemma-4-31b and qwen3.8-flash-next.
    req = StepRequest(phase="prefill", tokens=PREFILL_TOKENS, context=0, sequences=1, scope="body")
    for model_name, n_layers in (("gemma-4-31b", GEMMA_COMPACT_LAYERS), ("qwen3.8-flash-next", FLASH_NEXT_LAYERS)):
        raw_bef = compose_case(
            model_name, GEN5_DT_SKU, tp=1, ep=1, dp=1, layers=n_layers,
            prompt_tokens=PREFILL_TOKENS, output_tokens=2, quant="mxfp4", feature_track="merged",
        )
        raw_aft = compose_case(
            model_name, GEN5_DT_SKU, tp=1, ep=1, dp=1, layers=n_layers,
            prompt_tokens=PREFILL_TOKENS, output_tokens=2, quant="mxfp4", feature_track="expected_ideal",
        )
        b_bef = build_export_bundle(raw_bef, request=req, apply_target_envelope=False)
        b_aft = build_export_bundle(raw_aft, request=req, apply_target_envelope=True)
        assert b_aft.step_span_us <= b_bef.step_span_us + NON_REGRESSION_TOL_US
        for row in b_aft.operators:
            if row.dur_us > 0.0:
                assert row.dur_us + L_STAR_PARITY_TOL_US >= row.l_star_lean_us

        # Verify resolve_case_matrix_precision matches Engine rate for every packed FP4 op.
        parsed_bef = parse_case(raw_bef)
        body_ops = target_graphs(parsed_bef, scope="body", phase="prefill", tokens=PREFILL_TOKENS, context=0, sequences=1)[0]
        fp4_ops = [op for op in body_ops if dict(op.metadata).get("weight_format") == "mxfp4"]
        assert fp4_ops
        for op in fp4_ops:
            resolved = resolve_case_matrix_precision(op, parsed_bef)
            assert resolved.dtype == "fp8_e4m3"
            assert resolved.matrix_flops == (("fp8_e4m3", op.flops),)


def test_per_pair_link_supply_and_multitier_bottleneck_bandwidth() -> None:
    """Scale per_pair mesh links by (W-1) and pick bottleneck tier on multi-tier non-per_pair fabrics."""
    for tp_val, expected_peers in ((TP_QUAD, TP_QUAD - 1), (TP_OCT, TP_OCT - 1)):
        case_mesh = parse_case(compose_case("qwen3-32b", GEN5_DT_SKU, tp=tp_val, dp=1, layers=1))
        _, _, _, _, link_bps = _effective_die_supply(
            case_mesh.machine.dies[0],
            case_mesh.machine.links,
            case_mesh.machine.transfer_channels,
            active_width=tp_val,
            ranks_per_node=case_mesh.machine.ranks_per_node,
            dies_per_package=case_mesh.machine.dies_per_package,
        )
        expected_bps = PER_PAIR_SINGLE_LINK_GBPS * BYTES_PER_GB * float(expected_peers)
        assert link_bps == pytest.approx(expected_bps, rel=1e-6)

    # Cross-node TP=16 on 8-rank nodes must resolve to cross-node bottleneck tier (50 GB/s), not 332.5 GB/s.
    case_2node = parse_case(compose_case("qwen3-32b", GEN5_DT_SKU, tp=TP_DUAL_NODE, dp=1, layers=1))
    _, _, _, _, link_2node_bps = _effective_die_supply(
        case_2node.machine.dies[0],
        case_2node.machine.links,
        case_2node.machine.transfer_channels,
        active_width=TP_DUAL_NODE,
        ranks_per_node=case_2node.machine.ranks_per_node,
        dies_per_package=case_2node.machine.dies_per_package,
    )
    assert link_2node_bps == pytest.approx(SYNTHETIC_ROCE_BPMS * MS_PER_SECOND, rel=1e-6)

    # Dual-die non-per_pair multi-tier links: TP=2 uses intra-package SIO; TP=4 uses HCCS bottleneck tier.
    case_gen3 = parse_case(compose_case("qwen3-32b", GEN3_SKU, tp=TP_QUAD, dp=1, layers=1))
    sio_link = Link(
        name="d2d_intra_package:0",
        members=(0, 1),
        capacity_bytes_per_ms=SYNTHETIC_SIO_BPMS,
        channel="d2d_intra_package",
        semantics="per_port",
    )
    hccs_link = Link(
        name="d2d_intra_node:0",
        members=(0, 1, 2, 3),
        capacity_bytes_per_ms=SYNTHETIC_HCCS_BPMS,
        channel="d2d_intra_node",
        semantics="per_port",
    )
    _, _, _, _, tp2_bps = _effective_die_supply(
        case_gen3.machine.dies[0], (sio_link, hccs_link), (),
        active_width=TP_DUAL, ranks_per_node=TP_QUAD, dies_per_package=TP_DUAL,
    )
    _, _, _, _, tp4_bps = _effective_die_supply(
        case_gen3.machine.dies[0], (sio_link, hccs_link), (),
        active_width=TP_QUAD, ranks_per_node=TP_QUAD, dies_per_package=TP_DUAL,
    )
    assert tp2_bps == pytest.approx(SYNTHETIC_SIO_BPMS * MS_PER_SECOND)
    assert tp4_bps == pytest.approx(SYNTHETIC_HCCS_BPMS * MS_PER_SECOND)


def test_directional_hbm_bus_and_mixed_mte2_proportional_split() -> None:
    """Issue directional memory@die:r/w matching Engine and split mte2 proportionally on mixed Cube+Vector."""
    case_gen5 = parse_case(compose_case("qwen3-32b", GEN5_DT_SKU, tp=1, dp=1, layers=1))
    die = case_gen5.machine.dies[0]
    pt = load_primitives(die.sku)
    dp_cap, rates, peaks, svc_modes, link_bps = _effective_die_supply(
        die, case_gen5.machine.links, case_gen5.machine.transfer_channels, pt,
    )
    hbm_r_bps, hbm_w_bps, alpha_rw = _directional_hbm_bus_params(dp_cap, rates)
    assert hbm_r_bps > hbm_w_bps
    assert 0.0 <= alpha_rw <= 1.0

    # Read-dominated GEMM must not collapse onto slower hbm_rw21_dram half-duplex scalar resource.
    occ = OpOccurrence(replica=0, stage=0, rank=0, op_id="s0.l0.megamoe")
    spec = OpSpec(
        op_id="s0.l0.megamoe", family="grouped_gemm", recipe="megamoe_arch35",
        stage=0, layer=0, phase="prefill", requests=(),
        semantic_shape=(PREFILL_TOKENS, 4096, 4096), local_shape=(PREFILL_TOKENS, 4096, 4096),
        padded_shape=(PREFILL_TOKENS, 4096, 4096), launch_shape=(PREFILL_TOKENS, 4096, 4096),
        dtype="fp8_e4m3", flops=MIXED_CUBE_FLOPS, matrix_flops=(("fp8_e4m3", MIXED_CUBE_FLOPS),),
        vector_ops=MIXED_VEC_OPS, mte2_bytes=MIXED_MTE2_BYTES, mte3_bytes=HBM_WRITE_TINY_BYTES,
        hbm_bytes=HBM_READ_DOM_BYTES + HBM_WRITE_TINY_BYTES,
    )
    node = TargetOpNode(
        0, occ, "s0.l0.megamoe", "arch0", 1, "grouped_gemm", "megamoe_arch35",
        "main", 0, "AI_CORE", (PREFILL_TOKENS, 4096, 4096), "fp8_e4m3", (),
        0.0, 120.0, 120.0, 120.0, False, "",
        MIXED_CUBE_FLOPS, MIXED_VEC_OPS,
        HBM_READ_DOM_BYTES, HBM_WRITE_TINY_BYTES, HBM_READ_DOM_BYTES + HBM_WRITE_TINY_BYTES,
        MIXED_MTE2_BYTES, HBM_WRITE_TINY_BYTES, 0.0, 0.0,
        "cube_bound", "Cube", 80.0, 40.0, 1.5, 0.67,
        ("s0.l0.megamoe",), ("moe",), "MegaMoE", "aclnnMegaMoe", "CANN", "", False, "", "megamoe",
        spec=spec,
    )
    demand, _place, _lat_ms, _uses = _node_physical_demand_and_placement(
        node, die_tag="d0", lane_id="main", datapath=dp_cap, rates=rates, peaks=peaks,
        service_modes=svc_modes, link_bps=link_bps, owners=None, has_topology_plan=False,
        launch_floor_us=0.0, die=die, case=case_gen5,
    )
    assert "memory@d0" not in demand
    exp_r_ms = (
        HBM_READ_DOM_BYTES / (hbm_r_bps / MS_PER_SECOND)
        + alpha_rw * HBM_WRITE_TINY_BYTES / (hbm_w_bps / MS_PER_SECOND)
    )
    exp_w_ms = (
        HBM_WRITE_TINY_BYTES / (hbm_w_bps / MS_PER_SECOND)
        + alpha_rw * HBM_READ_DOM_BYTES / (hbm_r_bps / MS_PER_SECOND)
    )
    assert demand["memory@d0:r"] == pytest.approx(exp_r_ms)
    assert demand["memory@d0:w"] == pytest.approx(exp_w_ms)
    # Faster than half-duplex hbm_rw21_dram collapse.
    rw21_bps = float(svc_modes["hbm_rw21_dram"][1])
    old_half_duplex_ms = ((HBM_READ_DOM_BYTES + HBM_WRITE_TINY_BYTES) / rw21_bps) * MS_PER_SECOND
    assert max(demand["memory@d0:r"], demand["memory@d0:w"]) < old_half_duplex_ms

    # Mixed Cube+Vector mte2_bytes splits proportionally across mte2_aic and mte2_aiv and conserves bytes.
    aic_m2 = demand.get("mte2_aic@d0", 0.0)
    aiv_m2 = demand.get("mte2_aiv@d0", 0.0)
    assert aic_m2 > aiv_m2 > 0.0
    assert aic_m2 + aiv_m2 == pytest.approx(MIXED_MTE2_BYTES)


def test_strict_baseline_non_regression_sentinel_and_prefill_models() -> None:
    """Revert cleanly to native baseline when orig_sched exceeds native Before; hold After <= Before."""
    # 1. Explicit regression scenario: native Before has faster overlapped link span than target replay.
    case_gen5 = parse_case(compose_case("qwen3-32b", GEN5_DT_SKU, tp=TP_DUAL, dp=1, layers=1))
    occ = OpOccurrence(replica=0, stage=0, rank=0, op_id="s0.l0.allreduce")
    spec = OpSpec(
        op_id="s0.l0.allreduce", family="allreduce", recipe="allreduce",
        stage=0, layer=0, phase="prefill", requests=(),
        semantic_shape=(PREFILL_TOKENS, 4096), local_shape=(PREFILL_TOKENS, 4096),
        padded_shape=(PREFILL_TOKENS, 4096), launch_shape=(PREFILL_TOKENS, 4096),
        dtype="bf16", link_bytes=SYNTHETIC_LINK_BYTES,
    )
    native_node = TargetOpNode(
        0, occ, "s0.l0.allreduce", "arch0", 1, "allreduce", "allreduce",
        "main", 0, "HCCL", (PREFILL_TOKENS, 4096), "bf16", (),
        0.0, SYNTHETIC_BASE_DUR_US, SYNTHETIC_BASE_DUR_US, SYNTHETIC_BASE_DUR_US, False, "",
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SYNTHETIC_LINK_BYTES,
        "link_bound", "Link", SYNTHETIC_LSTAR_US, SYNTHETIC_BASE_DUR_US - SYNTHETIC_LSTAR_US,
        2.0, 0.5, ("s0.l0.allreduce",), ("comm",), "AllReduce", "HcclAllReduce", "CANN",
        "", False, "", "allreduce", spec=spec,
    )
    peaks = {0: extract_hardware_peaks(case_gen5, stage=0)}
    dems = ((0, 0, 0, 0, int(SYNTHETIC_LSTAR_US * US_PER_MS)),)
    orig_sched = schedule_target_graph_on_engine((native_node,), case_gen5)
    assert weighted_step_span_us(orig_sched.nodes) > SYNTHETIC_BASE_DUR_US + NON_REGRESSION_TOL_US

    solved = solve_target_graph((native_node,), dems, case_gen5, peaks, phase="prefill", allow_unbound_formula=False)
    assert solved.reverted_to_native_baseline is True
    assert weighted_step_span_us(solved.nodes) == pytest.approx(SYNTHETIC_BASE_DUR_US)
    assert solved.step_span_us == pytest.approx(SYNTHETIC_BASE_DUR_US)
    assert solved.stage_waterfall
    assert all(row["step_span_us"] == pytest.approx(SYNTHETIC_BASE_DUR_US) for row in solved.stage_waterfall)
    assert all(row.get("reverted_to_native_baseline") is True for row in solved.stage_waterfall)

    # 2. Live multi-model prefill non-regression across previously regressing per_pair A5 topologies.
    req = StepRequest(phase="prefill", tokens=PREFILL_TOKENS, context=0, sequences=1, scope="body")
    for model_name, tp_ep, n_layers in (
        ("qwen3-235b-a22b", TP_QUAD, COMPACT_LAYERS),
        ("glm-5.1", TP_OCT, MOE_COMPACT_LAYERS),
        ("minimax-m3", TP_OCT, MOE_COMPACT_LAYERS),
    ):
        raw_bef = compose_case(
            model_name, GEN5_DT_SKU, tp=tp_ep, ep=tp_ep, dp=1, layers=n_layers,
            prompt_tokens=PREFILL_TOKENS, output_tokens=2, feature_track="merged",
        )
        raw_aft = compose_case(
            model_name, GEN5_DT_SKU, tp=tp_ep, ep=tp_ep, dp=1, layers=n_layers,
            prompt_tokens=PREFILL_TOKENS, output_tokens=2, feature_track="expected_ideal",
        )
        b_bef = build_export_bundle(raw_bef, request=req, apply_target_envelope=False)
        b_aft = build_export_bundle(raw_aft, request=req, apply_target_envelope=True)
        assert b_aft.step_span_us <= b_bef.step_span_us + NON_REGRESSION_TOL_US


def test_hbm_sustained_cap_and_multinode_link_supply_eliminate_lower_bound_inversion() -> None:
    """Clamp raw wire HBM to die sustained rate and align multi-node per_pair link supply."""
    # 1. Without primitive_table, directional HBM read/write rates stay bounded by die.rates["hbm"].
    raw_glm = compose_case(
        "glm-5.3", GEN5_DT_SKU, tp=TP_OCT, ep=TP_OCT, dp=1, layers=FLASH_NEXT_LAYERS,
    )
    case_glm = parse_case(raw_glm)
    dp_glm, rates_glm, _, _, _ = _effective_die_supply(
        case_glm.machine.dies[0], case_glm.machine.links, case_glm.machine.transfer_channels,
        primitive_table=None, active_width=TP_OCT, ranks_per_node=case_glm.machine.ranks_per_node,
        dies_per_package=case_glm.machine.dies_per_package,
    )
    read_bps, write_bps, _ = _directional_hbm_bus_params(dp_glm, rates_glm)
    sustained_hbm = float(dict(case_glm.machine.dies[0].rates)["hbm"])
    assert read_bps <= sustained_hbm and write_bps <= sustained_hbm

    # 2. Multi-node tp=16 > ranks_per_node=8 aligns peaks.py link_peak with causal_tail supply.
    raw_ds = compose_case(
        "deepseek-v3.2", GEN5_DT_SKU, tp=TP_DUAL_NODE, ep=TP_DUAL_NODE, dp=1, layers=FLASH_NEXT_LAYERS,
    )
    case_ds = parse_case(raw_ds)
    pk_ds = extract_hardware_peaks(case_ds, stage=0)
    _, _, _, _, ct_link_bps = _effective_die_supply(
        case_ds.machine.dies[0], case_ds.machine.links, case_ds.machine.transfer_channels,
        primitive_table=None, active_width=TP_DUAL_NODE,
        ranks_per_node=case_ds.machine.ranks_per_node,
        dies_per_package=case_ds.machine.dies_per_package,
    )
    assert pk_ds["link_peak_bytes_per_s"] == pytest.approx(ct_link_bps, rel=1e-6)

    # 3. End-to-end prefill & decode bundles have zero superluminal lower_bound_inversion.
    raw_q38 = compose_case(
        "qwen3.8-2.4t-a95b", GEN5_DT_SKU, tp=TP_DUAL_NODE, ep=TP_DUAL_NODE, dp=1, layers=FLASH_NEXT_LAYERS,
    )
    for raw_case in (raw_glm, raw_ds, raw_q38):
        for req in (
            StepRequest(phase="prefill", tokens=PREFILL_TOKENS // TP_OCT, context=0, sequences=1, scope="body"),
            StepRequest(phase="decode", tokens=1, context=PREFILL_TOKENS // TP_DUAL, sequences=1, scope="body"),
        ):
            bundle = build_export_bundle(raw_case, request=req, apply_target_envelope=True)
            assert all(op.dur_us + NON_REGRESSION_TOL_US >= op.l_star_lean_us for op in bundle.operators)
            assert bundle.step_span_us + NON_REGRESSION_TOL_US >= bundle.step_l_star_lean_us


def test_formula_kernel_link_supply_matches_per_pair_mesh_without_inversion() -> None:
    """Scale formula-kernel per_pair link supply to match causal_tail and avoid L* inversion."""
    expected_bps = PER_PAIR_SINGLE_LINK_GBPS * BYTES_PER_GB * float(TP_OCT - 1)
    req = StepRequest(phase="prefill", tokens=PREFILL_TOKENS, context=0, sequences=1, scope="body")
    for model_name in ("deepseek-v3", "deepseek-r1", "deepseek-v3.2"):
        raw_case = compose_case(
            model_name, GEN5_DT_SKU, tp=TP_OCT, ep=TP_OCT, dp=1,
            prompt_tokens=PREFILL_TOKENS, output_tokens=2, mtp_proposal_tokens=1,
        )
        parsed = parse_case(raw_case)
        die0 = parsed.machine.dies[0]
        _, _, _, _, _, _, synth_link_bps, _ = _extract_hardware_supply(parsed)
        ct_link_bps = _resolve_die_link_supply_bps(
            int(die0.rank),
            parsed.machine.links,
            active_width=_active_comm_width(parsed),
            ranks_per_node=parsed.machine.ranks_per_node,
            dies_per_package=parsed.machine.dies_per_package,
            fallback_bps=float(dict(die0.rates).get("link") or 0.0),
        )
        assert synth_link_bps == pytest.approx(expected_bps, rel=1e-6)
        assert synth_link_bps == pytest.approx(ct_link_bps, rel=1e-6)
        bundle = build_export_bundle(raw_case, request=req, apply_target_envelope=True)
        assert any("shared.down" in op.op_id for op in bundle.operators)
        assert all(op.dur_us + NON_REGRESSION_TOL_US >= op.l_star_lean_us for op in bundle.operators)
        assert bundle.step_span_us + NON_REGRESSION_TOL_US >= bundle.step_l_star_lean_us
