"""Silicon calibration, UB hardware wall, lowering, and co-design tests for TileLang 0.1.15.

[Role] Verify sealed TileLang 0.1.15 calibration loading, SHA-256 and privacy
  hygiene, Plane-1 sealer round-trip reproduction, strict 216 KiB vs 248 KiB
  UB hardware wall enforcement (ACL 107000 guard), Cube T.gemm and
  T.gemm_blockscaled MXFP8 attainment, T.Pipelined DMA stage monotonicity,
  ACLGraph vs eager tvm_ffi dispatch floors, SimtVF vs SimdVF vector ceilings,
  and dual-tier MegaFusionBound / CoDesignOpportunity reporting.
[Invariants] Pure Tier-0 in-memory/file assertions; zero device contexts,
  zero private paths, exact FLOP/byte conservation, and strict A5 scoping.
[Timeouts & Locks] Fast CPU-only execution (< 5 s); no subprocess or network.
[Side-Effect Policy] Read-only over packaged tables; mutates only ``tmp_path``.
"""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from _support.axes import GEN5_DT_SKU, REPRESENTATIVE_SKUS
from _support.linear_cases import (
    REPO_ROOT,
    run_template,
    stages_by_name,
    template_context,
    template_datapath,
)
from mindie_sim.compose import compose_case
from mindie_sim.optimization import kernel_synthesis as ks_mod
from mindie_sim.prediction.program.stages import Stage
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, TileInfeasibleError
from mindie_sim.prediction.structure import parse_case
from mindie_sim.primitives import (
    TILELANG_ACL_UB_OVERFLOW_CODE,
    TILELANG_CALIBRATION_SCHEMA_VERSION,
    TILELANG_SEALED_ENVELOPE_SHA256,
    compute_tilelang_envelope_sha256,
    default_primitives_dir,
    load_tilelang_calibration,
    try_load_tilelang_calibration,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.scenario import CoDesignOpportunity, MegaFusionBound
from mindie_sim.templates import (
    SemanticWorkItem,
    TensorRangeObligation,
    compile_fuse_region,
    resolve_tilelang_profile,
)
from tools.bench_tilelang_primitives import (
    assert_tilelang_privacy_clean,
    build_tilelang_calibration_payload,
    describe_tilelang_probe_suite,
    verify_tilelang_calibration_file,
)

pytestmark = pytest.mark.tier0

KIB = 1024
SEALED_JSON_PATH = default_primitives_dir() / "arch35_a5_950dt_tilelang.json"
EXPECTED_SHA256 = "b19cc4d48a11aa4ec41a715276828c31dc709d0bd94f73760a0211e2d9ade0ae"


def _a5_datapath():
    return template_datapath(REPRESENTATIVE_SKUS["a5"])


def _a5_template_context(dp, **extra_rates):
    ctx = template_context(dp, None, **extra_rates)
    peaks = {
        **dict(ctx.compute_peaks),
        "bf16": dp.cube_peak("bf16"),
        "fp16": dp.cube_peak("fp16"),
        "fp8_e4m3": dp.cube_peak("fp8_e4m3"),
        "mxfp8": dp.cube_peak("mxfp8"),
        "fp32": dp.cube_peak("fp32"),
    }
    return replace(ctx, compute_peaks=peaks)


def test_sealed_tilelang_json_sha256_schema_privacy_and_loader_invariants(tmp_path) -> None:
    """Sealed JSON digest, schema fields, caching, tamper rejection, and Plane-1 sealer round-trip."""
    assert SEALED_JSON_PATH.is_file()
    raw_text = SEALED_JSON_PATH.read_text(encoding="utf-8")
    assert_tilelang_privacy_clean(raw_text, label="arch35_a5_950dt_tilelang.json")
    raw = json.loads(raw_text)
    assert compute_tilelang_envelope_sha256(raw) == EXPECTED_SHA256
    assert raw["envelope_sha256"] == EXPECTED_SHA256
    assert TILELANG_SEALED_ENVELOPE_SHA256["npu_arch_3510"] == EXPECTED_SHA256

    verified = verify_tilelang_calibration_file(SEALED_JSON_PATH)
    assert verified["status"] == "verified"
    assert verified["envelope_sha256"] == EXPECTED_SHA256
    assert verified["simt_usable_ub_bytes"] == 221_184
    assert verified["simd_usable_ub_bytes"] == 253_952

    cal = load_tilelang_calibration("npu_arch_3510")
    assert load_tilelang_calibration("arch35") is cal
    assert cal.schema_version == TILELANG_CALIBRATION_SCHEMA_VERSION
    assert cal.tilelang_version == "0.1.15"
    assert cal.simd_usable_ub_bytes == 253_952 == 248 * KIB
    assert cal.simt_usable_ub_bytes == 221_184 == 216 * KIB
    assert cal.aclgraph_supported is True
    assert cal.aclgraph_issue_us == pytest.approx(0.6117)
    assert cal.eager_tvm_ffi_prealloc_us == pytest.approx(11.7728)
    assert cal.eager_tvm_ffi_callee_alloc_us == pytest.approx(15.7746)
    assert cal.simt_reg_fma_gops_per_aiv == pytest.approx(105.6613)
    assert cal.simt_ub_fma_gops_per_aiv == pytest.approx(13.0979)
    assert cal.simd_reg_alu_gops_per_aiv == pytest.approx(174.5558)
    assert cal.simd_ub_stream_vadd_gops_per_aiv == pytest.approx(57.7441)
    assert cal.simd_sfu_exp_gops_per_aiv == pytest.approx(70.417)
    assert cal.eta_copy_rw21_simt == pytest.approx(0.9793)
    assert cal.eta_copy_rw21_simd == pytest.approx(1.0164)
    assert cal.dma_attainment(2, uses_simt=False, rw_pattern="rw21") == pytest.approx(1.0)
    assert cal.dma_attainment(2, uses_simt=True, rw_pattern="rw21") == pytest.approx(0.9793)
    assert cal.cube_attainment("bf16") == pytest.approx(0.9892)
    assert cal.cube_attainment("fp8_e4m3") == pytest.approx(0.9793)
    assert cal.cube_attainment("mxfp8", blockscaled=True) == pytest.approx(0.9681)

    # Unsupported architecture fails closed; try_load returns None.
    with pytest.raises(ModelingError, match="tilelang_calibration_arch_unsupported"):
        load_tilelang_calibration("npu_arch_2201")
    assert try_load_tilelang_calibration("npu_arch_2201") is None

    # Tampered JSON copy fails SHA-256 verification.
    tampered = json.loads(raw_text)
    tampered["onchip_ub_ceilings_bytes"]["simt_usable_ub_bytes"] = 230_000
    (tmp_path / "arch35_a5_950dt_tilelang.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ModelingError, match="tilelang_calibration_sha256_mismatch"):
        load_tilelang_calibration("npu_arch_3510", primitives_dir=tmp_path)

    # Plane-1 producer round-trip reproduces exact sealed SHA-256 from L0 table + level payloads.
    l0_table = json.loads((REPO_ROOT / "data" / "primitives" / "arch35_a5_950dt.json").read_text(encoding="utf-8"))
    levels = raw["levels"]
    vec_all = dict(levels["L2_vector_simt_vs_simd"])
    simt_reg = {k: vec_all.pop(k) for k in ("simt_reg_fma_t1024", "simt_reg_exp_t1024")}
    cube_all = dict(levels["L3_cube_gemm_and_blockscaled"])
    mxfp8 = cube_all.pop("mxfp8_blockscaled_8192")
    rebuilt = build_tilelang_calibration_payload(
        part1={"sections": {"TL00_host_and_graph_dispatch": levels["L0_dispatch"], "TL01_dma_copy": levels["L1_dma_copy"]}},
        part2={"sections": {"TL03_cube_gemm": cube_all, "TL05_composite_anchors": levels["L5_composite_anchors"]}},
        part3={"TL02_vector_onchip_low_cv": vec_all, "TL04_sync_low_cv": levels["L4_sync_primitives"]},
        part4={"simt_register_resident": simt_reg},
        mxfp8=mxfp8,
        l0_table=l0_table,
    )
    assert rebuilt["envelope_sha256"] == EXPECTED_SHA256
    plan = describe_tilelang_probe_suite()
    assert set(plan["levels"]) == set(levels)


def test_tilelang_ub_hardware_wall_216kib_vs_248kib_and_acl_107000_rejection() -> None:
    """Strict 216 KiB SimtVF wall vs 248 KiB SimdVF ceiling: 220 KB fits SimtVF, 230 KiB fails SimtVF & passes SimdVF, 256 KiB fails both."""
    dp = _a5_datapath()
    cal = load_tilelang_calibration(str(dp.arch_id))

    def _compile_with_ub(ub_bytes: int, *, backend: str, uses_simt: bool = False):
        stage = Stage("vec", {"vector": 4096}, placement={"vector": (0,)})
        semantic = (SemanticWorkItem("vec", "vector_alu", "fp32", 4096, ("aiv_simt" if uses_simt else "aiv_simd",)),)
        return compile_fuse_region(
            "ub_probe", ("vec",), dp,
            dependencies={"vec": ()},
            op_stages={"vec": stage},
            edge_routes=(),
            edge_bytes={},
            live_buffers=(LiveBufferSpec("tile_ub", "ub", ub_bytes, 0, 1, owner_kind="aiv"),),
            backend=backend,
            uses_simt=uses_simt,
            semantic_work=semantic,
        )

    # 1) <= 216 KiB (220,000 B = 214.8 KiB and exact 221,184 B = 216 KiB): fits BOTH SimtVF and SimdVF without warning.
    for ok_bytes in (220_000, 216 * KIB):
        spec_simt, _ = _compile_with_ub(ok_bytes, backend="TileLang", uses_simt=True)
        spec_simd, _ = _compile_with_ub(ok_bytes, backend="TileLang", uses_simt=False)
        assert spec_simt.backend == "tilelang_simt"
        assert spec_simd.backend == "tilelang_simd"
        assert spec_simt.tile_solution.pool_peaks[0].capacity_bytes == 216 * KIB
        assert spec_simd.tile_solution.pool_peaks[0].capacity_bytes == 248 * KIB
        assert cal.simt_vs_simd_ub_wall_warning(ok_bytes) is None

    # 2) 230 KiB (235,520 B in (216 KiB, 248 KiB]): REJECTS tilelang_simt cleanly, PASSES tilelang_simd with ACL 107000 warning.
    mid_bytes = 230 * KIB
    with pytest.raises(TileInfeasibleError) as exc_simt:
        _compile_with_ub(mid_bytes, backend="tilelang_simt", uses_simt=True)
    rej, = exc_simt.value.rejections
    assert (rej.pool, rej.required_bytes, rej.available_bytes) == ("ub", mid_bytes, 216 * KIB)

    spec_simd_mid, _ = _compile_with_ub(mid_bytes, backend="tilelang_simd", uses_simt=False)
    assert spec_simd_mid.tile_solution.pool_peaks[0].peak_bytes == mid_bytes
    assert spec_simd_mid.tile_solution.pool_peaks[0].capacity_bytes == 248 * KIB
    warn_mid = cal.simt_vs_simd_ub_wall_warning(mid_bytes)
    assert warn_mid is not None
    assert TILELANG_ACL_UB_OVERFLOW_CODE in warn_mid and "T.SimtVF" in warn_mid and "T.SimdVF" in warn_mid and "AscendC" in warn_mid

    # 3) 256 KiB (262,144 B > 248 KiB): FAILS BOTH SimtVF and SimdVF.
    over_bytes = 256 * KIB
    with pytest.raises(TileInfeasibleError):
        _compile_with_ub(over_bytes, backend="tilelang_simt", uses_simt=True)
    with pytest.raises(TileInfeasibleError) as exc_simd:
        _compile_with_ub(over_bytes, backend="tilelang_simd", uses_simt=False)
    rej_simd, = exc_simd.value.rejections
    assert (rej_simd.pool, rej_simd.required_bytes, rej_simd.available_bytes) == ("ub", over_bytes, 248 * KIB)

    # 4) Explicit contradiction (tilelang_simd + uses_simt=True) fails closed.
    with pytest.raises(ModelingError, match="tilelang_simd_incompatible_with_uses_simt"):
        resolve_tilelang_profile(dp, backend="tilelang_simd", uses_simt=True)


def test_tilelang_cube_gemm_and_mxfp8_blockscaled_attainment_and_conservation() -> None:
    """T.gemm (BF16/FP8) and T.gemm_blockscaled (MXFP8 uint16 paired_e8m0) conserve FLOPs and apply sealed attainment."""
    dp = _a5_datapath()
    flops = 2 * 4096 * 4096 * 4096
    ctx = _a5_template_context(dp, cube=1.0)
    cases = (
        ("bf16", False, 0.9892),
        ("fp8_e4m3", False, 0.9793),
        ("mxfp8", True, 0.9681),
    )
    for dtype, blockscaled, expected_eta in cases:
        stage = Stage(
            "gemm", {"cube": flops},
            placement={"cube": tuple(range(dp.aic_count))},
            metadata={"cube_dtype": dtype, "tilelang_blockscaled": blockscaled},
        )
        semantic = (SemanticWorkItem("gemm", "matrix_mac", dtype, flops, ("cube",)),)
        spec, result = compile_fuse_region(
            f"tl_{dtype}", ("gemm",), dp,
            dependencies={"gemm": ()},
            op_stages={"gemm": stage},
            edge_routes=(),
            edge_bytes={},
            backend="TileLang",
            uses_simt=False,
            tilelang_blockscaled=blockscaled,
            semantic_work=semantic,
            context=ctx,
        )
        assert spec.tilelang_profile is not None
        assert dict(result.summary.issued_cube_flops_by_dtype) == {dtype: flops}
        lowered = stages_by_name(result)["p0:t0:gemm"]
        assert lowered.metadata["tilelang_cube_attainment"] == pytest.approx(expected_eta)
        assert lowered.metadata["tilelang_blockscaled"] is blockscaled
        expected_peak_ms = (1000.0 * flops / dp.cube_peak(dtype)) / expected_eta
        assert lowered.work["cube"] == pytest.approx(expected_peak_ms, rel=1e-6)
        eng = run_template(dp, ctx, result)
        assert eng.serviced["cube"] == pytest.approx(expected_peak_ms, rel=1e-6)


def test_tilelang_dma_pipelined_stages_monotonicity_and_rw21_simt_vs_simd() -> None:
    """T.Pipelined stages=1/2/3 improve DMA attainment monotonically while conserving exact byte ledgers."""
    dp = _a5_datapath()
    payload_bytes = 4 * 1024 * 1024
    hbm_rate = dp.hbm_read_bytes_per_s / 1000.0
    ctx = _a5_template_context(dp, hbm=hbm_rate, mte2_aiv=hbm_rate, mte3=hbm_rate, vector=1e12)
    read_ob = TensorRangeObligation(0, "src", 0, ((0, payload_bytes),), "hbm_cold", "device")
    write_ob = TensorRangeObligation(0, "dst", 1, ((0, payload_bytes),), "hbm_cold", "device")

    stage_etas: list[float] = []
    stage_times: list[float] = []
    for stages in (1, 2, 3):
        spec, result = compile_fuse_region(
            f"dma_s{stages}", ("copy",), dp,
            dependencies={"copy": ()},
            op_stages={"copy": Stage("copy")},
            edge_routes=(),
            edge_bytes={},
            live_in_obligations=(read_ob,),
            live_out_obligations=(write_ob,),
            backend="tilelang_simd",
            tilelang_stages=stages,
            tilelang_dma_pattern="rw11",
            context=ctx,
        )
        assert spec.tilelang_profile is not None
        stage_etas.append(spec.tilelang_profile.dma_attainment)
        assert result.summary.logical_transfer_bytes == 2 * payload_bytes
        assert result.summary.dram_sector_bytes == 2 * payload_bytes
        eng = run_template(dp, ctx, result)
        stage_times.append(eng.now - 1.0)

    assert stage_etas == pytest.approx([0.7469, 0.9466, 0.9665])
    assert stage_etas[0] < stage_etas[1] < stage_etas[2]
    assert stage_times[0] > stage_times[1] > stage_times[2]

    # RW 2:1: SimtVF attains 0.9793 while SimdVF caps 1.0164 at 1.0.
    rw21_times: dict[str, float] = {}
    for tier in ("tilelang_simt", "tilelang_simd"):
        spec_rw, res_rw = compile_fuse_region(
            f"dma_{tier}", ("copy",), dp,
            dependencies={"copy": ()},
            op_stages={"copy": Stage("copy")},
            edge_routes=(),
            edge_bytes={},
            live_in_obligations=(read_ob,),
            live_out_obligations=(write_ob,),
            backend=tier,
            tilelang_stages=2,
            tilelang_dma_pattern="rw21",
            context=ctx,
        )
        assert spec_rw.tilelang_profile is not None
        expected_rw = 0.9793 if tier == "tilelang_simt" else 1.0
        assert spec_rw.tilelang_profile.dma_attainment == pytest.approx(expected_rw)
        rw21_times[tier] = run_template(dp, ctx, res_rw).now - 1.0
    assert rw21_times["tilelang_simd"] < rw21_times["tilelang_simt"]


def test_tilelang_dispatch_floors_aclgraph_vs_eager_tvm_ffi_and_vector_tiers() -> None:
    """ACLGraph 0.6117 us vs eager tvm_ffi 11.7728 us dispatch floors and SimtVF vs SimdVF vector ceilings."""
    dp = _a5_datapath()
    ctx = _a5_template_context(dp, vector=dp.aiv_count * 1e8, vec_sfu=dp.aiv_count * 1e8)

    # 1) Dispatch floor verification on empty kernel stage.
    disp_latencies_us: dict[str, float] = {}
    for mode in ("aclgraph", "eager_prealloc", "eager_callee_alloc"):
        spec, res = compile_fuse_region(
            f"disp_{mode}", ("noop",), dp,
            dependencies={"noop": ()},
            op_stages={"noop": Stage("noop")},
            edge_routes=(),
            edge_bytes={},
            backend="tilelang_simd",
            tilelang_dispatch_mode=mode,
            context=ctx,
        )
        assert spec.tilelang_profile is not None
        launch_stage = stages_by_name(res)["launch"]
        assert launch_stage.latency * 1000.0 == pytest.approx(spec.tilelang_profile.dispatch_us, rel=1e-6)
        disp_latencies_us[mode] = (run_template(dp, ctx, res).now - 1.0) * 1000.0

    assert disp_latencies_us["aclgraph"] == pytest.approx(0.6117, rel=1e-4)
    assert disp_latencies_us["eager_prealloc"] == pytest.approx(11.7728, rel=1e-4)
    assert disp_latencies_us["eager_callee_alloc"] == pytest.approx(15.7746, rel=1e-4)
    assert disp_latencies_us["aclgraph"] < disp_latencies_us["eager_prealloc"] < disp_latencies_us["eager_callee_alloc"]

    # 2) Vector execution tiers on 1 AIV: logical ops conserved, Engine time reflects calibrated Gops/s/AIV.
    vec_ops = 100_000_000
    tier_times_ms: dict[str, float] = {}
    for label, backend, vec_mode, pipe, expected_gops in (
        ("simt_ub", "tilelang_simt", "ub_fma", "vector", 13.0979),
        ("simt_reg", "tilelang_simt", "reg_fma", "vector", 105.6613),
        ("simd_ub", "tilelang_simd", "ub_stream", "vector", 57.7441),
        ("simd_reg", "tilelang_simd", "reg_alu", "vector", 174.5558),
        ("simd_sfu", "tilelang_simd", "sfu_exp", "vec_sfu", 70.417),
    ):
        st = Stage("v", {pipe: vec_ops}, placement={pipe: (0,)})
        _spec, res = compile_fuse_region(
            f"vec_{label}", ("v",), dp,
            dependencies={"v": ()},
            op_stages={"v": st},
            edge_routes=(),
            edge_bytes={},
            backend=backend,
            tilelang_vector_mode=vec_mode,
            tilelang_dispatch_mode="aclgraph",
            context=ctx,
        )
        assert dict(res.summary.vector_ops_by_pipe)[pipe] == vec_ops
        lowered = stages_by_name(res)["p0:t0:v"]
        assert lowered.metadata["tilelang_vector_gops_per_aiv"] == pytest.approx(expected_gops)
        eng = run_template(dp, ctx, res)
        expected_ms = (0.6117 / 1000.0) + (vec_ops / (expected_gops * 1e6))
        tier_times_ms[label] = eng.now - 1.0
        assert tier_times_ms[label] == pytest.approx(expected_ms, rel=1e-4)

    assert tier_times_ms["simd_reg"] < tier_times_ms["simt_reg"] < tier_times_ms["simd_sfu"] < tier_times_ms["simd_ub"] < tier_times_ms["simt_ub"]


def test_codesign_opportunity_dual_tier_tilelang_gains_and_ub_wall_warning() -> None:
    """Dual-tier MegaFusionBound gains, (216 KiB, 248 KiB] ACL 107000 warning, and zero A2 leakage."""
    # 1) MegaFusionBound preserves backward parity and rejects inverted dual-tier bounds.
    mf_plain = MegaFusionBound(
        region_op_ids=("l0.a", "l0.b"),
        exposed_before_ms=0.20, boundary_lower_bound_ms=0.08, tiled_engine_ms=0.10,
        gain_ceiling_ms=0.12, gain_expected_ms=0.10,
        eliminated_gm_bytes=65536, ub_live_bytes=131_072, ub_capacity_bytes=221_184,
    )
    assert (mf_plain.tilelang_simt_gain_ms, mf_plain.tilelang_simd_or_ascendc_gain_ms, mf_plain.ub_wall_warning) == (None, None, None)
    with pytest.raises(ModelingError, match="mega_fusion_simt_gain_exceeds_simd"):
        replace(mf_plain, tilelang_simt_gain_ms=0.11, tilelang_simd_or_ascendc_gain_ms=0.10)
    with pytest.raises(ModelingError, match="mega_fusion_simd_gain_exceeds_ceiling"):
        replace(mf_plain, tilelang_simd_or_ascendc_gain_ms=0.15)

    # 2) FormulaKernelSpec SRAM verification: 230 KiB passes under SimdVF/AscendC (248 KiB), 256 KiB fails both.
    a5_case = parse_case(compose_case("qwen3-32b", GEN5_DT_SKU, layers=1, dp=1, tp=1, ep=1, prompt_tokens=32, output_tokens=1))
    req = StepRequest(phase="decode", tokens=1, context=32, sequences=1, scope="body")
    # Row-reduction width N=19660 -> 19660 * 12 B (in+out+accum FP32) = 235,920 B (~230.4 KiB in (216 KiB, 248 KiB]).
    f_mid = ks_mod.FormulaKernelSpec(
        name="fused_rmsnorm_mid_ub",
        family="norm",
        equations=("y = rmsnorm(x + r)",),
        inputs=(ks_mod.FormulaTensorSpec("x", (1, 19660), "bf16"),),
        outputs=(ks_mod.FormulaTensorSpec("y", (1, 19660), "bf16"),),
        row_reduction=True,
        vector_ops_fp32=19660.0 * 8.0,
    )
    ev_mid = ks_mod.evaluate_formula_kernel(f_mid, a5_case)
    assert ev_mid.sram_verified is True
    assert 216 * KIB < ev_mid.ub_peak_bytes <= 248 * KIB
    assert ev_mid.ub_budget_bytes == 248 * KIB

    # Width N=22000 -> 22000 * 12 B = 264,000 B (> 248 KiB) fails both SimtVF and SimdVF.
    f_over = replace(
        f_mid,
        name="fused_rmsnorm_over_ub",
        inputs=(ks_mod.FormulaTensorSpec("x", (1, 22000), "bf16"),),
        outputs=(ks_mod.FormulaTensorSpec("y", (1, 22000), "bf16"),),
    )
    ev_over = ks_mod.evaluate_formula_kernel(f_over, a5_case)
    assert ev_over.sram_verified is False
    assert ev_over.tile_algorithm == "ub_infeasible"

    # 3) _collect_codesign_opportunities emits dual-tier gains and UB-wall warnings cleanly.
    base_spec = ks_mod.KernelSynthesisSpec(
        spec_id="norm_res_fit",
        kernel_name="tilelang_add_rmsnorm",
        abi_tier="K3_mega_fusion",
        suggested_backend=ks_mod.BACKEND_TILELANG_LABEL,
        status_tag=ks_mod.STATUS_NEW_OP,
        opt_pillar="pillar1_fusion",
        constituent_op_ids=("l0.add", "l0.rmsnorm"),
        constituent_traits=("elementwise", "row_reduction"),
        symbolic_equations=("y = rmsnorm(x + r)",),
        input_tensors=("x[bf16, [1, 7168]]",),
        output_tensors=("y[bf16, [1, 7168]]",),
        elided_intermediates=("x_add",),
        elided_hbm_bytes=14_336,
        tile_algorithm="resident_1pass",
        tile_shape="(M=1, N=7168, K=1)",
        ub_precision_mode="fp32_upcast",
        ub_peak_bytes=144 * KIB,
        ub_budget_bytes=216 * KIB,
        l1_peak_bytes=0,
        l1_budget_bytes=512 * KIB,
        l0_peak_bytes=0,
        l0_budget_bytes=256 * KIB,
        sram_verified=True,
        l_star_us=30.0,
        before_us=100.0,
        target_us=45.0,
        target_attainment_pct=66.7,
        exposed_before_us=100.0,
    )
    wall_spec = replace(
        base_spec,
        spec_id="norm_res_ub_wall",
        ub_peak_bytes=230 * KIB,
        ub_budget_bytes=248 * KIB,
    )
    opps_a5 = ks_mod._collect_codesign_opportunities(
        a5_case, (), (base_spec, wall_spec), (), primitive_table=None, step_request=req,
    )
    by_id: dict[str, CoDesignOpportunity] = {o.opportunity_id: o for o in opps_a5}
    mf_fit = by_id["mega_norm_res_fit"].mega_fusion
    mf_wall = by_id["mega_norm_res_ub_wall"].mega_fusion
    assert mf_fit is not None and mf_wall is not None

    # <= 216 KiB: both tiers positive, SimtVF < SimdVF/AscendC <= gain_ceiling, zero UB warning.
    assert mf_fit.ub_wall_warning is None
    assert mf_fit.tilelang_simt_gain_ms is not None and mf_fit.tilelang_simd_or_ascendc_gain_ms is not None
    assert 0.0 < mf_fit.tilelang_simt_gain_ms < mf_fit.tilelang_simd_or_ascendc_gain_ms <= mf_fit.gain_ceiling_ms + 1e-9

    # (216 KiB, 248 KiB]: SimtVF hits ACL 107000 (0.0 ms quick-win), SimdVF/AscendC retains full gain + explicit warning.
    assert mf_wall.tilelang_simt_gain_ms == pytest.approx(0.0)
    assert mf_wall.tilelang_simd_or_ascendc_gain_ms == pytest.approx(mf_wall.gain_expected_ms)
    assert mf_wall.ub_wall_warning is not None
    assert TILELANG_ACL_UB_OVERFLOW_CODE in mf_wall.ub_wall_warning
    assert "T.SimtVF" in mf_wall.ub_wall_warning and "T.SimdVF" in mf_wall.ub_wall_warning and "AscendC" in mf_wall.ub_wall_warning

    # Non-A5 hardware receives zero A5 TileLang dual-tier leakage.
    a2_case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a2"], layers=1, dp=1, tp=1, ep=1, prompt_tokens=32, output_tokens=1))
    opps_a2 = ks_mod._collect_codesign_opportunities(
        a2_case, (), (base_spec,), (), primitive_table=None, step_request=req,
    )
    mf_a2 = next(o.mega_fusion for o in opps_a2 if o.opportunity_id == "mega_norm_res_fit")
    assert mf_a2 is not None
    assert (mf_a2.tilelang_simt_gain_ms, mf_a2.tilelang_simd_or_ascendc_gain_ms, mf_a2.ub_wall_warning) == (None, None, None)
