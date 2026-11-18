"""Wave 4C: grouped matrix resplit, router Split-K, gating waves and floors.

The paired ``#62`` silicon records are the acceptance target.  Every case is
priced by the common Engine from source geometry and hardware parameters, and
the data source is the archived Cold-L2 ``cold_p50_us`` record.  A5 row and
workspace plans are registered ``PlannedRecipe`` builders; A2/A3, plain gating
and non-MX quantize recipes keep their existing physical mechanisms untouched.
"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.catalog.ops import vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.matrix_splitk import (_allocate, _per_owner_cap,
                                                 build_router_splitk,
                                                 router_output_tiles,
                                                 router_split_count)
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.moe_cann_fusion import (_fused_gating,
                                                   build_moe_gating_topk)
from mindie_sim.prediction.mxfp4_geometry import (
    Config, activation_l2_fraction, columns, effective_l2_bytes,
    resplit_columns, weight_l2_fraction,
)
from mindie_sim.prediction.mxfp4_grouped import build as build_mxfp4
from mindie_sim.prediction.mxfp4_grouped import recipe as mxfp4_recipe
from mindie_sim.prediction.mxfp8_grouped import recipe as mxfp8_recipe
from mindie_sim.prediction.mxfp8_grouped import build_swiglu_mx_quant
from mindie_sim.prediction.passes_ffn_fusion import (activation_quant_fusion,
                                                     router_gemm)
from mindie_sim.prediction.program import (KernelProgram, ParallelProgram,
                                           PipelineProgram, RecipeBook,
                                           RecipeContext)
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]
V41 = ROOT / "examples/cases/v41-a5-dp4-body-current.json"
GLM = (ROOT / "examples/cases/high-concurrency-20261030/glm-flash-ws9/"
       "a5_glm53flash_tp8_c32_p4k.json")
A3_GLM = (ROOT / "examples/cases/high-concurrency-20261030/glm78/"
          "a3_glm53_w4a8_tp16_tpot88_c8.json")
DT = "Ascend950DT_9582"
PR = "Ascend950PR_958b"
TOLERANCE = 0.12
#: Registered Split-K router workspace and the narrow resplit's per-pass small
#: column tiles use the declared small-kernel bound.
SMALL_TOLERANCE = 0.15
#: The narrow ``N=4608`` resplit traverses 32 128/160-column passes; its cold
#: per-pass panel read is a retained source-order effect the compact loop
#: bounds conservatively.  It stays the small-kernel 0.15 bound, never 0.30.
NARROW_RESPLIT_TOLERANCE = SMALL_TOLERANCE
ROUTER_PR_TOLERANCE = SMALL_TOLERANCE


def _case(path, sku):
    raw = json.loads(path.read_text())
    raw["machine"]["sku"] = sku
    raw["machine"]["hardware_key"] = sku
    return raw


def _price(raw, op, *, recipe=None, family=None):
    recipes = (RecipeBook(families={family: recipe}) if recipe is not None else None)
    return operator_costs(raw, (op,), recipes=recipes)["operators"][0]["duration_ms"] * 1000


def _geometry(raw):
    return Geometry.from_devices(parse_case(raw).machine.dies)


def _context(raw, op):
    device = parse_case(raw).machine.dies[0]
    rates = dict(device.rates)
    names = {role: f"{role}@r0:s0" for role in rates}
    names["cube"] = "cube@r0:s0"
    return RecipeContext(
        op, {}, names, {names[role]: rates[role] for role in rates},
        device.compute_units,
        resource_units={role: device.units(role) for role in rates},
        compute_peaks=dict(device.compute_peaks),
        pool_names={pool: f"{pool}@r0:s0" for pool, _ in device.buffers},
        pool_units={pool: device.pool_units(pool) for pool, _ in device.buffers})


def _stages(program):
    """Every finite Stage a registered plan issues, including nested windows."""
    if isinstance(program, KernelProgram):
        for stage in program.stages:
            yield stage
            if stage.program is not None:
                yield from _stages(stage.program)
    elif isinstance(program, PipelineProgram):
        for index in range(program.count):
            for stage in program.tile(index):
                yield stage
                if stage.program is not None:
                    yield from _stages(stage.program)
    elif isinstance(program, ParallelProgram):
        for branch in program.programs:
            yield from _stages(branch)


def _stage_work(program):
    issued = {}
    for stage in _stages(program):
        for name, amount in stage.work.items():
            issued[name] = issued.get(name, 0.0) + float(amount)
    return issued


@pytest.fixture(scope="module")
def mxfp4_base():
    raw = _case(V41, DT)
    raw["runtime"]["routing"] = {"policy": "balanced"}
    raw["runtime"].setdefault("kernels", {}).setdefault("families", {})[
        "grouped_gemm"] = {"kind": "ascend_mxa8w4_nznk"}
    ops = target_graphs(raw, phase="decode", tokens=1, context=8, scope="body",
                        resource_fusion=False)[0]
    return next(op for op in ops if op.family == "grouped_gemm")


@pytest.fixture(scope="module")
def mxfp8_base():
    raw = _case(GLM, DT)
    ops = target_graphs(raw, phase="decode", tokens=32, context=4096, sequences=32,
                        scope="body", resource_fusion=False)[0]
    return next(op for op in ops
                if op.family == "grouped_gemm" and op.op_id.endswith("expert_up"))


def _gmm(base, rows, n, k):
    metadata = dict(base.metadata)
    metadata.update(matrix_group_rows=rows, matrix_groups=len(rows),
                    matrix_weight_bank_groups=len(rows),
                    activation_scale_shape=(sum(rows), (k + 63) // 64, 2))
    shape = (sum(rows), n, k)
    return replace(base, local_shape=shape, semantic_shape=shape,
                   launch_shape=shape, metadata=tuple(metadata.items()))


MX4_SHAPES = {
    "gate_up_4096": ((480,) * 12, 4096, 5120, 292.00, 316.69, TOLERANCE),
    "gate_up_4608": ((480,) * 12, 4608, 5120, 330.89, 434.84,
                     NARROW_RESPLIT_TOLERANCE),
    "down_2304": ((480,) * 12, 5120, 2304, 169.20, 246.60, TOLERANCE),
    "small_m90": ((30, 30, 30), 4096, 7168, 32.73, 43.95, SMALL_TOLERANCE),
}
MX8_SHAPES = {
    "glm_up_3072": ((470, 470, 470, 469, 469, 469, 469, 469, 469), 3072, 4096,
                    141.28, 184.83),
    "glm_up_4096": ((470, 470, 470, 469, 469, 469, 469, 469, 469), 4096, 4096,
                    178.83, 245.85),
    "glm_down_2048": ((470, 470, 470, 469, 469, 469, 469, 469, 469), 4096, 2048,
                      98.71, 135.71),
}


@pytest.mark.parametrize("label", sorted(MX4_SHAPES))
def test_mxfp4_resplit_and_small_m_banks_match_paired_silicon(mxfp4_base, label):
    rows, n, k, dt_measured, pr_measured, tolerance = MX4_SHAPES[label]
    op = _gmm(mxfp4_base, rows, n, k)
    for sku, measured in ((DT, dt_measured), (PR, pr_measured)):
        duration = _price(_case(V41, sku), op, recipe=mxfp4_recipe(),
                          family="grouped_gemm")
        assert duration == pytest.approx(measured, rel=tolerance), (sku, label)


@pytest.mark.parametrize("label", sorted(MX8_SHAPES))
def test_mxfp8_grouped_cold_l2_pipeline_matches_paired_silicon(mxfp8_base, label):
    rows, n, k, dt_measured, pr_measured = MX8_SHAPES[label]
    op = _gmm(mxfp8_base, rows, n, k)
    for sku, measured in ((DT, dt_measured), (PR, pr_measured)):
        duration = _price(_case(GLM, sku), op, recipe=mxfp8_recipe(l1_k=256),
                          family="grouped_gemm")
        assert duration == pytest.approx(measured, rel=TOLERANCE), (sku, label)


def test_l2_fractions_derive_from_declared_capacity_not_empirical_shares():
    wide = Config(5120, 4096, (480,) * 12)
    narrow = Config(5120, 4608, (480,) * 12)
    geometry = _geometry(_case(V41, DT))
    l2_bytes = effective_l2_bytes(geometry)
    assert l2_bytes == pytest.approx(
        geometry.physical_mechanisms.memory_hierarchy.effective_l2_bytes)
    assert next(iter(columns(wide, 32))) == (0, 256)
    assert [width for _, width in columns(narrow, 32)] == [128] * 16 + [160] * 16
    assert resplit_columns(wide, 32) is False
    assert resplit_columns(narrow, 32) is True
    # Doubling the declared effective capacity can only retain more, and a
    # zero-capacity declaration retains nothing; no fixed share is embedded.
    assert weight_l2_fraction(wide, 32, 0) == 0.0
    assert activation_l2_fraction(wide, 32, 0) == 0.0
    assert weight_l2_fraction(wide, 32, 2 * l2_bytes) >= \
        weight_l2_fraction(wide, 32, l2_bytes)
    assert 0.0 < weight_l2_fraction(wide, 32, l2_bytes) <= 1.0
    assert 0.0 < activation_l2_fraction(narrow, 32, l2_bytes) <= 1.0


def _router(m, n, k):
    plan = SimpleNamespace(router_dtype="", router_output_dtype="fp32",
                           num_experts=n, router_input_size=k, index=0)
    model = SimpleNamespace(
        quant=QuantSpec(name="mx", weight_format="mxfp8",
                        activation_format="mxfp8"),
        activation_dtype="bf16")
    _, router, _, _ = router_gemm(
        prefix="p", plan=plan, deps=("x",), stage=0, requests=(), tokens=m,
        model=model, hidden=k, mode="decode")
    return router


@pytest.mark.parametrize("shape,dt_measured,pr_measured", [
    ((528, 288, 4096), 16.14, 19.66),
    ((240, 384, 7168), 15.52, 20.63),
])
def test_narrow_bf16_router_engages_split_k(shape, dt_measured, pr_measured):
    op = _router(*shape)
    metadata = dict(op.metadata)
    assert metadata["split_k_required"] is True
    assert metadata["weight_format"] == "bf16"
    assert metadata["output_format"] == "fp32"
    tiles = router_output_tiles(shape[0], shape[1])
    assert tiles > 0
    assert router_split_count(tiles) > 1
    for sku, measured, tolerance in ((DT, dt_measured, TOLERANCE),
                                     (PR, pr_measured, ROUTER_PR_TOLERANCE)):
        duration = _price(_case(V41, sku), op)
        assert duration == pytest.approx(measured, rel=tolerance), sku


def test_router_gemm_declares_bf16_fp32_tier4_and_split_k():
    plan = SimpleNamespace(router_dtype="", router_output_dtype="fp32",
                           num_experts=288, router_input_size=4096, index=0)
    model = SimpleNamespace(quant=QuantSpec(name="mx", weight_format="mxfp8",
                                            activation_format="mxfp8"),
                            activation_dtype="bf16")
    prefix_ops, router, weight_dtype, _ = router_gemm(
        prefix="p", plan=plan, deps=("x",), stage=0, requests=(), tokens=528,
        model=model, hidden=4096, mode="decode")
    metadata = dict(router.metadata)
    assert prefix_ops == []
    assert metadata["weight_format"] == "bf16"
    assert metadata["output_format"] == "fp32"
    assert metadata["split_k_required"] is True
    assert weight_dtype == "fp32"


def test_router_split_k_plan_conserves_base_and_workspace_work():
    raw = _case(V41, DT)
    op = _router(528, 288, 4096)
    plan = build_router_splitk(op, _geometry(raw))
    assert plan.metadata["split_k"] > 1
    assert plan.metadata["output_tiles"] == router_output_tiles(528, 288)
    partial = float(plan.metadata["partial_write_bytes"])
    assert plan.work.resources["hbm"] == pytest.approx(
        op.hbm_bytes + 2 * partial)
    assert plan.work.matrix["bf16"] == pytest.approx(op.flops)
    program = plan.program(_context(raw, plan.work.apply(op)))
    names = [stage.name for stage in _stages(program)]
    assert "narrow_matrix" in names
    assert "splitk_workspace" in names


def test_split_k_allocation_normalizes_padded_macro_areas():
    shares = _allocate(1000.0, [1.5, 1.5, 0.5])
    assert sum(shares) == pytest.approx(1000.0)
    assert shares == pytest.approx([428.5714285714, 428.5714285714, 142.8571428571])


def _gating_op(tokens, experts, top_k):
    plan = SimpleNamespace(num_experts=experts, top_k=top_k,
                           router_mode="sigmoid", router_norm_topk=True,
                           router_groups=8, router_group_topk=4,
                           routed_scaling_factor=2.5, index=0)
    members = (SimpleNamespace(vector_ops=1.0, recipe="a"),
               SimpleNamespace(vector_ops=1.0, recipe="b"))
    return _fused_gating(prefix="p", plan=plan, logits_input="logits",
                         members=members, stage=0, requests=(), tokens=tokens,
                         mode="decode", weight_dtype="fp32", weight_bytes=4,
                         hash_table=False, noaux_tc=True)


@pytest.mark.parametrize("tokens,experts", [
    (1, 384), (63, 384), (64, 384), (65, 384), (72, 896), (127, 288),
    (128, 288), (129, 288), (240, 384), (528, 288), (960, 384),
])
def test_gating_op_conserves_logical_work_without_a_token_cliff(tokens, experts):
    op = _gating_op(tokens, experts, 6)
    metadata = dict(op.metadata)
    assert op.mte2_bytes == float(tokens * experts * 4 + experts * 4)
    assert op.mte3_bytes == float(tokens * 6 * 8)
    assert op.hbm_bytes == op.mte2_bytes + op.mte3_bytes
    # Two declared member expressions, never a token-count prologue.
    assert op.vector_ops == pytest.approx(2.0)
    assert metadata["row_tokens"] == tokens
    assert metadata["row_experts"] == experts


@pytest.mark.parametrize("tokens,experts,top_k,dt_measured,pr_measured", [
    (960, 384, 6, 31.39, 31.70),
    (528, 288, 8, 21.20, 20.60),
    (240, 384, 6, 14.46, 13.50),
    (72, 896, 8, 11.06, 10.00),
])
def test_grouped_moe_gating_topk_row_waves(tokens, experts, top_k, dt_measured,
                                           pr_measured):
    op = _gating_op(tokens, experts, top_k)
    for sku, measured in ((DT, dt_measured), (PR, pr_measured)):
        assert _price(_case(V41, sku), op) == pytest.approx(
            measured, rel=TOLERANCE), sku


def test_gating_row_wave_program_is_finite_and_conserving():
    raw = _case(V41, DT)
    op = _gating_op(960, 384, 6)
    plan = build_moe_gating_topk(op, _geometry(raw))
    program = plan.program(_context(raw, plan.work.apply(op)))
    names = [stage.name for stage in _stages(program)]
    for required in ("kernel_prologue_drain", "logits_bias_load", "sigmoid_bias",
                     "group_top2_scan", "group_mask", "expert_topk",
                     "renorm_store"):
        assert required in names
    assert names.count("logits_bias_load") == 15
    issued = _stage_work(program)
    expected = _context(raw, plan.work.apply(op)).bind(plan.work)
    for name, amount in expected.items():
        assert issued.get(name, 0.0) == pytest.approx(amount, rel=1e-9)
    # A2/A3 keeps the registered Class D vector plan byte-for-byte.
    a3 = _geometry(_case(A3_GLM, "Ascend910_9391"))
    fallback = build_moe_gating_topk(op, a3)
    assert fallback.metadata["physical_class"] == "AIVTiledVectorAndGatherRecipe"


def _swiglu_pair(rows, width):
    act = vector_op(
        op_id="act", deps=("x",), stage=0, layer=0, requests=(),
        elements=rows * width, ops_per_element=2, dtype="bf16",
        family="swiglu", phase="decode", recipe="grouped_swiglu",
        shape=(rows, width // 2), read_bytes_override=rows * width * 2,
        write_bytes_override=rows * width,
        metadata=(("activation_group_size", 32), ("scale_format", "fp8_e8m0")))
    quant = vector_op(
        op_id="act.quantize", deps=(act.op_id,), stage=0, layer=0, requests=(),
        elements=rows * width // 2, ops_per_element=4, dtype="bf16",
        family="quantize", phase="decode", recipe="dynamic_quantize",
        shape=(rows, width // 2), read_bytes_override=rows * width,
        write_bytes_override=rows * width // 2 * 33 // 32,
        metadata=(("activation_group_size", 32), ("scale_format", "fp8_e8m0")))
    return act, quant


@pytest.mark.parametrize("rows,width", [
    (5760, 4608), (4224, 4096), (1440, 6144), (1152, 6144),
])
def test_swiglu_quantize_conserves_act_mte2_and_quant_mte3(rows, width):
    act, quant = _swiglu_pair(rows, width)
    op = activation_quant_fusion((act, quant), "bf16")[0]
    assert op.mte2_bytes == pytest.approx(float(act.mte2_bytes))
    assert op.mte3_bytes == pytest.approx(float(quant.mte3_bytes))
    assert op.hbm_bytes == pytest.approx(act.mte2_bytes + quant.mte3_bytes)
    assert op.vector_ops == pytest.approx(
        float(act.vector_ops) + float(quant.vector_ops))
    assert dict(op.metadata)["fused_kernel"] == "npu_swiglu_mx_quant"
    assert dict(op.metadata)["row_program"]["group_size"] == 32
    raw = _case(V41, DT)
    plan = build_swiglu_mx_quant(op, _geometry(raw))
    assert plan.work.resources["mte2"] == pytest.approx(float(act.mte2_bytes))
    assert plan.work.resources["mte3"] == pytest.approx(float(quant.mte3_bytes))
    program = plan.program(_context(raw, plan.work.apply(op)))
    issued = _stage_work(program)
    expected = _context(raw, plan.work.apply(op)).bind(plan.work)
    for name, amount in expected.items():
        assert issued.get(name, 0.0) == pytest.approx(amount, rel=1e-9)


@pytest.mark.parametrize("rows,width,dt_measured,pr_measured", [
    (5760, 4608, 34.27, 74.80),
    (4224, 4096, 25.36, 50.30),
    (1440, 6144, 14.95, 27.30),
    (1152, 6144, 13.47, 22.60),
])
def test_standalone_swiglu_mx_quant_row_program(rows, width, dt_measured,
                                                pr_measured):
    act, quant = _swiglu_pair(rows, width)
    op = activation_quant_fusion((act, quant), "bf16")[0]
    for sku, measured in ((DT, dt_measured), (PR, pr_measured)):
        assert _price(_case(V41, sku), op) == pytest.approx(
            measured, rel=TOLERANCE), sku


def test_small_nz_qmm_prologue_lifts_wkv_into_the_measured_band():
    raw = _case(V41, DT)
    ops = target_graphs(raw, phase="decode", tokens=960, context=8, scope="body",
                        resource_fusion=False)[0]
    op = next(op for op in ops if op.op_id == "s0.l0.kv_a")
    assert dict(op.metadata)["weight_representation"] == \
        "fp8_block32x32_fp8_e8m0"
    assert _price(raw, op) == pytest.approx(13.37, rel=TOLERANCE)
    assert _price(_case(V41, PR), op) == pytest.approx(15.52, rel=TOLERANCE)


def test_nz_block32_prologue_is_monotonic_in_work_size():
    durations = []
    for tokens in (64, 72, 128, 240, 384, 528, 960, 1920):
        raw = _case(V41, DT)
        ops = target_graphs(raw, phase="decode", tokens=tokens, context=8,
                            scope="body", resource_fusion=False)[0]
        op = next(op for op in ops if op.op_id == "s0.l0.kv_a")
        durations.append(_price(raw, op))
    assert all(later >= earlier for earlier, later in
               zip(durations, durations[1:])), durations


def test_nz_block32_prologue_is_a5_arch35_only():
    from mindie_sim.prediction.invocation_binding import _arch35
    a5 = _context(_case(V41, DT), _gating_op(64, 8, 2))
    a3 = _context(_case(A3_GLM, "Ascend910_9391"), _gating_op(64, 8, 2))
    assert _arch35(a5) is True
    assert _arch35(a3) is False


def test_router_workspace_scales_with_declared_dram_service():
    raw = _case(V41, DT)
    context = _context(raw, _router(240, 384, 7168))
    caps = _per_owner_cap(replace(
        context, service_modes={"mte2_aic_dram": ("mte2_aic", 1.2e9)}), 4)
    assert caps == {context.resource_names["hbm"]: 1.2e9 / 4000.0}


def test_grouped_gemm_resplit_branch_uses_the_source_schedule():
    raw = _case(V41, DT)
    op = _gmm(_base_gmm(), (480,) * 12, 4608, 5120)
    plan = build_mxfp4(op, _geometry(raw))
    detail = plan.metadata["mxa8w4"]
    assert detail["resplit_schedule"] is True
    assert detail["producer_mode"] == "dynamic_buffer4"
    assert detail["k_panel_execution"] == "detailed_panels"


def _base_gmm():
    source = _case(V41, DT)
    source["runtime"]["routing"] = {"policy": "balanced"}
    source["runtime"].setdefault("kernels", {}).setdefault("families", {})[
        "grouped_gemm"] = {"kind": "ascend_mxa8w4_nznk"}
    ops = target_graphs(source, phase="decode", tokens=1, context=8, scope="body",
                        resource_fusion=False)[0]
    return next(op for op in ops if op.family == "grouped_gemm")
