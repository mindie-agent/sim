"""Wave 4C: grouped matrix resplit, router Split-K, gating waves and floors.

The paired ``#62`` silicon records are the acceptance target.  Every case is
priced by the common Engine from source geometry and hardware parameters, and
the data source is the archived Cold-L2 ``cold_p50_us`` record.  A5 row and
workspace plans are registered ``PlannedRecipe`` builders; A2/A3, plain gating
and non-MX quantize recipes keep their existing physical mechanisms untouched.
"""
import json
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import compose_case, operator_costs, target_graphs
from mindie_sim.catalog.ops import vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction import gdn_decode_programs as gdn
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
from mindie_sim.prediction.mxfp8_grouped import (
    M_TILE_ROWS, NZ_K_GRANULE, NZ_N_GRANULE, align as mxfp8_align,
    build as build_mxfp8, build_swiglu_mx_quant,
    mapping as mxfp8_mapping, owner_tiles as mxfp8_owner_tiles,
    panel_work as mxfp8_panel_work, recipe as mxfp8_recipe,
)
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.passes_ffn_fusion import (activation_quant_fusion,
                                                     router_gemm)
from mindie_sim.prediction.program import (KernelProgram, ParallelProgram,
                                           PipelineProgram, RecipeBook,
                                           RecipeContext)
from mindie_sim.prediction.runtime import accel_bridge, accel_build
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec

ROOT = Path(__file__).resolve().parents[1]
V41 = ROOT / "examples/cases/v41-a5-dp4-body-current.json"
GLM = (ROOT / "examples/cases/high-concurrency-20261030/glm-flash-ws9/"
       "a5_glm53flash_tp8_c32_p4k.json")
A3_GLM = (ROOT / "examples/cases/high-concurrency-20261030/glm78/"
          "a3_glm53_w4a8_tp16_tpot88_c8.json")
DT = "Ascend950DT_9582"
PR = "Ascend950PR_958b"
MS_PER_SECOND = 1000
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


def _gating_context(raw, op):
    """Production-like service modes for the row-program issue-slot binding."""
    device = parse_case(raw).machine.dies[0]
    return replace(
        _context(raw, op),
        service_modes={mode: (role, rate)
                       for mode, (role, rate) in device.service_modes})


def _splitk_context(raw, op):
    """Per-ms rates and declared mechanisms, matching the priced context."""
    device = parse_case(raw).machine.dies[0]
    rates = {role: rate / MS_PER_SECOND
             for role, rate in dict(device.rates).items()}
    names = {role: f"{role}@r0:s0" for role in rates}
    names["cube"] = "cube@r0:s0"
    units = {role: device.units(role) for role in rates}
    units["cube"] = device.compute_units
    return RecipeContext(
        op, {}, names, {names[role]: rates[role] for role in rates},
        device.compute_units,
        resource_units=units,
        compute_peaks=dict(device.compute_peaks),
        service_modes={mode: (role, rate)
                       for mode, (role, rate) in device.service_modes},
        pool_names={pool: f"{pool}@r0:s0" for pool, _ in device.buffers},
        pool_units={pool: device.pool_units(pool) for pool, _ in device.buffers},
        pool_capacities=dict(device.buffers),
        physical_mechanisms=device.physical_mechanisms)


def _bound_context(raw, plan, op):
    context = _splitk_context(raw, plan.work.apply(op))
    return replace(context, work=context.bind(plan.work))


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


def _gating_op(tokens, experts, top_k, *, groups=8, group_topk=4,
               noaux_tc=True, vector_ops=(1.0, 1.0)):
    plan = SimpleNamespace(num_experts=experts, top_k=top_k,
                           router_mode="sigmoid", router_norm_topk=True,
                           router_groups=groups, router_group_topk=group_topk,
                           routed_scaling_factor=2.5, index=0)
    members = tuple(SimpleNamespace(vector_ops=value, recipe=recipe)
                    for value, recipe in zip(vector_ops, ("a", "b")))
    return _fused_gating(prefix="p", plan=plan, logits_input="logits",
                         members=members, stage=0, requests=(), tokens=tokens,
                         mode="decode", weight_dtype="fp32", weight_bytes=4,
                         hash_table=False, noaux_tc=noaux_tc)


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
    program = plan.program(_gating_context(raw, plan.work.apply(op)))
    names = [stage.name for stage in _stages(program)]
    for required in ("kernel_prologue_drain", "logits_bias_load", "sigmoid_bias",
                     "group_top2_scan", "group_mask", "expert_topk",
                     "renorm_store"):
        assert required in names
    assert names.count("logits_bias_load") == 15
    issued = _stage_work(program)
    expected = _gating_context(raw, plan.work.apply(op)).bind(plan.work)
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


# ---------------------------------------------------------------------------
# Mixed-placement runtime parity (v0.9.19 shard-2 Row 5)
# ---------------------------------------------------------------------------
def _native_engine(tmp_path, resources):
    previous = os.environ.get("MINDIE_CACHE_HOME")
    os.environ["MINDIE_CACHE_HOME"] = str(tmp_path / "accel-cache")
    try:
        accel_build.reset_build_memo()
        status = accel_build.build_accel()
        if not status["ok"]:
            pytest.skip("native accelerator unavailable: "
                        + str(status.get("reason") or status.get("detail")))
        accel_bridge.load_library(status)
        return Engine(resources, backend="c")
    finally:
        if previous is None:
            os.environ.pop("MINDIE_CACHE_HOME", None)
        else:
            os.environ["MINDIE_CACHE_HOME"] = previous
        accel_build.reset_build_memo()


def _mixed_unit_fixture(backend, tmp_path=None, peer_count=1):
    resources = [Resource("cube", 4.0, units=4), Resource("hbm", 4.0, units=4)]
    engine = (_native_engine(tmp_path, resources) if backend == "c"
              else Engine(resources, backend="python"))
    pinned = engine.submit("pinned", {"cube": 4.0},
                           placement={"cube": (0,)})
    peers = [engine.submit(f"peer{i}", {"cube": 4.0, "hbm": 4.0},
                           coupled=True) for i in range(peer_count)]
    timer = engine.submit("timer", {}, after=(peers[0],), latency=10.0)
    for _ in range(1000):
        if pinned.done and all(task.done for task in peers) and timer.done:
            break
        engine.advance(until=(timer,), max_steps=5000)
    return engine, pinned, peers, timer


def test_unpinned_coupled_peer_matches_c11_shared_unit_capacity(tmp_path):
    python = _mixed_unit_fixture("python")
    native = _mixed_unit_fixture("c", tmp_path / "mixed-peer")
    assert native[0].accel_status()["native_steps"] > 0
    for left, right in zip((python[1], *python[2]),
                           (native[1], *native[2])):
        assert left.time == pytest.approx(right.time, abs=1e-9)
    # The pinned Cube owner keeps 0.5 of unit 0 while the coupled peer owns
    # unit 0's other 0.5 plus units 1-3, so the peer completes in 2 ms even
    # though the pinned task still runs until 5 ms.
    assert python[2][0].time == pytest.approx(2.0)
    assert python[1].time == pytest.approx(5.0)
    assert python[3].time == pytest.approx(12.0)


@pytest.mark.parametrize("peer_count", (1, 2, 3, 4, 5, 6, 7, 8))
def test_mixed_unit_cohorts_match_c11_for_every_peer_count(peer_count, tmp_path):
    python = _mixed_unit_fixture("python", peer_count=peer_count)
    native = _mixed_unit_fixture(
        "c", tmp_path / f"cohort-{peer_count}", peer_count=peer_count)
    assert native[0].accel_status()["native_steps"] > 0
    for left, right in zip((python[1], *python[2]),
                           (native[1], *native[2])):
        assert left.time == pytest.approx(right.time, abs=1e-9)
    assert native[3].time == pytest.approx(11.0 + peer_count)


# ---------------------------------------------------------------------------
# MXFP8 per-expert cold-weight floor (Row 15)
# ---------------------------------------------------------------------------
def test_mxfp8_first_m_tile_of_every_expert_is_cold(mxfp8_base):
    rows, n, k = (512, 1), 256, 256
    op = _gmm(mxfp8_base, rows, n, k)
    raw = _case(GLM, DT)
    geometry = _geometry(raw)
    plan = build_mxfp8(op, geometry, l1_k=256)
    mapping = plan.metadata["source_mapping"]
    assert mapping["weight_l2_by_bank"] == pytest.approx((0.5, 0.0))
    assert mapping["weight_l2_fraction"] == pytest.approx(0.5)
    assert -(-rows[0] // M_TILE_ROWS) == 2
    assert -(-rows[1] // M_TILE_ROWS) == 1
    payloads = [mxfp8_align(n, NZ_N_GRANULE) * mxfp8_align(k, NZ_K_GRANULE)
                for _ in rows]
    distinct = sum(payloads)
    assert distinct == pytest.approx(131072.0)
    tile = mxfp8_mapping(sum(rows), n, k, geometry, 256)
    owners = mxfp8_owner_tiles(rows, n, tile, geometry.compute_units)
    shapes = {}
    for items in owners:
        for bank, m, width, _, _ in items:
            key = (bank, m, width)
            shapes[key] = shapes.get(key, 0) + 1
    cold = 0.0
    for (bank, m, width), occurrences in shapes.items():
        load, _ = mxfp8_panel_work(
            m, width, k, k, activation_l2=0.0,
            weight_l2=mapping["weight_l2_by_bank"][bank])
        cold += (load.resources["hbm"] - m * k) * occurrences
    assert cold >= distinct


# ---------------------------------------------------------------------------
# Split-K nonempty slices and routed operand conservation (Row 16)
# ---------------------------------------------------------------------------
def test_router_split_k_prices_only_nonempty_k_slices():
    raw = _case(V41, DT)
    op = _router(1, 16, 8)
    plan = build_router_splitk(op, _geometry(raw))
    assert plan.metadata["split_k"] == 8
    assert plan.metadata["partial_write_bytes"] == pytest.approx(512.0)
    assert plan.work.resources["hbm"] == pytest.approx(1360.0)
    context = _bound_context(raw, plan, op)
    expected = context.bind(plan.work)
    issued = _stage_work(plan.program(context))
    for name, amount in expected.items():
        assert issued.get(name, 0.0) == pytest.approx(amount, rel=1e-9)


def test_router_split_k_conserves_routed_matrix_operand_service():
    raw = _case(V41, DT)
    op = _router(528, 288, 4096)
    plan = build_router_splitk(op, _geometry(raw))
    assert plan.work.resources["mte2"] == pytest.approx(42860544.0)
    context = _bound_context(raw, plan, op)
    expected = context.bind(plan.work)
    assert expected[context.resource_names["mte2_aic"]] == pytest.approx(
        42860544.0)
    issued = _stage_work(plan.program(context))
    for name, amount in expected.items():
        assert issued.get(name, 0.0) == pytest.approx(amount, rel=1e-9)


# ---------------------------------------------------------------------------
# Gating Vector Op conservation, AIV owners and n_group <= 1 (Row 17)
# ---------------------------------------------------------------------------
def _bound_gating_context(raw, plan, op):
    context = _gating_context(raw, plan.work.apply(op))
    return replace(context, work=context.bind(plan.work))


def test_gating_plan_conserves_declared_vector_ops_on_one_aiv():
    raw = _case(V41, DT)
    op = _gating_op(1, 256, 6, vector_ops=(1552.0, 1552.0))
    assert op.vector_ops == pytest.approx(3104.0)
    plan = build_moe_gating_topk(op, _geometry(raw))
    assert plan.work.resources["vector"] == pytest.approx(3104.0)
    assert plan.metadata["row_program"]["aiv_units"] == 1
    context = _bound_gating_context(raw, plan, op)
    program = plan.program(context)
    expected = context.bind(plan.work)
    assert expected[context.resource_names["vector"]] == pytest.approx(3104.0)
    issued = _stage_work(program)
    for name, amount in expected.items():
        assert issued.get(name, 0.0) == pytest.approx(amount, rel=1e-9)
    for stage_name in ("logits_bias_load", "sigmoid_bias"):
        stage = next(s for s in _stages(program) if s.name == stage_name)
        assert stage.placement[context.resource_names["vector"]] == (0,)
        assert stage.work[context.resource_names["vector"]] > 0.0

    # End-to-end DeepSeek-V3.2 1-token gating keeps serviced vector == 3,104.
    dsv32 = parse_case(compose_case(
        "deepseek-v3.2", hardware=DT, tp=1, dp=8, ep=8, num_devices=8,
        layers=4, concurrency=1))
    dsv32_ops = target_operators(
        dsv32, phase="decode", tokens=1, context=128, sequences=1, scope="body")
    real_gating = next(o for o in dsv32_ops if o.recipe == "moe_gating_topk")
    assert real_gating.vector_ops == pytest.approx(3104.0)
    row = operator_costs(dsv32, (real_gating,))["operators"][0]
    assert row["serviced"]["vector@r0:s0"] == pytest.approx(3104.0)


@pytest.mark.parametrize("groups,group_topk", ((0, 0), (1, 0), (1, 1)))
def test_gating_row_program_supports_zero_or_one_group(groups, group_topk):
    raw = _case(V41, DT)
    op = _gating_op(1, 384, 6, groups=groups, group_topk=group_topk)
    plan = build_moe_gating_topk(op, _geometry(raw))
    assert plan.metadata["row_program"]["effective_groups"] == 1
    assert plan.work.resources["vector"] == pytest.approx(op.vector_ops)
    context = _bound_gating_context(raw, plan, op)
    expected = context.bind(plan.work)
    issued = _stage_work(plan.program(context))
    for name, amount in expected.items():
        assert issued.get(name, 0.0) == pytest.approx(amount, rel=1e-9)


FULL_MODEL_GATING = (
    ("deepseek-v4.1-flash", 960, 0, 31.39, 31.70),
    ("glm-5.3-flash", 528, 1, 21.20, 20.6),
    ("deepseek-v4-pro", 240, 0, 14.46, 13.5),
    ("kimi-k3", 72, 1, 11.06, 10.0),
)


def _full_model_gating_case(model, groups, sku):
    raw = compose_case(model, hardware="a5", tp=1, dp=8, ep=8, layers=4,
                       concurrency=16)
    raw["machine"]["sku"] = sku
    raw["machine"]["hardware_key"] = sku
    case = parse_case(raw)
    plans = tuple(
        replace(plan, router_mode="sigmoid", router_noaux_tc=True,
                router_groups=groups, router_group_topk=max(1, groups))
        if getattr(plan, "ffn", "") == "moe" and getattr(plan, "num_experts", 0)
        else plan for plan in case.model.layer_plans)
    kernels = {**dict(case.runtime.kernels),
               "moe_gating": {"implementation": "vllm_ascend_cann_gating"}}
    return replace(case, model=replace(case.model, layer_plans=plans),
                   runtime=replace(case.runtime, kernels=kernels))


@pytest.mark.parametrize(
    "model,tokens,groups,dt_measured,pr_measured", FULL_MODEL_GATING)
def test_full_model_grouped_gating_row_waves_match_paired_silicon(
        model, tokens, groups, dt_measured, pr_measured):
    for sku, measured in ((DT, dt_measured), (PR, pr_measured)):
        case = _full_model_gating_case(model, groups, sku)
        ops = target_operators(case, phase="decode", tokens=tokens,
                               context=128, sequences=1, scope="body")
        gating = next(op for op in ops if op.recipe == "moe_gating_topk")
        metadata = dict(gating.metadata)
        assert metadata["groups"] == groups
        assert metadata["noaux_tc"] is True
        plan = build_moe_gating_topk(
            gating, Geometry.from_devices(case.machine.dies))
        assert plan.metadata["row_program"]["effective_groups"] == max(1, groups)
        assert plan.work.resources["vector"] == pytest.approx(
            float(gating.vector_ops))
        duration = operator_costs(
            case, (gating,))["operators"][0]["duration_ms"] * 1000
        assert duration == pytest.approx(measured, rel=TOLERANCE), (sku, model)


# ---------------------------------------------------------------------------
# CausalConv1d UPDATE state-ring ordering (Row 18)
# ---------------------------------------------------------------------------
def test_causal_conv_state_write_follows_the_final_token_input():
    case = parse_case(compose_case("kimi-k3", hardware="a5", tp=1, dp=1,
                                   ep=1, layers=1))
    device = case.machine.dies[0]
    geometry = Geometry.from_devices(case.machine.dies)
    rates = dict(device.rates)
    names = {role: f"{role}@r0:s0" for role in rates}
    names["cube"] = "cube@r0:s0"
    units = {role: device.units(role) for role in rates}
    units["cube"] = device.compute_units
    context = RecipeContext(
        None, {}, names, {names[role]: rates[role] for role in rates},
        device.compute_units, resource_units=units,
        compute_peaks=dict(device.compute_peaks))
    shape = (6, 1024)
    op = OpSpec(
        op_id="conv", family="short_convolution", phase="decode", stage=0,
        layer=0, requests=(), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, dtype="bf16", recipe=gdn.CONV,
        metadata=(("sequence_tokens", (6,)),
                  ("physical_sequence_tokens", (6,)),
                  ("kernel_width", 4), ("speculative", False),
                  ("bias", False), ("state_history_slots", 3),
                  ("padding_slots", ())))
    plan = gdn.conv_plan(op, geometry)
    engine = Engine([Resource(names[role], rates[role], units=units[role])
                     for role in rates], trace_limit=10000)
    context = replace(context, op=plan.work.apply(op))
    done = plan.program(context).start(engine, "conv", engine.submit("ready"))
    engine.advance(until=(done,), max_steps=100000)
    assert plan.metadata["state_write_bytes"] == pytest.approx(6144.0)
    inputs = [event for event in engine.trace
              if event["demand"] and event["task"].endswith(":input")]
    writes = [event for event in engine.trace
              if event["demand"] and event["task"].endswith(":state_write")]
    assert inputs and writes
    final_input = max(inputs, key=lambda event: event["start_ms"])
    final_input_end = max(event["end_ms"] for event in inputs)
    assert all(write["start_ms"] >= final_input["start_ms"] for write in writes)
    assert all(write["start_ms"] >= final_input_end for write in writes)
