"""CPU execution and conservation checks for the kernel template catalog.
[Role] Exercise CATLASS, DSL reference kernels, and FuseRegion compilation.
[Invariants] Check exact tile, dependency, and traffic contracts independently.
[Timeouts & Locks] Bounded schedules on the Python engine without external work.
[Side-Effect Policy] Read packaged inputs and write Git fixtures under temporary roots.
"""
import tracemalloc
from dataclasses import replace
from fractions import Fraction
from itertools import product
from math import ceil
from time import perf_counter
from types import SimpleNamespace

import pytest

from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS, SKUS
from _support.hardware import machine
from _support.linear_cases import (
    REPO_ROOT as ROOT, compile_tile_region as _region, enumerate_matrix as _matrix,
    load_template_provenance, lower_reference as _reference, mounted_catalog_sources,
    provenance_git_fixture, run_template as _run, stage_ancestors as _ancestors,
    stage_bytes as _bytes, stages_by_name as _stages, template_context as _context,
    recipe_context, serpentine, sku_planning_context, template_datapath as _datapath, template_live_bindings, tile_edge as _edge,
    verify_template_declaration, visitor_graph, with_mx_modes,
)
from mindie_sim.optimization.reload_bounds import compute_compulsory_hbm_cut
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.program import Allocation, Stage
from mindie_sim.prediction.runtime.memory_physics import AddressRun, AddressRunPlan, AdmissionMemoryView, MemoryAccessPattern, ReuseKey
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, TileInfeasibleError
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.device_input import MemoryAccessBurstPenalty
from mindie_sim.templates import (
    CatlassProblem, SemanticWorkItem, TensorRangeObligation, builtin_template_catalog,
    compile_fuse_region, compose_step_obligations, enumerate_dsl_specs, lower_catlass_spec, lower_dsl_spec,
)
from mindie_sim.templates import catlass_catalog as cat, dsl_catalog as dsl
from mindie_sim.templates.catlass_lower import solve_catlass_tile
from mindie_sim.templates.contracts import evg_order
from mindie_sim.templates.fuse_region import Lowering, validate_live_buffer_owners

pytestmark = pytest.mark.tier0

KIB = 1024
SECTOR = 32
KV_BYTES = 672
ONE_SLOT = (1, 1, 1, 1, 1)
TURNAROUND = MemoryAccessBurstPenalty("turnaround", 0, None, 0.1, "test-geometry")
@pytest.fixture(scope="module")
def datapaths(): return tuple(_datapath(REPRESENTATIVE_SKUS[gen]) for gen in ("a2", "a5"))
@pytest.fixture(scope="module")
def direct(datapaths): return datapaths[1]
@pytest.mark.parametrize("sku", SKUS)
def test_pingpong_sram_fit_and_ring_credit_execution(sku):
    dp, p = _datapath(sku), CatlassProblem(32, 32, 160)
    options = dict(tile=(32, 32, 64), l0_tiles=((32, 32, 32),), stage_options=(ONE_SLOT,))
    limits = {"l1": 2 * 32 * 64 * 2, "l0a": 32 * 32 * 2, "l0b": 32 * 32 * 2, "l0c": 32 * 32 * 4}
    exact = replace(dp, buffers=tuple((key, limits.get(key, size)) for key, size in dp.buffers))
    single, = _matrix("basic_pingpong", p, exact, **options)
    assert {row.pool: row.peak_bytes for row in solve_catlass_tile(single, p, exact).pool_peaks} == limits
    for pool, required in limits.items():
        short = replace(exact, buffers=tuple((key, size - (key == pool)) for key, size in exact.buffers))
        assert not _matrix("basic_pingpong", p, short, **options)
        with pytest.raises(TileInfeasibleError) as error:
            solve_catlass_tile(single, p, short)
        rejection, = error.value.rejections
        assert (rejection.pool, rejection.required_bytes, rejection.available_bytes) == (pool, required, required - 1)
    spec, = _matrix("basic_pingpong", p, dp, tile=options["tile"], l0_tiles=options["l0_tiles"])
    context = _context(dp, p)
    result = lower_catlass_spec(spec, p, dp, context=context, die=1)
    summary = result.summary
    assert {entry.die for entry in (*summary.semantic_work, *summary.read_obligations, *summary.write_obligations)} == {1}
    stages = _stages(result)
    for operand in ("a", "b"):
        assert f"tile:0:l1{operand}:g0:last_use" in stages[f"tile:0:k2:gm_{operand}"].after
        assert f"tile:0:l0{operand}:g0:last_use" in stages[f"tile:0:l0{operand}:g2:load"].after
    assert sum(s.metadata.get("logical_flops", 0) for s in stages.values()) == 2 * p.m * p.n * p.k
    assert _run(dp, context, result).serviced["cube"] * context.rates["cube"] == pytest.approx(
        dict(result.summary.issued_cube_flops_by_dtype)["bf16"])
@pytest.mark.parametrize("has_bias", (False, True))
def test_mutex_budget_unit_flag_and_l1_resident_identity(direct, has_bias):
    p = CatlassProblem(32, 32, 32, has_bias=has_bias)
    fits = (9, 7, 6, 5 - 2 * has_bias, 1)
    assert sum(fits) + 2 * has_bias == 28 and len(set(fits)) == 5
    spec, = _matrix("mutex_block_mmad", p, direct, stage_options=(fits,), enable_unit_flag=True)
    peaks = dict(lower_catlass_spec(spec, p, direct).summary.sram_peaks_bytes)
    assert (spec.stages, peaks["l1"]) == (fits, (fits[0] + fits[1]) * 32 * 32 * 2 + has_bias * 32 * 4)
    with pytest.raises(ModelingError, match="catlass_mutex_ids_exhausted"):
        solve_catlass_tile(replace(spec, stages=(fits[0] + 1, *fits[1:])), p, direct)
    two_acc = (fits[0] - 1, *fits[1:-1], 2)
    assert solve_catlass_tile(replace(spec, stages=two_acc, enable_unit_flag=False), p, direct).depth == max(two_acc)
    with pytest.raises(ModelingError, match="catlass_unit_flag_requires_single_l0c_stage"):
        solve_catlass_tile(replace(spec, stages=two_acc), p, direct)
    for same_addr, same_coord, enabled in product((False, True), repeat=3):
        assert cat.should_elide_l1_reload("X", (1, 0), "X" if same_addr else "Y",
                                         (1, 0) if same_coord else (0, 1), enabled) == all((same_addr, same_coord, enabled))
    reuse = replace(p, n=2 * p.n)
    resident, = _matrix("mutex_block_mmad", reuse, direct, aic_budget=1, stage_options=(ONE_SLOT,), enable_l1_resident=True)
    stages = _stages(lower_catlass_spec(resident, reuse, direct))
    assert "tile:1:k0:gm_a" not in stages and "tile:1:k0:gm_b" in stages
    context = _context(direct, reuse, fixpipe=1)
    result = lower_catlass_spec(replace(resident, stages=(*ONE_SLOT[:-1], 2)), reuse, direct, context=context)
    trace = {r["task"].removeprefix("matrix:"): r for r in _run(direct, context, result, trace_limit=200).trace}
    mma, fix = trace["tile:1:micro:0:mma"], trace["tile:0:fixpipe"]
    assert mma["start_ms"] < fix["end_ms"] and fix["start_ms"] < mma["end_ms"]
@pytest.mark.parametrize("blocks,gk,cores,total", [(8, 3, 4, 0), (2, 2, 4, 4), (13, 7, 5, 7), (15, 9, 6, 8)])
def test_streamk_counts_intersected_output_tiles(blocks, gk, cores, total):
    partition = cat.streamk_partition(blocks, gk, cores)
    assert (partition.b_sk, partition.b_norm) == (blocks % cores, blocks - blocks % cores)
    assert partition.intervals[0][0] == 0 and partition.intervals[-1][1] == partition.b_sk * gk
    assert partition.partial_counts == tuple(len({i // gk for i in range(s, e)}) for s, e in partition.intervals)
    assert partition.partial_count == total
def test_streamk_signals_before_the_last_normal_work_and_reduction(direct):
    p = CatlassProblem(32, 320, 64)
    spec, = _matrix("streamk", p, direct)
    stages = _stages(lower_catlass_spec(spec, p, direct))
    flag = "cross_core_set_flag"
    assert set(stages[flag].after) == {f"tile:{i}:partial" for i in range(4, 8)}
    assert all({f"tile:{i}:fixpipe" for i in range(4)} <= _ancestors(stages, partial) for partial in stages[flag].after)
    assert all(flag in stages[f"tile:{i}:k0:gm_a"].after for i in range(8, 12))
    for i in range(2):
        deps = _ancestors(stages, f"reduce:{i}:add")
        assert flag in deps and not deps.intersection(f"tile:{j}:fixpipe" for j in range(8, 12))
        assert f"reduce:{i}:add" not in _ancestors(stages, "tile:8:fixpipe")
def test_splitk_uses_ceil_grid_and_rebinds_workspace_under_pressure(direct):
    p = CatlassProblem(32, 32, 192)
    options = dict(tile=(32, 32, 128), aic_budget=2, aiv_budget=4, stage_options=(ONE_SLOT,))
    spec, = _matrix("multi_core_splitk", p, direct, **options)
    assert spec.splitk_factor == 2 and spec.workspace_bytes == 2 * p.m * p.n * 4
    with pytest.raises(ModelingError, match="catlass_splitk_factor_invalid"):
        lower_catlass_spec(spec, replace(p, k=128), direct)
    context = _context(direct, p)
    inputs, output = (p.m + p.n) * p.k * 2, p.m * p.n * 2
    for pressured in (False, True):
        view = AdmissionMemoryView({0: KIB**2}, granule_bytes=SECTOR)
        result = lower_catlass_spec(spec, p, direct, context=context, memory_view=view, live_admission=True)
        assert result.summary.logical_transfer_bytes == inputs + output + 2 * spec.workspace_bytes
        assert result.dram_miss_bytes == inputs + output
        if pressured:
            view.grant_reservation(0, "concurrent", KIB**2)
        delivered = _run(direct, context, result).delivered
        assert result.summary.realized_hbm_bytes == pytest.approx(delivered["hbm:ctr_r"] + delivered["hbm:ctr_w"])
        assert result.summary.realized_hbm_bytes == (43_008 if pressured else inputs + output)
        assert result.summary.dram_sector_bytes == result.summary.turnaround_service_bytes == result.summary.realized_hbm_bytes
        assert (result.l2_hit_bytes, result.dram_miss_bytes) == (0 if pressured else 16_384, result.summary.dram_sector_bytes)
        assert result.l2_hit_bytes + result.dram_miss_bytes == result.summary.issued_copy_bytes
        assert delivered["hbm:ctr_r"] == pytest.approx(inputs + pressured * spec.workspace_bytes)
        assert delivered["hbm:ctr_w"] == pytest.approx(output + pressured * spec.workspace_bytes)
    wide = replace(p, n=2 * p.n, dtype_a="fp16", dtype_b="fp16")
    spec, = _matrix("multi_core_splitk", wide, direct, **options)
    result = lower_catlass_spec(spec, wide, direct)
    stages = _stages(result)
    for block in range(2):
        assert f"tile:{block}:k0:gm_b" in _ancestors(stages, f"tile:{block}:partial")
        assert f"tile:{block}:partial" in _ancestors(stages, f"tile:{block+2}:partial")
    item, = result.summary.semantic_work
    assert item.logical_ops == 2 * wide.m * wide.n * wide.k and item.eligible_engines == ("cube", "aiv_simt")
    assert dict(result.summary.vector_ops_by_pipe)["vec_reduce"] == wide.m * wide.n
def test_swizzle_serpentine_coordinates_and_admission_reuse(direct):
    for direction, expected in (("Zn", serpentine(5, 3)), ("Nz", [(j, i) for i, j in serpentine(3, 5)])):
        assert [cat.swizzle_block_coord(i, 5, 3, direction, 2) for i in range(15)] == expected
        for active in (1, 4, 7, 15, 20):
            assert cat.wave_band_counts(5, 3, active, direction, 2) == tuple(map(len, map(set, zip(*expected[:active]))))
    p = CatlassProblem(48, 32, 32)
    specs = _matrix("swizzle", p, direct, tile=(16, 16, 32), swizzle_offsets=(1, 2))
    assert [cat.swizzle_block_coord(i, 3, 2, "Nz", 2) for i in range(6)] == list(product(range(3), range(2)))
    for direction, offset, expected in (("Nz", 2, 2 * KIB), ("Zn", 1, 3 * KIB)):
        spec, = (s for s in specs if (s.swizzle_direction, s.swizzle_offset) == (direction, offset))
        assert lower_catlass_spec(spec, p, direct, memory_view=AdmissionMemoryView({0: 2 * KIB}, granule_bytes=SECTOR)).l2_hit_bytes == expected
def test_evg_visitor_dag_and_direct_epilogue_traffic(datapaths):
    tree, exp = visitor_graph()
    indirect, direct = datapaths
    positions = {n.node_id: i for i, n in enumerate(evg_order(tuple(reversed(tree))))}
    assert all(positions[parent] < positions[n.node_id] for n in tree for parent in n.inputs)
    with pytest.raises(ModelingError, match="evg_dag_unknown_or_duplicate_node"):
        evg_order(tree[1:])
    with pytest.raises(ModelingError, match="evg_dag_cycle"):
        evg_order((*tree[:4], replace(exp, inputs=("add",)), *tree[5:]))
    p = CatlassProblem(37, 39, 33)
    gm, ub = _matrix("evg", p, direct, evg_tree=tree)
    old_gm, = _matrix("evg", p, indirect, evg_tree=tree)
    with pytest.raises(UnmodeledCapabilityError, match="requires_l0c_to_ub"):
        lower_catlass_spec(replace(old_gm, epilogue_kind="evg_ub", workspace_bytes=0), p, indirect)
    gm_result, ub_result = (lower_catlass_spec(spec, p, direct) for spec in (gm, ub))
    assert gm_result.summary.logical_transfer_bytes - ub_result.summary.logical_transfer_bytes == 2 * p.m * p.n * 4
    assert dict(ub_result.summary.vector_ops_by_pipe) == {"vec_alu": 9 * p.m * p.n, "vec_sfu": 4 * p.m * p.n}
def test_grouped_scheduler_carries_core_index_and_bypasses_single_row_tiles(direct):
    p = CatlassProblem(65, 64, 32, groups=(16, 0, 33, 16))
    assert tuple(cat.grouped_start_cores(p.groups, p.n, 32, 32, 4)) == ((0, 0, 2), (2, 2, 4), (3, 2, 2))
    spec, = _matrix("grouped_slice_m", p, direct)
    stages = _stages(lower_catlass_spec(spec, p, direct, **template_live_bindings(direct, p)))
    reads = [stages[f"tile:{i}:k0:gm_b"].program.pattern for i in range(8)]
    assert [read.owner for read in reads] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [i for i, read in enumerate(reads) if read.reuse_key is not None] == [2, 3, 4, 5]
    only, = _matrix("grouped_slice_m", replace(p, m=32, groups=(16, 0, 16)), direct)
    assert only.l2_cache_hint_b == "disable"
@pytest.mark.parametrize("acc,width", [("fp32", 4), ("bf16", 2)])
def test_grouped_swiglu_quantization_avoids_gate_up_spills(direct, acc, width):
    dp, m, f = with_mx_modes(direct), 48, 64
    p = CatlassProblem(m, 2 * f, 32, "mxfp8", "mxfp8", dtype_acc=acc, groups=(16, 0, 32))
    spec, = _matrix("grouped_mx_swiglu_mx_quant", p, dp)
    result = lower_catlass_spec(spec, p, dp)
    assert (spec.l2_cache_hint_b, spec.workspace_bytes) == ("disable", 0)
    assert sum(s.metadata.get("saved_logical_bytes", 0) for s in result.program.stages) == 4 * m * f * width
    # Four A reads per row, two B experts, one scale per K block.
    assert result.summary.logical_transfer_bytes == (4 * m + 2 * p.n) * (p.k + 1) + m * (2 * f + f // 32)
    assert next(x.ranges for x in result.summary.write_obligations if x.allocation_id == p.c_id) == ((0, m * f * 2),)
    assert result.summary.semantic_work[0].logical_ops == 2 * m * (2 * f) * p.k
@pytest.mark.parametrize("mode,a,b,a_width,b_width,factor", [
    ("mxfp8", "mxfp8", "mxfp8", 1, 1, 1), ("mxfp4", "mxfp4", "mxfp4", 0.5, 0.5, 1),
    ("a8w4_mx", "fp8_e4m3", "fp4_e2m1", 1, 0.5, 16),
])
def test_mx_scale_storage_and_decompression_follow_declared_mode(direct, mode, a, b, a_width, b_width, factor):
    mx_datapath = with_mx_modes(direct)
    p = CatlassProblem(32, 32, 128, a, b, mx_mode=mode)
    spec, = _matrix("mx_matmul", p, mx_datapath, tile=(32, 32, 128), stage_options=(ONE_SLOT,))
    elements, panel = p.n * p.k, p.n * 64
    expanded_width = a_width if factor == 16 else b_width
    peaks = {row.pool: row for row in solve_catlass_tile(spec, p, mx_datapath).pool_peaks}
    assert (spec.l1_scale_factor_k, peaks["l1"].peak_bytes) == (factor, elements * (a_width + expanded_width) + 2 * elements // 32 * factor)
    assert peaks["l0b"].peak_bytes == panel * expanded_width + panel // 32
    assert all(peaks[pool].owner_kind == "aic" for pool in ("l1", "l0a", "l0b"))
    stages = _stages(lower_catlass_spec(spec, p, mx_datapath))
    assert sum(_bytes(stages, f":scale_{x}", "mte2_aic") for x in ("a", "b")) == 2 * elements // 32
    if factor == 16:
        assert (spec.prologue_b, _bytes(stages, ":gm", "mte2_aiv")) == ("DecompressFp4ToFp8", elements * b_width)
        assert (_bytes(stages, ":decompress_b", "vec_alu"), _bytes(stages, ":ub_to_l1", "mte3")) == (elements, elements * expanded_width)
    dec = direct.compute_arch.declared
    no_mx = replace(direct, compute_arch=replace(direct.compute_arch, declared=replace(
        dec, native_modes=tuple(m for m in dec.native_modes if not (set(m.operands) & {"mxfp4", "mxfp8", "fp4_e2m1", "a8w4_mx"})))))
    with pytest.raises(UnmodeledCapabilityError, match="native_mx_mode_unsupported"):
        lower_catlass_spec(spec, p, no_mx, context=replace(_context(no_mx, p), compute_peaks={mode: 10**12, a: 10**12}))
@pytest.mark.parametrize("policy,threads", [("tla_dsl", 2048), ("MatmulGatherScatterSimt", 1024),
                                            ("GroupedMxMatmulFinalizeRoutingTla", 2048)])
def test_simt_thread_ceilings_carveout_and_routing_workspace(datapaths, policy, threads):
    indirect, direct = datapaths
    indexed = policy == "MatmulGatherScatterSimt"
    p = CatlassProblem(1, 32, 32, "fp16", "fp16", "fp16", indices=(7,), physical_m=8) if indexed else CatlassProblem(32, 32, 32)
    options = dict(simt_variants=(policy,), simt_thread_blocks=((32, threads // 32, 1),))
    spec, = _matrix("a5_simt", p, direct, **options)
    assert direct.usable_bytes("ub") == 248 * KIB and direct.usable_bytes("ub", kernel=True) == 216 * KIB
    with pytest.raises(UnmodeledCapabilityError, match="simt_capability_missing"):
        solve_catlass_tile(replace(spec, arch_tag="AtlasA2"), p, indirect)
    with pytest.raises(ModelingError, match="simt_thread_block_invalid"):
        solve_catlass_tile(replace(spec, simt_thread_block_dim=(threads + 1, 1, 1)), p, direct)
    lowered = lower_catlass_spec(spec, p, direct)
    stages = _stages(lowered)
    if policy == "GroupedMxMatmulFinalizeRoutingTla":
        assert spec.workspace_bytes == 32 * 32 * spec.aic_budget * 2
        assert _bytes(stages, ":routing_read", "mte2_aiv") == _bytes(stages, ":routing_write", "fixpipe") == p.m * p.n * 2
    else:
        assert lowered.summary.semantic_work == (SemanticWorkItem(
            "catlass.a5_simt:matrix", "matrix_mac", p.dtype_a, 2 * p.m * p.n * p.k,
            ("cube", "aiv_simt") if indexed else ("cube",), ("dataparallel",),
        ),)
        ub, = solve_catlass_tile(spec, p, direct).pool_peaks
        assert (ub.peak_bytes, ub.capacity_bytes) == (64 * KIB if indexed else threads * 4, 216 * KIB)
        assert tuple(stages["simt:0:simt_mac"].metadata[f"thread_block_{axis}"] for axis in "xyz") == options["simt_thread_blocks"][0]
    if indexed:
        reads = {entry.allocation_id: entry.ranges for entry in lowered.summary.read_obligations}
        row = ((7 * 32 * 2, 8 * 32 * 2),)
        assert (reads[p.a_id], reads[p.indices_id], _bytes(stages, ":indices", "mte2_aiv")) == (row, ((0, p.m * 4),), SECTOR)
        assert {"simt:0:indices", "simt:zero_output:barrier"} <= _ancestors(stages, "simt:0:simt_mac")
        assert sum(s.work["mte3"] for n, s in stages.items() if n.startswith("simt:zero_output:") and s.work) == 8 * 32 * 2
        live = lower_catlass_spec(spec, p, direct, **template_live_bindings(direct, p, aiv_simt=10**6))
        assert _stages(live)["simt:0:scatter"].program.pattern.runs.sector_intervals() == row
        geometry, _ = sku_planning_context(REPRESENTATIVE_SKUS["a5"])
        shipping = lower_catlass_spec(spec, p, direct, context=recipe_context(geometry, p))
        ops = dict(shipping.summary.vector_ops_by_pipe)["aiv_simt"]
        assert _stages(shipping)["simt:0:simt_mac"].work == {"vector": ops}
        assert _stages(live)["simt:0:simt_mac"].work == {"aiv_simt": ops}
        small = replace(direct, buffers=tuple((key, 16 * KIB if key == "ub" else size) for key, size in direct.buffers))
        with pytest.raises(TileInfeasibleError):
            solve_catlass_tile(spec, p, small)
        with pytest.raises(ModelingError, match="gather_scatter_requires_fp16"):
            solve_catlass_tile(spec, replace(p, dtype_a="fp32"), direct)
        cube_spec, = _matrix("full_load_a_gather_scatter", p, direct)
        cube = lower_catlass_spec(cube_spec, p, direct, **template_live_bindings(direct, p))
        assert dict(cube.summary.issued_cube_flops_by_dtype)["fp16"] >= 2 * p.m * p.n * p.k
        assert _stages(cube)["tile:0:scatter"].program.pattern.runs.sector_intervals() == row
@pytest.mark.parametrize("arch_id", ARCH_IDS)
def test_reference_admission_follows_declared_isa(datapaths, arch_id):
    dp = next(path for path in datapaths if path.arch_id == arch_id)
    for name, p in zip(dsl.DSL_KERNEL_IDS, (
        dsl.FlashKdaProblem((1,)), dsl.MegaRecurrentKdaProblem(1, 4, 4), dsl.CommitRecurrentKdaProblem((1,)),
        dsl.MlaPrologProblem(1), dsl.AttnResProblem(1), dsl.QsliProblem(1, 32, 8),
        dsl.MixedQuantSparseMlaProblem(1, 1),
    ), strict=True):
        if dp is datapaths[0]:
            with pytest.raises(UnmodeledCapabilityError, match="dsl_requires_"):
                enumerate_dsl_specs(name, p, dp, aic_budget=8, aiv_budget=16)
        else:
            options = dict(aic_budget=8, aiv_budget=16)
            if name == dsl.QSLI:
                shipping = _datapath("Ascend950DT_9582")
                spec, = enumerate_dsl_specs(name, p, shipping, **options)
                dec = shipping.compute_arch.declared
                no_mx = replace(shipping, compute_arch=replace(shipping.compute_arch, declared=replace(
                    dec, native_modes=tuple(m for m in dec.native_modes if not (set(m.operands) & {"mxfp4", "fp4_e2m1"})))))
                with pytest.raises(UnmodeledCapabilityError, match="qsli_requires_native_mxfp4"):
                    enumerate_dsl_specs(name, p, no_mx, **options)
                with pytest.raises(UnmodeledCapabilityError, match="qsli_requires_native_mxfp4"):
                    lower_dsl_spec(spec, p, no_mx)
            else:
                assert enumerate_dsl_specs(name, p, dp, **options)
def test_flash_kda_cost_threshold_and_saved_output_visibility(direct):
    # Chunks (3, 1, 2) give cyclic peak 5, LPT peak 3 and two staging rounds.
    staged = (2 + 3 * 3 + 5 * 5 + 7 * 2) * 64 + (11 + 17) * 384
    for p0, margin in ((154, 0), (155, 0), (156, 0), (154, 5000)):
        costs = (2, 3, 5, 7, 11, 13, 17, p0, 3, 2, 3, 2, 5, 7, margin, 0)
        persistent = (p0 + 3 * 3 + 2 * 3) * 64 + 3 * 384 + 10 * 192
        expected = "persistent" if persistent < staged - staged * margin // 1_000_000 else "staged"
        assert dsl.select_flash_kda_template((192, 64, 128), 2, cost_coefficients=costs, workspace_slots=3) == expected
    p = dsl.FlashKdaProblem((65,), metadata_ms=0.1, eager_d2h_bytes=256, save_matrices=True,
                            save_kernel_caches=True, save_chunk_states=True, requested_template="staged", max_chunk_group=1)
    spec, result = _reference(dsl.FLASH_KDA, p, direct)
    assert spec.template_variant == "staged" and spec.uses_aicpu_metadata and spec.has_eager_d2h_readback
    assert result.summary.stars_launch_count == 2 and result.summary.aicpu_metadata_ms == Fraction(1, 10)
    stages = _stages(result)
    assert stages["metadata_eager_d2h"].after == ("aicpu_metadata",) and stages["launch"].after == ("metadata_eager_d2h",)
    assert stages["metadata_eager_d2h"].work["d2h"] == p.eager_d2h_bytes
    for tensor, size in {"Akk": 65 * 64 * 2, "Aqk": 65 * 64 * 2, "qg": 65 * 128 * 2,
                         "g_cumsum": 65 * 128 * 4, "h": 2 * 128**2 * 4}.items():
        assert dict(spec.l2_cache_policies)[tensor] == "disable"
        carry = _bytes(stages, ":carry_h", "hbm") if tensor == "h" else 0
        assert _bytes(stages, f":save_{tensor}", "hbm") + carry == size
    assert "kda:0:0:0:save_h" in _ancestors(stages, "kda:0:0:0:fp32_state_update")
    reload = stages["kda:0:0:1:reload_h"]
    assert reload.work["hbm"] == 128**2 * 4 and "kda:0:0:0:carry_h" in _ancestors(stages, reload.name)
    _, persistent_result = _reference(dsl.FLASH_KDA, replace(p, requested_template="persistent"), direct)
    assert _bytes(stages, ":q_decayed:read", "mte2_aic") == 65 * 128 * 2
    assert _bytes(_stages(persistent_result), ":q_decayed:read", "mte2_aic") == 0
@pytest.mark.parametrize("tp,sequence", tuple(product((4, 8, 16), (4, 8))))
def test_mega_recurrent_variant_workspace_reads_and_launch_savings(direct, tp, sequence):
    slots, capacity, heads = tuple(range(7, 7 + tp)), sequence + 1, 96 // tp
    stride = heads * capacity * 128 + 128
    p = dsl.MegaRecurrentKdaProblem(1, tp, sequence, state_indices=slots, replay_capacity=capacity, k_stride0=stride)
    spec, result = _reference(dsl.MEGA_KDA, p, direct)
    stages = _stages(result)
    assert spec.template_variant == f"tp{tp}_s{sequence}"
    for tensor, width in (("Q", 128), ("K", 128), ("V", 128), ("decay", 128), ("beta", 1), ("output_gate", 128)):
        assert _bytes(stages, f":{tensor}:gm:read", "mte2_aiv") == tp * SECTOR * ceil(sequence * heads * width * 2 / SECTOR)
        assert stages[f"mega:0:{tensor}:gm:read"].after == (f"mega:0:{tensor}:gm:write",)
    assert spec.workspace_bytes_total == tp * sequence * 2 * (5 * heads * 128 + 16 * ceil(heads / 16) + 128) + 128**2 * 4
    assert len(p.unfused_launches) - result.summary.stars_launch_count == stages["mega:unfused_baseline"].metadata["saved_launches"]
    obligations = {entry.allocation_id: entry for entry in result.summary.write_obligations}
    assert "checkpoint" not in obligations
    for tensor, element in (("replay_u", 4), ("replay_k", 2), ("replay_decay", 4)):
        pitch = stride if tensor == "replay_k" else heads * capacity * 128
        bases = ((slot * pitch + head * capacity * 128) * element for slot in slots for head in range(heads))
        entry = obligations[tensor]
        ranges = (AddressRun(lo, 1, hi - lo, hi - lo) for lo, hi in entry.ranges)
        expected = tuple((base, base + sequence * 128 * element) for base in bases)
        assert AddressRunPlan((*entry.address_runs, *ranges), sector_bytes=1).sector_intervals() == expected
@pytest.mark.parametrize("mode", tuple(dsl.QkvQuantMode))
def test_mla_four_products_workspace_and_quantization_barrier(direct, mode):
    p = dsl.MlaPrologProblem(3, 192, 96, 128, 2, 32, 64, mode=mode, weight_format="nd")
    spec, result = _reference(dsl.MLA_PROLOG, p, direct)
    stages = _stages(result)
    widths = dict(wqa=192 * 96, wkva=192 * 128, wqb=96 * 2 * (32 + 64), wkb=2 * 32 * (128 - 64))
    assert spec.template_variant == mode.value
    assert {k: stages[f"mla:0:{k}"].work["cube"] for k in widths} == {k: 2 * p.tokens * v for k, v in widths.items()}
    for tensor, width, element, pipe in (("ws_qa", 96, 16, "mte2_aiv"), ("ws_kva", 192, 2, "mte2_aiv"),
                                         ("ws_q_proj", 192, 2, "mte2_aic")):
        name = f"mla:0:{tensor}"
        assert stages[name + ":read"].work[pipe] == stages[name + ":write"].work["fixpipe"] == 8 * width * element
        assert stages[name + ":read"].after == (name + ":write",)
    barrier = "q_global_quant_launch_barrier"
    if mode == dsl.QkvQuantMode.HiF8:
        assert set(stages[barrier].after) == {f"mla:ws_amax_write:{i}" for i in range(spec.aic_budget)}
        assert all(f"mla:0:partial_max:{i}" in _ancestors(stages, barrier) for i in range(spec.aic_budget))
        assert barrier in _ancestors(stages, "mla:0:q_output")
    else:
        assert barrier not in stages
    assert all(stages[f"mla:0:{matrix}:l0{bank}"].work["mte1"] > 0 for matrix, bank in product(widths, "ab"))
    if mode == dsl.QkvQuantMode.FP8:
        packed = replace(p, tokens=2, kvr=576, mode=1, cache_index=(7, -1))
        _, live = _reference(dsl.MLA_PROLOG, packed, direct, live_rates={"vec_alu": 10**6})
        indexed = _stages(live)
        cache = indexed["mla:0:kv_cache"]
        assert cache.after == ("mla:0:cache_index",) and indexed["mla:0:cache_index"].program.pattern.logical_bytes == 16
        assert cache.program.pattern.logical_bytes == 656 and cache.program.pattern.runs.runs[0].base_byte == 7 * 656
        qr, = (entry for entry in live.summary.write_obligations if entry.allocation_id == "mla:qr")
        reads, _ = compose_step_obligations((live.summary, replace(live.summary, read_obligations=(qr,))))
        assert all(entry.allocation_id != "mla:qr" for entry in reads)
        with pytest.raises(ModelingError, match="group_fp8_requires_pa_bsnd"):
            enumerate_dsl_specs(dsl.MLA_PROLOG, replace(packed, cache_layout="PA_NZ"), direct)
@pytest.mark.parametrize("tokens", (1, 5))
def test_attention_residual_ub_and_inplace_savings(direct, tokens):
    p, d = dsl.AttnResProblem(tokens), 7168
    spec, result = _reference(dsl.ATTN_RES, p, direct, aiv_budget=1)
    tile, depth = spec.tile_shape[0], 1 if tokens <= spec.tile_shape[0] else 2
    assert (tile, depth) == (1, 1 if tokens == 1 else 2)
    assert spec.ub_peak_bytes == 6 * d + depth * (10 * d * tile + 12 * 8 * ceil(tile / 8))
    assert dsl.attn_res_tiling(8190, 1, ub_bytes=2**32) == (1, 8190, 4095, 2)
    stages, payload = _stages(result), 4 * tokens * d
    assert (_bytes(stages, ":score_reduce", "vec_reduce"), _bytes(stages, ":score_sqrt", "vec_sfu")) == (2 * tokens * d, tokens)
    assert stages["attn_res:inplace_contract"].metadata["saved_logical_bytes"] == payload
    assert _bytes(stages, ":partial", "mte2_aiv") == _bytes(stages, ":inplace_partial", "mte3") == payload
    assert [(x.allocation_id, x.tensor_version, x.ranges) for x in result.summary.write_obligations
            if x.allocation_id.endswith(":partial_block")] == [("attn_res:partial_block", 1, ((0, 4 * tokens * d),))]
    for bad in (replace(p, hidden_size=d - 1), replace(p, dtype="fp16")):
        with pytest.raises(ModelingError, match="attn_res_requires_fixed_bf16_width"):
            enumerate_dsl_specs(dsl.ATTN_RES, bad, direct)
@pytest.mark.parametrize("splits", (1, 2))
def test_sparse_indexer_packed_rings_histogram_and_ld_workspace(direct, splits):
    direct, trunk, bins, topk = with_mx_modes(direct), 16_384, 256, 32
    p = dsl.QsliProblem(1, trunk * splits + 1, topk, splits)
    spec, result = _reference(dsl.QSLI, p, direct)
    stages = _stages(result)
    assert dict(result.summary.issued_cube_flops_by_dtype) == {"mxfp4": 2 * 32 * 512 * 128 * (32 * splits + 1)}
    rings = {name: (pool, depth) for name, pool, depth in spec.ring_depths}
    assert all(rings.get(name) == (pool, 2) for name, pool in (("k_l1", "l1"), ("l0b", "l0b"), ("l0c", "l0c")))
    assert "qsli:0:0:l0c:g0:last_use" in stages["qsli:0:0:tile2:QsliCube"].after
    low = [s for n, s in stages.items() if n.endswith(":masked_low_histogram")]
    assert sorted(s.work["sort_topk"] for s in low) == [1 + bins] + [trunk + bins] * splits
    assert all(s.metadata == {"bins": bins, "counter_bytes": 2, "low_byte_masked": True} for s in low)
    for tag, pipe, size in (("publish_packed_k", "mte2_aic", 512 * 64), ("select_keys", "mte2_aiv", trunk * 2 + (SECTOR if splits == 1 else 0)),
                            ("positions", "mte2_aiv", topk * 4)):
        name = "qsli:0:0:" + ("tile0:" if tag == "publish_packed_k" else "") + tag
        assert stages[name + ":read"].work[pipe] == stages[name + ":write"].work["mte3"] == size
    assert spec.workspace_bytes_total == 36 * 4 * 512 * (64 + 4) + spec.aic_budget * (2 * p.candidates + 16 * topk)
    assert ("qsli:0:LdMergeStage" in stages) == (splits > 1)
    for tensor, width in (("indices", 4), ("values", 2)):
        assert _bytes(stages, f":partial_{tensor}_read", "mte2_aiv") == (splits > 1) * splits * topk * width
        assert _bytes(stages, f":partial_{tensor}", "mte3") == (splits > 1) * splits * topk * width
    if splits > 1:
        assert "qsli:0:ld_barrier" in _ancestors(stages, "qsli:0:LdMergeStage")
    else:
        _, empty = _reference(dsl.QSLI, dsl.QsliProblem(1, 0, 1), direct)
        stages = _stages(empty)
        assert stages["qsli:0:pad_topk"].metadata == {"index": -1, "value": "-inf"}
        assert all(stages[f"qsli:0:{out}"].after == ("qsli:0:pad_topk",) for out in ("indices", "values"))
        assert not empty.summary.issued_cube_flops_by_dtype
@pytest.mark.parametrize("splits", (1, 2))
def test_mixed_sparse_mla_bank_bytes_and_split_row_reduction(direct, splits):
    p = dsl.MixedQuantSparseMlaProblem(129, 72, split_parts=splits)
    spec, result = _reference(dsl.MQSMLA, p, direct)
    stages = _stages(result)
    assert tuple(_bytes(stages, tag, "mte2_aiv") for tag in (":ori", ":cmp", ":sink")) == (544 * p.n_ori, 320 * p.n_cmp, splits * 256)
    slot = (64 * 512 + 2 * 64) * 4
    assert spec.workspace_bytes_total == 2 * slot * spec.aic_budget == 263_168 * spec.aic_budget
    writes = {name for name in stages if name.endswith(":fd_partial_write")}
    assert len(writes) == (splits > 1) * splits and sum(stages[n].work["mte3"] for n in writes) == len(writes) * slot
    barrier, reduce = "mqsmla:fd_barrier", "mqsmla:0:fd_reduce"
    if splits > 1:
        assert set(stages[barrier].after) == writes
        for name in stages[reduce].after:
            assert (stages[name].after, stages[name].work["mte2_aiv"]) == ((barrier,), slot)
        assert stages[reduce].work["vec_reduce"] == splits * slot // 4
        assert stages["mqsmla:0:output"].after == (reduce,)
    else:
        assert barrier not in stages and reduce not in stages
        p = replace(p, n_ori=1, n_cmp=1, topk_indices=(0, 7), ori_block_table=(1,), cmp_block_table=(3,))
        _, indexed = _reference(dsl.MQSMLA, p, direct)
        reads = {entry.allocation_id: entry.ranges for entry in indexed.summary.read_obligations}
        assert (reads["mqsmla:ori"], reads["mqsmla:cmp"]) == (((128 * 544, 129 * 544),), ((391 * 320, 392 * 320),))
        assert {name for name in reads if name.endswith(":block_table")} == {"mqsmla:ori:block_table", "mqsmla:cmp:block_table"}
        assert _bytes(_stages(indexed), ":sink", "mte2_aiv") == p.heads * 4
def test_region_rejects_invalid_graph_and_unread_alias(direct):
    for options, error in (
        ({"dependencies": {"make": (), "outside": ("make",), "take": ("outside",)}}, "not_convex"),
        ({"collective_ops": ("make",)}, "collective_barrier"),
        ({"waits": (("take", "make"),)}, "credit_publish_deadlock"),
    ):
        with pytest.raises(ModelingError, match=error):
            _region(direct, _edge(), **options)
    with pytest.raises(ModelingError, match="in_place_unread_alias"):
        compile_fuse_region("fused", ("make", "take", "read"), direct, dependencies={
            "make": (), "take": ("make",), "read": ("make",),
        }, op_stages={}, edge_bytes={}, edge_routes=(replace(_edge(), in_place_mutates=True), _edge(dst="read", view="alias")),
            alias_views={"tile": ("x", 0, 64), "alias": ("x", 32, 96)})
def test_region_routes_admit_only_valid_pools_and_owner_pairs(datapaths):
    indirect, direct = datapaths
    for pool, owner in (("l1", "aiv"), ("ub", "aic")):
        with pytest.raises(ModelingError, match=f"live_buffer_pool_owner_mismatch:panel:{pool}:{owner}"):
            validate_live_buffer_owners((LiveBufferSpec("panel", pool, 64, 0, 1, owner_kind=owner),))
    assert _region(indirect, _edge("gm_spill"))[1].summary.logical_transfer_bytes == 2 * 64
    for route, axes, pipe in (("l0c->ub", ("aic", "aiv"), "fixpipe"), ("ub->l1", ("aiv", "aic"), "mte3")):
        edge = _edge(route, axes)
        with pytest.raises(UnmodeledCapabilityError, match="does not declare directed route"):
            _region(indirect, edge)
        spec, result = _region(direct, edge)
        assert _stages(result)["p0:t0:edge:0"].work[pipe] == 64 and result.summary.logical_transfer_bytes == 0
        assert sorted(p.peak_bytes for p in spec.tile_solution.pool_peaks) == [64, 128]
        with pytest.raises(ModelingError, match="owner_pair_unreachable"):
            _region(direct, edge, edge_owners={"tile": (0, 2) if axes[0] == "aic" else (2, 0)})
def test_region_cube_rate_and_full_barrier_release_between_tiles(direct):
    direct = replace(direct, buffers=tuple((key, 64 if key == "l0c" else size) for key, size in direct.buffers))
    make = Stage("make", {"cube": 131_072}, acquire=(Allocation("accumulator", "l0c", 64, (0,)),), placement={"cube": (0,)})
    _, result = _region(direct, replace(_edge(), iter_kind="full_barrier"), tile_count=2,
                        op_stages={"make": make, "take": Stage("take", free=make.acquire)})
    cube, peaks = _stages(result)["p0:t0:make"], dict(machine(REPRESENTATIVE_SKUS["a5"]).dies[0].compute_peaks)
    assert (cube.work_units["cube"], cube.work["cube"]) == ("peak_ms", pytest.approx(1000 * make.work["cube"] / peaks["bf16"]))
    assert dict(result.summary.issued_cube_flops_by_dtype) == {"bf16": 2 * make.work["cube"]}
    assert _run(direct, _context(direct, None), result).summary()["peak_allocated_per_unit"]["l0c:unit0"] == 64
def test_region_compiler_preserves_coupling_metadata_and_step_obligations(direct):
    stage = Stage("compute", {"cube": 8, "aiv_simt": 4}, coupled=True,
                  placement={"cube": (0,), "aiv_simt": (0, 1)}, metadata={"rows": 2, "cols": 4, "launch": True})
    semantic = SemanticWorkItem("matrix", "matrix_mac", "bf16", 8, ("cube", "aiv_simt"), ("tiles",))
    dp = replace(direct, memory_access=replace(direct.memory_access, burst_turn_penalty=(TURNAROUND,)))
    summaries = []
    for start in (0, SECTOR):
        read = TensorRangeObligation(0, "shared", 0, ((start, start + 2 * SECTOR),), "hbm_cold", "device")
        options = dict(dependencies={"compute": ()}, edge_routes=(), edge_bytes={}, backend="tla_simt",
                       semantic_work=(semantic,), live_in_obligations=(read,),
                       live_buffers=(LiveBufferSpec("tile", "ub", 216 * KIB, 0, 1, owner_kind="aiv"),))
        spec, result = compile_fuse_region("fused", ("compute",), dp, op_stages={"compute": stage}, **options)
        assert [(p.peak_bytes, p.capacity_bytes) for p in spec.tile_solution.pool_peaks] == [(216 * KIB, (248 - 32) * KIB)]
        lowered, summary = _stages(result)["p0:t0:compute"], result.summary
        assert lowered.coupled and lowered.placement == stage.placement and lowered.metadata == stage.metadata
        with pytest.raises(ModelingError, match="coupled_placement_not_proportional"):
            compile_fuse_region("fused", ("compute",), dp, op_stages={"compute": replace(stage, placement={"cube": (0,), "aiv_simt": (0,)})}, **options)
        assert summary.dram_sector_bytes == 2 * SECTOR and summary.turnaround_service_bytes == 2 * SECTOR + Fraction(1, 10)
        assert (summary.semantic_work, summary.spec, dict(summary.issued_cube_flops_by_dtype)) == ((semantic,), spec, {"bf16": 8})
        summaries.append(summary)
    assert compose_step_obligations(summaries) == ((replace(read, ranges=((0, 3 * SECTOR),)),), ())
    cold, warm = replace(read, ranges=((0, SECTOR),)), replace(read, ranges=((SECTOR, 2 * SECTOR),), initial_residency="l2_warm")
    assert _region(dp, _edge(), live_in_obligations=(cold, warm))[1].summary.read_obligations == (cold, warm)
    assert compose_step_obligations((_region(dp, _edge(), live_in_obligations=(cold, warm))[1].summary,)) == ((cold,), ())
def _prime(view, tensor, run):
    pattern = MemoryAccessPattern(tensor, 0, 0, 0, "read", "mte2_aiv", run.rows * run.row_bytes,
                                  AddressRunPlan((run,)), ReuseKey(0, tensor, 0, "main"))
    assert view.admit_access(pattern, pattern.runs.aligned_copy_bytes()) == (0, pattern.logical_bytes)
def test_periodic_template_obligations_and_service_follow_resident_sectors(direct):
    rows, stride = 128 * KIB * 32, 2 * SECTOR
    dp = replace(direct, memory_access=replace(direct.memory_access, burst_turn_penalty=(TURNAROUND,)))
    run, view = AddressRun(0, rows, SECTOR, stride), AdmissionMemoryView({0: 2 * rows * SECTOR})
    context, primed = _context(dp, None), replace(run, rows=rows // 2, stride_bytes=2 * stride)
    _prime(view, "rows", primed)
    plan = Lowering(dp, context=context, memory_view=view, live_admission=True)
    tracemalloc.start()
    try:
        start = perf_counter()
        plan.access("read", "rows", rows * SECTOR, runs=(run,))
        result = plan.result("K3", "read", (), None, (), SimpleNamespace(pool_peaks=()))
        assert perf_counter() - start < 0.5 and tracemalloc.get_traced_memory()[1] < 8 * KIB**2
    finally:
        tracemalloc.stop()
    warm, cold = result.summary.read_obligations
    assert warm.address_runs == (primed,) and warm.initial_residency == "l2_warm" and not (warm.ranges or cold.ranges)
    assert cold.address_runs == (replace(primed, base_byte=stride),) and compose_step_obligations((result.summary,)) == ((cold,), ())
    delivered, service = _run(dp, context, result).delivered, rows // 2 * (SECTOR + Fraction("0.1"))
    assert result.dram_miss_bytes == result.summary.dram_sector_bytes == result.l2_hit_bytes == rows // 2 * SECTOR
    assert (result.summary.turnaround_service_bytes, delivered["hbm:ctr_r"]) == (service, pytest.approx(float(result.summary.dram_sector_bytes)))
def test_templates_keep_producer_stream_and_release_owned_streams_at_last_use(direct):
    view, context = AdmissionMemoryView({0: 2 * KV_BYTES}, granule_bytes=SECTOR), _context(direct, None)
    _prime(view, "kv", AddressRun(0, 1, KV_BYTES, KV_BYTES))
    plan = Lowering(direct, context=context, memory_view=view, live_admission=True)
    first = plan.access("first", "first", SECTOR)
    second = plan.access("second", "second", SECTOR, after=(first,))
    plan.access("read_kv", "kv", KV_BYTES, after=(second,))
    result = plan.result("K3", "read", (), None, (), SimpleNamespace(pool_peaks=()))
    assert plan.state.active_stream_count(0) == view.active_stream_count(0) == 1
    assert (result.l2_hit_bytes, result.dram_miss_bytes) == (KV_BYTES, 2 * SECTOR)
    assert _run(direct, context, result).delivered["hbm:ctr_r"] == pytest.approx(2 * SECTOR)
    assert (result.l2_hit_bytes, view.active_stream_count(0)) == (KV_BYTES, 1)
def test_catalog_provenance_validates_mounted_catalog(monkeypatch):
    catalog = builtin_template_catalog()
    refs = catalog.all_template_source_refs()
    assert catalog.ids("K1") and len(catalog.ids("K2")) == 11 and len(catalog.ids("K3")) == 8
    mounted = mounted_catalog_sources(ROOT, refs)
    checked, findings = load_template_provenance(monkeypatch).verify(ROOT, refs_root=mounted)
    assert checked == len(refs) == len(set(refs)) == 36
    assert [f.reason.split(":")[0] for f in findings] == ["source_repository_missing"] * (len(refs) if mounted is None else 0)
def test_catalog_provenance_rejects_invalid_sources(monkeypatch, tmp_path):
    audit = load_template_provenance(monkeypatch)
    repo, bare, ref, tree = provenance_git_fixture(
        tmp_path, "include/fixture.hpp", b"// prefix\nconstexpr int TOTAL_MUTEX_IDS = 28;\n// suffix\n", "TOTAL_MUTEX_IDS", (2, 2))
    cases = (
        (repo, {}, None), (bare, {}, None), (None, {}, "source_repository_missing"),
        (repo / "include", {}, "source_repository_not_mount_root"),
        (None, {"source_sha256": "0" * 64}, "source_repository_missing"),
        (repo, {"revision": tree}, "source_revision_not_commit"),
        (repo, {"symbol": "OTAL_MUTEX_ID"}, "symbol_missing_from_range"),
        (repo, {"source_sha256": "f" * 64}, "source_sha256_mismatch"),
        (repo, {"line_range": (3, 2)}, "invalid_line_range"),
        (repo, {"line_range": (2, 4)}, "source_line_range_exceeds_blob"), (repo, {"symbol": ""}, "invalid_symbol"),
    )
    for mounted, changes, reason in cases:
        assert verify_template_declaration(audit, tmp_path, ref.to_dict() | changes, mounted) == ([reason] if reason else [])
def test_scalar_gemm_logical_obligations_and_compulsory_sectors(direct):
    problem = CatlassProblem(1, 1, 1)
    spec, = _matrix("basic_pingpong", problem, direct, stage_options=(ONE_SLOT,))
    bindings = template_live_bindings(direct, problem)
    result = lower_catlass_spec(spec, problem, direct, **bindings)
    context, pools = bindings["context"], dict(direct.buffers)
    resources = [Resource(role, context.rates[role], units=units) for role, units in context.resource_units.items()]
    punits = {pool: 1 if pool == "l2" else direct.aiv_count if pool == "ub" else direct.aic_count for pool in pools}
    engine = Engine(resources, pools=pools, backend="c", pool_units=punits)
    for hits, hbm in ((0, 3 * SECTOR), (2 * SECTOR, SECTOR)):
        steps = engine.accel_status()["native_steps"]
        engine.advance(until=(result.program.start(engine, "gemm", engine.submit("ready", release=engine.now + 1)),))
        summary, cut = result.summary, compute_compulsory_hbm_cut((result.summary,))[0]
        assert (summary.l2_hit_bytes, summary.dram_sector_bytes, summary.logical_transfer_bytes) == (hits, hbm, 6)
        assert (cut.read_bytes, cut.write_bytes) == (hbm - SECTOR, SECTOR) and engine.accel_status()["native_steps"] > steps and not engine.allocations
        assert {entry.initial_residency for entry in summary.read_obligations} == {"l2_warm" if hits else "hbm_cold"}
