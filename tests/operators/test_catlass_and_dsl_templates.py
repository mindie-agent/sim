"""CPU execution and conservation checks for the kernel template catalog.
[Role] Exercise CATLASS, DSL reference kernels, and FuseRegion compilation.
[Invariants] Check exact tile, dependency, and traffic contracts independently.
[Timeouts & Locks] Bounded schedules on the Python engine without external work.
[Side-Effect Policy] Read packaged profiles and source. Use in-memory fixtures.
"""
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from math import ceil
from pathlib import Path
import re
import sys

import pytest

from _support.axes import ARCH_IDS, REPRESENTATIVE_SKUS, SKUS
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.program.binding import RecipeContext
from mindie_sim.prediction.program.stages import Stage
from mindie_sim.prediction.runtime.memory_physics import AdmissionMemoryView
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, TileInfeasibleError
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_arch import NativeComputeMode, RepresentationCapability
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.device_input import MemoryAccessBurstPenalty
from mindie_sim.schema.hardware import Hardware
from mindie_sim.templates import (
    CatlassProblem, EdgeRouteBinding, EvgVisitorNode, FusedRegionSummary, SemanticWorkItem,
    TensorRangeObligation, builtin_template_catalog, compile_fuse_region, compose_step_obligations,
    enumerate_catlass_specs, enumerate_dsl_specs, lower_catlass_spec, lower_dsl_spec,
)
from mindie_sim.templates import catlass_catalog as cat, dsl_catalog as dsl
from mindie_sim.templates.catlass_lower import solve_catlass_tile
from mindie_sim.templates.contracts import evg_order
from mindie_sim.templates.fuse_region import validate_live_buffer_owners

pytestmark = pytest.mark.tier0

KIB = 1024
SECTOR = 32
ONE_SLOT = (1, 1, 1, 1, 1)
SIMT = "catlass.a5_simt"
ROOT = Path(__file__).resolve().parents[2]

def _datapath(sku):
    return datapath_for_die(Hardware.from_mapping({"backend": "ascend", "sku": sku, "num_devices": 1}).dies[0])

@pytest.fixture(scope="module")
def datapaths():
    return tuple(_datapath(REPRESENTATIVE_SKUS[generation]) for generation in ("a2", "a5"))

@pytest.fixture(scope="module")
def direct(datapaths):
    return datapaths[1]

def _matrix(name, p, dp, *, tile=(32, 32, 32), **options):
    return enumerate_catlass_specs(p, dp, template_ids=("catlass." + name,), l1_tiles=(tile,),
                                   **{"aic_budget": 4, "aiv_budget": 8, **options})

def _reference(name, p, dp, **options):
    spec, = enumerate_dsl_specs(name, p, dp, **{"aic_budget": 8, "aiv_budget": 16, **options})
    return spec, lower_dsl_spec(spec, p, dp)

def _stages(result):
    return {stage.name: stage for stage in result.program.stages}

def _ancestors(stages, name):
    seen, pending = set(), list(stages[name].after)
    while pending:
        parent = pending.pop()
        if parent not in seen:
            seen.add(parent)
            pending.extend(stages[parent].after)
    return seen

def _bytes(stages, suffix, pipe):
    return sum(stage.work.get(pipe, 0) for name, stage in stages.items() if name.endswith(suffix))

def _context(dp, p):
    units = dict.fromkeys(("cube", "mte1", "mte2_aic", "fixpipe"), dp.aic_count)
    units.update(dict.fromkeys(("mte2_aiv", "mte3", "vec_reduce"), dp.aiv_count), hbm=1, l2=1)
    rates = dict.fromkeys(units, 10**6)
    return RecipeContext(p, {}, dict(zip(units, units)), rates, dp.aic_count, resource_units=units,
                         compute_peaks={"bf16": 1000 * rates["cube"]},
                         pool_names={pool: pool for pool, _ in dp.buffers})

def _run(dp, context, result):
    resources = [Resource(role, context.rates[role], units=units)
                 for role, units in context.resource_units.items() if role != "hbm"]
    pools = dict(dp.buffers)
    engine = Engine(resources, pools=pools, backend="python", pool_units={
        pool: 1 if pool == "l2" else dp.aiv_count if pool == "ub" else dp.aic_count for pool in pools
    })
    done = result.program.start(engine, "matrix", engine.submit("ready", release=1))
    engine.advance(until=(done,))
    assert done.time > 1 and not engine.allocations
    return engine

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
    result = lower_catlass_spec(spec, p, dp, context=context)
    stages = _stages(result)
    for operand in ("a", "b"):
        assert f"tile:0:l1{operand}:g0:last_use" in stages[f"tile:0:k2:gm_{operand}"].after
        assert f"tile:0:l0{operand}:g0:last_use" in stages[f"tile:0:l0{operand}:g2:load"].after
    assert sum(s.metadata.get("logical_flops", 0) for s in stages.values()) == 2 * p.m * p.n * p.k
    issued, = dict(result.summary.issued_cube_flops_by_dtype).values()
    assert _run(dp, context, result).serviced["cube"] * context.rates["cube"] == pytest.approx(issued)

@pytest.mark.parametrize("has_bias", (False, True))
def test_mutex_budget_unit_flag_and_l1_resident_identity(direct, has_bias):
    p = CatlassProblem(32, 32, 32, has_bias=has_bias)
    fits = (9, 7, 6, 5 - 2 * has_bias, 1)
    assert sum(fits) + 2 * has_bias == 28 and len(set(fits)) == 5
    spec, = _matrix("mutex_block_mmad", p, direct, stage_options=(fits,), enable_unit_flag=True)
    assert spec.stages == fits
    assert dict(lower_catlass_spec(spec, p, direct).summary.sram_peaks_bytes)["l1"] == (
        (fits[0] + fits[1]) * 32 * 32 * 2 + has_bias * 32 * 4
    )
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
    assert stages[flag].metadata["streamk_partial_count"] == 4
    assert set(stages[flag].after) == {f"tile:{i}:partial" for i in range(4, 8)}
    early = {f"tile:{i}:fixpipe" for i in range(4)}
    assert all(early <= _ancestors(stages, partial) for partial in stages[flag].after)
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
    short = replace(p, k=128)
    with pytest.raises(ModelingError, match="catlass_splitk_factor_invalid"):
        lower_catlass_spec(spec, short, direct)
    context = _context(direct, p)
    for pressured in (False, True):
        view = AdmissionMemoryView({0: KIB**2}, granule_bytes=SECTOR)
        result = lower_catlass_spec(spec, p, direct, context=context, memory_view=view, live_admission=True)
        inputs, output = (p.m + p.n) * p.k * 2, p.m * p.n * 2
        assert result.summary.logical_transfer_bytes == inputs + output + 2 * spec.workspace_bytes
        assert result.dram_miss_bytes == inputs + output
        if pressured:
            view.grant_reservation(0, "concurrent", KIB**2)
        delivered = _run(direct, context, result).delivered
        assert delivered["hbm:ctr_r"] == pytest.approx(inputs + pressured * spec.workspace_bytes)
        assert delivered["hbm:ctr_w"] == pytest.approx(output + pressured * spec.workspace_bytes)

def test_swizzle_serpentine_coordinates_and_admission_reuse(direct):
    def serpentine(major, minor):
        bands = [range(start, min(start + 2, major)) for start in range(0, major, 2)]
        return [(i, j) for band, strip in enumerate(bands)
                for j in (range(minor) if band % 2 == 0 else reversed(range(minor))) for i in strip]
    for direction, expected in (("Zn", serpentine(5, 3)), ("Nz", [(j, i) for i, j in serpentine(3, 5)])):
        assert [cat.swizzle_block_coord(i, 5, 3, direction, 2) for i in range(15)] == expected
        for active in (1, 4, 7, 15, 20):
            assert cat.wave_band_counts(5, 3, active, direction, 2) == tuple(map(len, map(set, zip(*expected[:active]))))
    p = CatlassProblem(48, 32, 32)
    specs = _matrix("swizzle", p, direct, tile=(16, 16, 32), swizzle_offsets=(1, 2))
    assert [cat.swizzle_block_coord(i, 3, 2, "Nz", 2) for i in range(6)] == list(product(range(3), range(2)))
    for direction, offset, expected in (("Nz", 2, 2 * KIB), ("Zn", 1, 3 * KIB)):
        spec, = (s for s in specs if (s.swizzle_direction, s.swizzle_offset) == (direction, offset))
        view = AdmissionMemoryView({0: 2 * KIB}, granule_bytes=SECTOR)
        assert lower_catlass_spec(spec, p, direct, memory_view=view).l2_hit_bytes == expected

def test_evg_visitor_dag_and_direct_epilogue_traffic(datapaths):
    indirect, direct = datapaths
    roots = (EvgVisitorNode("acc", "acc_load", "fp32"), *(EvgVisitorNode(kind, kind, "bf16", aux_tensor_id=kind)
              for kind in ("aux_load", "row_broadcast", "col_broadcast")))
    unary = ("exp", "relu", "sqrt", "rsqrt_fast", "leaky_relu", "silu", "cast", "muls")
    chain = tuple(EvgVisitorNode(kind, kind, "fp32", inputs=(parent,), scalar_param=0.25)
                  for kind, parent in zip(unary, ("acc", *unary)))
    tree = (*roots, *chain, EvgVisitorNode("mul", "mul", "fp32", inputs=("muls", "row_broadcast")),
            EvgVisitorNode("add", "add", "fp32", inputs=("mul", "col_broadcast")),
            EvgVisitorNode("store", "aux_store", "bf16", inputs=("add",), aux_tensor_id="aux_output"))
    positions = {n.node_id: i for i, n in enumerate(evg_order(tuple(reversed(tree))))}
    assert len(positions) == 15 and all(positions[parent] < positions[n.node_id] for n in tree for parent in n.inputs)
    with pytest.raises(ModelingError, match="evg_dag_unknown_or_duplicate_node"):
        evg_order(tree[1:])
    with pytest.raises(ModelingError, match="evg_dag_cycle"):
        evg_order((*tree[:4], replace(chain[0], inputs=("add",)), *tree[5:]))
    p = CatlassProblem(37, 39, 33)
    gm, ub = _matrix("evg", p, direct, evg_tree=tree)
    old_gm, = _matrix("evg", p, indirect, evg_tree=tree)
    with pytest.raises(UnmodeledCapabilityError, match="requires_l0c_to_ub"):
        lower_catlass_spec(replace(old_gm, epilogue_kind="evg_ub", workspace_bytes=0), p, indirect)
    gm_result, ub_result = (lower_catlass_spec(spec, p, direct) for spec in (gm, ub))
    saved = 2 * p.m * p.n * 4
    assert gm_result.summary.logical_transfer_bytes - ub_result.summary.logical_transfer_bytes == saved
    assert dict(ub_result.summary.vector_ops_by_pipe) == {"vec_alu": 9 * p.m * p.n, "vec_sfu": 4 * p.m * p.n}

@pytest.fixture
def mx_datapath(direct):
    arch = direct.compute_arch.declared
    formats = (("mxfp8", 8), ("mxfp4", 4), ("a8w4_mx", 4))
    arch = replace(arch, representations=(*arch.representations, *(RepresentationCapability(name, bits, ("operand",))
                   for name, bits in formats)), native_modes=(*arch.native_modes, *(
                       NativeComputeMode(name, "matrix", (name,), "cube", ("fp32",)) for name, _ in formats)))
    return replace(direct, compute_arch=replace(direct.compute_arch, declared=arch))

def test_grouped_scheduler_carries_core_index_and_bypasses_single_row_tiles(direct):
    groups = (16, 0, 33, 16)
    p = CatlassProblem(sum(groups), 64, 32, groups=groups)
    assert tuple(cat.grouped_start_cores(groups, p.n, 32, 32, 4)) == ((0, 0, 2), (2, 2, 4), (3, 2, 2))
    spec, = _matrix("grouped_slice_m", p, direct)
    result = lower_catlass_spec(spec, p, direct, context=_context(direct, p), live_admission=True,
                                memory_view=AdmissionMemoryView({0: KIB**2}))
    stages = _stages(result)
    reads = [stages[f"tile:{i}:k0:gm_b"].program.pattern for i in range(8)]
    assert [read.owner for read in reads] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [i for i, read in enumerate(reads) if read.reuse_key is not None] == [2, 3, 4, 5]
    single = replace(p, m=32, groups=(16, 0, 16))
    only, = _matrix("grouped_slice_m", single, direct)
    assert only.l2_cache_hint_b == "disable"

@pytest.mark.parametrize("acc,width", [("fp32", 4), ("bf16", 2)])
def test_grouped_swiglu_quantization_avoids_gate_up_spills(mx_datapath, acc, width):
    m, f = 48, 64
    p = CatlassProblem(m, 2 * f, 32, "mxfp8", "mxfp8", dtype_acc=acc, groups=(16, 0, 32))
    spec, = _matrix("grouped_mx_swiglu_mx_quant", p, mx_datapath)
    result = lower_catlass_spec(spec, p, mx_datapath)
    assert spec.l2_cache_hint_b == "disable" and spec.workspace_bytes == 0
    assert sum(s.metadata.get("saved_logical_bytes", 0) for s in result.program.stages) == 4 * m * f * width
    operands = (4 * m + 2 * p.n) * (p.k + 1)  # Four A reads per row, two B experts, one scale per K block.
    assert result.summary.logical_transfer_bytes == operands + m * (2 * f + f // 32)
    output = next(x for x in result.summary.write_obligations if x.allocation_id == p.c_id)
    assert output.ranges == ((0, m * f * 2),)
    assert result.summary.semantic_work[0].logical_ops == 2 * m * (2 * f) * p.k

@pytest.mark.parametrize("mode,a,b,a_width,b_width,factor", [
    ("mxfp8", "mxfp8", "mxfp8", 1, 1, 1), ("mxfp4", "mxfp4", "mxfp4", 0.5, 0.5, 1),
    ("a8w4_mx", "fp8_e4m3", "fp4_e2m1", 1, 0.5, 16),
])
def test_mx_scale_storage_and_decompression_follow_declared_mode(mx_datapath, direct, mode, a, b, a_width, b_width, factor):
    p = CatlassProblem(32, 32, 128, a, b, mx_mode=mode)
    spec, = _matrix("mx_matmul", p, mx_datapath, tile=(32, 32, 128), stage_options=(ONE_SLOT,))
    elements, panel = p.n * p.k, p.n * 64
    expanded_width = a_width if factor == 16 else b_width
    assert spec.l1_scale_factor_k == factor
    peaks = {row.pool: row for row in solve_catlass_tile(spec, p, mx_datapath).pool_peaks}
    assert peaks["l1"].peak_bytes == elements * (a_width + expanded_width) + 2 * elements // 32 * factor
    assert peaks["l0b"].peak_bytes == panel * expanded_width + panel // 32
    assert all(peaks[pool].owner_kind == "aic" for pool in ("l1", "l0a", "l0b"))
    stages = _stages(lower_catlass_spec(spec, p, mx_datapath))
    assert sum(_bytes(stages, f":scale_{x}", "mte2_aic") for x in ("a", "b")) == 2 * elements // 32
    if factor == 16:
        assert spec.prologue_b == "DecompressFp4ToFp8"
        assert _bytes(stages, ":gm", "mte2_aiv") == elements * b_width
        assert _bytes(stages, ":decompress_b", "vec_alu") == elements
        assert _bytes(stages, ":ub_to_l1", "mte3") == elements * expanded_width
    misleading_peaks = replace(_context(direct, p), compute_peaks={mode: 10**12, a: 10**12})
    with pytest.raises(UnmodeledCapabilityError, match="native_mx_mode_unsupported"):
        lower_catlass_spec(spec, p, direct, context=misleading_peaks)

@pytest.mark.parametrize("policy,threads", [("tla_dsl", 2048), ("MatmulGatherScatterSimt", 1024),
                                            ("GroupedMxMatmulFinalizeRoutingTla", 2048)])
def test_simt_thread_ceilings_carveout_and_routing_workspace(datapaths, policy, threads):
    indirect, direct = datapaths
    p = CatlassProblem(32, 32, 32)
    options = dict(simt_variants=(policy,), simt_thread_blocks=((32, threads // 32, 1),))
    spec, = _matrix(SIMT.removeprefix("catlass."), p, direct, **options)
    assert direct.usable_bytes("ub") == 248 * KIB and direct.usable_bytes("ub", kernel=True) == 232 * KIB
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
            f"{SIMT}:matrix", "matrix_mac", "bf16", 2 * p.m * p.n * p.k, ("aiv_simt",), ("dataparallel",),
        ),)
        ub, = solve_catlass_tile(spec, p, direct).pool_peaks
        assert (ub.peak_bytes, ub.capacity_bytes) == (threads * 4, 232 * KIB)
        assert stages["simt:0:simt_mac"].metadata["thread_block_dim"] == options["simt_thread_blocks"][0]

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
            assert enumerate_dsl_specs(name, p, dp, aic_budget=8, aiv_budget=16)

def test_flash_kda_cost_threshold_and_saved_output_visibility(direct):
    # Chunks (3, 1, 2) give cyclic peak 5, LPT peak 3 and two staging rounds.
    staged = (2 + 3 * 3 + 5 * 5 + 7 * 2) * 64 + (11 + 17) * 384
    for p0, margin in ((154, 0), (155, 0), (156, 0), (154, 5000)):
        costs = (2, 3, 5, 7, 11, 13, 17, p0, 3, 2, 3, 2, 5, 7, margin, 0)
        persistent = (p0 + 3 * 3 + 2 * 3) * 64 + 3 * 384 + 10 * 192
        expected = "persistent" if persistent < staged - staged * margin // 1_000_000 else "staged"
        assert dsl.select_flash_kda_template((192, 64, 128), 2, cost_coefficients=costs, workspace_slots=3) == expected
    p = dsl.FlashKdaProblem((65,), metadata_ms=0.1, eager_d2h_bytes=256, save_matrices=True,
                            save_kernel_caches=True, save_chunk_states=True)
    spec, result = _reference(dsl.FLASH_KDA, p, direct)
    assert spec.template_variant == "staged" and spec.uses_aicpu_metadata and spec.has_eager_d2h_readback
    assert result.summary.stars_launch_count == 2 and result.summary.aicpu_metadata_ms == Fraction(1, 10)
    stages = _stages(result)
    assert stages["metadata_eager_d2h"].after == ("aicpu_metadata",) and stages["launch"].after == ("metadata_eager_d2h",)
    assert stages["metadata_eager_d2h"].work["d2h"] == p.eager_d2h_bytes
    for tensor, size in {"Akk": 65 * 64 * 2, "Aqk": 65 * 64 * 2, "qg": 65 * 128 * 2,
                         "g_cumsum": 65 * 128 * 4, "h": 2 * 128**2 * 4}.items():
        assert dict(spec.l2_cache_policies)[tensor] == "disable"
        assert _bytes(stages, f":save_{tensor}", "hbm") == size
    _, persistent_result = _reference(dsl.FLASH_KDA, replace(p, requested_template="persistent"), direct)
    assert _bytes(stages, ":q_decayed:read", "mte2_aic") == 65 * 128 * 2
    assert _bytes(_stages(persistent_result), ":q_decayed:read", "mte2_aic") == 0

@pytest.mark.parametrize("tp,sequence", tuple(product((4, 8, 16), (4, 8))))
def test_mega_recurrent_variant_workspace_reads_and_launch_savings(direct, tp, sequence):
    p = dsl.MegaRecurrentKdaProblem(1, tp, sequence)
    spec, result = _reference(dsl.MEGA_KDA, p, direct)
    stages, heads = _stages(result), 96 // tp
    assert spec.template_variant == f"tp{tp}_s{sequence}"
    for tensor, width in (("Q", 128), ("K", 128), ("V", 128), ("decay", 128), ("beta", 1), ("output_gate", 128)):
        payload = SECTOR * ceil(sequence * heads * width * 2 / SECTOR)
        assert _bytes(stages, f":{tensor}:gm:read", "mte2_aiv") == tp * payload
        assert stages[f"mega:0:{tensor}:gm:read"].after == (f"mega:0:{tensor}:gm:write",)
    assert spec.workspace_bytes_total == tp * sequence * 2 * (5 * heads * 128 + 16 * ceil(heads / 16) + 128) + 128**2 * 4
    assert result.summary.stars_launch_count + stages["mega:unfused_baseline"].metadata["saved_launches"] == len(p.unfused_launches)

@pytest.mark.parametrize("partitions", (1, 2))
def test_commit_zero_accept_and_partitioned_replay_bytes(direct, partitions):
    p = dsl.CommitRecurrentKdaProblem((0, 2), row_partitions=partitions)
    _, result = _reference(dsl.COMMIT_KDA, p, direct)
    stages = _stages(result)
    assert not any(name.startswith("commit:0:") for name in stages)
    assert _bytes(stages, ":checkpoint_read", "mte2_aiv") == _bytes(stages, ":checkpoint_write", "mte3") == 4 * 128**2
    replay = 4 * 128 + 6 * partitions * 128
    assert sum(_bytes(stages, f":replay_{tensor}:read", "mte2_aiv") for tensor in ("u", "k", "decay")) == 2 * replay
    assert dsl.commit_replay_bytes(partitions) == replay
    _, empty = _reference(dsl.COMMIT_KDA, replace(p, accepted=(0, 0)), direct)
    assert empty.summary.logical_transfer_bytes == 0 and not empty.summary.read_obligations and not empty.summary.write_obligations
    for budget, quota in ((8, 8), (23, 8), (24, 24), (27, 24), (28, 28), (31, 28), (32, 32)):
        spec, = enumerate_dsl_specs(dsl.COMMIT_KDA, p, direct, aic_budget=budget)
        assert (spec.aic_budget, spec.aiv_budget) == (quota, 2 * quota)

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
    width = 1 if mode in {dsl.QkvQuantMode.FP8, dsl.QkvQuantMode.HiF8} else 2
    assert stages["mla:0:q_output"].work["mte3"] == p.tokens * p.heads * p.kvr * width

@pytest.mark.parametrize("tokens", (1, 5))
def test_attention_residual_ub_and_inplace_savings(direct, tokens):
    p = dsl.AttnResProblem(tokens)
    d = 7168
    spec, result = _reference(dsl.ATTN_RES, p, direct, aiv_budget=1)
    tile = spec.tile_shape[0]
    depth = 1 if ceil(tokens / tile) == 1 else 2
    assert (tile, depth) == (1, 1 if tokens == 1 else 2)
    assert spec.ub_peak_bytes == 6 * d + depth * (10 * d * tile + 12 * 8 * ceil(tile / 8))
    assert dsl.attn_res_tiling(8190, 1, ub_bytes=2**32) == (1, 8190, 4095, 2)
    stages = _stages(result)
    assert stages["attn_res:inplace_contract"].metadata["saved_logical_bytes"] == 4 * tokens * d
    assert _bytes(stages, ":partial", "mte2_aiv") == _bytes(stages, ":inplace_partial", "mte3") == 4 * tokens * d
    assert [(x.allocation_id, x.tensor_version, x.ranges) for x in result.summary.write_obligations
            if x.allocation_id.endswith(":partial_block")] == [("attn_res:partial_block", 1, ((0, 4 * tokens * d),))]
    for bad in (replace(p, hidden_size=d - 1), replace(p, dtype="fp16")):
        with pytest.raises(ModelingError, match="attn_res_requires_fixed_bf16_width"):
            enumerate_dsl_specs(dsl.ATTN_RES, bad, direct)

@pytest.mark.parametrize("splits", (1, 2))
def test_sparse_indexer_packed_rings_histogram_and_ld_workspace(direct, splits):
    trunk, bins, topk = 16_384, 256, 32
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

@pytest.mark.parametrize("splits", (1, 2))
def test_mixed_sparse_mla_bank_bytes_and_split_row_reduction(direct, splits):
    p = dsl.MixedQuantSparseMlaProblem(129, 72, split_parts=splits)
    spec, result = _reference(dsl.MQSMLA, p, direct)
    stages = _stages(result)
    assert (_bytes(stages, ":ori", "mte2_aiv"), _bytes(stages, ":cmp", "mte2_aiv")) == (544 * p.n_ori, 320 * p.n_cmp)
    slot = (64 * 512 + 2 * 64) * 4
    assert spec.workspace_bytes_total == 2 * slot * spec.aic_budget == 263_168 * spec.aic_budget
    writes = {name for name in stages if name.endswith(":fd_partial_write")}
    assert len(writes) == (splits > 1) * splits and sum(stages[n].work["mte3"] for n in writes) == len(writes) * slot
    barrier, reduce = "mqsmla:fd_barrier", "mqsmla:0:fd_reduce"
    if splits > 1:
        reads = stages[reduce].after
        assert set(stages[barrier].after) == writes
        assert all(stages[n].after == (barrier,) and stages[n].work["mte2_aiv"] == slot for n in reads)
        assert stages[reduce].work["vec_reduce"] == splits * slot // 4
        assert stages["mqsmla:0:output"].after == (reduce,)
    else:
        assert barrier not in stages and reduce not in stages

def _edge(route="l0c->ub", axes=("aic", "aiv"), dst="take", view="tile"):
    return EdgeRouteBinding("make", 0, view, dst, "tile_pipelined", route, *axes, "fp32", "RowMajor", 2)

def _region(dp, edge, **overrides):
    return compile_fuse_region("fused", ("make", "take"), dp, **{
        "dependencies": {"make": (), "take": ("make",)}, "edge_routes": (edge,), "edge_bytes": {edge.view_id: 64},
        "op_stages": {name: Stage(name) for name in ("make", "take")}, **overrides,
    })

def test_region_rejects_invalid_graph_and_unread_alias(direct):
    edge = _edge()
    for options, error in (
        ({"dependencies": {"make": (), "outside": ("make",), "take": ("outside",)}}, "not_convex"),
        ({"collective_ops": ("make",)}, "collective_barrier"),
        ({"waits": (("take", "make"),)}, "credit_publish_deadlock"),
    ):
        with pytest.raises(ModelingError, match=error):
            _region(direct, edge, **options)
    mutate, read = replace(edge, in_place_mutates=True), _edge(dst="read", view="alias")
    with pytest.raises(ModelingError, match="in_place_unread_alias"):
        compile_fuse_region("fused", ("make", "take", "read"), direct, dependencies={
            "make": (), "take": ("make",), "read": ("make",),
        }, op_stages={}, edge_bytes={}, edge_routes=(mutate, read), alias_views={"tile": ("x", 0, 64), "alias": ("x", 32, 96)})

def test_region_routes_admit_only_valid_pools_and_owner_pairs(datapaths):
    indirect, direct = datapaths
    for pool, owner in (("l1", "aiv"), ("ub", "aic")):
        with pytest.raises(ModelingError, match=f"live_buffer_pool_owner_mismatch:panel:{pool}:{owner}"):
            validate_live_buffer_owners((LiveBufferSpec("panel", pool, 64, 0, 1, owner_kind=owner),))
    _, spill = _region(indirect, _edge("gm_spill"))
    assert spill.summary.logical_transfer_bytes == 2 * 64
    for route, axes, pipe in (("l0c->ub", ("aic", "aiv"), "fixpipe"), ("ub->l1", ("aiv", "aic"), "mte3")):
        edge = _edge(route, axes)
        with pytest.raises(UnmodeledCapabilityError, match="does not declare directed route"):
            _region(indirect, edge)
        spec, result = _region(direct, edge)
        assert _stages(result)["p0:t0:edge:0"].work[pipe] == 64 and result.summary.logical_transfer_bytes == 0
        assert sorted(p.peak_bytes for p in spec.tile_solution.pool_peaks) == [64, 128]
        with pytest.raises(ModelingError, match="owner_pair_unreachable"):
            _region(direct, edge, edge_owners={"tile": (0, 2) if axes[0] == "aic" else (2, 0)})

def test_region_compiler_preserves_coupling_metadata_and_step_obligations(direct):
    stage = Stage("compute", {"cube": 8, "aiv_simt": 4}, coupled=True,
                  placement={"cube": (0,), "aiv_simt": (0, 1)}, metadata={"geometry": (2, 4), "launch": True})
    semantic = SemanticWorkItem("matrix", "matrix_mac", "bf16", 8, ("cube", "aiv_simt"), ("tiles",))
    penalty = MemoryAccessBurstPenalty("contiguous", 0, None, 0.1, "test-geometry")
    dp = replace(direct, memory_access=replace(direct.memory_access, burst_turn_penalty=(penalty,)))
    summaries = []
    for start in (0, SECTOR):
        read = TensorRangeObligation(0, "shared", 0, ((start, start + 2 * SECTOR),), "hbm_cold", "device")
        options = dict(dependencies={"compute": ()}, edge_routes=(), edge_bytes={}, backend="tla_simt",
                       semantic_work=(semantic,), live_in_obligations=(read,),
                       live_buffers=(LiveBufferSpec("tile", "ub", 232 * KIB, 0, 1, owner_kind="aiv"),))
        spec, result = compile_fuse_region("fused", ("compute",), dp, op_stages={"compute": stage}, **options)
        peak, = spec.tile_solution.pool_peaks
        assert (peak.peak_bytes, peak.capacity_bytes) == (232 * KIB, (248 - 16) * KIB)
        lowered = _stages(result)["p0:t0:compute"]
        assert lowered.coupled and lowered.placement == stage.placement and lowered.metadata == stage.metadata
        with pytest.raises(ModelingError, match="coupled_placement_not_proportional"):
            bad = replace(stage, placement={"cube": (0,), "aiv_simt": (0,)})
            compile_fuse_region("fused", ("compute",), dp, op_stages={"compute": bad}, **options)
        summary = result.summary
        assert isinstance(summary, FusedRegionSummary) and summary.semantic_work == (semantic,)
        assert summary.dram_sector_bytes == 2 * SECTOR and summary.turnaround_service_bytes == 2 * SECTOR + Fraction(1, 10)
        assert summary.spec == spec and dict(summary.issued_cube_flops_by_dtype) == {"bf16": 8}
        summaries.append(summary)
    reads, writes = compose_step_obligations(summaries)
    assert reads == (replace(read, ranges=((0, 3 * SECTOR),)),) and not writes
    for invalid in ((True, 2), {"inner": 1}, Fraction(1, 8)):
        with pytest.raises(ModelingError, match="stage_metadata_invalid"):
            replace(stage, metadata={"nested": invalid})

def test_catalog_provenance_inventory_and_blob_validation(monkeypatch):
    catalog = builtin_template_catalog()
    assert catalog.ids("K1") and len(catalog.ids("K2")) == 10 and len(catalog.ids("K3")) == 8
    refs = catalog.all_template_source_refs()
    assert len(refs) == len(set(refs)) == 34
    for ref in refs:
        assert re.fullmatch(r"[0-9a-f]{40}", ref.revision) and re.fullmatch(r"[0-9a-f]{64}", ref.source_sha256)
        assert ref.symbol.strip() and 1 <= ref.line_range[0] <= ref.line_range[1]
    spec = spec_from_file_location("template_provenance", ROOT / "scripts" / "check_template_provenance.py")
    audit = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, audit)
    spec.loader.exec_module(audit)
    assert audit.verify(ROOT) == (34, [])
    blob = b"// prefix\nvoid pinned_symbol();\n// suffix\n"
    digest = sha256(blob).hexdigest()
    ref = replace(refs[0], source_sha256=digest, symbol="pinned_symbol", line_range=(2, 2)).to_dict()
    monkeypatch.setattr(audit, "_load_blob", lambda *_: (digest, blob.splitlines()))
    for changes, reason in (({}, None), ({"source_sha256": "f" * 64}, "source_sha256_mismatch"),
                            ({"line_range": (3, 2)}, "invalid_line_range"),
                            ({"line_range": (2, 4)}, "source_line_range_exceeds_blob"),
                            ({"symbol": "other_symbol"}, "symbol_missing_from_range"), ({"symbol": ""}, "invalid_symbol")):
        declaration = f"TemplateSourceRef{tuple((ref | changes).values())!r}\n"
        with monkeypatch.context() as fixture:
            fixture.setattr(Path, "rglob", lambda folder, _: (folder / "fixture.py",))
            fixture.setattr(Path, "read_text", lambda *_args, **_kwargs: declaration)
            _, findings = audit.verify(ROOT, runtime=False, refs_root=Path("fixtures"))
        assert [f.reason.split(":")[0] for f in findings] == ([reason] if reason else [])
