"""Check grouped matrix work, ABI legality, and the fused MoE execution ledger.

[Role] Exercise source scheduling and the public governed grouped MoE resolver.
[Invariants] Expert histograms preserve useful work, issued fractals, buffer
  ownership, and the architecture-specific matrix-to-vector handoff.
[Timeouts & Locks] Bounded analytical schedules and small Engine replays.
[Side-Effect Policy] No device access. C11 uses the session-isolated cache.
"""
from collections import Counter
from dataclasses import replace
from math import ceil
from pathlib import Path
from types import SimpleNamespace
import json
import pickle

import pytest

from _support.axes import ARCH_IDS, PHASES, REPRESENTATIVE_SKUS, SKUS
from _support.hardware import REGBASE_ISA, isa_of, machine
from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.operators.moe_routing.moe_gating_fusion import _fused_gating, build_moe_gating_topk
from mindie_sim.prediction.sharding.ffn_rules import router_gemm
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.operators import FamilyDispatcher, FamilyRegistry, grouped_moe
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.grouped_moe.lowering import StageDemand
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.operators import OpSpec

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
FRACTAL = 16
K0S = (("fp32", 8), ("bf16", 16), ("fp16", 16), ("int8", 32), ("mxfp8", 32), ("mxfp4", 64))
AIC_PIPES = ("cube", "mte1", "mte2_aic", "fixpipe")
AIV_PIPES = ("mte2_aiv", "vector", "mte3")
XLLM_ABI = "aclnnMoeGroupedMatmulSwigluQuant"
MX_ABI = "aclnnGroupedMatmulSwigluMxQuant"
MEGA_ABI = "aclnnMegaMoe"
MX_SCALE_GROUP_WIDTH = 32
MX_SCALE_PAIR_WIDTH = 64
ROUTER_TILE_M = 128
ROUTER_TILE_N = 256
INT8_DISPATCH_ROW_BYTES = 36
BF16_COMBINE_ROW_BYTES = 64


def hardware(sku):
    return Geometry.from_devices(machine(sku).dies[:1])


@pytest.fixture
def geometry(arch):
    return hardware(next(sku for sku in REPRESENTATIVE_SKUS.values() if isa_of(sku) == arch))


def make_op(rows, n, k, *, mode="grouped_gemm", dtype="bf16", **fields):
    shape = (sum(rows), n, k)
    metadata = dict(mode=mode, group_rows=rows, n=n, k=k, **fields)
    return OpSpec("matrix", "grouped_moe", "decode", 0, 0, (), shape, shape, shape, shape,
                  dtype=dtype, recipe=mode, metadata=tuple(metadata.items()))


def resolve(op, geometry):
    ctx = OperatorPlanningContext(OpOccurrence(0, 0, geometry.rank, op.op_id), geometry,
                                  geometry.datapath)
    return grouped_moe.resolve_operator_plan(op, ctx)


def bind(op, geometry, work):
    roles = set(work.resources) | {"cube"}
    return RecipeContext(op, {}, {role: role for role in roles}, {}, geometry.compute_units,
                         compute_peaks=dict.fromkeys(work.matrix, 1000),
                         datapath=geometry.datapath, resource_units=geometry.resource_units,
                         pool_names={"ub": "ub"}, pool_units={"ub": geometry.datapath.aiv_count})


def check_ledger(resolved, context):
    plan = resolved.recipe
    stages = plan.program(context).stages
    phases = {phase.phase_id: phase for phase in resolved.descriptor.phases}
    assert len(phases) == len(stages)
    assert tuple(phases) == tuple(stage.name for stage in stages)
    matrix, resources = Counter(), Counter()
    for stage in stages:
        phase = phases[stage.name]
        assert stage.after == phase.after
        typed = Work({item.dtype: item.amount for item in phase.work if item.resource == "cube"},
                     {item.resource: item.amount for item in phase.work if item.resource != "cube"})
        matrix.update(typed.matrix)
        resources.update(typed.resources)
        assert stage.work == context.bind(typed)
    assert matrix == Counter(plan.work.matrix) and resources == Counter(plan.work.resources)
    return stages


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("m,row_fractals", [(1, 1), (15, 1), (16, 1), (17, 2)])
@pytest.mark.parametrize("dtype,k0", K0S)
def test_short_rows_separate_useful_flops_from_issued_fractals(geometry, arch, m, row_fractals, dtype, k0):
    n, k = FRACTAL + 1, k0 + 1
    issues = row_fractals * ceil(n / FRACTAL) * ceil(k / k0)
    issued = 2 * FRACTAL**2 * k0 * issues
    assert grouped_moe.fractal_issues(m, n, k, dtype) == (issues, issued)
    op = make_op((0, m, 0), n, k, dtype=dtype, output_dtype="bf16")
    if arch != REGBASE_ISA and dtype.startswith("mx"):
        with pytest.raises(UnmodeledCapabilityError, match="compute_dtype|mx_operands"):
            resolve(op, geometry)
        return
    plan = resolve(op, geometry).recipe
    assert plan.metadata["useful_flops"] == 2 * m * n * k
    assert plan.metadata["issued_fractals"] == issues
    assert plan.work.matrix == {dtype: issued}
    assert plan.work.resources["fixpipe"] == m * n * 2


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("split", [False, True])
def test_blaze_virtual_intervals_wrap_and_keep_empty_experts(geometry, split):
    rows, n, k, cores, start = (0, 19, 0, 16, 0), 32, 48, 3, 2
    sched = grouped_moe.schedule_blaze_groups(
        rows, n, k, aic_cores=cores, base_m=16, base_n=16, start_core=start,
        tail_split=split, n_tail_cnt=2,
    )
    offset = 0
    for expert, (group, m) in enumerate(zip(sched.groups, rows, strict=True)):
        logical = ceil(m / 16) * ceil(n / 16)
        total = logical + (logical % cores if split else 0)
        assert (group.expert, group.offset, group.rows) == (expert, offset, m)
        assert (group.start_core, group.end_block) == (start, start + total)
        assert (group.logical_tiles, group.total_tiles) == (logical, total)
        offset += m
        start = (start + total) % cores
    assert sched.useful_flops == 2 * sum(rows) * n * k
    assert sum(sched.owner_output_elements) == sum(rows) * n
    assert sum(sched.owner_tiles) == sum(group.total_tiles for group in sched.groups)
    assert sched.issued_flops == sched.issued_fractals * 2 * FRACTAL**3
    offsets = tuple((sum(rows[:i]), m) for i, m in enumerate(rows))
    assert grouped_moe.normalize_group_rows((0, 19, 19, 35, 35), cumulative=True) == offsets
    for histogram, expected_rows in (((32, 0, 0), 2), ((15, 16, 1), 3)):
        op = make_op(histogram, n, k, base_m=16, base_n=n)
        plan = resolve(op, geometry).recipe
        assert plan.metadata["useful_flops"] == 2 * 32 * n * k
        assert plan.metadata["issued_fractals"] == expected_rows * (n // 16) * (k // 16)


@pytest.mark.parametrize("hardware_sku", SKUS)
def test_fused_swiglu_handoff_matches_physical_datapath(hardware_sku):
    geom = hardware(hardware_sku)
    on_chip = geom.datapath.has_l0c_to_ub
    rows, width, k = (3, 0, 5), 64, 32
    m, n = sum(rows), 2 * width
    abi, compute = (MX_ABI, "mxfp8") if on_chip else (XLLM_ABI, "int8")
    op = make_op(rows, n, k, mode="gmm_swiglu_quant", requested_abi=abi, dtype=compute,
                 layout_b="ND" if on_chip else "NZ", handoff_dtype="int32",
                 output_dtype="fp8_e4m3" if on_chip else "bf16")
    resolved = resolve(op, geom)
    plan = resolved.recipe
    stages = check_ledger(resolved, bind(op, geom, plan.work))
    assert resolved.implementation == abi
    assert plan.metadata["useful_flops"] == 2 * m * n * k
    gm = 16 * m * width
    resources = plan.work.resources
    assert resources["fixpipe"] == gm // 2
    assert resources["vector"] == m * width * (5 + 4)
    assert resources["mte1"] == resources["mte2_aic"]
    fix = [stage for stage in stages if "fixpipe" in stage.work]
    if on_chip:
        assert plan.metadata["gm_handoff_eliminated_bytes"] == gm
        assert "mte2_aiv" not in resources
        assert resources["mte3"] == m * (width + width // 32)
        assert resources["hbm"] == resources["l2"] == resources["mte2_aic"] + resources["mte3"]
        assert all(set(stage.work) == {"fixpipe"} and len(stage.after) == 2 for stage in fix)
        assert {slot.name for stage in stages for slot in stage.acquire} == {
            slot.name for stage in stages for slot in stage.free}
    else:
        assert plan.metadata["gm_handoff_bytes"] == gm
        scale_bytes = 4 * m + 4 * n * sum(count > 0 for count in rows)
        assert resources["mte2_aiv"] - scale_bytes == gm // 2
        assert resources["mte3"] == m * (width + 4)
        assert float(geom.rates.get("l2", 0.0) or 0.0) == 0.0
        assert "l2" not in resources
        assert resources["hbm"] == resources["mte2_aic"] + gm + scale_bytes + resources["mte3"]
        assert all(stage.work["hbm"] == stage.work["fixpipe"] for stage in fix)
        assert next(stage for stage in stages if stage.name == "weight_scales").after == ("gmm1",)


@pytest.mark.parametrize("arch", ARCH_IDS[:1])
@pytest.mark.parametrize("fields,reason", [
    ({"layout_b": "ND"}, "xllm_requires_int8_nd_by_int8_nz"),
    ({"layout_b": "NZ", "layout_a": "CONTIGUOUS"}, "xllm_requires_int8_nd_by_int8_nz"),
    ({"layout_b": "NZ", "dtype_b": "int4"}, "xllm_requires_int8_nd_by_int8_nz"),
    ({"layout_b": "NZ", "dtype_a": "bf16"}, "xllm_requires_int8_nd_by_int8_nz"),
    ({"layout_b": "NZ", "handoff_path": "l0c->ub"}, "requires_gm_handoff"),
    ({"layout_b": "NZ", "mask_routing": True}, "mask_routing"),
])
def test_xllm_and_on_chip_abi_guards(geometry, fields, reason):
    op = make_op((3,), 64, 32, mode="gmm_swiglu_quant", dtype="int8",
                 requested_abi=XLLM_ABI, **fields)
    with pytest.raises(UnmodeledCapabilityError, match=reason) as error:
        resolve(op, geometry)
    assert error.value.onboarding_hint["schema"] == "mindie-unmodeled-structure-hint/1"


@pytest.mark.parametrize("arch", ARCH_IDS[:1])
@pytest.mark.parametrize("dtype,weight,gap", [
    ("bf16", "bf16", None), ("int8", "int8", None), ("w4a8", "int4", None),
    ("int8", "int32", "int32_weight"), ("int8", "mxfp8", "mx_operands"), ("int8", "mxfp4", "mx_operands"),
])
def test_mega_moe_materializes_supported_operands_before_quant(geometry, dtype, weight, gap):
    op = make_op((4, 0), 64, 32, mode="mega_moe", dtype=dtype, dtype_b=weight,
                 requested_abi=MEGA_ABI, output_dtype="bf16", soc_tier="a2",
                 ep=2, dispatch_link_bytes=0, combine_link_bytes=0)
    if gap:
        with pytest.raises(UnmodeledCapabilityError, match=gap):
            resolve(op, geometry)
        return
    if dtype == "bf16":
        ep1 = make_op((4, 0), 64, 32, mode="mega_moe", dtype=dtype, dtype_b=weight,
                      requested_abi=MEGA_ABI, output_dtype="bf16", soc_tier="a2", ep=1)
        with pytest.raises(UnmodeledCapabilityError, match="arch22_mega_moe_ep_world_size:1"):
            resolve(ep1, geometry)
    resolved = resolve(op, geometry)
    plan = resolved.recipe
    stages = {stage.name: stage for stage in check_ledger(resolved, bind(op, geometry, plan.work))}
    assert resolved.implementation == MEGA_ABI
    assert plan.metadata["gm_handoff_bytes"] == 2 * 4 * 64 * (2 if dtype == "bf16" else 4)
    assert stages["gmm1"].metadata["useful_flops"] == 2 * 4 * 64 * 32
    assert stages["gmm2"].metadata["useful_flops"] == 2 * 4 * 32 * 32
    assert stages["combine:local"].after == ("gmm2",)


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
@pytest.mark.parametrize("activation,weight,specialization", [
    ("fp8_e4m3", "fp8_e4m3", "MegaMoeA8W8Wave"), ("mxfp8", "mxfp4", "MegaMoeA8W4Wave"),
    ("mxfp4", "mxfp4", "MegaMoeA4W4Wave"), ("bf16", "bf16", "MegaMoeA16W16Wave"),
])
@pytest.mark.parametrize("shared", [0, 32])
def test_mega_moe_wave_partitions_and_shared_expert(geometry, activation, weight, specialization, shared):
    counts, width, k = (3, 0, 6), 64, 32
    shared_compute, shared_weight = ("bf16", "bf16") if activation == "fp8_e4m3" else (activation, weight)
    layout_b = "NZ" if (activation, weight) == ("mxfp8", "mxfp4") else "ND"
    op = make_op(counts, 2 * width, k, mode="mega_moe", requested_abi=MEGA_ABI,
                 dtype=activation, dtype_b=weight, layout_b=layout_b, output_dtype="bf16",
                 ep=2, dispatch_link_bytes=0, combine_link_bytes=0,
                 wave_rows=4, enable_fused_mc2=3 if shared else 2, shared_intermediate=shared,
                 shared_compute_dtype=shared_compute, shared_weight_dtype=shared_weight)
    resolved = resolve(op, geometry)
    plan = resolved.recipe
    stages = {stage.name: stage for stage in check_ledger(resolved, bind(op, geometry, plan.work))}
    assert plan.metadata["wave_specialization"] == specialization
    if specialization == "MegaMoeA8W4Wave":
        assert plan.metadata["gm_handoff_eliminated_bytes"] == 0
        assert plan.metadata["gm_handoff_bytes"] == 8 * sum(counts) * width
    else:
        assert plan.metadata["gm_handoff_eliminated_bytes"] == 16 * sum(counts) * width
    assert "wave3_gmm1" not in stages
    for wave, rows in enumerate((4, 4, 1)):
        prefix = f"wave{wave}_"
        assert stages[prefix + "gmm1"].metadata["useful_flops"] == 2 * rows * (2 * width) * k
        assert stages[prefix + "gmm2"].metadata["useful_flops"] == 2 * rows * k * width
        assert prefix + "swiglu_quant" in stages["swiglu_quant"].after
        assert stages[prefix + "dispatch:local"].metadata["source_tokens"] == rows
        assert stages[prefix + "combine:local"].after == (prefix + "gmm2",)
        assert stages[prefix + "combine:local"].metadata["output_tokens"] == 0
    deps = stages["combine:local"].after
    assert set(deps) == {"gmm2", "wave0_combine", "wave1_combine", "wave2_combine"} | (
        {"shared_gmm2"} if shared else set())
    assert stages["combine:local"].metadata["output_tokens"] == sum(counts)
    if shared:
        assert stages["shared_gmm2"].metadata["useful_flops"] == 2 * sum(counts) * k * shared
    else:
        assert all(not name.startswith("shared_") for name in stages)


@pytest.mark.parametrize("arch", ARCH_IDS)
def test_architecture_bounds_fused_datapath(geometry):
    op = make_op((3,), 64, 32, mode="mega_moe", requested_abi=MEGA_ABI,
                 enable_fused_mc2=3, shared_intermediate=32,
                 ep=2, dispatch_link_bytes=0, combine_link_bytes=0)
    ep1 = make_op((3,), 64, 32, mode="mega_moe", requested_abi=MEGA_ABI,
                  enable_fused_mc2=3, shared_intermediate=32, ep=1)
    with pytest.raises(UnmodeledCapabilityError, match="mega_moe_ep_world_size:1"):
        resolve(ep1, geometry)
    if not geometry.datapath.has_l0c_to_ub:
        with pytest.raises(UnmodeledCapabilityError, match="shared_mc2"):
            resolve(op, geometry)
        with pytest.raises(UnmodeledCapabilityError, match="mx_epilogue"):
            resolve(make_op((3,), 64, 32, mode="gmm_swiglu_quant", dtype="int8",
                            requested_abi=MX_ABI), geometry)
    else:
        dp = geometry.datapath
        limited = replace(dp, matrix_output_paths=dp.matrix_output_paths - {"ub"},
                          usable_paths=tuple(p for p in dp.usable_paths if (p.source, p.target) != ("l0c", "ub")))
        with pytest.raises(UnmodeledCapabilityError, match="l0c_to_ub_fixpipe"):
            resolve(op, replace(geometry, datapath=limited))


@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("shared_hbm", [True, False])
def test_descriptor_roundtrip_and_causal_replay_match_native(geometry, shared_hbm):
    on_chip = geometry.datapath.has_l0c_to_ub
    dtype = "mxfp8" if on_chip else "int8"
    op = make_op((3, 0, 1), 64, 32, mode="gmm_swiglu_quant", dtype=dtype,
                 layout_b="ND" if on_chip else "NZ",
                 absorbed_source_op_ids=("activation",), output_dtype="fp8_e4m3" if on_chip else "bf16")
    resolved = resolve(op, geometry)
    plan, descriptor = resolved.recipe, resolved.descriptor
    context = bind(op, geometry, plan.work)
    stages = check_ledger(resolved, context)
    assert descriptor.owner == resolved.occurrence
    assert descriptor.member_op_ids == (op.op_id, "activation")
    assert pickle.loads(pickle.dumps(descriptor)) == descriptor
    copied = pickle.loads(pickle.dumps(plan))
    assert copied.work == plan.work
    assert copied.program(context).stages == stages
    assert copied.pipeline_spec.phases == descriptor.phases

    durations = []
    program = plan.program(context)
    units = geometry.resource_units | ({"hbm": 1} if shared_hbm else {})
    for backend in ("python", "c"):
        resources = [Resource(role, 128, units=units.get(role, 1))
                     for role in context.resource_names]
        engine = Engine(resources, pools={"ub": geometry.datapath.usable_bytes("ub")},
                        pool_units=context.pool_units, backend=backend, trace_limit=len(stages) + 8)
        ready = engine.submit("ready", release=2)
        engine.advance(until=(ready,))
        entry_steps = engine.accel_status()["native_steps"]
        done = program.start(engine, op.op_id, ready)
        engine.advance(until=(done,))
        assert engine.serviced == pytest.approx(context.bind(plan.work), rel=1e-12)
        summary = engine.summary()
        assert summary["live_tasks"] == summary["live_programs"] == 0
        assert not any(summary["live_allocated"].values())
        if backend == "c" and shared_hbm:
            assert engine.accel_status()["native_steps"] > entry_steps, engine.accel_status()
        assert all(row["start_ms"] >= ready.time for row in engine.trace if row["demand"])
        durations.append(done.time)
    assert durations[1] == pytest.approx(durations[0], rel=1e-12)


def read_case(name):
    return json.loads((ROOT / "examples" / name).read_text())


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_nz_tail_aligns_each_subtile_to_c0(geometry, dtype):
    op = make_op((16,), 48, 16, dtype=dtype, layout_b="NZ", tail_split=True, n_tail_cnt=2)
    resolved = resolve(op, geometry)
    stages = check_ledger(resolved, bind(op, geometry, resolved.recipe.work))
    writes = [stage.work["fixpipe"] for stage in stages if "fixpipe" in stage.work]
    assert writes == [16 * 32 * 2, 16 * 16 * 2]
    assert resolved.recipe.metadata["issued_fractals"] == 3
    assert resolved.recipe.metadata["useful_flops"] == 2 * 16 * 48 * 16


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
@pytest.mark.parametrize("rows", [(8,), (8, 8), (1,)])
def test_mx_v2_publishes_fp32_handoff_and_acknowledges_both_consumers(geometry, rows):
    abi = "aclnnGroupedMatmulSwigluQuantV2"
    op = make_op(rows, 128, 32, mode="gmm_swiglu_quant", dtype="mxfp8",
                 requested_abi=abi, output_dtype="fp8_e4m3fn", handoff_dtype="bf16")
    resolved = resolve(op, geometry)
    plan = resolved.recipe
    stages = check_ledger(resolved, bind(op, geometry, plan.work))
    assert resolved.implementation == resolved.descriptor.implementation == abi
    assert plan.work.resources["fixpipe"] == sum(rows) * 128 * 4
    assert plan.work.resources["mte3"] == sum(rows) * (64 + 2)
    pairs = dict(geometry.datapath.aic_aiv_pairs)
    for issue in (stage for stage in stages if "fixpipe" in stage.work):
        aic, = issue.placement["fixpipe"]
        consumers = [stage for stage in stages if stage.name.endswith(":sync")
                     and stage.name.split(":")[1] == issue.name.split(":")[1]]
        assert len(consumers) == 2
        assert tuple(stage.metadata["aiv_consumer"] for stage in consumers) == pairs[aic]
        assert sum(stage.metadata["rows"] for stage in consumers) == rows[aic]
    vectors = [stage for stage in stages if "vector" in stage.work]
    assert len(vectors) == sum(min(rows_on_aic, 2) for rows_on_aic in rows)
    assert sum(stage.work["vector"] for stage in vectors) == sum(rows) * 64 * 9


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
@pytest.mark.parametrize("input_dtype,n,output_dtype,expected_bytes", [
    ("mxfp4", 128, "fp8_e4m3fn", 528),
    ("mxfp4", 128, "fp4_e2m1", 272),
    ("mxfp8", 192, "fp8_e4m3fn", 800),
])
@pytest.mark.parametrize("base_n", [MX_SCALE_GROUP_WIDTH, MX_SCALE_PAIR_WIDTH, 2 * MX_SCALE_PAIR_WIDTH])
def test_mx_quant_stores_declared_payload_and_paired_scales(geometry, input_dtype, n, output_dtype,
                                                          expected_bytes, base_n):
    op = make_op((8,), n, MX_SCALE_PAIR_WIDTH, mode="gmm_swiglu_quant", dtype=input_dtype,
                 output_dtype=output_dtype, requested_abi="aclnnGroupedMatmulSwigluQuantV2", base_n=base_n)
    resolved = resolve(op, geometry)
    stages = check_ledger(resolved, bind(op, geometry, resolved.recipe.work))
    stores = [stage.work["mte3"] for stage in stages if "mte3" in stage.work]
    assert sum(stores) == expected_bytes
    if input_dtype == "mxfp8" and base_n == MX_SCALE_GROUP_WIDTH:
        assert stores == [132, 132, 132, 132, 136, 136]


@pytest.mark.parametrize("hardware_sku", SKUS[:2])
@pytest.mark.parametrize("declaration", [None, "soc_tier", "transport_abi", "soc_variant"])
def test_packed_int32_weights_follow_the_declared_soc_contract(hardware_sku, declaration):
    geometry = hardware(hardware_sku)
    legacy = hardware_sku == REPRESENTATIVE_SKUS["a2"]
    identity = {
        "soc_tier": "a2" if legacy else "a3",
        "transport_abi": "a2_hccl_client" if legacy else "a3_sio_hccs",
        "soc_variant": hardware_sku,
    }
    operands = dict(mode="mega_moe", dtype="int8", output_dtype="bf16", requested_abi=MEGA_ABI,
                    ep=2, dispatch_link_bytes=0, combine_link_bytes=0)
    fields = {declaration: identity[declaration]} if declaration else {}
    op = make_op((4, 0), 64, 32, dtype_b="int32", **operands, **fields)
    if legacy:
        with pytest.raises(UnmodeledCapabilityError, match="int32_weight"):
            resolve(op, geometry)
    else:
        plan = resolve(op, geometry).recipe
        explicit_int4 = make_op((4, 0), 64, 32, dtype_b="int4", compute_dtype="w4a8", **operands)
        assert plan.work == resolve(explicit_int4, geometry).recipe.work
        assert plan.work.matrix["w4a8"] > 0


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
def test_mega_moe_wave_readiness_releases_each_matrix_before_the_last_dispatch(geometry):
    rows = (15, 0, 1)
    dispatch = sum(rows) * INT8_DISPATCH_ROW_BYTES
    combine = sum(rows) * BF16_COMBINE_ROW_BYTES

    def transport(name, after, _amount, metadata):
        row_bytes = BF16_COMBINE_ROW_BYTES if dict(metadata)["transport_direction"] == "combine" else INT8_DISPATCH_ROW_BYTES
        return (StageDemand(name, after, Work(resources={"link": row_bytes})),)

    int8_op = make_op(rows, 64, 32, mode="mega_moe", dtype="int8",
                      requested_abi=MEGA_ABI, output_dtype="bf16", wave_rows=1, ep=2,
                      dispatch_link_bytes=dispatch, combine_link_bytes=combine, cohort_transport_builder=transport)
    with pytest.raises(UnmodeledCapabilityError, match="arch35_mega_moe_operand_pair:int8:int8"):
        resolve(int8_op, geometry)
    op = make_op(rows, 64, 32, mode="mega_moe", dtype="fp8_e4m3",
                 requested_abi=MEGA_ABI, output_dtype="bf16", wave_rows=1, ep=2,
                 dispatch_link_bytes=dispatch, combine_link_bytes=combine, cohort_transport_builder=transport)
    resolved = resolve(op, geometry)
    stages = check_ledger(resolved, bind(op, geometry, resolved.recipe.work))
    ancestry = {}
    for stage in stages:
        ancestry[stage.name] = set(stage.after).union(*(ancestry[parent] for parent in stage.after))
    for wave in range(16):
        prefix = f"wave{wave}_"
        chain = ("dispatch", "gmm1", "swiglu_quant", "gmm2", "combine")
        assert all(prefix + left in ancestry[prefix + right] for left, right in zip(chain, chain[1:]))
        if wave < 15:
            assert "wave15_dispatch" not in ancestry[prefix + "gmm1"]
    assert resolved.recipe.work.resources["link"] == dispatch + combine


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
@pytest.mark.parametrize("shared_dtype,shared_weight", [
    ("bf16", "bf16"), ("fp32", "fp32"), ("fp16", "fp16"), ("fp8_e5m2", "fp4_e2m1"),
])
def test_shared_expert_uses_its_own_precision_for_both_matmuls(geometry, shared_dtype, shared_weight):
    common = dict(mode="mega_moe", dtype="fp8_e4m3", output_dtype="bf16",
                  ep=2, dispatch_link_bytes=0, combine_link_bytes=0)
    op = make_op((8,), 128, 32, **common,
                 enable_fused_mc2=3, shared_intermediate=32,
                 shared_compute_dtype=shared_dtype, shared_weight_dtype=shared_weight)
    if shared_dtype != "bf16":
        with pytest.raises(ValueError, match=f"unsupported_shared_expert_precision:{shared_dtype}:{shared_weight}"):
            resolve(op, geometry)
        return
    resolved = resolve(op, geometry)
    routed = resolve(make_op((8,), 128, 32, **common), geometry)
    phases = resolved.descriptor.phases
    shared_mmad = [item for phase in phases if phase.phase_id.startswith("shared_")
                   for item in phase.work if item.resource == "cube"]
    assert len(shared_mmad) == 2 and all(item.dtype == shared_dtype for item in shared_mmad)
    assert resolved.recipe.work.matrix == {
        "fp8_e4m3": routed.recipe.work.matrix["fp8_e4m3"], shared_dtype: 2 * 16 * 32 * (64 + 32),
    }
    down_reads = [item.amount for phase in phases if phase.phase_id.startswith("shared_gmm2:")
                  for item in phase.work if item.resource == "mte2_aic"]
    assert down_reads == [(8 * 32 + 32 * 32) * 2]


@pytest.mark.parametrize("arch", ARCH_IDS[1:])
@pytest.mark.parametrize("bins,width", [(32, 64), (68, 128)])
def test_special_dequant_assigns_whole_groups_and_reads_channel_scales(geometry, bins, width):
    rows = tuple({0: 2, 2: 7, bins - 1: 1}.get(group, 0) for group in range(bins))
    allocated = 2 * bins
    op = make_op(rows, 2 * width, 0, mode="dequant_swiglu_quant", dtype="int32",
                 allocated_rows=allocated, requested_abi="aclnnDequantSwigluQuant")
    resolved = resolve(op, geometry)
    plan = resolved.recipe
    stages = check_ledger(resolved, bind(op, geometry, plan.work))
    reads = [stage for stage in stages if "mte2_aiv" in stage.work]
    assert [stage.metadata["group"] for stage in reads] == [0, 2, bins - 1]
    owners = min(allocated, geometry.datapath.aiv_count)
    for stage in reads:
        group = stage.metadata["group"]
        assert stage.placement["mte2_aiv"] == (group % owners,)
        assert stage.metadata["row_start"] == sum(rows[:group])
        assert stage.metadata["rows"] == rows[group]
        assert stage.work["mte2_aiv"] == rows[group] * (2 * width * 4 + 4) + 2 * width * 4
    assert plan.work.resources["mte3"] == sum(rows) * (width + 4)


@pytest.mark.parametrize("arch", ARCH_IDS)
def test_dense_gemm_is_owned_by_the_governed_dispatcher(geometry):
    op = replace(make_op((16,), 32, 16, mode="dense_gemm"), family="matmul")
    registry = FamilyRegistry((grouped_moe.register_family(),))
    ctx = OperatorPlanningContext(OpOccurrence(0, 0, 0, op.op_id), geometry, geometry.datapath)
    result = FamilyDispatcher(registry).resolve_plan(op, ctx)
    assert result.op.flops == result.recipe.work.matrix["bf16"] == 2 * 16 * 32 * 16
    assert result.descriptor.implementation == "aclnnGroupedMatmul"


@pytest.mark.parametrize("phase", PHASES)
def test_grouped_mla_projection_override_stays_local(phase):
    raw = read_case("cases/high-concurrency-20261030/new17/v4flash-a3-tp8-c4.json")
    raw["model"]["layers"] = 1
    quant = raw["model"]["quant"]
    float_operands = {"weight_format": "bf16", "activation_format": "bf16"}
    quant["category_overrides"] = {"mla": float_operands}
    opts = dict(scope="body", phase=phase, tokens=2, context=64)
    baseline = {op.recipe: op for op in target_graphs(raw, **opts)[0]}
    quant["category_overrides"]["mla.o_a"] = {"weight_format": "int8", "activation_format": "int8"}
    changed = {op.recipe: op for op in target_graphs(raw, **opts)[0]}
    a = changed["grouped_output_a"]
    assert a.dtype == "int8" and baseline[a.recipe].dtype == "bf16"
    assert a.flops == baseline[a.recipe].flops
    for role in ("grouped_output_b", "q_b", "grouped_gate_up"):
        assert changed[role].dtype == baseline[role].dtype
        assert changed[role].hbm_bytes == baseline[role].hbm_bytes


@pytest.mark.parametrize("m,n,k,admitted", [
    (1, 16, 8, False), (1, 16, 4096, True), (528, 288, 4096, True), (240, 384, 7168, True),
])
def test_router_split_k_preserves_matrix_and_workspace_service(m, n, k, admitted):
    from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul

    layer = SimpleNamespace(router_dtype="", router_output_dtype="fp32",
                            num_experts=n, router_input_size=k, index=0)
    model = SimpleNamespace(quant=QuantSpec("mx", weight_format="mxfp8"), activation_dtype="bf16")
    _, op, _, _ = router_gemm(prefix="route", plan=layer, deps=(), stage=0, requests=(),
                              tokens=m, model=model, hidden=k, mode="decode")
    prepared_op, plan = RecipeBook().prepare(op, hardware(REPRESENTATIVE_SKUS["a5"]))
    if not admitted:
        # Below DAV_3510 Stream-K K-axis threshold, router stays on SteadyMatmul with no workspace.
        assert isinstance(plan, SteadyMatmul)
        assert prepared_op.flops == 2 * m * n * k
        return
    tiles = ceil(m / ROUTER_TILE_M) * ceil(n / ROUTER_TILE_N)
    partial = m * n * 4 * plan.metadata["split_k"]
    assert plan.metadata["output_tiles"] == tiles
    assert 1 < plan.metadata["split_k"] <= k
    assert plan.metadata["partial_write_bytes"] == partial
    assert plan.work.matrix["bf16"] == 2 * m * n * k
    assert plan.work.resources["hbm"] == pytest.approx(op.hbm_bytes)
    assert plan.work.resources["l2"] == pytest.approx(2 * partial)
    assert plan.work.resources["fixpipe"] == pytest.approx(partial)
    assert plan.work.resources["mte2_aiv"] == pytest.approx(partial)


@pytest.mark.parametrize("tokens,groups,group_topk", [(1, 0, 0), (65, 1, 1), (960, 8, 4)])
def test_gating_waves_preserve_declared_logits_and_routing_work(tokens, groups, group_topk):
    experts, top_k = 384, 6
    layer = SimpleNamespace(num_experts=experts, top_k=top_k, router_mode="sigmoid",
                            router_norm_topk=True, router_groups=groups, router_group_topk=group_topk,
                            routed_scaling_factor=2.5, index=0)
    members = tuple(SimpleNamespace(vector_ops=tokens * experts, recipe=role) for role in ("score", "select"))
    op = _fused_gating(prefix="route", plan=layer, logits_input="logits", members=members, stage=0,
                       requests=(), tokens=tokens, mode="decode", weight_dtype="fp32",
                       weight_bytes=4, hash_table=False, noaux_tc=True)
    geom = hardware(REPRESENTATIVE_SKUS["a5"])
    plan = build_moe_gating_topk(op, geom)
    assert plan.program(bind(op, geom, plan.work)).stages[-1].program.count == ceil(tokens / 64)
    assert op.mte2_bytes == 4 * (tokens + 1) * experts
    assert op.mte3_bytes == 8 * tokens * top_k
    assert plan.work.resources["vector"] == 2 * tokens * experts
    assert plan.metadata["row_program"]["effective_groups"] == max(1, groups)


@pytest.mark.parametrize("rows", [1, 7])
def test_shared_int32_fusion_keeps_per_row_and_channel_scale_boundaries(rows):
    width, hidden = 16, 32
    quant = QuantSpec("w8a8", weight_format="int8", activation_format="int8")
    common = dict(stage=0, layer=0, requests=(), phase="prefill")

    def meta(role):
        return (("ffn_category", "shared_expert"), ("ffn_role", role))

    gate = gemm_op(op_id="p", deps=(), m=rows, n=2 * width, k=hidden, quant=quant,
                   category="shared_expert", family="gate_up_projection", recipe="dense_gate_up",
                   metadata=meta("gate_up"), **common)
    act = vector_op(op_id="a", deps=("p",), elements=rows * width, ops_per_element=4,
                    dtype="bf16", family="swiglu", recipe="swiglu", shape=(rows, width),
                    metadata=(*meta("activation"), ("activation_kind", "silu")), **common)
    down = gemm_op(op_id="q", deps=("a",), m=rows, n=hidden, k=width, quant=quant,
                   category="shared_expert", family="down_projection", recipe="dense_down",
                   metadata=meta("down"), **common)
    runtime = replace(parse_case(read_case("calls/quantized-grouped-small.json")).runtime,
                      kernels={"lowerings": {"shared_expert": "ascend_shared_gate_up_int32"}})
    ops = lower_implementations((gate, act, down), runtime, "bf16")
    by_id = {op.op_id: op for op in ops}
    fused = by_id[by_id["q"].deps[0]]
    assert by_id["p"].mte3_bytes == 8 * rows * width and by_id["p"].vector_ops == 0
    assert fused.deps[0] == "p" and "activation_scales" in by_id[fused.deps[1]].returns
    assert fused.mte2_bytes == 8 * rows * width + 8 * width + 4 * rows
    assert fused.mte3_bytes == rows * (width + 4)
    assert by_id["q"].deps == (fused.op_id,)


@pytest.mark.parametrize("batch,query", [(1, "analytic_outside_domain"), (16, "observable_response")])
def test_two_bank_bf16_service_preserves_analytical_work(batch, query):
    raw = read_case("cases/high-concurrency-20261030/kimi-source-complete/"
                    f"a5_kimi_tp8_e16_bf16_c{batch}_p4k_source_complete.json")
    raw["model"]["layers"] = 2
    kernels = raw["runtime"]["kernels"]
    kernels.update(read_case("service-regimes/a5-bf16-grouped-two-bank-20261030.json"))
    ops = target_graphs(raw, scope="body", phase="decode", tokens=1, context=4104,
                        sequences=batch, apply_body_defaults=True)[0]
    op = replace(next(op for op in ops if op.family == "grouped_gemm"), deps=())
    response = graph_costs(raw, (op,), input_ready_ms=7)
    del kernels["service_regimes"]
    control = graph_costs(raw, (op,), input_ready_ms=7)
    assert response["execution"]["work"] == control["execution"]["work"]
    assert response["execution"]["work"]["hbm_bytes"] >= dict(op.metadata)["packed_weight_bytes"]
    assert any(row["queries"].get(query) == 1 for row in response["parameters"]["service_regimes"])


def test_router_splitk_preserves_single_gemm_flops_on_typed_and_reprepared_ops():
    from mindie_sim.prediction.operators.matrix.matrix_splitk import build_router_splitk

    m, n, k = 16, 256, 4096
    layer = SimpleNamespace(
        router_dtype="", router_output_dtype="fp32",
        num_experts=n, router_input_size=k, index=0,
    )
    model = SimpleNamespace(quant=QuantSpec("mx", weight_format="mxfp8"), activation_dtype="bf16")
    _, op, _, _ = router_gemm(
        prefix="route", plan=layer, deps=(), stage=0, requests=(),
        tokens=m, model=model, hidden=k, mode="decode",
    )
    expected_flops = float(2 * m * n * k)
    geom = hardware(REPRESENTATIVE_SKUS["a5"])

    typed_op = replace(op, flops=expected_flops, matrix_flops=(("bf16", expected_flops),))
    plan_typed = build_router_splitk(typed_op, geom)
    assert plan_typed.work.matrix == {"bf16": pytest.approx(expected_flops)}
    assert sum(plan_typed.work.matrix.values()) == pytest.approx(expected_flops)

    reprepared_op = plan_typed.work.apply(op)
    plan_reprepared = build_router_splitk(reprepared_op, geom)
    assert plan_reprepared.work.matrix == {"bf16": pytest.approx(expected_flops)}
    assert sum(plan_reprepared.work.matrix.values()) == pytest.approx(expected_flops)


def test_splitk_workspace_charges_fixpipe_and_routes_warm_vs_spill_reads():
    from mindie_sim.prediction.operators.matrix.matrix_splitk import build_router_splitk

    def _exec_plan(op_spec, geom, plan):
        roles = set(geom.rates) | set(plan.work.resources) | {"cube"}
        rates = {r: float(geom.rates.get(r, 1e12)) / 1000.0 for r in roles}
        rates["cube"] = 1.0
        units = {r: int(geom.resource_units.get(r, 1)) for r in roles}
        ctx = RecipeContext(
            op_spec,
            {},
            {r: r for r in roles},
            rates,
            geom.compute_units,
            resource_units=units,
            compute_peaks=dict(geom.compute_peaks),
            pool_names={p: p for p in geom.buffers},
            pool_capacities=dict(geom.buffers),
            pool_units=dict(geom.buffer_units),
            physical_mechanisms=geom.physical_mechanisms,
            datapath=geom.datapath,
        )
        ctx = replace(ctx, work=ctx.bind(plan.work))
        eng = Engine(
            [Resource(r, rates[r], units=units[r]) for r in roles],
            pools=dict(geom.buffers), pool_units=dict(geom.buffer_units),
        )
        done = plan.program(ctx).start(eng, "splitk", eng.submit("in"))
        eng.advance(until=(done,))
        return eng

    m, n, k = 32, 256, 4096
    layer = SimpleNamespace(
        router_dtype="", router_output_dtype="fp32",
        num_experts=n, router_input_size=k, index=0,
    )
    model = SimpleNamespace(quant=QuantSpec("mx", weight_format="mxfp8"), activation_dtype="bf16")
    _, op, _, _ = router_gemm(
        prefix="route", plan=layer, deps=(), stage=0, requests=(),
        tokens=m, model=model, hidden=k, mode="decode",
    )
    geom_l2 = hardware(REPRESENTATIVE_SKUS["a5"])
    plan_l2 = build_router_splitk(op, geom_l2)
    partial_l2 = float(plan_l2.metadata["partial_write_bytes"])
    assert plan_l2.metadata["workspace_residency"] == "l2"
    assert plan_l2.work.resources["hbm"] == pytest.approx(op.hbm_bytes)
    assert plan_l2.work.resources["l2"] == pytest.approx(2.0 * partial_l2)
    assert plan_l2.work.resources["fixpipe"] == pytest.approx(partial_l2)
    assert plan_l2.work.resources["mte2_aiv"] == pytest.approx(partial_l2)

    eng_l2 = _exec_plan(op, geom_l2, plan_l2)
    assert eng_l2.serviced["hbm"] == pytest.approx(op.hbm_bytes)
    assert eng_l2.serviced["l2"] == pytest.approx(2.0 * partial_l2)
    assert eng_l2.serviced["fixpipe"] == pytest.approx(partial_l2)
    assert eng_l2.serviced["mte2_aiv"] == pytest.approx(partial_l2)

    # Force workspace spill by shrinking effective L2 below partial_write_bytes.
    mech = geom_l2.physical_mechanisms
    tiny_hierarchy = replace(mech.memory_hierarchy, l2_capacity_bytes=1024)
    geom_spill = replace(
        geom_l2,
        buffers={**dict(geom_l2.buffers), "l2": 1024},
        physical_mechanisms=replace(mech, memory_hierarchy=tiny_hierarchy),
    )
    plan_spill = build_router_splitk(op, geom_spill)
    partial_spill = float(plan_spill.metadata["partial_write_bytes"])
    assert plan_spill.metadata["workspace_residency"] == "hbm"
    assert plan_spill.work.resources["hbm"] == pytest.approx(op.hbm_bytes + 2.0 * partial_spill)
    assert plan_spill.work.resources.get("l2", 0.0) == 0.0
    assert plan_spill.work.resources["fixpipe"] == pytest.approx(partial_spill)
    assert plan_spill.work.resources["mte2_aiv"] == pytest.approx(partial_spill)

    eng_spill = _exec_plan(op, geom_spill, plan_spill)
    assert eng_spill.serviced["hbm"] == pytest.approx(op.hbm_bytes + 2.0 * partial_spill)
    assert eng_spill.serviced.get("l2", 0.0) == 0.0
    assert eng_spill.serviced["fixpipe"] == pytest.approx(partial_spill)
    assert eng_spill.serviced["mte2_aiv"] == pytest.approx(partial_spill)

    # Full-grid router (32 tiles on 32 AICs) must NOT allocate Split-K workspace or partials.
    full_layer = SimpleNamespace(
        router_dtype="", router_output_dtype="fp32",
        num_experts=256, router_input_size=6144, index=0,
    )
    _, full_op, _, _ = router_gemm(
        prefix="route", plan=full_layer, deps=(), stage=0, requests=(),
        tokens=4096, model=model, hidden=6144, mode="prefill",
    )
    full_plan = build_router_splitk(full_op, geom_l2)
    assert full_plan.metadata.get("execution_abstraction") != "finite_split_k_workspace"
    assert full_plan.work.resources.get("l2", 0.0) == 0.0

    # Real split executes Cube work inside split_k K-slices before vector merge (never empty-work).
    ctx_l2 = RecipeContext(
        op, {}, {r: r for r in set(geom_l2.rates) | set(plan_l2.work.resources) | {"cube"}},
        {**{r: float(geom_l2.rates.get(r, 1e12)) / 1000.0 for r in set(geom_l2.rates) | set(plan_l2.work.resources)}, "cube": 1.0},
        geom_l2.compute_units,
        resource_units={r: int(geom_l2.resource_units.get(r, 1)) for r in set(geom_l2.rates) | set(plan_l2.work.resources) | {"cube"}},
        compute_peaks=dict(geom_l2.compute_peaks),
        pool_names={p: p for p in geom_l2.buffers}, pool_capacities=dict(geom_l2.buffers),
        pool_units=dict(geom_l2.buffer_units),
        physical_mechanisms=geom_l2.physical_mechanisms, datapath=geom_l2.datapath,
    )
    ctx_l2 = replace(ctx_l2, work=ctx_l2.bind(plan_l2.work))
    prog_l2 = plan_l2.program(ctx_l2)
    assert [s.name for s in prog_l2.stages] == ["setup", "splits", "split_sync", "merge"]
    assert eng_l2.serviced["cube"] == pytest.approx(ctx_l2.bind(plan_l2.work)["cube"])


def test_grouped_rows_l2_read_credit_conserved_across_matrix_recipes():
    from mindie_sim.catalog.ops import QuantSpec, gemm_op
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.operators.grouped_moe.grouped_rows import adapt as adapt_rows
    from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
    from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul
    from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul

    op = gemm_op(
        op_id="g", deps=(), stage=0, layer=0, requests=(), m=3, n=32, k=32,
        quant=QuantSpec("bf16"), category="routed_expert", phase="decode", weight_count=2,
        metadata=(("matrix_group_rows", (1, 2)),), family="grouped_gemm",
    )
    names = {r: r for r in ("hbm", "cube", "mte1", "mte2", "mte3", "vector")}
    work = dict(hbm=op.hbm_bytes - 1024, mte2=op.mte2_bytes, mte3=op.mte3_bytes, cube=op.flops)
    ctx = RecipeContext(op, work, names, dict.fromkeys(names, 1e6), 1,
                        compute_peaks={"bf16": 1e9}, native_work=work)
    for recipe in (
        TiledMatmul(template_execution=False),
        TiledMatmul(template_execution=True),
        HierarchicalMatmul(),
        MappedMatmul(),
        SteadyMatmul(),
    ):
        eng = Engine([Resource(r, 1e6) for r in names], backend="python")
        prog = adapt_rows(recipe, op).lower(ctx)
        eng.advance(until=(prog.start(eng, "g", eng.submit("r")),))
        assert eng.serviced["hbm"] == pytest.approx(work["hbm"])

    # Split-K grouped wrapper bills parent independent HBM once (setup only, never duplicated in children).
    op_sk = gemm_op(
        op_id="gsk", deps=(), stage=0, layer=0, requests=(), m=3, n=64, k=4096,
        quant=QuantSpec("bf16"), category="routed_expert", phase="decode", weight_count=2,
        metadata=(("matrix_group_rows", (1, 2)),), family="grouped_gemm",
    )
    extra_hbm = 192.0
    base_sk = dict(hbm=op_sk.hbm_bytes, mte2=op_sk.mte2_bytes, mte3=op_sk.mte3_bytes, cube=op_sk.flops)
    plus_sk = {**base_sk, "hbm": op_sk.hbm_bytes + extra_hbm}
    serviced_sk = []
    for wk in (base_sk, plus_sk):
        ctx_sk = RecipeContext(op_sk, wk, names, dict.fromkeys(names, 1e6), 4,
                               compute_peaks={"bf16": 1e9}, native_work=wk)
        eng_sk = Engine([Resource(r, 1e6, units=1 if r == "hbm" else 4) for r in names], backend="python")
        prog_sk = adapt_rows(HierarchicalMatmul(k_splits=2), op_sk).lower(ctx_sk)
        eng_sk.advance(until=(prog_sk.start(eng_sk, "gsk", eng_sk.submit("r")),))
        serviced_sk.append(eng_sk.serviced["hbm"])
    assert serviced_sk[1] - serviced_sk[0] == pytest.approx(extra_hbm)


def test_expected_grouped_scales_stream_refill_and_deterministic_topk_rows():
    from mindie_sim.catalog.ops import QuantSpec, gemm_op
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.operator_analysis import target_operators
    from mindie_sim.prediction.operators.grouped_moe.expected_grouped import adapt as adapt_exp
    from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
    from mindie_sim.prediction.structure import parse_case

    names = {r: r for r in ("hbm", "cube", "mte1", "mte2", "mte3", "vector")}
    meta = dict(routing_basis="probability_moments", matrix_mapping_domain="aggregate_experts",
                expected_rows=2.0, expected_active_experts=1.5, matrix_weight_bank_groups=2)
    op = gemm_op(
        op_id="g", deps=(), stage=0, layer=0, requests=(), m=2, n=16, k=16, family="grouped_gemm",
        quant=QuantSpec("bf16"), category="routed_expert", phase="prefill",
        weight_bytes_override=1.5 * 16 * 16 * 2, metadata=tuple(meta.items()),
    )
    work = dict(hbm=op.hbm_bytes, mte2=op.mte2_bytes, mte3=op.mte3_bytes, cube=op.flops)
    ctx = RecipeContext(op, work, names, dict.fromkeys(names, 1e6), 1,
                        compute_peaks={"bf16": 1000.0}, native_work=work)
    eng = Engine([Resource(r, 1e6) for r in names], backend="python")
    eng.advance(until=(adapt_exp(HierarchicalMatmul(), op).lower(ctx).start(eng, "g", eng.submit("r")),))
    assert eng.serviced["mte3"] == pytest.approx(op.mte3_bytes)
    assert eng.serviced["hbm"] == pytest.approx(op.hbm_bytes)

    model = dict(
        name="two", layers=1, hidden_size=32, num_heads=2, num_kv_heads=2, head_dim=16,
        intermediate_size=64, vocab_size=128, ffn="moe", num_experts=2, top_k=2, moe_intermediate_size=32,
    )
    case = parse_case(dict(
        machine=dict(backend="ascend", sku="Ascend910B1", num_devices=1), model=model,
        workload=dict(requests=[dict(prompt_tokens=16, output_tokens=1)]), runtime={},
    ))
    gate_op = next(o for o in target_operators(case, phase="prefill", tokens=16) if o.recipe == "grouped_gate_up")
    gate_work = dict(hbm=gate_op.hbm_bytes, mte2=gate_op.mte2_bytes, mte3=gate_op.mte3_bytes, cube=gate_op.flops)
    records = []
    gate_ctx = RecipeContext(gate_op, gate_work, names, dict.fromkeys(names, 1e6), 1,
                             compute_peaks={"bf16": 1e9}, record_mapping=records.append)
    adapt_exp(TiledMatmul(), gate_op).lower(gate_ctx)
    rec = next(r for r in records if "expected_active_experts" in r)
    assert rec["padded_m"] == 32

    # Fractional bank-count interpolation caps the 1-bank branch at routing_valid_tokens (never 17 > 16).
    p_b = 1.0 - (15.0 / 16.0) ** 16
    meta_cap = dict(
        routing_basis="probability_moments", matrix_mapping_domain="aggregate_experts",
        expected_rows=17.0, expected_active_experts=1.0 + p_b,
        matrix_weight_bank_groups=2, routing_valid_tokens=16,
    )
    op_cap = gemm_op(
        op_id="gcap", deps=(), stage=0, layer=0, requests=(), m=17, n=16, k=16, family="grouped_gemm",
        quant=QuantSpec("bf16"), category="routed_expert", phase="prefill",
        weight_bytes_override=(1.0 + p_b) * 16 * 16 * 2, metadata=tuple(meta_cap.items()),
    )
    w_cap = dict(hbm=op_cap.hbm_bytes, mte2=op_cap.mte2_bytes, mte3=op_cap.mte3_bytes, cube=op_cap.flops)
    rec_cap = []
    ctx_cap = RecipeContext(op_cap, w_cap, names, dict.fromkeys(names, 1e6), 1,
                            compute_peaks={"bf16": 1e9}, record_mapping=rec_cap.append)
    adapt_exp(TiledMatmul(), op_cap).lower(ctx_cap)
    entry_cap = next(r for r in rec_cap if "expected_active_experts" in r)
    assert entry_cap["padded_m"] == pytest.approx(16.0 * (1.0 + p_b))


def test_grouped_moe_contract_abi_scales_msd_and_hifloat8():
    from mindie_sim.catalog.ops import gemm_op
    from mindie_sim.catalog.quant import quant_preset
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.mechanism import Geometry, Work
    from mindie_sim.prediction.operators.grouped_moe import build as build_gmoe
    from mindie_sim.prediction.operators.grouped_moe.call_grouped_source import build_quantized_grouped_source_call
    from mindie_sim.prediction.operators.mc2 import _rebind_fused_mc2_op
    from mindie_sim.prediction.operators.mc2.megamoe import make_op as make_mc2_op
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.hardware import Hardware
    from mindie_sim.schema.operators import OpSpec, TensorPrecisionABI

    geo_a2 = hardware("Ascend910B1")
    geo_a3 = Geometry.from_devices(Hardware.from_mapping(dict(backend="ascend", sku="Ascend910_9391", num_devices=2)).dies[:1])
    geo_a5 = hardware("Ascend950DT_9581")

    call = build_quantized_grouped_source_call(dict(
        expert_rows=(1, 0), n=32, k=32, tiling=dict(base_m=16, base_n=32, base_k=32, used_cores=1),
    ))
    src_op, = call.ops
    _, src_plan = call.recipes.prepare(src_op, geo_a5)
    wave_i8 = next(x for x in geo_a5.datapath.subtile_wave if x.native_mode == "int8")
    expected_flops = 2 * wave_i8.tile_m * 32 * wave_i8.tile_k
    assert int(src_plan.work.matrix["int8"]) == expected_flops
    roles = ("hbm", "cube", "mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe", "vector")
    rates = {r: (1.0 if r == "cube" else 1e9) for r in roles}
    units = {r: int(geo_a5.resource_units.get(r, 1)) for r in roles}
    src_ctx = RecipeContext(
        src_op, {}, {r: r for r in roles}, rates, geo_a5.compute_units,
        resource_units=units, compute_peaks={"int8": 1e6},
        pool_names={p: p for p in geo_a5.buffers}, pool_capacities=dict(geo_a5.buffers),
        pool_units=dict(geo_a5.buffer_units), datapath=geo_a5.datapath,
    )
    eng = Engine([Resource(r, rates[r], units=units[r]) for r in roles],
                 pools=dict(geo_a5.buffers), pool_units=dict(geo_a5.buffer_units), backend="python")
    eng.advance(until=(src_plan.program(src_ctx).start(eng, "src", eng.submit("r")),))
    assert eng.serviced["cube"] == pytest.approx(src_ctx.bind(src_plan.work)["cube"])
    assert eng.serviced["cube"] == pytest.approx(expected_flops * 1000.0 / 1e6)

    # Uneven tail tile across 2 AICs (16 rows + 1 row) gives each owner its equal 32,768 issued FLOPs.
    call2 = build_quantized_grouped_source_call(dict(
        expert_rows=(17,), n=32, k=32, tiling=dict(base_m=16, base_n=32, base_k=32, used_cores=2),
    ))
    src2_op, = call2.ops
    _, src2_plan = call2.recipes.prepare(src2_op, geo_a5)
    assert int(src2_plan.work.matrix["int8"]) == 2 * expected_flops
    src2_ctx = replace(src_ctx, op=src2_op)
    eng2 = Engine([Resource(r, rates[r], units=units[r]) for r in roles],
                  pools=dict(geo_a5.buffers), pool_units=dict(geo_a5.buffer_units), backend="python", trace_limit=200)
    eng2.advance(until=(src2_plan.program(src2_ctx).start(eng2, "src2", eng2.submit("r")),))
    cube_by_owner = {}
    for ev in eng2.trace:
        if ev["task"].endswith(":matrix"):
            owner_key = ev["task"].split(":cores:program:")[1].split(":")[0]
            cube_by_owner[owner_key] = cube_by_owner.get(owner_key, 0.0) + (ev["end_ms"] - ev["start_ms"])
    assert len(cube_by_owner) == 2
    per_aic_ms = expected_flops * 1000.0 / 1e6 * geo_a5.compute_units
    assert all(dur == pytest.approx(per_aic_ms) for dur in cube_by_owner.values())
    assert eng2.serviced["cube"] == pytest.approx(src2_ctx.bind(src2_plan.work)["cube"])

    msd_op = make_mc2_op(op_id="moe", tokens=2, hidden=128, moe_inter=64, top_k=1, ep=2,
                         routed_rows=2, w13_bytes=8192, w2_bytes=4096, compute_dtype="w4a8")
    msd_op, = _rebind_fused_mc2_op([msd_op], 2, "int4", experts=2, ep=2)
    msd_plan = build_gmoe(msd_op, geo_a3)
    for prefix, expected_bytes in (("gmm1", 2048), ("gmm2", 1024)):
        pub = sum(s.work.resources.get("fixpipe", 0) for s in msd_plan.program.demands if s.name.startswith(prefix + ":"))
        con = sum(s.work.resources.get("mte2_aiv", 0) for s in msd_plan.program.demands if s.name.startswith(prefix + "_msd_recombine:"))
        assert pub == con == expected_bytes

    for routed_rows, expected_cube_max in ((1.25, 786432), (1.75, 786432)):
        frac_op = make_mc2_op(op_id="moe", tokens=2, hidden=128, moe_inter=64, top_k=1,
                              routed_rows=routed_rows, w13_bytes=128 * 128, w2_bytes=64 * 128, ep=2, compute_dtype="int8")
        frac_op, = _rebind_fused_mc2_op([frac_op], 2, "int8", experts=2, ep=2)
        cube_total = sum(build_gmoe(frac_op, geo_a3).work.matrix.values())
        assert 0 < cube_total <= expected_cube_max

    abi_i32 = TensorPrecisionABI("w8a8", "int8", "int8", "int32", "int8", "int32")
    s_i32 = (8, 128, 64)
    op_i32 = OpSpec("gmm", "grouped_gemm", "decode", 0, 0, (), s_i32, s_i32, s_i32, s_i32,
                    recipe="grouped_gemm", dtype="int8", precision_abi=abi_i32, metadata=(("group_rows", (8,)),))
    assert build_gmoe(op_i32, geo_a3).work.resources["fixpipe"] == 8 * 128 * 4

    s_gl = (8, 64, 64)
    for kind, groups in ((0, (2, 3)), (1, (2, 1)), (2, ((0, 2), (1, 1)))):
        op_gl = OpSpec("g", "grouped_gemm", "decode", 0, 0, (), s_gl, s_gl, s_gl, s_gl,
                       dtype="bf16", recipe="grouped_gemm", metadata=(("group_list_type", kind), ("group_list", groups)))
        assert build_gmoe(op_gl, geo_a5).work.resources["fixpipe"] == 384
    bad_sparse = OpSpec("g", "grouped_gemm", "decode", 0, 0, (), s_gl, s_gl, s_gl, s_gl, dtype="bf16",
                        recipe="grouped_gemm", metadata=(("group_list_type", 2), ("group_list", ((0, 2), (3, 1))), ("num_experts", 2)))
    with pytest.raises(ModelingError, match="sparse_group_index_out_of_range"):
        build_gmoe(bad_sparse, geo_a5)

    s_act = (1, 512)
    for geo in (geo_a2, geo_a5):
        for scaled in (False, True):
            meta_act = dict(quant_mode=1, output_dtype="int8", has_quant_scale=scaled, quant_scale_dtype="fp32")
            op_act = OpSpec("a", "grouped_moe", "decode", 0, 0, (), s_act, s_act, s_act, s_act,
                            dtype="bf16", recipe="dequant_swiglu_quant", metadata=tuple(meta_act.items()))
            assert build_gmoe(op_act, geo).work.resources["mte2_aiv"] == 1024 * (1 + scaled)
    meta_static = dict(quant_mode=0, output_dtype="int8", has_quant_scale=True, quant_scale_dtype="fp32",
                       has_quant_offset=True, quant_offset_dtype="fp32")
    op_static = OpSpec("as", "grouped_moe", "decode", 0, 0, (), s_act, s_act, s_act, s_act,
                       dtype="bf16", recipe="dequant_swiglu_quant", metadata=tuple(meta_static.items()))
    assert build_gmoe(op_static, geo_a3).work.resources["mte2_aiv"] == 1024 + 8

    s_hif = (16, 512, 256)
    for dt in ("hifloat8", "hif8"):
        md_hif = dict(group_rows=(16,), quant_mode=0, dequant_mode=0, output_dtype=dt, quant_dtype=dt,
                      weight_scale_dtype="fp32", requested_abi="aclnnGroupedMatmulSwigluQuantV2")
        op_hif = OpSpec("g", "grouped_moe", "prefill", 0, 0, (), s_hif, s_hif, s_hif, s_hif,
                        dtype=dt, recipe="gmm_swiglu_quant", metadata=tuple(md_hif.items()))
        ctx_hif = RecipeContext(op_hif, {}, {"cube": "cube"}, {}, 1, compute_peaks=geo_a5.compute_peaks)
        assert ctx_hif.bind(Work(build_gmoe(op_hif, geo_a5).work.matrix))["cube"] > 0

    w8_op = gemm_op(
        op_id="g", deps=(), stage=0, layer=0, requests=(), m=16, n=128, k=128,
        quant=quant_preset("w8a8_int8"), category="routed_expert", phase="prefill",
        recipe="grouped_gemm", family="grouped_gemm", weight_count=2, metadata=(("group_rows", (8, 8)),),
    )
    _, w8_plan = RecipeBook().prepare(w8_op, geo_a5)
    mte2_total = sum(v for k, v in w8_plan.work.resources.items() if k.startswith("mte2"))
    assert mte2_total == int(w8_op.mte2_bytes) == 16 * 128 + 2 * 128 * 128 + 16 * 4 + 2 * 128 * 2
    assert w8_plan.work.resources["vector"] == pytest.approx(w8_op.vector_ops)

    # Block-scaled FP8 grouped GEMM uses 2D activation (16x2) and block weight (1x2) scale tensors (136 B).
    abi_fp8 = TensorPrecisionABI(
        "fp8", "fp8_e4m3", "fp8_e4m3", "bf16", "fp8_e4m3", "fp32",
        weight_layout="fp8_block", weight_block=(128, 128), weight_scale_dtype="fp32",
        act_group_size=128, act_scale_dtype="fp32", act_scale_pack="block",
    )
    s_fp8 = (16, 128, 256)
    op_fp8 = OpSpec("gf8", "grouped_gemm", "prefill", 0, 0, (), s_fp8, s_fp8, s_fp8, s_fp8,
                    recipe="grouped_gemm", dtype="fp8_e4m3", precision_abi=abi_fp8,
                    metadata=(("group_rows", (16,)),))
    assert build_gmoe(op_fp8, geo_a5).work.resources["mte2_aiv"] == (16 * 2 + 1 * 2) * 4
