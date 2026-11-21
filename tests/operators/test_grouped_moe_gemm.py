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
from mindie_sim.prediction.rewrite_ffn_rules import router_gemm
from mindie_sim.prediction.engine import Engine, Resource
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
        assert resources["hbm"] == resources["l2"] == resources["mte2_aic"] + gm + scale_bytes + resources["mte3"]
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
                 requested_abi=MEGA_ABI, output_dtype="bf16", soc_tier="a2")
    if gap:
        with pytest.raises(UnmodeledCapabilityError, match=gap):
            resolve(op, geometry)
        return
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
    ("int8", "int8", "MegaMoeA8W8Wave"), ("mxfp8", "mxfp4", "MegaMoeA8W4Wave"),
    ("mxfp4", "mxfp4", "MegaMoeA4W4Wave"), ("bf16", "bf16", "MegaMoeA16W16Wave"),
])
@pytest.mark.parametrize("shared", [0, 32])
def test_mega_moe_wave_partitions_and_shared_expert(geometry, activation, weight, specialization, shared):
    counts, width, k = (3, 0, 6), 64, 32
    shared_compute, shared_weight = ("bf16", "bf16") if activation == "int8" else (activation, weight)
    op = make_op(counts, 2 * width, k, mode="mega_moe", requested_abi=MEGA_ABI,
                 dtype=activation, dtype_b=weight, output_dtype="bf16",
                 wave_rows=4, enable_fused_mc2=3 if shared else 2, shared_intermediate=shared,
                 shared_compute_dtype=shared_compute, shared_weight_dtype=shared_weight)
    resolved = resolve(op, geometry)
    plan = resolved.recipe
    stages = {stage.name: stage for stage in check_ledger(resolved, bind(op, geometry, plan.work))}
    assert plan.metadata["wave_specialization"] == specialization
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
                 enable_fused_mc2=3, shared_intermediate=32)
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
    operands = dict(mode="mega_moe", dtype="int8", output_dtype="bf16", requested_abi=MEGA_ABI)
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

    op = make_op(rows, 64, 32, mode="mega_moe", dtype="int8",
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
    op = make_op((8,), 128, 32, mode="mega_moe", dtype="int8", output_dtype="bf16",
                 enable_fused_mc2=3, shared_intermediate=32,
                 shared_compute_dtype=shared_dtype, shared_weight_dtype=shared_weight)
    if shared_dtype != "bf16":
        with pytest.raises(ValueError, match=f"unsupported_shared_expert_precision:{shared_dtype}:{shared_weight}"):
            resolve(op, geometry)
        return
    resolved = resolve(op, geometry)
    routed = resolve(make_op((8,), 128, 32, mode="mega_moe", dtype="int8", output_dtype="bf16"), geometry)
    phases = resolved.descriptor.phases
    shared_mmad = [item for phase in phases if phase.phase_id.startswith("shared_")
                   for item in phase.work if item.resource == "cube"]
    assert len(shared_mmad) == 2 and all(item.dtype == shared_dtype for item in shared_mmad)
    assert resolved.recipe.work.matrix == {
        "int8": routed.recipe.work.matrix["int8"], shared_dtype: 2 * 16 * 32 * (64 + 32),
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


@pytest.mark.parametrize("m,n,k", [(1, 16, 8), (528, 288, 4096), (240, 384, 7168)])
def test_router_split_k_preserves_matrix_and_workspace_service(m, n, k):
    layer = SimpleNamespace(router_dtype="", router_output_dtype="fp32",
                            num_experts=n, router_input_size=k, index=0)
    model = SimpleNamespace(quant=QuantSpec("mx", weight_format="mxfp8"), activation_dtype="bf16")
    _, op, _, _ = router_gemm(prefix="route", plan=layer, deps=(), stage=0, requests=(),
                              tokens=m, model=model, hidden=k, mode="decode")
    _, plan = RecipeBook().prepare(op, hardware(REPRESENTATIVE_SKUS["a5"]))
    tiles = ceil(m / ROUTER_TILE_M) * ceil(n / ROUTER_TILE_N)
    partial = m * n * 4 * plan.metadata["split_k"]
    assert plan.metadata["output_tiles"] == tiles
    assert 1 < plan.metadata["split_k"] <= k
    assert plan.metadata["partial_write_bytes"] == partial
    assert plan.work.matrix["bf16"] == 2 * m * n * k
    assert plan.work.resources["hbm"] == pytest.approx(op.hbm_bytes + 2 * partial)


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
