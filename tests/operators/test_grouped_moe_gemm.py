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
from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction import bf16_grouped, family_shared_quant, grouped_dequant_arch35, mxfp4_grouped
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.call_builders import build_call
from mindie_sim.prediction.family_grouped_quant import RECIPE as DEQUANT_RECIPE, lower as lower_int32
from mindie_sim.prediction.grouped_rows import row_classes
from mindie_sim.prediction.matrix_splitk import build_router_splitk, router_output_tiles
from mindie_sim.prediction.moe_cann_fusion import _fused_gating, build_moe_gating_topk
from mindie_sim.prediction.mxfp8_grouped import build_swiglu_mx_quant
from mindie_sim.prediction.passes_ffn_fusion import activation_quant_fusion, router_gemm
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.operators import grouped_gemm, grouped_moe
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.quant_grouped_schedule import QuantGroupedTiling, plan_quant_grouped
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
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
                 layout_b="ND" if on_chip else "NZ", handoff_dtype="int32", output_dtype="bf16")
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
                 requested_abi=MEGA_ABI, output_dtype="bf16")
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
    op = make_op(counts, 2 * width, k, mode="mega_moe", requested_abi=MEGA_ABI,
                 dtype=activation, dtype_b=weight, output_dtype="bf16",
                 wave_rows=4, enable_fused_mc2=3 if shared else 2, shared_intermediate=shared)
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
    deps = stages["combine:local"].after
    assert deps == (("gmm2", "shared_gmm2") if shared else ("gmm2",))
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
def test_descriptor_roundtrip_and_causal_replay_match_native(geometry):
    on_chip = geometry.datapath.has_l0c_to_ub
    dtype = "mxfp8" if on_chip else "int8"
    op = make_op((3, 0, 1), 64, 32, mode="gmm_swiglu_quant", dtype=dtype,
                 absorbed_source_op_ids=("activation",), output_dtype="bf16")
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
    for backend, numeric_only in (("python", False), ("c", False), ("c", True)):
        resources = [Resource(role, 128, units=geometry.resource_units.get(role, 1))
                     for role in context.resource_names]
        engine = Engine(resources, pools={"ub": geometry.datapath.usable_bytes("ub")},
                        pool_units=context.pool_units, backend=backend, trace_limit=len(stages) + 8)
        ready = engine.submit("ready", release=2)
        # Reproduce identical work and dependencies in a numeric C11 frontier.
        engine.advance(until=(ready,))
        replay = replace(program, stages=tuple(replace(s, acquire=(), free=()) for s in stages)) if numeric_only else program
        done = replay.start(engine, op.op_id, ready)
        engine.advance(until=(done,))
        assert engine.serviced == pytest.approx(context.bind(plan.work), rel=1e-12)
        summary = engine.summary()
        assert summary["live_tasks"] == summary["live_programs"] == 0
        assert not any(summary["live_allocated"].values())
        if numeric_only:
            assert engine.accel_backend == "c_native", engine.accel_status()
        assert all(row["start_ms"] >= ready.time for row in engine.trace if row["demand"])
        durations.append(done.time)
    assert durations[1:] == pytest.approx(durations[:1] * 2, rel=1e-12)


def read_case(name):
    return json.loads((ROOT / "examples" / name).read_text())


@pytest.mark.parametrize("hardware_sku", SKUS)
@pytest.mark.parametrize("rows,n,k", [((0, 4, 0), 128, 64), ((84, 0, 136), 272, 193)])
def test_bf16_source_compaction_and_tile_bytes(hardware_sku, rows, n, k):
    op = grouped_gemm.make_op(dict(expert_rows=rows, n=n, k=k, dtype="bf16",
                                  implementation="ascendc_grouped_nd_bf16", tiling_key=10))
    geom = hardware(hardware_sku)
    classified = replace(op, metadata=(("matrix_group_rows", rows),))
    assert row_classes(classified) == tuple((m, rows.count(m)) for m in sorted(set(rows) - {0}))
    if geom.datapath.has_l0c_to_ub:
        op = classified
        plan = bf16_grouped.build(op, geom)
        reads = 2 * k * sum(m * ceil(n / 256) + n * ceil(m / 256) for m in rows)
        writes = 2 * sum(rows) * n
        assert plan.work.matrix["bf16"] == 2 * sum(rows) * n * k
        assert plan.work.resources["mte2_aic"] == plan.work.resources["mte1"] == reads
        assert plan.work.resources["fixpipe"] == writes
        assert plan.work.resources["hbm"] == reads + writes + geom.compute_units * 8 * len(rows)
        assert bf16_grouped.build(op, geom, compact=False).work == plan.work
        return
    plan = grouped_gemm.build(op, geom)
    assert grouped_gemm.compact_histogram(rows) == tuple((i, m) for i, m in enumerate(rows) if m)
    raw = grouped_gemm.interpret_raw_group_list(tuple(enumerate(rows)), n=n, k=k)
    assert raw["processed_rows"] == sum(rows[:rows.index(0)])
    reads = 2 * k * sum(m * ceil(n / 256) + n * ceil(m / 128) for m in rows)
    writes = 2 * sum(rows) * n
    assert plan.work == Work({"bf16": 2 * sum(rows) * n * k},
                             {"hbm": reads + writes, "mte2": reads, "mte3": writes})
    tiles = list(grouped_gemm.iter_tiles(op, geom.compute_units))
    for idx, tile in enumerate(tiles):
        assert tile["core"] == idx % geom.compute_units
        assert tile["expert_id"] in {i for i, m in enumerate(rows) if m}
    assert sum(tile["tile_rows"] * tile["tile_cols"] for tile in tiles) == sum(rows) * n


@pytest.mark.parametrize("rows,expected_count", [((64,), 8), ((64, 0), 4), ((17, 0, 33, 5), 8)])
def test_quantized_tiles_keep_bank_offsets_and_cover_valid_rows(rows, expected_count):
    n, k = 64, 32
    tiling = QuantGroupedTiling(32, 32, 32, 8)
    schedule = plan_quant_grouped(rows, n, k, tiling)
    tiles = [tile for owner in range(tiling.used_cores) for tile in schedule.iter_owner_tiles(owner)]
    assert len(tiles) == expected_count
    for bank, m in enumerate(rows):
        cells = Counter()
        for tile in (tile for tile in tiles if tile.group == bank):
            offset = sum(rows[:bank]) + tile.m_offset
            assert tile.x_offset == offset * k
            assert tile.output_offset == (offset * n + tile.n_offset) * 2
            assert tile.row_scale_offset == offset * 4
            assert tile.column_scale_offset == (bank * n + tile.n_offset) * 4
            assert list(schedule.iter_weight_spans(tile)) == [
                (bank * n * k + tile.n_offset * k, tile.n * k)]
            cells.update((i, j) for i in range(tile.m_offset, tile.m_offset + tile.m)
                         for j in range(tile.n_offset, tile.n_offset + tile.n))
        assert cells == dict.fromkeys(((i, j) for i in range(m) for j in range(n)), 1)


@pytest.mark.parametrize("source", [False, True])
def test_grouped_call_conserves_payload_scales_and_output_publication(source):
    rows, n, k = (3, 0, 3, 0), 256, 64
    args = dict(expert_rows=rows, n=n, k=k)
    if source:
        args["tiling"] = {"base_m": 16, "base_n": 128, "base_k": 32, "used_cores": 32}
    name = "quantized_grouped_source" if source else "quantized_grouped"
    call = build_call({"builder": name, "arguments": args})
    (op,) = call.ops
    assert op.flops == 2 * sum(rows) * n * k
    assert op.plan_required is source
    if source:
        generic = RecipeBook.from_mapping({"programs": {"matmul": {"kind": "phase_matmul"}}})
        with pytest.raises(ModelingError, match="source_plan_binding_required"):
            generic.select(op)
    else:
        meta = dict(op.metadata)
        assert meta["active_expert_ids"] == (0, 2)
        assert meta["activation_scale_bytes"] == 4 * sum(rows)
        assert meta["weight_scale_bytes"] == 4 * 2 * n
        assert meta["weight_bank_residency_bytes"] == len(rows) * k * n
        assert meta["group_list_scan_bytes"] == 8 * len(rows)
        assert meta["output_bytes"] == 2 * sum(rows) * n
    report = call_costs(read_case("calls/quantized-grouped-small.json"), call.ops, recipes=call.recipes,
                        mode="graph", invocations=2, input_ready_ms=5)
    assert report["execution"]["work"]["flops"] == 2 * op.flops
    assert report["complete_ms"] > 5
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert not any(report["execution"]["live_allocated"].values())


@pytest.mark.parametrize("capacity,owners", [(512, (8,) * 64), (2048, (64,) * 8 + (0,) * 56)])
def test_dequant_capacity_changes_owner_assignment_without_changing_rows(capacity, owners):
    rows = (8,) * 64
    cfg = grouped_dequant_arch35.Config(rows, 512, capacity, 64, 262144)
    assignments = [grouped_dequant_arch35.assignments(cfg, owner) for owner in range(cfg.blocks)]
    assert tuple(sum(m for _, m in entries) for entries in assignments) == owners
    assert Counter({bank: sum(dict(entry).get(bank, 0) for entry in assignments) for bank in range(len(rows))}) == (
        dict(enumerate(rows)))
    dequant, reduction, quant = grouped_dequant_arch35.inventories(3, 65)
    assert dequant.counts["Cast32"].active_lanes == 3 * 65 * 2
    assert reduction.counts["ReduceMax"].active_lanes == 3 * 64
    assert quant.counts["Store"].write_bytes == 3 * 65


@pytest.mark.parametrize("batch", [1, 3])
def test_int32_caller_exchanges_activation_intermediate_for_scale_dependency(batch):
    raw = read_case("cases/high-concurrency-20261030/qwen-source-gdn-native/"
                    "qwen35-35b-a5-tp4-c128-graph-source-gdn-native.json")
    raw["model"]["layers"] = 1
    raw["runtime"]["kernels"] = {"lowerings": {}}
    before = target_graphs(raw, scope="body", phase="decode", tokens=4, sequences=batch, context=256)[0]
    binding = {"grouped_int8_caller": "ascend_grouped_int32_dequant_swiglu"}
    after = lower_int32(before, binding)
    consumer = next(op for op in after if op.recipe == DEQUANT_RECIPE)
    by_id = {op.op_id: op for op in after}
    gate, scale = (by_id[key] for key in consumer.deps)
    down = next(op for op in after if op.recipe == "balanced_grouped_down")
    rows, n, k = gate.local_shape
    histogram = dict(gate.metadata)["matrix_group_rows"]
    assert sum(histogram) == rows
    assert gate.mte2_bytes == rows * k + sum(m > 0 for m in histogram) * n * k
    assert gate.mte3_bytes == rows * n * 4 and gate.vector_ops == 0
    assert "activation_scales" in scale.returns and down.deps == (consumer.op_id,)
    assert {op.op_id for op in before} - by_id.keys() == {down.op_id + ".quantize"}
    origins = dict(consumer.metadata)["fused_origin_scopes"]
    assert {(origin["stage"], origin["layer"], origin["block"]) for origin in origins} == {
        (consumer.stage, consumer.layer, "ffn")}
    assert lower_int32(after, binding) == after
    reader = replace(gate, op_id="extra_reader", deps=(gate.op_id,), family="tensor_copy", recipe="copy")
    with pytest.raises(ModelingError, match="exclusive_activation_chain"):
        lower_int32((*before, reader), binding)


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


@pytest.mark.parametrize("scale,program", [("bf16", "JointNativeService"), ("fp32", "ExpectedGroupedMatmul")])
def test_column_scale_format_controls_service_domain(scale, program):
    raw = read_case("cases/longctx-decode-20261030/qwen35-a5-tp4-ep1-grouped-joint.json")
    raw["model"]["layers"] = 1
    for rule in raw["runtime"]["kernels"]["service_regimes"]:
        rule["match"].update({"metadata.weight_scale_format": "bf16", "metadata.weight_scale_bytes": 2})
    if scale == "fp32":
        raw["model"]["quant"]["category_overrides"]["routed_expert"].update(scale_format=scale, scale_bytes=4)
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4, context=126976)[0]
    op = next(op for op in ops if op.recipe == "grouped_gate_up")
    priced = operator_costs(raw, [op], input_ready_ms=7)["operators"][0]
    assert dict(op.metadata)["weight_scale_format"] == scale
    assert priced["program"] == program
    assert not priced["service_before_input"] and priced["live_tasks"] == 0


def test_mx_grouped_resplit_preserves_matrix_work():
    rows, n, k = (480,) * 12, 4608, 5120
    op = make_op(rows, n, k, dtype="fp8_e4m3", matrix_group_rows=rows, **mxfp4_grouped.SOURCE_INTERFACE)
    op = replace(op, family="grouped_gemm")
    plan = mxfp4_grouped.build(op, hardware(REPRESENTATIVE_SKUS["a5"]))
    detail = plan.metadata["mxa8w4"]
    assert detail["resplit_schedule"] and detail["k_panel_execution"] == "detailed_panels"
    assert sum(plan.work.matrix.values()) == 2 * sum(rows) * n * k


@pytest.mark.parametrize("m,n,k", [(1, 16, 8), (528, 288, 4096), (240, 384, 7168)])
def test_router_split_k_preserves_matrix_and_workspace_service(m, n, k):
    layer = SimpleNamespace(router_dtype="", router_output_dtype="fp32",
                            num_experts=n, router_input_size=k, index=0)
    model = SimpleNamespace(quant=QuantSpec("mx", weight_format="mxfp8"), activation_dtype="bf16")
    _, op, _, _ = router_gemm(prefix="route", plan=layer, deps=(), stage=0, requests=(),
                              tokens=m, model=model, hidden=k, mode="decode")
    plan = build_router_splitk(op, hardware(REPRESENTATIVE_SKUS["a5"]))
    tiles = router_output_tiles(m, n)
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


@pytest.mark.parametrize("rows,width", [(1, 64), (65, 192)])
def test_standalone_swiglu_mx_quant_conserves_io_and_arithmetic(rows, width):
    shaped = dict(stage=0, layer=0, requests=(), dtype="bf16", phase="decode", shape=(rows, width),
                  metadata=(("activation_group_size", 32), ("scale_format", "fp8_e8m0")))
    act = vector_op(op_id="silu", deps=(), elements=rows * width, ops_per_element=4,
                    family="swiglu", recipe="grouped_swiglu", read_bytes_override=4 * rows * width, **shaped)
    quant = vector_op(op_id="quant", deps=(act.op_id,), elements=rows * width, ops_per_element=4,
                      family="quantize", recipe="dynamic_quantize",
                      write_bytes_override=rows * (width + width // 32), **shaped)
    (op,) = activation_quant_fusion((act, quant), "bf16")
    plan = build_swiglu_mx_quant(op, hardware(REPRESENTATIVE_SKUS["a5"]))
    assert plan.work.resources["mte2"] == 4 * rows * width
    assert plan.work.resources["mte3"] == rows * (width + width // 32)
    assert plan.work.resources["vector"] == 8 * rows * width


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
                      kernels={"lowerings": {"shared_expert": family_shared_quant.BINDING}})
    ops = lower_implementations((gate, act, down), runtime, "bf16")
    fused = next(op for op in ops if op.recipe == family_shared_quant.RECIPE)
    by_id = {op.op_id: op for op in ops}
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
