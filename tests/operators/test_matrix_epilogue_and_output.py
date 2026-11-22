"""Matrix family geometry, scale ABI, and output dependencies.

[Role] Exercise source and analytical matrix programs from typed input to publication.
[Invariants] Preserve exact rows, FLOPs, scale bytes, and dependency boundaries.
[Timeouts & Locks] Local bounded simulations and retained capture reads, with no locks.
[Side-Effect Policy] Reads checked-in cases and companion profiler fixtures.
"""

from dataclasses import replace
from functools import lru_cache
from importlib.util import find_spec
import json
import math
from pathlib import Path

import pytest
from _support.captures import REPO_ROOT
from _support.matrix_cases import matrix_setup, quantized_context
from _support.moe_hc_mc2_fixtures import explicit_route_case

from mindie_sim import graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.operators.mx_quant_matrix import family_gate_up_situ_mxfp4 as gate_situ
from mindie_sim.prediction.operators.grouped_moe import family_grouped_gemm; from mindie_sim.prediction.operators.mx_quant_matrix import family_mx_matrix, shared_w4a8; from mindie_sim.prediction.operators.matrix import matrix_wrappers
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.operators.grouped_moe.gmmsq_quant_typed import work as gmmsq_work
from mindie_sim.prediction.implementations import Selection
from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
from mindie_sim.prediction.operators.matrix.matrix_epilogue import (
    MatrixEpilogue, plan_matrix_epilogue, reconcile_native_scale_storage, reconcile_scale_storage,
)
from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul, mapping_bound
from mindie_sim.prediction.operators.matrix.matrix_output import PartitionedMatrixOutput
from mindie_sim.prediction.operators.matrix.matrix_phase import PhaseMatmul
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.native_binding import bind_native_call, declare_native_call
from mindie_sim.prediction.operators.hyper_connections.native_fusion import _groups
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure import export_op_structure, export_structure
from mindie_sim.structure_contracts import comparison_scope, predicted_contract
from mindie_sim.structure.native import grouped_situ_contract

pytestmark = pytest.mark.tier0

CASES = REPO_ROOT / "examples/cases"
WRAPPER_CASE = "kimi-k3-a5-current-checkpoint-conditional.json"
TYPED_CASE = "high-concurrency-20261030/new9/qwen36-a5-tp2-c1.json"
PREPARED_CASE = ("high-concurrency-20261030/v41-mx32-conditional/"
                 "a5_v41_fp8_tp1_dp8_ep8_c128_p2k-prepared-mx32-source.json")
TYPED_DTYPES = ("int8", "int8", "bf16", "fp32")
CAPTURED_TOTAL, CAPTURED_COLUMN, CAPTURED_OUTPUT = 44_650_512, 10_240, 40_960
TYPED_MISMATCH = "typed_scale_representation_mismatch"
PRODUCER_CONFLICT = "producer_scale_conflict"
DRAM_SERVICE_FACTOR = 1 / 0.8316
FAST = 1e12
AIV_COUNT = 64
OUTPUT = PartitionedMatrixOutput("fixpipe", 2, 2, 8)

def _launch(engine, ctx, recipe, *, release=7, **options):
    return recipe.lower(ctx, **options).start(engine, "matrix", engine.submit("ready", release=release))
def _finish(engine, done, release=7):
    engine.advance(horizon=release)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not any(engine.allocated.values()) and engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
def _engine(ctx, trace=0):
    return Engine([Resource(name, rate, units=ctx.resource_units.get(name, 1)) for name, rate in ctx.rates.items()],
                  pools=ctx.pool_capacities, pool_units=ctx.pool_units, trace_limit=trace)
@pytest.mark.parametrize("tile_k", [2, 6])
@pytest.mark.parametrize("copy,traffic", [({"mte2": 64}, {"mte2": 220}),
                                          ({"mte2_aic": 64, "mte2_aiv": 32}, {"mte2_aic": 132, "mte2_aiv": 88})])
def test_row_and_column_scales_are_charged_only_at_output(tile_k, copy, traffic):
    _, ctx = quantized_context(rates=dict(cube=128, vector=32, hbm=1024, mte1=128, mte3=128, **copy))
    units = dict(ctx.resource_units, mte2_aiv=6) if "mte2_aiv" in copy else ctx.resource_units
    ctx = replace(ctx, resource_units=units, pool_units=dict.fromkeys(ctx.pool_capacities, 3))
    engine, recipe = _engine(ctx), HierarchicalMatmul(4, 4, tile_k, 2, 2)
    done = _launch(engine, ctx, recipe)
    _finish(engine, done)
    # Two [3,6] by [6,5] groups repeat A and row scales across the N tail on private pipes.
    assert engine.serviced == pytest.approx(dict(cube=360, vector=120, hbm=220, mte1=228, mte3=60, **traffic))
    assert mapping_bound(ctx, recipe) <= done.time - 7
@pytest.mark.parametrize("m,k,cube,vector,copy,span", [(2, 4, 32, 8, FAST, 3), (1, 1, FAST, FAST, 1, 10)])
def test_scale_follows_reduction_and_shares_the_operand_pipe(m, k, cube, vector, copy, span):
    rates = dict(cube=cube, vector=vector, mte2=copy, hbm=FAST, mte1=FAST, mte3=FAST)
    engine, ctx = quantized_context(m=m, n=m, k=k, groups=1, cores=1, rates=rates)
    recipe = HierarchicalMatmul(m, m, m, m, m)
    done = _launch(engine, ctx, recipe, release=0)
    if m == 2:
        for time, pipes in ((0.99, ("vector", "mte3")), (2.9, ("mte3",))):
            engine.advance(horizon=time)
            assert all(engine.serviced[pipe] == 0 for pipe in pipes)
    else:
        assert mapping_bound(ctx, recipe) == 2 + 2 * 4
    engine.advance(until=(done,))
    assert engine.serviced["hbm"] == 2 * m * k + 8 * m + 2 * m * m
    assert done.time == pytest.approx(span, abs=1e-8)
@pytest.mark.parametrize("recipe", [TiledMatmul(4, 4, 3), HierarchicalMatmul(4, 4, 3, 2, 2),
    MappedMatmul(((4, 4, 3),), 2, 2), PhaseMatmul(((4, 4, 3),), 2, 2), SteadyMatmul(((4, 4, 3),), 2, 2)])
def test_integer_output_casts_after_complete_reduction(recipe):
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(), m=2, n=3, k=7,
                 quant=QuantSpec("dynamic", weight_format="int8"), category="dense_mlp", phase="prefill")
    rates = dict(cube=100, vector=8, hbm=1000, mte1=1000, mte2=1000, mte3=1000)
    ctx = RecipeContext(op, {"cube": op.flops, "vector": op.vector_ops}, {n: n for n in rates}, rates, 1)
    engine = _engine(ctx, 1000)
    done = _launch(engine, ctx, recipe)
    _finish(engine, done)
    assert (engine.serviced["cube"], engine.serviced["vector"]) == (2*2*3*7, 4*2*3)
    assert engine.serviced["hbm"] == pytest.approx(op.hbm_bytes)
    vector = min(row["start_ms"] for row in engine.trace if row["demand"].get("vector"))
    cube = max(row["end_ms"] for row in engine.trace if row["demand"].get("cube"))
    assert vector >= cube - 1e-10 and done.time >= 7 + 24 / 8
def test_partial_quantization_group_and_explicit_narrow_tiles():
    tail = gemm_op(op_id="tail", deps=(), stage=0, layer=0, requests=(), m=2, n=3, k=64, category="dense_mlp", phase="decode",
                   quant=QuantSpec("dynamic", weight_format="int8", activation_group_size=128))
    assert dict(tail.metadata)["activation_scale_shape"] == (2,) and tail.vector_ops == 4 * 2 * 3
    _, ctx = matrix_setup(32, 24, 256, cores=16, pools=dict(l1=512*1024, l0a=64*1024, l0b=64*1024, l0c=256*1024))
    ctx = replace(ctx, rates=dict(dict.fromkeys(ctx.rates, FAST), cube=1))
    narrow, fine = MappedMatmul(narrow_candidates=((16, 16, 64),)).select(ctx)
    _, coarse = MappedMatmul(narrow_candidates=(), underfilled_candidates=()).select(ctx)
    assert narrow.tile_m == narrow.tile_n == narrow.compute_m == narrow.compute_n == 16
    bounds = [min(r["resource_bound_ms"] for r in entry["candidates"] if r["resource_bound_ms"] is not None)
              for entry in (fine, coarse)]
    assert bounds[0] < bounds[1]
    assert MappedMatmul(((32, 32, 64),), 16, 16).select(ctx)[1]["candidates"][0]["tile"] == [32, 32, 64]
@pytest.mark.parametrize("m,n,k,groups,rectangles,expected", [
    (3, 5, 4, 1, (), {"hbm": 94, "mte1": 76, "mte2_aic": 32, "mte2_aiv": 52, "fixpipe": 80}),
    (1, 8, 2, 1, (), {"fixpipe": 64}),
    (6, 5, 4, 2, ((3, 2), (3, 3), (3, 2), (3, 3)), {"hbm": 188, "mte2_aic": 88, "fixpipe": 160}),
])
def test_partitioned_output_handoff_and_serial_acknowledgement(m, n, k, groups, rectangles, expected):
    rates = dict(hbm=1024, cube=128, mte1=128, mte2_aic=64, mte2_aiv=64, vector=32, mte3=128, fixpipe=64)
    if m == 1:
        rates = dict(dict.fromkeys(rates, FAST), vector=8, fixpipe=32)
    _, ctx = quantized_context(m=m, n=n, k=k, groups=groups, cores=1, rates=rates)
    pools = dict(ctx.pool_capacities, ub=4096)
    ctx = replace(ctx, pool_capacities=pools, pool_units=dict(dict.fromkeys(pools, 1), ub=2),
                  pool_names={name: name for name in pools},
                  resource_units=dict(ctx.resource_units, vector=2, mte2_aiv=2, mte3=2))
    engine = _engine(ctx, 2000)
    tile = HierarchicalMatmul(2 if m == 1 else 4, 4 if m == 1 or rectangles else 8, 2, 2, 2)
    options = {"owner_tiles": ((4, iter(rectangles)),)} if rectangles else {}
    done = _launch(engine, ctx, tile, output=OUTPUT, serial_tiles=True, **options)
    _finish(engine, done)
    expected = dict(expected, cube=2 * m * n * k, vector=4 * m * n, mte3=2 * m * n)
    assert {name: engine.serviced[name] for name in expected} == pytest.approx(expected)
    if m == 1:
        assert done.time - 7 == pytest.approx(2 * (32 / 32 + 16 / (8 / 2)), abs=1e-8)
        assert {row["placement"]["vector"] for row in engine.trace if row["demand"].get("vector")} == {(0,)}
    elif not rectangles:
        assert [engine.peak_unit_allocated["ub", owner] for owner in (0, 1)] == [160, 160]

def _typed_case(enabled=False):
    raw = json.loads((CASES / TYPED_CASE).read_text())
    raw["model"]["layers"] = 1
    if enabled:
        raw["model"]["quant"].update(runtime_weight_scale_format="bf16", runtime_weight_scale_bytes=2,
                                     runtime_activation_scale_format="fp32", runtime_int8_weight_metadata="scale_only")
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4, sequences=1, context=4096)[0]
    matrix = next(op for op in ops if op.op_id == "s0.l0.down")
    return raw, matrix, next((op for op in ops if op.op_id in matrix.deps), None)
def _scale_record(op, shape, *, offset=False):
    m, n, k = shape
    fmt, width = ("fp32", 4) if offset else ("bf16", 2)
    row, column, output = 4 * m, n * width * (2 if offset else 1), 2 * m * n
    packed, reads = n * k + column, (m + n) * k + row + column
    meta = dict(op.metadata, packed_weight_bytes=packed, weight_bytes=packed,
                weight_scale_format=fmt, weight_scale_bytes=float(width), activation_scale_format="fp32",
                activation_scale_bytes=row, int8_weight_metadata="scale_offset" if offset else "scale_only",
                weight_scale_storage_format=fmt, weight_scale_storage_bytes=float(width),
                matrix_epilogue=dict(kind="row_column_scale", row_scale_bytes=row, column_scale_bytes=column, operations_per_output=4))
    if not offset:
        meta.update(activation_scale_shape=(m,), weight_scale_shape=(n,))
    return replace(op, local_shape=shape, semantic_shape=shape, launch_shape=shape, padded_shape=shape,
                   metadata=tuple(meta.items()), mte2_bytes=reads, hbm_bytes=reads + output, mte3_bytes=output, vector_ops=4*m*n)
def _typed(op):
    return declare_native_call(
        op, api={"namespace": "cann_kernel", "entry": "QuantBatchMatmulV3"},
        source={"repository": "mindie-sim-data", "path": "native_qbmm_row120", "sha256": "0" * 64,
                "symbol": "QuantBatchMatmulV3", "revision": "20261030", "state": "task_artifact"},
        typed={"attrs": {"input_dtypes": list(TYPED_DTYPES), "input_formats": ["ND", "FRACTAL_NZ", "ND", "ND"],
                         "output_formats": ["ND"]}},
        kernels=[{"name": "QuantBatchMatmulV3", "role": "cube", "evidence": "source_declared"}])
@pytest.fixture(scope="module")
def typed_records():
    raw, op, _ = _typed_case()
    legacy = _scale_record(op, op.local_shape, offset=True)
    return raw, legacy, reconcile_native_scale_storage(_typed(legacy))
def test_declared_scale_offset_requires_typed_reconciliation(typed_records):
    _, declared, _ = typed_records
    assert reconcile_native_scale_storage(declared) is declared
    assert MatrixEpilogue.from_op(declared) == MatrixEpilogue(16, 40_960, 4)
    assert declared.hbm_bytes == 44_681_232
    with pytest.raises(ModelingError, match=TYPED_MISMATCH):
        MatrixEpilogue.from_op(_typed(declared))

@pytest.mark.parametrize("recipe", [TiledMatmul(128, 128, 128), SteadyMatmul()])
def test_reconciled_scale_bytes_feed_both_matrix_programs(typed_records, recipe):
    raw, _, fixed = typed_records
    assert (fixed.hbm_bytes, fixed.mte2_bytes, fixed.mte3_bytes) == (CAPTURED_TOTAL, CAPTURED_TOTAL - CAPTURED_OUTPUT, CAPTURED_OUTPUT)
    assert MatrixEpilogue.from_op(fixed) == MatrixEpilogue(16, CAPTURED_COLUMN, 4)
    meta = dict(fixed.metadata)
    assert (meta["weight_scale_format"], meta["weight_scale_bytes"], meta["packed_weight_bytes"]) == ("bf16", 2, 44_574_720)
    assert meta["native_scale_reconciliation"]["hbm_delta_bytes"] == -30_720
    assert reconcile_native_scale_storage(fixed) == fixed
    row = operator_costs(raw, (fixed,), recipes=RecipeBook({"dense_down": recipe}), input_ready_ms=0)["operators"][0]
    assert row["program"] == type(recipe).__name__ and row["operator"]["hbm_bytes"] == CAPTURED_TOTAL
    for view in ("declared_service", "serviced"):
        assert row[view]["memory@r0:s0"] == pytest.approx(CAPTURED_TOTAL, rel=1e-12)
    binding = bind_native_call(_typed(fixed))
    assert binding.physical is None and tuple(binding.typed["attrs"]["input_dtypes"]) == TYPED_DTYPES
@pytest.mark.parametrize("mutation,omitted,reject", [
    ({"weight_scale_format": "fp32", "weight_scale_bytes": 4.0, "int8_weight_metadata": "scale_offset"}, (), False),
    ({"activation_scale_format": "bf16", "activation_scale_bytes": 8}, (), True),
    ({"activation_scale_shape": (4, 2)}, (), True), ({}, ("activation_scale_shape", "weight_scale_shape"), True),
])
def test_reconciliation_repairs_metadata_without_repricing(typed_records, mutation, omitted, reject):
    _, _, fixed = typed_records
    stale = _typed(replace(fixed, metadata=tuple((k, v) for k, v in dict(fixed.metadata, **mutation).items() if k not in omitted)))
    if reject:
        with pytest.raises(ModelingError, match=TYPED_MISMATCH):
            MatrixEpilogue.from_op(stale)
    restored = reconcile_native_scale_storage(stale)
    assert restored.hbm_bytes == fixed.hbm_bytes and restored.mte2_bytes == fixed.mte2_bytes
    meta = dict(restored.metadata)
    keys = ("weight_scale_format", "weight_scale_bytes", "int8_weight_metadata", "activation_scale_format",
            "activation_scale_bytes", "activation_scale_shape", "weight_scale_shape")
    assert tuple(meta[k] for k in keys) == ("bf16", 2, "scale_only", "fp32", 16, (4,), (5120,))
    assert meta["matrix_epilogue"]["row_scale_bytes"] == 16 and meta["native_scale_reconciliation"]["hbm_delta_bytes"] == 0
    assert reconcile_native_scale_storage(restored) == restored
def test_typed_tail_scale_splits_operands_and_epilogue(typed_records):
    tail = _typed(_scale_record(typed_records[2], (5, 7, 11)))
    epilogue = MatrixEpilogue.from_op(tail)
    assert epilogue == MatrixEpilogue(5 * 4, 7 * 2, 4)
    assert epilogue.operands(tail) == (5 * 11, 7 * 11) and reconcile_native_scale_storage(tail) == tail
@pytest.mark.parametrize("typed", [False, True])
def test_quantize_producer_dependency_and_exact_matrix_storage(typed):
    raw, matrix, producer = _typed_case(enabled=True)
    assert producer.family == "quantize" and dict(matrix.metadata)["activation_producer"] == "explicit_quantization"
    options = {}
    if typed:
        _, prepared = reconcile_scale_storage((producer, _typed(matrix)))
        matrix, epilogue = plan_matrix_epilogue(prepared, producer=producer)
        assert epilogue == MatrixEpilogue(16, CAPTURED_COLUMN, 4) and epilogue.operands(matrix) == (4*8704, 5120*8704)
        options["recipes"] = RecipeBook({matrix.recipe: TiledMatmul(128, 128, 128)})
    assert matrix.hbm_bytes == CAPTURED_TOTAL and dict(matrix.metadata)["activation_scale_shape"] == (4,)
    joint = graph_costs(raw, (producer, matrix), input_ready_ms=0, external_inputs={producer.deps[0]: 0}, **options)
    rows = {row["op_id"]: row for row in joint["timeline"]}
    assert rows[matrix.op_id]["predecessors"] == [f"r0:{producer.op_id}"]
    assert rows[matrix.op_id]["ready_ms"] == pytest.approx(rows[producer.op_id]["complete_ms"])
    assert joint["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(CAPTURED_TOTAL + producer.hbm_bytes * DRAM_SERVICE_FACTOR, rel=1e-12)
@pytest.mark.parametrize("entry", [MatrixEpilogue.from_op, reconcile_native_scale_storage, reconcile_scale_storage])
def test_bound_producer_scale_conflicts_are_visible(entry):
    _, matrix, producer = _typed_case(enabled=True)
    declared = _typed(matrix)
    meta = dict(producer.metadata, activation_scale_format="bf16", activation_scale_bytes=8)
    if entry is not reconcile_scale_storage:
        declared = reconcile_native_scale_storage(declared, producer=producer)
        assert MatrixEpilogue.from_op(declared, producer=producer).row_scale_bytes == 16
        meta["activation_scale_shape"] = (4, 2)
    bad = replace(producer, metadata=tuple(meta.items()))
    with pytest.raises(ModelingError, match=PRODUCER_CONFLICT):
        if entry is reconcile_scale_storage:
            entry((bad, declared))
        else:
            entry(declared, producer=bad)

@pytest.fixture(scope="module")
def family_case():
    case = parse_case(json.loads((CASES / "kimi-k3-a5-sp-right-pad-conditional.json").read_text()))
    ops = target_graphs(case, scope="body", phase="decode", tokens=1, context=2048,
                        sequences=1, resource_fusion=False)[0]
    shared = tuple(next(op for op in ops if op.family == family and dict(op.metadata).get("ffn_category") == "shared_expert")
                   for family in ("gate_up_projection", "down_projection"))
    return case, shared, next(op for op in ops if op.family == "grouped_gemm")
def _served(case, op, **options):
    row = operator_costs(case, (op,), input_ready_ms=7, **options)["operators"][0]
    assert not row["service_before_input"] and row["live_tasks"] == 0
    for role, delta in row["service_delta"].items():
        assert abs(delta) <= max(1e-6, abs(row["serviced"].get(role, 0)) * 1e-9)
    return row
@pytest.mark.parametrize("shape,index", [(None, 0), (None, 1), ((1, 64, 64), 0), ((3, 320, 448), 0),
                                         ((17, 1088, 1088), 0), ((257, 320, 512), 0)])
def test_shared_matrix_geometry_conserves_issued_and_published_work(family_case, shape, index):
    case, shared, _ = family_case
    op = shared[index]
    m, n, k = shape or op.local_shape
    if shape:
        scales = (math.ceil(k / 64), 2)
        op = _meta(replace(op, local_shape=shape, semantic_shape=shape, launch_shape=shape),
                   activation_scale_shape=(m, *scales), weight_scale_shape=(n, *scales))
    geometry = Geometry.from_devices(case.machine.dies)
    assert geometry.resource_units["vector"] == AIV_COUNT and shared_w4a8.matches_source_interface(op)
    plan, row = shared_w4a8.build(op, geometry), _served(case, op)
    detail = row["operator"]["metadata"]["program_execution"]["shared_w4a8"]
    step = math.ceil(m / math.ceil(m / 256))
    extents = [min(step, m - start) for start in range(0, m, step)]
    padded = [16 * math.ceil(rows / 16) for rows in extents]
    issued = 2 * n * k * sum(padded)
    assert detail["mmad_parameter_m"] == detail["fixpipe_m"] == tuple(sorted(set(extents)))
    assert detail["l0a_issue_fractal_m"] == tuple(sorted(set(padded)))
    assert detail["logical_flops"] == 2*m*n*k and detail["issued_cube_flops"] == issued
    assert detail["init_mx_bytes"] == shared_w4a8.INIT_PERMUTATION_BYTES * AIV_COUNT
    assert plan.work.matrix == pytest.approx({"fp8_e4m3": issued})
    assert row["serviced"]["fixpipe@r0:s0"] == pytest.approx(2*m*n)
@pytest.mark.parametrize("change", [
    {"weight_format": "fp8_e4m3"}, {"activation_format": "bf16"}, {"output_format": "fp32"},
    {"matrix_groups": 2}, {"matrix_group_rows": (8,)}, {"bias": {"present": True}},
    {"activation_scale_shape": (8, 1, 2)}, {"weight_layout": "FRACTAL_Z"},
    {"ffn_category": "routed_expert"}, {"quant_category": "routed_expert"},
])
def test_shared_matrix_rejects_ineligible_source_interfaces(family_case, change):
    op = family_case[1][0]
    candidate = replace(op, metadata=tuple({**dict(op.metadata), **change}.items()))
    assert not shared_w4a8.matches_source_interface(candidate)
    with pytest.raises(ModelingError, match="shared_w4a8_requires_fp8_mxfp4_group32_e8m0_bf16"):
        shared_w4a8.config_from_op(candidate)

@pytest.mark.parametrize("rows,width,k", [((1,), 128, 64), ((2, 0, 3), 320, 448),
                         ((5, 7), 192, 512), ((33,), 256, 1024), ((0, 0), 128, 64)])
def test_gate_up_banks_conserve_issued_tiles_and_padded_publication(family_case, rows, width, k):
    case, _, base = family_case
    exclude = {"matrix_group_rows", "matrix_weight_bank_groups", "matrix_groups", "grouped_dequant"}
    meta = {key: value for key, value in base.metadata if key not in exclude}
    meta["mxfp4_gate_config"] = dict(group_rows=rows, output_width=width, k=k)
    shape = (sum(rows), 2 * width, k)
    op = replace(base, recipe=gate_situ.RECIPE, flops=0, matrix_flops=(), vector_ops=0, hbm_bytes=0, mte2_bytes=0,
                 mte3_bytes=0, deps=(), local_shape=shape, semantic_shape=shape, launch_shape=shape, metadata=tuple(meta.items()))
    plan = gate_situ.build(op, Geometry.from_devices(case.machine.dies))
    detail = plan.metadata
    tiles = [(min(128, r - start), min(128, width - col)) for r in rows
             for start in range(0, r, 128) for col in range(0, width, 128)]
    assert detail["source_output_tiles"] == len(tiles)
    if not tiles:
        assert not plan.work.matrix and not plan.work.resources
        assert detail["active_aic_owners"] == detail["producer_packed_weight_bytes"] == 0
        return
    issued = 4 * k * sum(16 * math.ceil(r / 16) * n for r, n in tiles)
    fixpipe = 16 * sum(math.ceil(r / 2) * n for r, n in tiles)
    assert detail["physical_cube_flops"] == detail["physical_cube_flops_derived"] == issued
    assert detail["source_api_fixpipe_bytes"] == fixpipe and detail["logical_valid_flops"] == 4 * sum(rows) * width * k
    row = _served(case, op, recipes=RecipeBook({gate_situ.RECIPE: PlannedRecipe(gate_situ.build)}))
    assert row["serviced"]["fixpipe@r0:s0"] == pytest.approx(fixpipe)
@pytest.mark.parametrize("groups", [0, 1, 63, 64, 65, 257])
def test_u8_widen_inventory_charges_packing_once(groups):
    typed, counts, inventory = gmmsq_work(groups)
    vf = math.ceil(groups / 64)
    assert (counts.vf_groups, counts.packed_bytes, counts.expanded_bytes) == (vf, 256 * groups, 512 * groups)
    assert {name: inventory.counts[name].calls for name in ("LoadAlignU8", "StoreAlignU8", "CreateMask", "DuplicateU8")} == {
        "LoadAlignU8": groups, "StoreAlignU8": 2 * groups, "CreateMask": vf, "DuplicateU8": 3 * vf}
    expected = {"reg_issue_slots": 4 * vf + 12 * groups, "reg_data_writes": 3 * vf + 8 * groups,
                "reg_ub_accesses": 3 * groups, "reg_ub_stores": 2 * groups}
    assert all(typed.resources.get(name, 0) == value for name, value in expected.items())
def test_prepared_grouped_matrix_replaces_checkpoint_scale_storage():
    raw = json.loads((CASES / PREPARED_CASE).read_text())
    raw["runtime"]["kernels"]["lowerings"]["mx_matrix"] = dict(implementation="ascend_prepared_mx32_matrix",
        options=dict(projection_recipes=["q_b", "grouped_output_a", "grouped_output_b"], scale_storage="replace_checkpoint_scales"))
    ops = target_graphs(raw, scope="body", phase="decode", tokens=1, context=2048, sequences=[4] + [0] * 7)[0]
    op = next(op for op in ops if dict(op.metadata).get("semantic_recipe") == "grouped_output_a")
    m, n, k = op.local_shape
    meta = dict(op.metadata)
    scales = 8 * n * (k // 64) * 2
    assert meta["matrix_groups"] == 8 and meta["packed_weight_bytes"] == 8*n*k + scales
    assert meta["prepared_weight_scale_bytes"] == scales and meta["prepared_scale_storage"] == "replace_checkpoint_scales"
    assert meta["fixpipe_bytes"] == 2*m*n and op.mte3_bytes == 0 and op.recipe == family_mx_matrix.PROFILE
    row = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
    assert row["serviced"]["fixpipe@r0:s0"] == pytest.approx(2*m*n)
    assert row["serviced"]["memory@r0:s0"] == pytest.approx(op.hbm_bytes)
    assert not any(row["serviced"].get(role + "@r0:s0", 0) for role in ("mte3", "vector"))
    assert not row["service_before_input"] and not any(row["remaining_buffers"].values())
@pytest.mark.parametrize("rank,placement,rows,extents", [
    (0, ((0, 0), (0, 1), (1, 0), (1, 1)), (2, 2), (4, 4, 8)), (1, "contiguous_ep", (2, 2), (4, 4, 4)),
    (0, "contiguous_ep", (16, 0), (16, 32, 64)), (1, ((0, 0), (0, 1), (1, 0), (1, 2)), (2, 2), (4, 4, 4)),
])
def test_grouped_token_routing_keeps_destination_bank_and_interface_rows(rank, placement, rows, extents):
    valid, received, capacity = extents
    ids, ledger = (2 * rank, 2 * rank + 1), (2, 2, 2, 2) if valid == 4 else rows
    realization = dict(basis="explicit_token_routing_integer", expert_placement=placement, local_expert_rows=rows,
                       representative_expert_ids=ids, local_weight_expert_ids=(0, 1), global_expert_rows=ledger,
                       representative_rank=rank, valid_rows=valid, received_rows=received, capacity_rows=capacity)
    if placement == "contiguous_ep":
        realization.update(represented_ep_ranks=(rank,), capacity_slack=capacity - received, padding_rows=received - valid)
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(), m=valid, n=8, k=4, quant=QuantSpec("bf16"),
                 category="routed_expert", phase="prefill", family="grouped_gemm", recipe="grouped_gate_up")
    op = replace(op, metadata=tuple(dict(op.metadata, routing_realization=realization, routing_basis=realization["basis"]).items()))
    if isinstance(placement, tuple) and rank == 1:
        with pytest.raises(ModelingError, match="explicit_local_bank_slot_out_of_range"):
            family_grouped_gemm.lower_op(op)
        return
    lowered = family_grouped_gemm.lower_op(op)
    meta = dict(lowered.metadata)
    assert lowered.local_shape == (valid, 8, 4) and meta["local_expert_rows"] == rows
    assert (meta["routing_valid_rows"], meta["routing_received_rows"], meta["routing_capacity_rows"]) == extents
    assert meta["representative_expert_ids"] == ids and meta["global_expert_slice"] == ledger[2 * rank:2 * rank + 2]
    assert meta["global_expert_rows"] == ledger and meta["routing_rows_basis"] == "explicit_destination_bank"
    if valid == 16:
        assert (meta["routing_capacity_slack"], meta["routing_padding_rows"]) == (32, 16)
def test_explicit_token_route_enters_public_grouped_program():
    raw = explicit_route_case(route=((6, 10, 8, 8), True))
    raw["runtime"]["kernels"] = {"lowerings": {"grouped_gemm": {"implementation": "ascendc_grouped_nd_bf16"}}}
    ops = target_operators(raw, phase="prefill", tokens=6, scope="body")
    grouped = [op for op in ops if op.family == "grouped_gemm"]
    assert len(grouped) == 2
    for op in grouped:
        meta = dict(op.metadata)
        assert meta["routing_rows_basis"] == "explicit_destination_bank"
        assert tuple(meta["matrix_group_rows"]) == meta["global_expert_slice"] == (8, 8)
    row = operator_costs(raw, (grouped[0],), input_ready_ms=0)["operators"][0]
    assert row["serviced"]["matrix@r0:s0"] > 0 and row["live_tasks"] == 0
@pytest.mark.parametrize("rows,interface,capacity", [((16, 0), 32, 64), ((1, 1), 2, 2), ((2, 0), 2, 2)])
def test_native_grouped_rows_keep_interface_capacity_and_issued_regions(rows, interface, capacity):
    valid, macs = sum(rows), sum(bool(count) for count in rows) * 65536
    geometry = dict(valid_rows=valid, received_rows=interface, interface_rows=interface, capacity_rows=capacity,
                    padding_rows=interface - valid, capacity_slack=capacity - interface)
    regions = [{"rows": n, "issued_macs": 65536 if n else 0} for n in rows]
    compiled = family_grouped_gemm.native_grouped_row_contract(
        {"geometry": geometry, "regions": regions, "work": {"cube": {"issued_macs": macs}}})
    assert {key: compiled[key] for key in geometry} == geometry
    assert compiled["group_rows"] == rows and compiled["nnz_rows"] == valid
    assert compiled["active_regions"] == tuple(i for i, n in enumerate(rows) if n) and compiled["issued_macs"] == macs

def _capture_path(relative):
    spec = find_spec("mindie_profiler")
    roots = [REPO_ROOT.parent / "profiler", REPO_ROOT.parent.parent / "profiler"]
    if spec and spec.origin:
        roots.insert(0, Path(spec.origin).resolve().parents[1])
    paths = [root / "tests/fixtures" / relative for root in roots]
    return next((path for path in paths if path.is_file()), paths[0])

MX_CAPTURE = _capture_path("fusion_contracts/kimi_matrix_branches.csv")
WRAPPER_CAPTURE = _capture_path("kimi_grouped_attnres/a5_kimi_k3.csv")
@lru_cache(maxsize=2)
def _capture_rows(path):
    from mindie_profiler.normalized import kernel_rows
    return tuple(kernel_rows(path, rank_id="rank_0"))
def _map_capture(path, events, structure, variant, *, step=None):
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.structure_mapping import map_structure
    stat, rank = path.stat(), events[0].rank_id
    source = SourceRef(rank, 0, "csv", str(path), path.name, None, stat.st_size, stat.st_mtime_ns, "retained_capture", variant)
    window = Window(min(e.start_ns for e in events), max(e.end_ns for e in events), step=step)
    return map_structure([Observation(rank, 0, window, None, events, source, QueryStats(rank, "retained_capture"))],
                         structure, mapping_limit=0 if step is not None else 100)
@pytest.fixture(scope="module")
def mx_case():
    raw = json.loads((CASES / "kimi-k3-a5.json").read_text())
    raw["model"]["layers"] = 4
    base = parse_case(raw)
    quant = replace(base.model.quant, weight_format="mxfp4", expert_weight_format="mxfp4",
                    activation_format="fp8_e4m3", activation_group_size=32,
                    activation_scale_format="fp8_e8m0", activation_scale_pack=2, exclusions=())
    return replace(base, model=replace(base.model, quant=quant))
@pytest.mark.requires_capture_mount(str(MX_CAPTURE))
def test_exported_matrix_boundaries_preserve_native_tensor_contracts(mx_case, profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract
    graph = export_structure(mx_case, phase="prefill", tokens=8192, scope="body")
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    contracts = [row["contract"] for row in (*graph["nodes"], *graph["subgraphs"])]
    for kind in ("QuantBatchMatmulV4", "SituMxQuant", "GroupedMatmulSituQuant"):
        observed = [observed_contract(row) for row in _capture_rows(MX_CAPTURE) if row.kind == kind]
        assert observed and all(json.loads(json.dumps(value)) in contracts for value in observed)
    groups = [row for row in graph["subgraphs"] if row["role"] == "grouped_gate_up_situ_quant"]
    assert groups
    for group in groups:
        assert group["contract"] == ["grouped_mlp_parameters_active", "GroupedMatmulSituQuant", [28, 3584, 3072],
                                     ["fp8_e4m3", "mxfp4", "fp8_e4m3"]]
        members = [nodes[node_id] for node_id in group["node_ids"]]
        assert sorted(row["operator"]["family"] for row in members) == ["grouped_gemm", "quantize", "situ"]
        assert group["contract_scope"]["work_assumption"] == "expected_routing_work_is_declared_not_measured"
        assert all(sum(row["physical_work"][key] for row in members) > 0 for key in ("flops", "vector_ops"))
def test_grouped_situ_requires_native_precision_and_scale_domain(mx_case):
    ops = target_operators(mx_case, phase="prefill", tokens=8)
    view = next(group["tensor_view"] for group, _ in _groups(ops) if group["role"] == "grouped_gate_up_situ_quant")
    assert grouped_situ_contract(view) is not None
    for key, value in (("weight_dtype", "fp8_e4m3"), ("input_dtype", "bf16"), ("input_scale_group_size", 64),
                       ("output_scale_tail", (48, 1)), ("intermediate_dtype", "fp32")):
        assert grouped_situ_contract({**view, key: value}) is None
    assert grouped_situ_contract(dict(view, activation=dict(view["activation"], activation_linear_beta=None))) is None
@pytest.mark.requires_capture_mount(str(MX_CAPTURE))
def test_native_matrix_boundaries_report_reference_costs(mx_case, profiler_api):
    from mindie_sim.cohort_accuracy import compare_cohorts
    block = next(plan for plan in mx_case.model.layer_plans if plan.ffn == "moe")
    model = replace(mx_case.model, layers=1, body_layers=1, layer_plans=(replace(block, index=0, kv_owner_layer_id=0),))
    lowerings = dict(mx_case.runtime.kernels.get("lowerings", {}), native_fusion={
        "implementation": "declared_native_fusion",
        "options": {"roles": ["grouped_gate_up_situ_quant"], "traffic_profile": "preserve_workspaces"}})
    case = replace(mx_case, model=model, runtime=replace(mx_case.runtime, pp_layer_partition=(1,),
                   kernels=dict(mx_case.runtime.kernels, lowerings=lowerings)))
    events = [next(row.event for row in _capture_rows(MX_CAPTURE) if row.kind == kind)
              for kind in ("GroupedMatmulSituQuant", "QuantBatchMatmulV4")]
    graph = export_structure(case, phase="prefill", tokens=8192, scope="body")
    mapping = _map_capture(MX_CAPTURE, events, graph, "op_summary", step=0)
    rows = compare_cohorts(case, mapping)["windows"][0]["cohorts"]
    assert {row["target_kind"] for row in rows} == {"node", "subgraph"}
    for row in rows:
        assert (row["status"], row["reason"], row["comparison"]) == ("reference_only", "mxfp4_native_execution_unmodeled", None)
        assert row["prediction"]["per_call_ms"] > 0 and row["prediction"]["physical_work_per_instance"]["launches"] == 1

@lru_cache(maxsize=4)
def _wrapper_ops(tokens=1024, phase="prefill", graph_mode=None):
    case = parse_case(json.loads((CASES / WRAPPER_CASE).read_text()))
    if graph_mode:
        case = replace(case, runtime=replace(case.runtime, graph_mode=graph_mode))
    ops = target_operators(case, phase=phase, tokens=tokens, context=32768 if phase == "decode" else 0)
    return case, tuple(op for op in ops if op.layer == 1)
def _projection(ops):
    return next(op for op in ops if op.recipe == "latent_projection")
def _meta(op, **change):
    return replace(op, metadata=tuple(dict(op.metadata, **change).items()))
def _wrapper(ops, case, recipe="latent_projection", graph_mode="eager", implementation=matrix_wrappers.IMPLEMENTATION):
    selection = Selection(implementation, matrix_wrappers, {"projection_recipes": (recipe,)})
    return matrix_wrappers.apply_matrix_wrapper(ops, selection, replace(case.runtime, graph_mode=graph_mode))
@pytest.mark.requires_capture_mount(str(WRAPPER_CAPTURE))
@pytest.mark.parametrize("tokens,phase", [(1024, "prefill"), (1, "decode")])
def test_wrapper_preserves_matrix_work_and_matches_retained_single_bank(tokens, phase, profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract
    case, ops = _wrapper_ops(tokens, phase, "eager")
    native = {observed_contract(row): row for row in _capture_rows(WRAPPER_CAPTURE) if row.kind == "GroupedMatmul"}
    bound = _wrapper(ops, case)
    assert all(replace(b, metadata=o.metadata) == o for o, b in zip(ops, bound, strict=True))
    projections = [op for op in bound if op.recipe == "latent_projection"]
    assert {op.local_shape[1:] for op in projections} == {(3584, 7168), (7168, 3584)}
    for op in projections:
        contract, scope = predicted_contract(op.to_dict()), comparison_scope(op.to_dict())
        assert contract in native and contract[0] == "grouped_matrix" and contract[1][0] == 1
        assert scope["declared_group_list"] == [tokens] and "native_dispatch" in scope["not_compared"]
    if tokens == 1024:
        op = projections[0]
        graph = export_op_structure((op,), model=case.model.to_dict(), execution={"phase": phase, "tokens": tokens})
        result = _map_capture(WRAPPER_CAPTURE, [native[predicted_contract(op.to_dict())].event], graph, "kernel_details")
        row = result["event_mappings"][0]
        assert (row["status"], row["candidate_node_ids"], row["node_ids"]) == ("candidate", ["rank:0/" + op.op_id], [])
        assert result["windows"][0]["bound_busy_union_ns"] == 0
def test_public_eager_wrapper_preserves_executed_matrix_cost():
    raw = json.loads((CASES / WRAPPER_CASE).read_text())
    raw["runtime"]["graph_mode"] = "eager"
    baseline = parse_case(raw)
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["matrix_wrapper"] = {
        "implementation": matrix_wrappers.IMPLEMENTATION, "options": {"projection_recipes": ["latent_projection"]}}
    reports = []
    for case, kind in ((baseline, "matrix"), (parse_case(raw), "grouped_matrix")):
        ops = target_operators(case, phase="decode", tokens=1, context=2048)
        projections = tuple(op for op in ops if op.recipe == "latent_projection")
        assert projections and all(predicted_contract(op.to_dict())[0] == kind for op in projections)
        report = operator_costs(case, projections[:2])
        reports.append((report["operator_sum_ms"], [row["serviced"] for row in report["operators"]]))
    assert reports[0] == reports[1]
@pytest.mark.parametrize("extra,reason", [
    ({"bias": True, "bias_shape": (3584,)}, "requires_bias_free_projection"),
    ({"activation_scale_shape": ()}, "requires_group32_e8m0_scales"), ({"weight_format": "mxfp4"}, "requires_mxfp8_operands"),
    ({"matrix_groups": 2}, "requires_single_dense_matrix"), ({"expected_rows": 512}, "requires_all_recorded_rows_active"),
])
def test_ineligible_projection_rejects_source_wrapper(extra, reason):
    case, ops = _wrapper_ops()
    with pytest.raises(ModelingError, match=reason):
        _wrapper((_meta(_projection(ops), **extra),), case)
@pytest.mark.parametrize("recipe,grouped", [("grouped_down", True), ("linear_qkvbfg_a", False)])
def test_source_wrapper_does_not_merge_routed_or_packed_calls(recipe, grouped):
    case, ops = _wrapper_ops()
    op = next(item for item in ops if str(item.recipe).startswith(recipe))
    contract = predicted_contract(op.to_dict())
    if grouped:
        assert contract[1][0] == 28 and contract[3] == "mxfp4"
        assert comparison_scope(op.to_dict())["prediction_coverage"]["reason"] == "mxfp4_native_execution_unmodeled"
    else:
        assert contract[0] == "matrix"
        assert {"q", "k", "v"} <= {row["role"] for row in dict(op.metadata)["packed_projection_slices"]}
    with pytest.raises(ModelingError, match="source_caller_unsupported"):
        _wrapper((op,), case, recipe)
    assert predicted_contract(op.to_dict()) == contract
def test_wrapper_checks_explicit_eager_dispatch_and_implementation():
    case, ops = _wrapper_ops(1, "decode")
    op = _projection(ops)
    assert dict(op.metadata)["graph_captured"] and predicted_contract(op.to_dict())[0] == "matrix"
    missing = tuple(replace(item, metadata=tuple((k, v) for k, v in item.metadata if not k.startswith("graph_"))) for item in ops)
    with pytest.raises(ModelingError, match="requires_explicit_eager_dispatch"):
        _wrapper(missing, case, graph_mode=case.runtime.graph_mode)
    for captured in (_meta(_projection(missing), graph_captured=True), op):
        with pytest.raises(ModelingError, match="requires_eager_dispatch"):
            _wrapper((captured,), case)
    eager, ops = _wrapper_ops(1, "decode", "eager")
    with pytest.raises(ModelingError, match="implementation_unsupported"):
        _wrapper((_projection(ops),), eager, implementation="some_other_projection_wrapper")
def test_source_wrapper_identity_and_protected_programs():
    case, ops = _wrapper_ops(1, "decode", "eager")
    op = _projection(ops)
    protected = (replace(op, source_boundary="kernel"),
                 _meta(op, implementation_binding={"implementation": "independent_source_program"}))
    assert matrix_wrappers.apply_matrix_wrapper(ops, None, case.runtime) == ops
    assert _wrapper(protected, case) == protected
    wrapper = dict(_wrapper((op,), case)[0].metadata)["matrix_wrapper"]
    identity = wrapper["source_identity"]
    assert identity["wrapper_image_id"] == matrix_wrappers.SOURCE["image_id"] and identity["current_executable_source"] is False
    assert identity["basis"] == wrapper["source_basis"] == "declared_source_wrapper_not_capture_dispatch_proof"
    assert all(token not in json.dumps(wrapper) for token in ("b57207c2", "measured"))
    assert wrapper["proven_predicates"]["active_forward_rows"] == 1
