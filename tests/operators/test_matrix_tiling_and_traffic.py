"""Matrix tiling, owner scheduling, and physical work contracts.

[Role] Exercise finite, phase, compact and public matrix execution.
[Invariants] Tail work, operand traffic, and owner capacity agree with analytical oracles.
[Timeouts & Locks] Bounded local event simulations, with no locks.
[Side-Effect Policy] Reads checked-in cases, profiles, and captures.
"""

from dataclasses import replace
from itertools import product
import json
import math
import pytest
from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.captures import REPO_ROOT, fixture_path
from _support.hardware import hardware_case
from _support.matrix_cases import matrix_run, matrix_setup, quantized_context

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
from mindie_sim.prediction.operators.matrix.matrix_cycle import matrix_cycle
from mindie_sim.prediction.operators.matrix.matrix_mapping import (
    DEFAULT_CANDIDATES, UNDERFILLED_CANDIDATES, MappedMatmul, grouped_owner_rectangles,
    grouped_owner_tiles, mapping_bound, matrix_geometry, owner_tiles, physical_owner_rate,
    simulate_matrix_operator,
)
from mindie_sim.prediction.operators.matrix.matrix_phase import PhaseMatmul
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, RecipeContext, Stage
from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
from mindie_sim.prediction.service_regime_domains.services import ResourceService, ServiceCurve, resolved_phase_work
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.graph_analysis import verify_operator
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from tools.analyze_a5_microbench_boundaries import analyze_window

pytestmark = pytest.mark.tier0
FAST = 1e12
ROOMY = 1 << 20
GENERATION_PROFILE = REPRESENTATIVE_SKUS["a5"]
OWNER_GRID_PROFILE = "Ascend950DT_9582"
FP32_PEAK = 27.0336e12
PANEL_COUNT = 10_000_000
DEFAULT_TILE = HierarchicalMatmul(4, 4, 2, 2, 2)
TWO_CANDIDATES = MappedMatmul(((2, 2, 2), (4, 4, 2)), 2, 2)
TRAFFIC_ROLES = ("cube", "hbm", "mte2", "mte1", "mte3")

def _op(m, n, k, groups=1, *, quant=None, deps=(), phase="prefill"):
    return gemm_op(op_id="matrix", deps=deps, stage=0, layer=0, requests=(), m=m, n=n, k=k,
                   weight_count=groups, quant=quant or QuantSpec("bf16"), category="dense_mlp", phase=phase)
def _context(op, cores=2, pools=None, rates=None):
    pools = pools or dict.fromkeys(("l1", "l0a", "l0b", "l0c"), 4096)
    _, base = matrix_setup(*op.local_shape[-3:], cores=cores, pools=pools)
    rates = rates or base.rates
    work = {"cube": op.flops, **({"vector": op.vector_ops} if op.vector_ops else {})}
    return replace(base, op=op, work=work, rates=rates, resource_names={name: name for name in rates},
                   pool_units=dict.fromkeys(pools, cores), resource_units={name: 1 if name == "hbm" else cores for name in rates})
def _engine(ctx, trace=0):
    return Engine([Resource(name, rate, units=ctx.resource_units[name]) for name, rate in ctx.rates.items()],
                  pools=ctx.pool_capacities, pool_units=ctx.pool_units or dict.fromkeys(ctx.pool_capacities, ctx.core_count), trace_limit=trace)
def _run(ctx, recipe, *, release=7, trace=0, **options):
    engine = _engine(ctx, trace)
    done = recipe.lower(ctx, **options).start(engine, "matrix", engine.submit("ready", release=release))
    engine.advance(horizon=release)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not any(engine.allocated.values()) and engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return engine, done.time - release
def _owners(rows, n, tm, tn, cores):
    tiles = [(min(tm, m - r), min(tn, n - c)) for m in rows for r in range(0, m, tm) for c in range(0, n, tn)]
    return tuple((len(tiles[i::cores]), tuple(tiles[i::cores])) for i in range(cores))
def _aggregate(sources):
    return tuple((count, sum(a for a, _ in tiles), sum(b for _, b in tiles), sum(a * b for a, b in tiles)) for count, tiles in sources if count)

def _refill_span(load, micros, panels, refill, consumer, tail=None):
    engine = Engine([Resource(name, 1) for name in ("load", "operand", "cube")])
    def panel(index):
        amount, steps = tail if index == panels else (load, micros)
        tasks = (((Stage("operand", {"operand": a}), Stage("cube", {"cube": b}, after=("operand",))),
                  i == len(steps) - 1) for i, (a, b) in enumerate(steps))
        return Stage("load", {"load": amount}), tasks, ()
    done = RefillPipeline(panels + bool(tail), refill, consumer, panel).start(engine, "matrix", engine.submit("ready", release=7))
    engine.advance(until=(done,))
    return done.time - 7
@pytest.mark.parametrize("panels,refill,consumer", tuple(product((1, 2, 7, 31), (1, 2, 3), (1, 2, 4))))
def test_cycle_matches_finite_dependencies(panels, refill, consumer):
    tail = (1.75, ((2, 0.25), (0.125, 3)))
    for load, micros in product((0.25, 3), (((1, 1),), ((0.25, 2), (3, 0.5), (0, 1)))):
        result = matrix_cycle(load, micros, panels, refill, consumer, tail_load_ms=tail[0], tail_micro_steps=tail[1])
        assert result.span_ms == pytest.approx(_refill_span(load, micros, panels, refill, consumer, tail))
        assert result.stepped_panels + result.jumped_panels == panels + 1
def test_cycle_overlap_and_period_with_exact_tail():
    micros = ((1, 1),)
    assert matrix_cycle(1, micros, 4, 2, 2).span_ms == _refill_span(1, micros, 4, 2, 2) == 7 > 6
    long = matrix_cycle(1, micros, PANEL_COUNT, 2, 2, tail_load_ms=0.5, tail_micro_steps=((0.5, 0.5),))
    assert (long.span_ms, long.period_panels, long.period_ms) == (3 * PANEL_COUNT / 2 + 1.5, 2, 3)
    assert long.stepped_panels < 16 and long.stepped_panels + long.jumped_panels == PANEL_COUNT + 1
    with pytest.raises(ModelingError, match="period_not_found_within_budget:1"):
        matrix_cycle(1, micros, PANEL_COUNT, max_steps=1)
    assert matrix_cycle(1, micros, 1, max_steps=1).span_ms == 3
    tail = (2, ((1, 3), (2, 1)))
    tail_only = matrix_cycle(4, ((1, 2),), 0, tail_load_ms=tail[0], tail_micro_steps=tail[1])
    assert (tail_only.prologue_ms, tail_only.prologue_panels, tail_only.span_ms) == (0, 0, _refill_span(4, ((1, 2),), 0, 2, 2, tail))
    assert all(matrix_cycle(load, ((0, 0),), panels).span_ms == 0 for load, panels in ((4, 0), (0, PANEL_COUNT)))

@pytest.mark.parametrize("shape,rates,spans", [
    ((4, 2, 40), (32, 24, FAST), (21, 40, 21)),
    ((4, 2, 40), (32 / 1.5, 240, FAST), (30.1, 32, 30.1)),
    ((4, 2, 6), (16, FAST, 8), (13, 18)),
    ((4, 4, 4), (16, 32 / 3, FAST), (11,)),
])
def test_overlap_and_serial_phase_bounds(shape, rates, spans):
    cube, load, operand = rates
    ctx = _context(_op(*shape), 1, rates=dict(zip(TRAFFIC_ROLES, (cube, FAST, load, operand, FAST))))
    records = []
    ctx = replace(ctx, record_mapping=records.append)
    tile = (4, shape[1], 2)
    recipes = (HierarchicalMatmul(*tile, 2, 2), PhaseMatmul((tile,), 2, 2), SteadyMatmul((tile,), 2, 2))
    engines, measured = zip(*(_run(ctx, recipe) for recipe in recipes[:len(spans)]))
    assert measured == pytest.approx(spans, abs=1e-8)
    assert all(engine.serviced == pytest.approx(engines[0].serviced) for engine in engines[1:])
    if shape[-1] == 40:
        assert matrix_cycle(24 / load, ((0, 16 / cube),) * 2, 20).span_ms == pytest.approx(spans[0])
        assert engines[0].serviced == pytest.approx(dict(zip(TRAFFIC_ROLES, (640, 496, 480, 640, 16))))
        assert engines[-1].completed_count < engines[0].completed_count / 5
@pytest.mark.parametrize("m,n,k,groups", [(7, 5, 3, 1), (6, 3, 2, 3)])
@pytest.mark.parametrize("reuse_a,reuse_b", tuple(product((False, True), repeat=2)))
def test_ragged_phase_and_detailed_traffic(m, n, k, groups, reuse_a, reuse_b):
    ctx = _context(_op(m, n, k, groups), 3)
    detail = replace(DEFAULT_TILE, reuse_inputs=reuse_a, reuse_weights=reuse_b)
    phase = PhaseMatmul(((4, 4, 2),), 2, 2, reuse_inputs=reuse_a, reuse_weights=reuse_b)
    a, b, c = 2*m*k, 2*groups*n*k, 2*m*n
    rows, cols = math.ceil(m / groups / 4), math.ceil(n / 4)
    expected = (2*m*n*k, a*(1 if reuse_a else cols) + b*(1 if reuse_b else rows) + c,
                a*cols + b*rows, a*math.ceil(n/2) + b*math.ceil(m/groups/2), c)
    for recipe in (detail, phase):
        engine, span = _run(ctx, recipe, trace=100)
        assert engine.serviced == pytest.approx(dict(zip(TRAFFIC_ROLES, expected)))
    assert mapping_bound(ctx, detail) <= span and not any(engine.peak_allocated.values())
    for owner, (_, tiles) in enumerate(_owners((m // groups,) * groups, n, 4, 4, 3)):
        copies = [r["serviced"]["mte2"] for r in engine.trace if r["placement"].get("mte2") == (owner,)]
        assert sum(copies) == pytest.approx(sum((r + c) * k * 2 for r, c in tiles))
@pytest.mark.parametrize("shape,cores,groups,serial,expected", [
    ((4, 2, 1), 1, 1, False, None), ((3, 2, 7), 1, 1, False, None), ((4, 3, 8), 1, 1, False, None),
    ((9, 7, 13), 3, 1, False, None), ((12, 7, 13), 4, 3, False, None), ((8, 2, 400), 2, 2, False, None),
    ((11, 17, 7), 1, 1, True, None), ((12, 17, 11), 1, 3, True, None), ((4, 43, 40), 1, 2, True, None),
    ((2, 2, 5), 1, 1, False, 12.775), ((2, 2, 6), 1, 1, False, 15.55),
])
def test_steady_owner_program_preserves_tails_and_serial_outputs(shape, cores, groups, serial, expected):
    tile = (2, 2, 4) if expected else (4, 4, 4)
    rates = (320, FAST, 3.2, 32, FAST) if expected else (32, FAST, 24, 20 if serial else 16, FAST)
    ctx = _context(_op(*shape, groups), cores, rates=dict(zip(TRAFFIC_ROLES, rates)))
    compact = SteadyMatmul((tile,), 2, 2)
    control = replace(compact, compact_output_tiles=False) if serial else HierarchicalMatmul(*tile, 2, 2)
    (before, a), (after, b) = (_run(ctx, recipe) for recipe in (control, compact))
    assert after.serviced == pytest.approx(before.serviced) and b > 0
    if cores == 1:
        assert b == pytest.approx(expected or a, abs=1e-8)
    if serial:
        assert after.completed_count < before.completed_count and after.peak_unit_allocated == before.peak_unit_allocated
@pytest.mark.parametrize("before,after,cores,recipe", [
    ((7, 5, 6), (7, 5, 60000), 3, PhaseMatmul(((4, 4, 2),), 2, 2)),
    ((4, 2, 40), (4, 2, 20_000_000), 1, SteadyMatmul(((4, 2, 2),), 2, 2)),
    ((12, 8, 40), (12000, 8, 40), 2, SteadyMatmul(((4, 4, 4),), 2, 2)), ((8, 8, 8), (8, 8, 128), 2, DEFAULT_TILE),
])
def test_active_task_window_is_bounded(before, after, cores, recipe):
    rates = dict(zip(TRAFFIC_ROLES, (32, FAST, 24, FAST, FAST)))
    a, b = (_run(_context(_op(*shape), cores, rates=rates), recipe)[0] for shape in (before, after))
    assert b.peak_tasks == a.peak_tasks and b.serviced["cube"] == pytest.approx(2 * math.prod(after))
    assert (b.peak_allocated == a.peak_allocated) if recipe is DEFAULT_TILE else (b.completed_count == a.completed_count)
@pytest.mark.parametrize("k,tile_k", [(2, 2), (6, 2), (6, 4)])
def test_compact_matrix_preserves_output_scale_work_and_dependencies(k, tile_k):
    _, ctx = quantized_context(k=k)
    for recipe in (replace(DEFAULT_TILE, tile_k=tile_k), SteadyMatmul(((4, 4, tile_k),), 2, 2),
                   PhaseMatmul(((4, 4, 2),), 2, 2)):
        engine, _ = _run(ctx, recipe, trace=100)
        assert engine.serviced == pytest.approx(dict(cube=60*k, vector=120, hbm=16*k+124, mte2=22*k+88, mte1=38*k, mte3=60))
    events = {row["task"]: row for row in engine.trace}
    for name in (name.removesuffix(":epilogue") for name in events if name.endswith(":epilogue")):
        assert events[name+":matrix"]["end_ms"] <= events[name+":epilogue"]["start_ms"]
        assert events[name+":epilogue"]["end_ms"] <= events[name+":store"]["start_ms"]
@pytest.mark.parametrize("cores", [1, 2])
def test_compact_residency_and_owner_contention(cores):
    ctx = _context(_op(4*cores, 2, 40), cores, {"l1": 48, "l0a": 16, "l0b": 16, "l0c": 32},
                   dict(zip(TRAFFIC_ROLES, (32*cores, FAST, 24*cores, FAST, FAST))))
    program = SteadyMatmul(((4, 2, 2),), 2, 2, owner_cohorts=cores == 1).lower(ctx)
    engine = _engine(ctx, 200)
    first = program.start(engine, "first", engine.submit("ready"))
    if cores == 1:
        later = program.start(engine, "later", engine.submit("ready", release=5))
        engine.advance(until=(first,))
        assert not later.done and first.time == pytest.approx(21, abs=1e-8)
        engine.advance(until=(later,))
        assert later.time == pytest.approx(42, abs=1e-8)
    else:
        noisy = engine.submit("external", {"cube": 640}, release=3, placement={"cube": (1,)})
        engine.advance(until=(first, noisy), first=False)
        stores = {r["placement"]["mte3"][0]: r["end_ms"] for r in engine.trace if r["demand"].get("mte3")}
        assert stores[0] == pytest.approx(21, abs=1e-8) and stores[1] > stores[0]
        assert engine.serviced["cube"] == pytest.approx(2*8*2*40 + 640)
        assert engine.serviced["hbm"] == pytest.approx(2 * (8*40 + 2*40 + 8*2))
    assert not any(engine.allocated.values())
def test_refill_reuse_capacity_and_owner_selection():
    small, before = matrix_run(HierarchicalMatmul(2, 2, 2, 2, 2))
    large, after = matrix_run(DEFAULT_TILE)
    assert small.serviced == dict(zip(TRAFFIC_ROLES, (768, 320, 768, 768, 128)))
    assert large.serviced == dict(small.serviced, mte2=384) and after < before
    serial, pre = matrix_run(replace(DEFAULT_TILE, slots=1), copy_rate=128)
    double, post = matrix_run(DEFAULT_TILE, copy_rate=128)
    assert serial.serviced == pytest.approx(double.serviced) and 768 / 128 < post < pre
    records = []
    for capacity, reads in ((16, 768), (64, 384), (256, 384)):
        engine, _ = matrix_run(TWO_CANDIDATES, records=records, pools={"l1": 256, "l0a": 32, "l0b": 32, "l0c": capacity})
        assert engine.serviced == pytest.approx(dict(large.serviced, mte2=reads))
def test_mapping_bounds_use_only_active_owners():
    _, ctx = matrix_setup(1, 8, 2, cores=4, copy_rate=4)
    chosen, report = TWO_CANDIDATES.select(ctx)
    assert chosen.tile_n == 2 and report["candidates"][0]["resource_bound_ms"] < report["candidates"][1]["resource_bound_ms"]
    assert _run(ctx, TWO_CANDIDATES)[0].serviced["mte2"] == 48
    ctx = _context(_op(3, 3, 2), rates=dict(zip(TRAFFIC_ROLES, (1e9, 1e9, 8, 8, 1e9))))
    chosen, report = TWO_CANDIDATES.select(ctx)
    assert [r["resource_bound_ms"] for r in report["candidates"]] == [7, 12] and chosen.tile_m == 2
def test_shared_memory_and_batched_domains():
    ctx = _context(_op(8, 8, 6), rates={"cube": 128, "hbm": 1024})
    ctx = replace(ctx, pool_names={}, pool_capacities={}, pool_units={})
    assert _run(ctx, DEFAULT_TILE)[0].serviced == {"cube": 768, "hbm": 320}
    batched, _ = _run(_context(_op(6, 3, 2, 3, phase="decode")), DEFAULT_TILE)
    assert batched.serviced == pytest.approx(dict(zip(TRAFFIC_ROLES, (72, 96, 60, 84, 36))))
@pytest.mark.parametrize("m,n,cores,groups", [(5, 7, 3, 1), (5, 7, 5, 1), (5, 7, 3, 2), (9, 5, 7, 3), (3, 8, 5, 4), (0, 7, 2, 1)])
def test_geometry_retains_useful_area_and_owner_counts(m, n, cores, groups):
    rows = (m,) * groups
    geometry = matrix_geometry(_context(_op(m*groups, n, 3, groups), cores), DEFAULT_TILE)
    assert geometry.owners == grouped_owner_tiles(rows, n, 4, 4, cores) == _aggregate(_owners(rows, n, 4, 4, cores))
    assert sum(row[3] for row in geometry.owners) == m * n * groups == geometry.useful_flops / 6
    assert geometry.output_tiles == sum(row[0] for row in geometry.owners)
@pytest.mark.parametrize("rows,n,k,cores,recipe", [
    ((1, 3), 4, 2, 1, DEFAULT_TILE), ((1, 3), 4, 2, 1, SteadyMatmul(((4, 4, 2),), 2, 2)),
    ((2, 5, 4, 3), 9, 3, 3, DEFAULT_TILE),
])
def test_unequal_owner_rectangles_preserve_group_tails(rows, n, k, cores, recipe):
    owners = _owners(rows, n, 4, 4, cores)
    assert grouped_owner_rectangles(rows, n, 4, 4, cores) == owners
    assert grouped_owner_tiles(rows, n, 4, 4, cores) == _aggregate(owners)
    ctx = _context(_op(sum(rows), n, k, len(rows)), cores)
    engine, _ = _run(ctx, recipe, group_rows=rows, **({"owner_tiles": owners} if cores == 3 else {}))
    assert engine.serviced["cube"] == 2 * sum(rows) * n * k
    assert engine.serviced["hbm"] == pytest.approx(2 * (sum(rows) * (k+n) + len(rows) * n * k))
    if cores == 1:
        assert [engine.serviced[key] for key in ("mte2", "mte1", "mte3")] == [48, 80, 32]
def test_owner_enumeration_ignores_empty_banks_and_bounds_work():
    assert grouped_owner_tiles((2, 0, 5, 7), 9, 4, 4, 3) == _aggregate(_owners((2, 0, 5, 7), 9, 4, 4, 3))
    with pytest.raises(ValueError, match="matrix_grouped_owner_enumeration_exceeds_bound"):
        grouped_owner_tiles((2, 5, 4, 3), 9, 1, 1, 3, max_tiles=1)
def test_public_geometry_and_occurrence_accounting():
    ctx = _context(_op(5, 7, 3))
    engine = _engine(ctx)
    fresh, shared, delayed = (simulate_matrix_operator(ctx, recipe=DEFAULT_TILE, engine=owner, release_ms=t)
                              for owner, t in ((engine, 3), (engine, 11), (None, 11)))
    assert matrix_geometry(ctx, DEFAULT_TILE).tile_counts == dict.fromkeys(("m", "n", "k"), 2)
    assert tuple(fresh.geometry[k] for k in ("logical_shape", "storage_shape", "issued_shape")) == ([5, 7, 3], [16, 16, 16], [8, 8, 4])
    assert fresh.geometry["macro_tile_envelope"] == 512 > fresh.serviced["cube"] == 2 * 5 * 7 * 3 and fresh.serviced["hbm"] == pytest.approx(2 * (5 * 3 + 7 * 3 + 5 * 7))
    assert shared.serviced == pytest.approx(fresh.serviced) and shared.duration_ms == delayed.duration_ms == pytest.approx(fresh.duration_ms)
    assert fresh.peak_allocated and shared.peak_allocated is None and shared.completed_tasks == fresh.completed_tasks
    mapped = MappedMatmul(((4, 4, 2),), 2, 2)
    mapped.select(ctx)
    _, row = mapped.select(replace(ctx, op=replace(ctx.op, padded_shape=(8, 8, 8), semantic_shape=(9, 7, 3))))
    assert (row["geometry"]["storage_shape"], row["geometry"]["logical_shape"]) == ([8, 8, 8], [9, 7, 3])
def test_public_entry_checks_idle_frontier_and_respects_existing_clock():
    ctx = _context(_op(4, 8, 6))
    engine = _engine(ctx)
    queued = engine.submit("queued", release=100)
    before = engine.summary()
    with pytest.raises(ExecutionError, match="matrix_operator_requires_idle_engine_frontier"):
        simulate_matrix_operator(ctx, recipe=DEFAULT_TILE, engine=engine)
    assert engine.summary() == before
    direct = tuple(DEFAULT_TILE.lower(ctx).start(engine, name, engine.submit(name)) for name in ("a", "b"))
    assert len(engine.advance(until=(queued, *direct), first=False)) == 3
    assert simulate_matrix_operator(ctx, recipe=DEFAULT_TILE, engine=engine).ready_ms == max(e.time for e in (queued, *direct))
def test_equal_shared_memory_bounds_keep_a_single_owner_wave():
    pools = {"l1": ROOMY, "l0a": 128*1024, "l0b": 128*1024, "l0c": 512*1024}
    recipe, spans = SteadyMatmul(underfilled_candidates=UNDERFILLED_CANDIDATES), []
    for factor in (0.5, 1, 2):
        ctx = _context(_op(258, 33, 4096, 2), 32, pools, dict(cube=39e6, hbm=34500, mte2_aic=802000,
                       mte2_aiv=1, mte1=192000*factor, mte3=102000, vector=1e9))
        _, report = recipe.select(ctx)
        tile = report["selected"]
        record = next(row for row in report["candidates"] if row["split_k"] == tile["k_splits"]
                      and row["tile"] == [tile[key] for key in ("tile_m", "tile_n", "tile_k")])
        assert record["maximum_owner_output_tiles"] == 1 and record["split_k"] == 1
        engine, span = _run(ctx, recipe)
        assert engine.serviced["hbm"] == pytest.approx(2 * (258*4096 + 2*33*4096 + 258*33))
        spans.append(span)
    assert spans[0] > spans[1] and spans[1] == pytest.approx(spans[2], rel=0.02)
@pytest.mark.parametrize("hardware", SKUS)
def test_matrix_service_tracks_profile_precision_and_capacity(hardware):
    device = json.loads((REPO_ROOT / "mindie_sim/backends/profiles/ascend.json").read_text())[hardware]["device"]
    op, peak, cores = _op(512, 1024, 256), device["compute_peaks"]["bf16"], device["compute_units"]
    rates = {name: value / 1000 for name, value in device["rates"].items()}
    names = {name: name for name in ("cube", "hbm", "mte2_aic", "mte2_aiv", "mte1", "mte3", "vector")}
    ctx = RecipeContext(op, {}, names, rates, cores, pool_names={r: r for r in device["buffers"]},
                        pool_capacities=device["buffers"], pool_units=device["buffer_units"], resource_units=device["resource_units"])
    work = Work(matrix={"bf16": op.flops}, resources={"hbm": op.hbm_bytes, "mte2": op.mte2_bytes, "mte3": op.mte3_bytes})
    results = []
    for factor in (1, 2):
        ctx = replace(ctx, compute_peaks={"bf16": peak / factor})
        result = simulate_matrix_operator(replace(ctx, work=ctx.bind(work), op=work.apply(op)), recipe=HierarchicalMatmul(*(128,) * 5))
        assert result.serviced["cube"] == pytest.approx(1000 * factor * 2 * 512 * 1024 * 256 / peak)
        results.append(result.duration_ms)
    assert results[1] > results[0]
def test_phase_service_converts_native_units_once():
    ctx = _context(_op(8, 8, 6), rates=dict(cube=128, hbm=1024, mte2=16, mte1=1024, mte3=1024, vector=1000))
    work = Work(resources={"ub_copy": 1000})
    identity, copy = {"kernel": "copy"}, {"ub_copy": ("vector", 2e6)}
    phase = replace(ctx, service_modes=copy, service_identity=identity, work_units={"ub_copy": "bytes"}).for_phase(work, kernel="copy")
    assert physical_owner_rate(ctx, "cube", (0,)) == 64 and physical_owner_rate(ctx, "hbm", (0,)) == 1024
    assert phase.work == resolved_phase_work(phase, work) == {"vector": 500}
    with pytest.raises(ValueError, match="service_work_does_not_match_phase_context"):
        resolved_phase_work(phase, Work(resources={"ub_copy": 2000}))
    curve = ServiceCurve("ub_copy", "bytes", identity, "resource_phase", 2, 0.25, ((500, 1), (2000, 3.5)))
    assert _run(phase, ResourceService(curve))[0].serviced["vector"] == 500

def _tiled(op, recipe, cores=2, pools=None, copy=16, store=4096, trace=0):
    rates = {"cube": 512, "hbm": 4096, "mte2": copy, "mte3": store, "vector": 4096}
    ctx = _context(op, cores, pools or {"l1": ROOMY, "l0c": ROOMY}, rates)
    return _run(replace(ctx, work=dict(ctx.work, hbm=op.hbm_bytes, mte2=op.mte2_bytes, mte3=op.mte3_bytes)), recipe, trace=trace)
def _tiled_oracle(op, recipe):
    total, n, k = op.local_shape
    groups = dict(op.metadata).get("matrix_groups", 1)
    tm, tn, tk = recipe.tile_m, recipe.tile_n, recipe.tile_k
    a, b, c = 2 * total * k, 2 * groups * n * k, 2 * total * n
    copy_a, copy_b = a * math.ceil(n / tn), b * math.ceil(total / groups / tm)
    expected = {"cube": 2 * total * n * k, "vector": op.vector_ops, "mte3": c, "mte2": copy_a + copy_b,
                "hbm": (a if recipe.reuse_inputs else copy_a) + (b if recipe.reuse_weights else copy_b) + c}
    rows, cols = min(total // groups, tm), min(n, tn)
    return expected, 2 * (rows + cols) * min(k, tk), 4 * rows * cols
@pytest.mark.parametrize("total,groups,n,k", [(5, 1, 4, 3), (12, 3, 7, 5), (20, 4, 5, 3), (7, 7, 5, 5)])
@pytest.mark.parametrize("cores", [1, 3, 5])
@pytest.mark.parametrize("reuse_a,reuse_b", tuple(product((False, True), repeat=2)))
def test_tiled_group_traffic_and_exact_private_allocations(total, groups, n, k, cores, reuse_a, reuse_b):
    op = replace(_op(total, n, k, groups), vector_ops=2 * total * n)
    tile = TiledMatmul(2, 3, 2, slots=1, reuse_inputs=reuse_a, reuse_weights=reuse_b)
    expected, panel, accumulator = _tiled_oracle(op, tile)
    engine, _ = _tiled(op, tile, cores)
    assert engine.serviced == pytest.approx(expected, rel=1e-12, abs=1e-9)
    peaks = engine.peak_unit_allocated
    assert [max(v for (pool, _), v in peaks.items() if pool == role) for role in ("l1", "l0c")] == [panel, accumulator]
@pytest.mark.parametrize("groups", [4, 0, -3])
def test_invalid_group_row_domains_fail_before_launch(groups):
    op = _op(10, 4, 2, 4)
    op = replace(op, metadata=tuple(dict(op.metadata, matrix_groups=groups).items()))
    with pytest.raises(ValueError, match="matrix_groups_require_equal_integer_rows"):
        TiledMatmul().lower(_context(op))
def test_tiled_group_count_empty_and_implicit_group():
    tile = TiledMatmul(4, 4, 4)
    a, b = (_tiled(_op(4*g, 4, 4, g), tile)[0] for g in (4, 16))
    assert b.peak_tasks == a.peak_tasks and b.peak_allocated == a.peak_allocated
    assert [b.serviced[r] for r in ("cube", "mte2")] == [4 * a.serviced[r] for r in ("cube", "mte2")]
    op = _op(8, 6, 4)
    bare = replace(op, metadata=tuple((k, v) for k, v in op.metadata if k != "matrix_groups"))
    assert _tiled(bare, tile, 3)[0].serviced == pytest.approx(_tiled_oracle(op, tile)[0])
    assert not any(_tiled(_op(0, 4, 3, 2), TiledMatmul())[0].serviced.values())
def test_tiled_aggregate_experts_and_packed_panel_capacity():
    op = _op(16, 4, 4)
    meta = dict(op.metadata, packed_weight_bytes=4096, matrix_mapping_domain="aggregate_experts")
    aggregate = replace(op, metadata=tuple(meta.items()), mte2_bytes=4224, hbm_bytes=4352)
    engine, _ = _tiled(aggregate, TiledMatmul(4, 4, 2, slots=1), pools={"l1": 32, "l0c": 64})
    assert (engine.serviced["hbm"], engine.serviced["mte2"], engine.peak_unit_allocated["l1", 0]) == (4352, 128 + 4 * 4096, 32)
    op = _op(6, 4, 6, 3, quant=quant_preset("w8a8_int8"))
    engine, _ = _tiled(op, TiledMatmul(2, 4, 3, slots=1), 3)
    assert (engine.peak_unit_allocated["l1", 0], engine.serviced["vector"]) == (3 * (2 + 4), 4 * 6 * 4)
    assert engine.serviced["hbm"] == pytest.approx(6*6 + 3*4*6 + 4*3*4 + 2*6*4)
@pytest.mark.parametrize("shape,groups,tile", [
    ((9, 7, 5), 1, TiledMatmul(4, 4, 3, reuse_inputs=False, reuse_weights=False)),
    ((16, 4, 8), 4, TiledMatmul(4, 4, 4)), ((8, 4, 4), 2, TiledMatmul(4, 4, 2)),
])
def test_tiled_panel_and_accumulator_residency(shape, groups, tile):
    op = _op(*shape, groups)
    expected, panel, acc = _tiled_oracle(op, tile)
    if groups == 2:
        engine, _ = _tiled(op, tile, 1, pools={"l1": ROOMY, "l0c": acc}, copy=ROOMY, store=2, trace=5000)
        trace = {row["task"].split(":tile")[-1]: row for row in engine.trace}
        assert engine.peak_unit_allocated["l0c", 0] == acc
        assert trace["2:compute"]["start_ms"] >= trace["1:store"]["end_ms"] - 1e-9
    else:
        _, before = _tiled(op, tile)
        engine, after = _tiled(op, tile, pools={"l1": panel, "l0c": ROOMY})
        assert engine.serviced == pytest.approx(expected) and after > before
        assert max(engine.peak_unit_allocated["l1", i] for i in range(2)) == panel
def test_frontend_grouped_mla_traffic_and_source_binding():
    raw = json.loads((REPO_ROOT / "examples/source-families/indexed-mla-a3-control.json").read_text())
    raw["machine"]["num_devices"] = raw["machine"]["ranks_per_node"] = raw["runtime"]["tp"] = 1
    raw["model"]["num_heads"] = 16
    raw["model"]["quantization"] = "bf16"
    raw["runtime"]["kernels"]["programs"] = {"matmul": {"kind": "tiled_matmul"}}
    ops = target_graphs(raw, apply_body_defaults=False, phase="decode", tokens=1, context=512, sequences=1, scope="model")[0]
    rows = {row["operator"]["op_id"]: row for row in operator_costs(raw, ops)["operators"]}
    grouped = [op for op in ops if op.op_id.endswith(("q_absorb", "v_expand"))]
    assert len(grouped) == 2
    for op in grouped:
        assert dict(op.metadata)["matrix_groups"] == 16 and rows[op.op_id]["program"] == "TiledMatmul"
        expected = _tiled_oracle(op, TiledMatmul())[0]
        for logical, physical in (("hbm", "memory"), ("mte2", "mte2_aic"), ("mte3", "fixpipe")):
            assert rows[op.op_id]["serviced"][physical + "@r0:s0"] == pytest.approx(expected[logical])
    assert {(row["operator"]["recipe"], row["program"]) for row in rows.values()
            if row["operator"].get("source_boundary") == "kernel"} == {("ascend_sparse_flash_attention_arch22", "OperatorPlan")}
    case = parse_case(raw)
    execution = Execution(case, CostModel(case), trace_limit=5000)
    op, recipe = execution.prepare(next(op for op in grouped if op.op_id.endswith("v_expand")))
    engine = execution.engine
    program = recipe.lower(execution._context(op, 0, "none", recipe=recipe))
    engine.advance(until=(program.start(engine, "mla", engine.submit("ready", release=7)),))
    compute = [r for r in engine.trace if r["demand"].get("matrix@r0:s0") and r["task"].endswith(":compute")]
    assert len({owner for row in compute for owner in row["placement"]["matrix@r0:s0"]}) == case.machine.dies[0].compute_units
    assert min(r["start_ms"] for r in compute) >= 7

def _price(m=128, n=288, k=4096, *, groups=1, options=None, scale=1, dtype="fp32", primitive_table=None):
    from mindie_sim.prediction.evaluation import EvaluationContext
    raw = json.loads((REPO_ROOT / "examples/quickstart.json").read_text())
    raw["machine"] = dict(backend="ascend", sku=OWNER_GRID_PROFILE, num_devices=1, device={"compute_peaks": {"fp32": FP32_PEAK * scale}})
    raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul", **(options or {})}}}
    op = _op(m, n, k, groups, deps=("input",), phase="decode",
             quant=QuantSpec(dtype, weight_format=dtype, activation_format=dtype, weight_block=(128, 128)))
    ev = EvaluationContext.create(parse_case(raw), primitive_table=primitive_table) if primitive_table is not None else None
    result = graph_costs(raw, (op,), external_inputs={"input": 7}, isolated=False, evaluation=ev)
    assert all(row["ready_ms"] >= 7 for row in result["timeline"])
    return result["duration_ms"], result["execution"]["serviced"], result["parameters"]["matrix_mappings"][0]
@pytest.mark.parametrize("m,n,dtype", [(128, 288, "fp32"), *product((64,), (24, 288), ("bf16", "fp8_e4m3", "int8"))])
def test_owner_grid_subtiles_and_explicit_domain(m, n, dtype):
    span, work, report = _price(m, n, dtype=dtype)
    assert _price(m, n, dtype=dtype, options={"candidates": DEFAULT_CANDIDATES})[:2] == (span, work)
    tile = report["selected"]
    if dtype != "fp32":
        if n == 288:
            # Dense BF16 admits Stream-K (128x128 k_splits=8); block-quantized FP8/INT8
            # stay non-Stream-K (k_splits=1) where 128x256 maximizes sector efficiency.
            assert (tile["tile_m"], tile["tile_n"]) == ((128, 128) if dtype == "bf16" else (128, 256))
            assert tile["k_splits"] == (8 if dtype == "bf16" else 1)
        else:
            assert (tile["tile_m"], tile["tile_n"]) == ((16, 32) if dtype == "bf16" else (16, 16))
            assert tile["k_splits"] == (8 if dtype == "bf16" else 1)
        if dtype == "bf16" or n == 24:
            assert _price(m, n, dtype=dtype, options={"underfilled_candidates": UNDERFILLED_CANDIDATES})[:2] == (span, work)
    else:
        control, coarse, coarse_report = _price(options={"underfilled_candidates": ()})
        assert span <= control and len(owner_tiles(m, n, tile["tile_m"], tile["tile_n"], 32)) * tile["k_splits"] >= 16
        partial = tile["k_splits"] * 4 * m * n
        assert work["memory@r0:s0"] == pytest.approx(6_963_200) and coarse["memory@r0:s0"] == pytest.approx(6_963_200)
        assert work["l2@r0:s0"] == pytest.approx(2 * partial) and work["fixpipe@r0:s0"] == pytest.approx(partial)
        # Tile-grid MTE2 traffic scales with output-tile operand reuse (ceil(N/tile_n)*A + ceil(M/tile_m)*B).
        for metrics, sel in ((work, tile), (coarse, coarse_report["selected"])):
            expected_mte2 = math.ceil(n / sel["tile_n"]) * (4 * m * 4096) + math.ceil(m / sel["tile_m"]) * (4 * n * 4096)
            assert metrics["mte2_aic@r0:s0"] == pytest.approx(expected_mte2)
def test_full_grid_and_compute_sweep_preserve_logical_work():
    assert _price(1024, 4096, 128)[:2] == _price(1024, 4096, 128, options={"underfilled_candidates": ()})[:2]
    _, work, mapping = _price(57, 145, 319, groups=3, primitive_table=False)
    assert mapping["groups"] == 3 and work["memory@r0:s0"] == pytest.approx(4 * (57*319 + 3*145*319 + 57*145))
    assert work["matrix@r0:s0"] == pytest.approx(2 * 57 * 145 * 319 / FP32_PEAK * 1000)
    fast, slow = (_price(scale=scale) for scale in (2, 0.5))
    assert fast[0] < slow[0] and slow[1]["matrix@r0:s0"] == pytest.approx(4 * fast[1]["matrix@r0:s0"])

@pytest.mark.parametrize("hardware", SKUS)
def test_small_matrix_service_domain_and_memory_supply(hardware):
    from mindie_sim.prediction.evaluation import EvaluationContext
    path = "examples/cases/high-concurrency-20261030/kimi-source-complete/a5_kimi_tp8_e16_bf16_c32_p4k_source_complete.json"
    raw = json.loads((REPO_ROOT / path).read_text())
    native = hardware == GENERATION_PROFILE
    raw["model"]["layers"], raw["machine"]["sku"] = 2, OWNER_GRID_PROFILE if native else hardware
    regimes = json.loads((REPO_ROOT / "examples/service-regimes/a5-small-m-bf16-matmul-20261030.json").read_text())
    for batch, rate, span in ((32, 2e12, 0.054272), (32, 4e12, 0.027136), (32, 8e12, "physical"),
                              (32, 1e9, None), (40, 4e12, None)):
        ops = target_graphs(raw, scope="body", phase="decode", tokens=1, context=4104, sequences=batch)[0]
        op = replace(next(op for op in ops if op.op_id == "s0.l1.latent_down"), op_id="independent_matrix", deps=(), layer=-1)
        raw["machine"]["device"] = {"hbm_bw_bytes_per_s": rate}
        raw["runtime"]["kernels"].update(regimes)
        ev_reg = EvaluationContext.create(parse_case(raw), primitive_table=False)
        result = graph_costs(raw, (op,), input_ready_ms=7, evaluation=ev_reg)
        raw["runtime"]["kernels"].pop("service_regimes")
        actual, work = result["duration_ms"], result["execution"]["work"]
        assert (work["hbm_bytes"], work["flops"], result["complete_ms"]) == (op.hbm_bytes, op.flops, pytest.approx(7 + actual))
        if native and batch == 32 and span == "physical":
            ev_plain = EvaluationContext.create(parse_case(raw), primitive_table=False)
            assert actual == pytest.approx(graph_costs(raw, (op,), input_ready_ms=7, evaluation=ev_plain)["duration_ms"])
            assert 0.027136 / 2 < actual < 0.027136
        elif native and batch == 32:
            assert actual == pytest.approx(span) if span else actual > 0.027136
        else:
            ev_plain = EvaluationContext.create(parse_case(raw), primitive_table=False)
            assert actual == graph_costs(raw, (op,), input_ready_ms=7, evaluation=ev_plain)["duration_ms"]
@pytest.mark.parametrize("capture,count,gap,duration,blocks", [
    ("bf16_matmulv3_m1_k64_n64.csv", 4, 69.581, 8.091, ("1", "0")),
    ("int8_quantbatchmatmulv3_m1_k512_n512.csv", 3, 0.022, 5.795 + 6.679 + 6.151, ("32", "64")),
])
def test_tiny_cube_capture_separates_device_work_and_submission(capture, count, gap, duration, blocks):
    report = analyze_window(fixture_path("a5_tiny_cube_20261030", capture), count)
    stream, desc = report["streams"][0], report["kernel_descriptors"]
    assert (report["physical_rows"], stream["first_transition_gap_us"], stream["duration_sum_us"]) == pytest.approx((count, gap, duration))
    assert (desc["Block Num"][0], desc["Mix Block Num"][0]) == blocks
    if count == 4:
        assert report["device_kernel_duration"]["min_us"] == 1.691 and stream["first_start_to_last_end_us"] == 92.107
    else:
        assert (stream["negative_gap_pairs"], stream["legacy_float_negative_gap_pairs"], desc["Accelerator Core"]) == (0, 1, ["MIX_AIC"])

def _mixed_op():
    return OpSpec("mixed", "fused", "prefill", 0, 0, ("r",), *((8, 8),) * 4,
                  flops=999, hbm_bytes=999, dtype="bf16", recipe="two_products",
                  metadata=(("resource_work", {"vector": 999}),))
def _mixed_plan(op, geometry):
    phases = (Work(resources={"hbm": 1000, "mte2": 1000}), Work(matrix={"bf16": 2000}),
              Work(matrix={"fp32": 2000}), Work(resources={"hbm": 1000, "mte3": 1000}))
    def lower(ctx):
        stages = tuple(Stage(name, ctx.bind(work), after=deps) for name, work, deps in
                       zip(("load", "bf16", "fp32", "store"), phases, ((), ("load",), ("load",), ("bf16", "fp32"))))
        for role, amount in ctx.bind(sum(phases, Work())).items():
            assert sum(stage.work.get(role, 0) for stage in stages) == amount
        return KernelProgram(stages)
    return OperatorPlan(sum(phases, Work()), lower)
def _children(op):
    return replace(op, op_id="first", deps=()), replace(op, op_id="second", deps=("first",))
def _iterate(case, book, *ops, scope="none"):
    execution = Execution(case, CostModel(case), recipes=book)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    result = execution.add(IterationWork(ops, plan, 1, 1, 1, None, "prefill", "one", replay_scope=scope), 0, 0)
    execution.engine.advance(until=(result.completion,))
    return execution, result.completion.time
def test_planned_precision_work_is_shared_between_public_entries():
    raw, op = hardware_case(), _mixed_op()
    case, book = parse_case(raw), RecipeBook({op.recipe: PlannedRecipe(_mixed_plan)})
    execution, span = _iterate(case, book, op)
    row = operator_costs(case, (op,), recipes=book)["operators"][0]
    assert (span, row["duration_ms"], execution.work["flops"]) == (5, 5, 4000)
    assert row["declared_service"] == row["serviced"] == {"matrix@r0:s0": 3, "memory@r0:s0": 2000}
    peaks = raw["machine"]["device"]["compute_peaks"]
    peaks["fp32"] = 2e6
    faster = verify_operator(raw, op, _mixed_plan)["result"]
    assert (faster["duration_ms"], faster["operator"]) == (4, row["operator"])
    del peaks["fp32"]
    with pytest.raises(ValueError, match="compute_precision_rate_required:fp32"):
        verify_operator(raw, op, _mixed_plan)
def test_source_kernel_expansion_preserves_launches_and_child_work():
    raw, op = hardware_case(), _mixed_op()
    raw["machine"]["device"]["launch_us"] = 50
    case, book = parse_case(raw), RecipeBook({op.recipe: PlannedRecipe(_mixed_plan, _children), "direct": PlannedRecipe(_mixed_plan)})
    report = operator_costs(case, (op,), recipes=book)
    assert (report["operator_occurrences"], report["operator_sum_ms"]) == (2, 10)
    consumer = replace(op, op_id="consumer", deps=(op.op_id,), recipe="direct")
    for scope, launches in (("none", 3), ("full", 1)):
        execution, span = _iterate(case, book, op, consumer, scope=scope)
        assert execution.engine.serviced["host@r0:s0"] == pytest.approx(launches * 0.05)
        assert span == pytest.approx(15.05) and execution.work["flops"] == 12000
    assert verify_operator(case, op, _mixed_plan, kernels=_children)["status"] == "passed"
def test_work_verification_detects_missing_workspace_and_opposite_child_errors():
    def missing(op, geometry):
        plan = _mixed_plan(op, geometry)
        return replace(plan, work=plan.work + Work(resources={"hbm": 128}))
    def distinct(op):
        return tuple(replace(child, metadata=(("part", index),)) for index, child in enumerate(_children(op), 1))
    def cancelled(op, geometry):
        amount = 0 if dict(op.metadata)["part"] == 1 else 2000
        return OperatorPlan(Work(resources={"hbm": 1000}), lambda ctx: KernelProgram((
            Stage("copy", ctx.bind(Work(resources={"hbm": amount}))),)))
    raw, op = hardware_case(), _mixed_op()
    single, pair = verify_operator(raw, op, missing), verify_operator(raw, op, cancelled, kernels=distinct)
    assert single["finite_completion"] and single["status"] == pair["status"] == "failed"
    assert single["work_mismatches"] == {"memory@r0:s0": {"declared": 2128, "executed": 2000}}
    assert pair["result"]["declared_service"] == pair["result"]["serviced"]
    assert pair["work_mismatches"] == {f"mixed/{name}": {"memory@r0:s0": {"declared": 1000, "executed": amount}}
                                       for name, amount in (("first", 0), ("second", 2000))}
def test_work_verification_detects_prefetch_before_input():
    class EarlyPrefetch:
        def start(self, engine, name, ready, *, owner=""):
            early = engine.submit(name + ":load", {"memory@r0:s0": 1000})
            return engine.submit(name + ":done", after=(early, ready))
    def build(op, geometry):
        return OperatorPlan(Work(resources={"hbm": 1000}), lambda _: EarlyPrefetch())
    raw, op = hardware_case(), _mixed_op()
    assert verify_operator(raw, op, build)["status"] == "passed"
    rejected = verify_operator(raw, op, build, input_ready_ms=2)
    assert rejected["finite_completion"] and not rejected["work_mismatches"] and rejected["status"] == "failed"
    assert rejected["result"]["service_before_input"] == {"memory@r0:s0": 1000}
    assert verify_operator(raw, op, _mixed_plan, input_ready_ms=2)["result"]["duration_ms"] == 5
@pytest.mark.parametrize("units,explicit", tuple(product((1, 4, 8), (False, True))))
def test_instruction_modes_share_physical_capacity_and_preserve_byte_counts(units, explicit):
    raw = hardware_case()
    device = raw["machine"]["device"]
    device["compute_units"], device["rates"]["copy"] = units, units * 1e6
    device["service_modes"] = {"copy_strided": {"resource": "copy", "rate": units * 0.5e6}}
    work = Work(resources={"copy": 1000, "copy_strided": 1000})
    def build(op, geo):
        def lower(ctx):
            owner = ctx.placement("copy_strided", (0,)) if explicit else ctx.owned(0, geo.compute_units, ("copy_strided",))
            return KernelProgram(tuple(Stage(role, ctx.bind(Work(resources={role: 1000})), placement=owner) for role in work.resources))
        return OperatorPlan(work, lower)
    slow = verify_operator(raw, _mixed_op(), build, input_ready_ms=7)["result"]
    device["service_modes"]["copy_strided"]["rate"] *= 2
    fast = verify_operator(raw, _mixed_op(), build)["result"]
    assert (slow["duration_ms"], fast["duration_ms"]) == pytest.approx((3, 2))
    assert slow["operator"] == fast["operator"] and slow["operator"]["metadata"]["resource_work"] == work.resources
def test_representative_rank_keeps_rate_and_owner_count_together():
    raw = hardware_case()
    raw["model"].update(hidden_size=16, num_heads=2, num_kv_heads=2)
    raw["runtime"]["tp"] = raw["machine"]["num_devices"] = 2
    def device(units, rate):
        return {"compute_units": units, "rates": {"memory": 1e9, "vector": 1e9, "copy": rate},
                "service_modes": {"slow": {"resource": "copy", "rate": rate}}}
    raw["machine"]["devices"] = [device(1, 1e6), device(4, 2e6)]
    work = Work(resources={"slow": 1000})
    def build(op, geo):
        return OperatorPlan(work, lambda ctx: KernelProgram((
            Stage("copy", ctx.bind(work), placement=ctx.owned(0, geo.compute_units, ("slow",))),)))
    with pytest.raises(ValueError, match="heterogeneous_owners_require_per_rank_execution"):
        verify_operator(raw, _mixed_op(), build)
    raw["machine"]["devices"][0] = device(4, 4e6)
    assert verify_operator(raw, _mixed_op(), build)["result"]["duration_ms"] == 2


def test_contiguous_narrow_matrix_panel_avoids_strided_subsector_penalty():
    from types import SimpleNamespace
    from mindie_sim.prediction.operators.matrix.matrix_mapping_owners import primitive_cold_hbm_bps

    p01_metrics = {"hbm_read_bytes_per_s": SimpleNamespace(p50=3.6e12, certified_upper_capacity=4.0e12)}
    p01 = SimpleNamespace(
        supported=True, modes={}, mode_status=lambda _m=None: "calibrated",
        band=lambda m, mode=None: p01_metrics.get(m),
        metrics=p01_metrics,
        parameters={"mte2_aic_cold_scaling_bps": {"1": 150e9, "8": 1.2e12, "32": 3.6e12}},
    )
    p05 = SimpleNamespace(
        supported=True, modes={}, mode_status=lambda _m=None: "calibrated",
        band=lambda _m, mode=None: None,
        metrics={}, parameters={"l2_sector_bytes": 128},
    )
    pt = SimpleNamespace(cell=lambda cid: {"P01_datacopy": p01, "P05_l2_residency_reuse": p05}.get(cid))
    ctx = SimpleNamespace(
        primitive_table=pt, core_count=32, op=SimpleNamespace(local_shape=(4, 16, 7168)),
        compute_arch=SimpleNamespace(arch_id="ascend_950"),
    )

    # Dense contiguous BF16 N=16 (row_bytes == stride_bytes == 32B) across K=7168 rows coalesces
    # across 128B sectors and must NOT take a 32/128 = 0.25x strided sub-sector penalty.
    dense_bps = primitive_cold_hbm_bps(
        ctx, active_cores=8, row_bytes=32, stride_bytes=32,
        total_tiles=8, is_grouped=False, panel_rows=7168,
    )
    wide_bps = primitive_cold_hbm_bps(
        ctx, active_cores=8, row_bytes=256, stride_bytes=256,
        total_tiles=8, is_grouped=False, panel_rows=7168,
    )
    assert dense_bps == pytest.approx(wide_bps)

    # Short contiguous panel (2 rows * 32B = 64B inside one 128B sector) matches exact sector union 64/128.
    short_bps = primitive_cold_hbm_bps(
        ctx, active_cores=8, row_bytes=32, stride_bytes=32,
        total_tiles=8, is_grouped=False, panel_rows=2,
    )
    assert short_bps == pytest.approx(0.5 * wide_bps)

    # Truly strided narrow slice (row_bytes=32 < stride_bytes=256) retains the 32/128 sub-sector penalty.
    strided_bps = primitive_cold_hbm_bps(
        ctx, active_cores=8, row_bytes=32, stride_bytes=256,
        total_tiles=8, is_grouped=False, panel_rows=7168,
    )
    assert strided_bps == pytest.approx(0.25 * wide_bps)

    # Narrow strided slice whose stride is smaller than a sector shares sectors across rows:
    # 32 B every 64 B across 128 rows gives exact 1/2 sector efficiency on a 512 B sector (not 1/16).
    p05_512 = SimpleNamespace(
        supported=True, modes={}, mode_status=lambda _m=None: "calibrated",
        band=lambda _m, mode=None: None, metrics={}, parameters={"l2_sector_bytes": 512},
    )
    ctx_512 = SimpleNamespace(
        primitive_table=SimpleNamespace(
            cell=lambda cid: {"P01_datacopy": p01, "P05_l2_residency_reuse": p05_512}.get(cid),
        ),
        core_count=32, op=SimpleNamespace(local_shape=(4, 32, 128)),
        compute_arch=SimpleNamespace(arch_id="ascend_910b"),
    )
    shared_sector_bps = primitive_cold_hbm_bps(
        ctx_512, active_cores=8, row_bytes=32, stride_bytes=64,
        total_tiles=2, is_grouped=False, panel_rows=128,
    )
    full_512_bps = primitive_cold_hbm_bps(
        ctx_512, active_cores=8, row_bytes=512, stride_bytes=512,
        total_tiles=2, is_grouped=False, panel_rows=128,
    )
    assert shared_sector_bps == pytest.approx(0.5 * full_512_bps)


def test_splitk_charges_unreused_hbm_operands_and_reserves_operand_buffers():
    base = json.loads((REPO_ROOT / "examples/calls/quantized-grouped-small.json").read_text())
    m, n, k = 64, 288, 4096
    op = matrix_setup(m, n, k)[1].op
    for reuse_inputs, reuse_weights, expected_hbm in (
        (False, False, 2 * (5 * m * k + n * k + m * n)),
        (True, True, 2 * (m * k + n * k + m * n)),
    ):
        case = json.loads(json.dumps(base))
        case["runtime"]["kernels"] = {"programs": {"matmul": dict(
            kind="steady_matmul", candidates=[[64, 64, 128]],
            underfilled_candidates=[], narrow_candidates=[],
            reuse_inputs=reuse_inputs, reuse_weights=reuse_weights,
        )}}
        report = graph_costs(case, (op,), isolated=False)
        mapping = report["parameters"]["matrix_mappings"][0]
        assert mapping["selected"]["k_splits"] == 6
        assert mapping["split_k_traffic"]["workspace_residency"] == "l2"
        assert report["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(expected_hbm)
        assert mapping["split_k_traffic"]["issued_hbm_work"] == pytest.approx(expected_hbm)
        declared_fp = mapping["candidates"][0]["footprint_bytes_per_owner"]
        for pool in ("l1", "l0a", "l0b", "l0c"):
            assert report["execution"]["peak_allocated"][pool + "@r0:s0"] == pytest.approx(declared_fp[pool])
        assert report["execution"]["live_tasks"] == 0

    # Admitted L2 read hit credit must be applied strictly once even when op.hbm_bytes is pre-discounted.
    half_reads = float(m * k + n * k)
    op_warm = replace(
        op, hbm_bytes=float(op.hbm_bytes) - half_reads,
        metadata=(*op.metadata, ("admitted_l2_read_hit_bytes", half_reads)),
    )
    case_warm = json.loads(json.dumps(base))
    case_warm["runtime"]["kernels"] = {"programs": {"matmul": dict(
        kind="steady_matmul", candidates=[[64, 64, 128]],
        underfilled_candidates=[], narrow_candidates=[],
    )}}
    rep_warm = graph_costs(case_warm, (op_warm,), isolated=False)
    expected_warm_hbm = float(op.hbm_bytes) - half_reads
    assert rep_warm["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(expected_warm_hbm)

    # Extra independent HBM traffic on top of a pre-discounted admitted L2 hit must be conserved.
    extra_hbm = 262_144.0
    op_warm_extra = replace(op_warm, hbm_bytes=float(op_warm.hbm_bytes) + extra_hbm)
    rep_warm_extra = graph_costs(case_warm, (op_warm_extra,), isolated=False)
    assert rep_warm_extra["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(
        expected_warm_hbm + extra_hbm
    )
