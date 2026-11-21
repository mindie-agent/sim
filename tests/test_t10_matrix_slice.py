"""T10 matrix geometry, shared-phase capacity and public operator entry.

[Role]
  Establish the common logical/local/storage/issued geometry, finite owner
  tails, real Cube/Vector/MTE capacity sharing, one-time service conversion,
  grouped owner reuse and the isolated matrix-operator entry.

[Invariants]
  * Logical, local, storage and issued extents are distinct values.
  * Owner records conserve output area and every useful tail.
  * A service mode is converted to its physical pipe once.
  * The public entry returns finite work, buffers and completion without
    reading a target duration, model name or observation.
  * A5 BF16/FP32 matrix service comes from the declared profile peaks.

[Timeouts & Locks]
  Pure arithmetic and bounded Engine runs; no device, weights or network.

[Side-Effect Policy]
  Reads the checked-in A5 profile.  Writes nothing.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul
from mindie_sim.prediction.operators.matrix.matrix_mapping import (
    MappedMatmul, MatrixGeometry, grouped_owner_rectangles, grouped_owner_tiles,
    matrix_geometry, owner_tiles, physical_owner_rate, simulate_matrix_operator,
)
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.services import (
    ResourceService, ServiceCurve, resolved_phase_work,
)

ROOT = Path(__file__).resolve().parents[1]
A5 = "Ascend950DT_9581"


def _op(m=8, n=8, k=6, *, groups=1, total_m=None, padded=True):
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                 m=(m * groups if total_m is None else total_m), n=n, k=k,
                 weight_count=groups,
                 quant=QuantSpec("bf16"), category="dense_mlp", phase="prefill")
    if not padded:
        op = replace(op, padded_shape=tuple(op.local_shape[-3:]))
    return op


def _context(op, *, cores=2, pools=None, rates=None):
    base_rates = {"cube": 128, "hbm": 1024, "mte2": 16, "mte1": 1024, "mte3": 1024}
    if rates:
        base_rates.update(rates)
    rates = {name: value for name, value in base_rates.items()
             if name != "vector" or op.vector_ops}
    units = {name: (1 if name == "hbm" else cores) for name in rates}
    pools = pools or {"l1": 4096, "l0a": 4096, "l0b": 4096, "l0c": 4096}
    context = RecipeContext(
        op, {}, {name: name for name in rates}, rates, cores,
        pool_names={name: name for name in pools},
        pool_capacities=dict(pools),
        pool_units={name: cores for name in pools},
        resource_units=units,
    )
    return replace(context, work={"cube": op.flops})


def _engine(context):
    return Engine(
        [Resource(name, rate, units=context.resource_units.get(name, 1))
         for name, rate in context.rates.items()],
        pools=context.pool_capacities, pool_units=context.pool_units)


def _a5_context(m, n, k):
    device = json.loads(
        (ROOT / "mindie_sim/backends/profiles/ascend.json").read_text()
    )[A5]["device"]
    op = _op(m=m, n=n, k=k)
    names = {"cube": "cube", "hbm": "hbm", "mte2_aic": "mte2_aic",
             "mte2_aiv": "mte2_aiv", "mte1": "mte1", "mte3": "mte3",
             "vector": "vector"}
    rates = {name: value / 1000 for name, value in device["rates"].items()}
    base = RecipeContext(
        op, {}, names, rates, device["compute_units"],
        pool_names={name: name for name in device["buffers"]},
        pool_capacities=dict(device["buffers"]),
        pool_units=dict(device["buffer_units"]),
        resource_units=dict(device["resource_units"]),
        compute_peaks=dict(device["compute_peaks"]),
    )
    work = Work(
        matrix={"bf16": op.flops},
        resources={"hbm": op.hbm_bytes, "mte2": op.mte2_bytes,
                   "mte3": op.mte3_bytes},
    )
    return replace(base, work=base.bind(work), op=work.apply(op),
                   native_work={**work.resources, "bf16": op.flops})


def _owner_sources(rows, n, tm, tn, cores):
    owners = [[] for _ in range(cores)]
    index = 0
    for row in rows:
        for start in range(0, row, tm):
            a = min(tm, row - start)
            for column in range(0, n, tn):
                b = min(tn, n - column)
                owners[index % cores].append((a, b))
                index += 1
    return tuple((len(shapes), tuple(shapes)) for shapes in owners)


def test_geometry_keeps_logical_local_storage_and_issued_extents_distinct():
    op = _op(m=5, n=7, k=3)
    context = _context(op)
    recipe = HierarchicalMatmul(4, 4, 2, 2, 2)
    geometry = matrix_geometry(context, recipe)
    assert isinstance(geometry, MatrixGeometry)
    assert geometry.logical_shape == (5, 7, 3)
    assert geometry.local_shape == (5, 7, 3)
    assert geometry.storage_shape == (16, 16, 16)
    assert geometry.issued_shape == (8, 8, 4)
    assert geometry.tile_counts == {"m": 2, "n": 2, "k": 2}
    assert geometry.useful_flops == 2 * 5 * 7 * 3
    assert geometry.macro_tile_area == 4 * 4 * 4
    assert geometry.macro_tile_envelope == 2 * 16 * 4 * 4
    assert sum(record[3] for record in geometry.owners) == 5 * 7


def test_geometry_zero_rows_is_finite_and_ownerless():
    op = _op(m=0, n=7, k=3)
    geometry = matrix_geometry(_context(op), HierarchicalMatmul(4, 4, 2, 2, 2))
    assert geometry.useful_flops == geometry.macro_tile_envelope == 0
    assert geometry.owners == ()
    assert geometry.tiles == 0


def test_equal_group_geometry_matches_independent_owner_enumeration():
    recipe = HierarchicalMatmul(4, 4, 2, 2, 2)
    for m, n, cores, groups in ((5, 7, 3, 1), (5, 7, 5, 1),
                                (5, 7, 3, 2), (9, 5, 7, 3)):
        op = _op(m=m, n=n, k=3, groups=groups)
        geometry = matrix_geometry(_context(op, cores=cores), recipe)
        expected = owner_tiles(m, n, 4, 4, cores, groups)
        assert geometry.owners == expected
        assert sum(record[3] for record in geometry.owners) == m * n * groups
        assert geometry.output_tiles == sum(record[0] for record in expected)


@pytest.mark.parametrize("recipe", [
    HierarchicalMatmul(4, 4, 2, 2, 2),
    SteadyMatmul(((4, 4, 2),), 2, 2),
])
def test_explicit_unequal_group_rows_launch_each_group_tail(recipe):
    op = _op(n=4, k=2, groups=2, total_m=4)
    context = _context(op, cores=1)
    assert grouped_owner_rectangles((1, 3), 4, 4, 4, 1) == (
        (2, ((1, 4), (3, 4))),)
    engine = _engine(context)
    done = recipe.lower(context, group_rows=(1, 3)).start(
        engine, "unequal-groups", engine.submit("ready"))
    engine.advance(until=(done,))
    # Group rows are 1 and 3, so the two tile row extents are 1 and 3.
    # Reconstructing both from the widest group would charge 2 * 3 * 4 * 2 * 2
    # = 96 FLOPs instead of the declared 2 * (1 + 3) * 4 * 2 = 64.
    assert engine.serviced["cube"] == 64
    assert engine.serviced["hbm"] == 80
    assert engine.serviced["mte2"] == 48
    assert engine.serviced["mte1"] == 80
    assert engine.serviced["mte3"] == 32
    assert not any(engine.allocated.values())


def test_grouped_owner_entry_matches_closed_form_for_equal_groups():
    for m, n, cores, groups in ((5, 7, 3, 2), (9, 5, 7, 3), (3, 8, 5, 4)):
        assert grouped_owner_tiles((m,) * groups, n, 4, 4, cores) == owner_tiles(
            m, n, 4, 4, cores, groups)


def test_explicit_unequal_group_rows_share_the_same_owner_entry():
    rows, n, cores = (2, 5, 4, 3), 9, 3
    sources = _owner_sources(rows, n, 4, 4, cores)
    records = grouped_owner_tiles(rows, n, 4, 4, cores)
    expected = tuple((count, sum(a for a, _ in shapes),
                      sum(b for _, b in shapes),
                      sum(a * b for a, b in shapes))
                     for count, shapes in sources)
    # The pure aggregate must match an independent group-major enumeration.
    assert records == expected
    assert sum(record[3] for record in records) == sum(rows) * n

    empty = grouped_owner_tiles((2, 0, 5, 7), n, 4, 4, cores)
    assert sum(record[3] for record in empty) == 14 * n
    assert all(record[0] > 0 for record in empty)

    with pytest.raises(ValueError, match="matrix_grouped_owner_enumeration_exceeds_bound"):
        grouped_owner_tiles(rows, n, 1, 1, cores, max_tiles=1)

    op = _op(n=n, k=3, groups=len(rows), total_m=sum(rows))
    context = _context(op, cores=cores)
    engine = Engine(
        [Resource(name, rate, units=units)
         for name, rate, units in (
             ("cube", context.rates["cube"], cores),
             ("hbm", context.rates["hbm"], 1),
             ("mte2", context.rates["mte2"], cores),
             ("mte1", context.rates["mte1"], cores),
             ("mte3", context.rates["mte3"], cores))],
        pools=context.pool_capacities, pool_units=context.pool_units)
    done = HierarchicalMatmul(4, 4, 2, 2, 2).lower(
        context, owner_tiles=sources, group_rows=rows).start(
            engine, "grouped", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced["cube"] == pytest.approx(2 * sum(rows) * n * 3)
    assert engine.serviced["hbm"] == pytest.approx(
        sum(rows) * 3 * 2 + len(rows) * n * 3 * 2 + sum(rows) * n * 2)
    for pool in ("l1", "l0a", "l0b", "l0c"):
        assert not engine.allocated.get(pool)
    assert engine.summary()["live_tasks"] == 0


def test_mapping_cache_separates_declared_storage_and_semantic_geometry():
    mapped = MappedMatmul(((4, 4, 2),), 2, 2)
    op = _op(m=5, n=7, k=3)
    _, first = mapped.select(_context(op))
    changed = replace(op, padded_shape=(8, 8, 8),
                      semantic_shape=(9, 7, 3))
    _, second = mapped.select(_context(changed))
    assert first["geometry"]["storage_shape"] == [16, 16, 16]
    assert second["geometry"]["storage_shape"] == [8, 8, 8]
    assert second["geometry"]["logical_shape"] == [9, 7, 3]
    assert first["geometry"]["macro_tile_envelope"] == 512
    assert first["geometry"]["macro_tile_envelope_basis"] == (
        "full_macro_tile_extent_not_costed_work")
    assert first["candidates"][0]["macro_tile_envelope"] == 512
    assert "issued_flops" not in first["geometry"]
    assert "issued_flops" not in first["candidates"][0]


def test_physical_owner_rate_projects_declared_aggregate_capacity():
    op = _op(m=8, n=8, k=6)
    context = _context(op, cores=2)
    assert physical_owner_rate(context, "cube", (0,)) == pytest.approx(128 / 2)
    assert physical_owner_rate(context, "hbm", (0,)) == pytest.approx(1024)
    units = dict(context.resource_units, vector=4)
    context = replace(context, resource_units=units,
                      resource_names={**context.resource_names, "vector": "vector"},
                      rates={**context.rates, "vector": 64})
    assert physical_owner_rate(context, "vector", (0, 1)) == pytest.approx(32)


def test_shared_engine_reports_occurrence_deltas_not_whole_history():
    context = _context(_op(m=4, n=8, k=6))
    recipe = HierarchicalMatmul(4, 4, 2, 2, 2)
    engine = _engine(context)
    first = simulate_matrix_operator(context, recipe=recipe, engine=engine)
    second = simulate_matrix_operator(context, recipe=recipe, engine=engine)
    assert first.peak_allocated
    assert first.peak_basis == "occurrence_on_fresh_engine"
    assert second.peak_allocated is None
    assert second.peak_basis == "unavailable_shared_engine_slice_boundary"
    assert second.historical_peak_allocated == first.historical_peak_allocated
    assert second.serviced == pytest.approx(first.serviced)
    assert second.completed_tasks == first.completed_tasks
    assert second.duration_ms == pytest.approx(first.duration_ms)
    assert second.ready_ms >= first.ready_ms + first.duration_ms - 1e-9


def test_shared_engine_already_advanced_uses_actual_ready_time():
    context = _context(_op(m=4, n=8, k=6))
    engine = _engine(context)
    warm = engine.submit("prior-work", release=5)
    engine.advance(until=(warm,))
    assert engine.now == 5
    result = simulate_matrix_operator(
        context, recipe=HierarchicalMatmul(4, 4, 2, 2, 2), engine=engine)
    assert result.ready_ms == 5
    assert result.duration_ms > 0
    assert result.peak_allocated is None
    assert result.peak_basis == "unavailable_shared_engine_slice_boundary"
    assert result.completed_tasks > 0


def test_public_entry_rejects_active_frontier_before_engine_mutation():
    context = _context(_op(m=4, n=8, k=6))
    recipe = HierarchicalMatmul(4, 4, 2, 2, 2)
    engine = _engine(context)
    queued = engine.submit("queued-work", release=100)
    before = engine.summary()
    with pytest.raises(
            ExecutionError, match="matrix_operator_requires_idle_engine_frontier"):
        simulate_matrix_operator(context, recipe=recipe, engine=engine)
    after = engine.summary()
    assert after["time_ms"] == before["time_ms"] == 0
    assert after["completed_tasks"] == before["completed_tasks"] == 0
    assert after["serviced"] == before["serviced"]
    assert after["live_allocated"] == before["live_allocated"]
    assert after["live_tasks"] == 1

    # The explicit lower/start combination remains available on the shared
    # Engine; only the isolated public entry rejects a non-idle frontier.
    first = recipe.lower(context).start(
        engine, "direct-a", engine.submit("ready-a"))
    second = recipe.lower(context).start(
        engine, "direct-b", engine.submit("ready-b"))
    engine.advance(until=(queued, first, second), first=False)
    assert queued.done and first.done and second.done


def test_public_operator_entry_returns_work_geometry_and_finite_buffers():
    op = _op(m=5, n=7, k=3)
    context = _context(op, pools={"l1": 256, "l0a": 64, "l0b": 64, "l0c": 256})
    result = simulate_matrix_operator(
        context, recipe=HierarchicalMatmul(4, 4, 2, 2, 2), release_ms=3)
    assert math.isfinite(result.duration_ms) and result.duration_ms > 0
    assert result.release_ms == 3
    assert result.geometry["logical_shape"] == [5, 7, 3]
    assert result.geometry["storage_shape"] == [16, 16, 16]
    assert result.geometry["issued_shape"] == [8, 8, 4]
    assert result.geometry["macro_tile_envelope"] == 512
    assert result.geometry["macro_tile_envelope_basis"] == (
        "full_macro_tile_extent_not_costed_work")
    assert result.serviced["cube"] == pytest.approx(context.work["cube"]) == 210
    assert result.geometry["macro_tile_envelope"] > context.work["cube"]
    assert result.serviced["hbm"] == pytest.approx(
        op.mte2_bytes + op.mte3_bytes)
    assert result.peak_allocated
    assert result.completed_tasks > 0

    delayed = simulate_matrix_operator(
        context, recipe=HierarchicalMatmul(4, 4, 2, 2, 2), release_ms=11)
    assert delayed.duration_ms == pytest.approx(result.duration_ms)


def test_small_shape_and_rate_perturbations_change_geometry_or_span_only():
    recipe = HierarchicalMatmul(4, 4, 2, 2, 2)
    small = simulate_matrix_operator(_context(_op(m=4, n=8, k=6)), recipe=recipe)
    tail = simulate_matrix_operator(_context(_op(m=5, n=8, k=6)), recipe=recipe)
    assert small.geometry["issued_shape"] != tail.geometry["issued_shape"]
    assert tail.geometry["useful_flops"] > small.geometry["useful_flops"]
    assert tail.duration_ms >= small.duration_ms

    fast = simulate_matrix_operator(
        _context(_op(m=4, n=8, k=6), rates={"cube": 256}), recipe=recipe)
    slow = simulate_matrix_operator(
        _context(_op(m=4, n=8, k=6), rates={"cube": 64}), recipe=recipe)
    assert fast.serviced == pytest.approx(slow.serviced)
    assert fast.duration_ms < slow.duration_ms


def test_service_mode_phase_binds_once_and_rejects_a_mismatched_native_amount():
    op = _op(m=8, n=8, k=6)
    context = _context(op)
    context = replace(
        context,
        service_modes={"ub_copy": ("vector", 2_000_000.0)},
        rates={**context.rates, "vector": 1000},
        resource_units={**context.resource_units, "vector": 2},
        resource_names={**context.resource_names, "vector": "vector"},
        service_identity={"kernel": "copy"},
        work_units={**context.work_units, "ub_copy": "bytes"},
    )
    phase = context.for_phase(Work(resources={"ub_copy": 1000}), kernel="copy")
    assert phase.work == {"vector": 500}
    assert resolved_phase_work(
        phase, Work(resources={"ub_copy": 1000})) == {"vector": 500}
    with pytest.raises(ValueError, match="service_work_does_not_match_phase_context"):
        resolved_phase_work(phase, Work(resources={"ub_copy": 2000}))

    curve = ServiceCurve(
        resource="ub_copy", work_unit="bytes", identity={"kernel": "copy"},
        boundary="resource_phase", active_units=2, startup_ms=0.25,
        knots=((500, 1.0), (2000, 3.5)),
    )
    engine = Engine([Resource("vector", 1000, units=2)])
    done = ResourceService(curve).lower(phase).start(
        engine, "copy", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced["vector"] == pytest.approx(500)


def test_a5_matrix_service_uses_profile_peaks_without_a_hardcoded_rate():
    context = _a5_context(m=512, n=1024, k=256)
    device = json.loads(
        (ROOT / "mindie_sim/backends/profiles/ascend.json").read_text()
    )[A5]["device"]
    peak = device["compute_peaks"]["bf16"]
    assert context.work["cube"] == pytest.approx(1000 * 256 * 512 * 1024 * 2 / peak)
    recipe = HierarchicalMatmul(128, 128, 128, 128, 128)
    result = simulate_matrix_operator(context, recipe=recipe)
    assert result.serviced["cube"] == pytest.approx(context.work["cube"])
    assert result.mapping["compute_precision_peaks"]["bf16"] == peak
    assert result.geometry["owner_count"] == 32
    assert result.geometry["useful_flops"] == 2 * 512 * 1024 * 256

    slower = replace(context, compute_peaks={**context.compute_peaks,
                                             "bf16": peak / 2})
    work = Work(matrix={"bf16": 512 * 1024 * 256 * 2},
                resources={"hbm": context.op.hbm_bytes,
                           "mte2": context.op.mte2_bytes,
                           "mte3": context.op.mte3_bytes})
    slower = replace(slower, work=slower.bind(work))
    reduced = simulate_matrix_operator(slower, recipe=recipe)
    assert reduced.serviced["cube"] == pytest.approx(2 * result.serviced["cube"])
    assert reduced.duration_ms > result.duration_ms
