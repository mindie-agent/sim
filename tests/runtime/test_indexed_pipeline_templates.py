"""Finite indexed ``PipelineTemplate`` catalogs produced by ``TiledMatmul``.

[Role] Exercise the producer side of ``PipelineTemplate.indexed_panels`` /
  ``index_of``: bounded catalog construction, absolute-index selection,
  stable allocation symbols, ABI-marker-gated selection, and execution
  equivalence against the detailed (non-template) program.
[Invariants] The catalog is built from quotient/remainder geometry, never by
  enumerating waves: both 20- and 80-panel sizes construct exactly 6 phases,
  18 stages and 6 builder calls. A reader without ``indexed_panels_abi == 1``
  gets the exact ordered template, never the catalog, and an explicit catalog
  refuses to start. Indexed work, dependencies, allocations, timing and peaks
  equal the detailed reference for equal groups, tails, single K and quant
  epilogues. The executable consumer below is a test-local producer control.
[Timeouts & Locks] In-memory Engine runs on <=80-panel geometries.
[Side-Effect Policy] ``scheduling.start_pipeline_template`` and constructor
  counters are swapped only through ``monkeypatch``.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.program import (
    PipelinePhase, PipelineTemplate, PipelineTemplateProgram, RecipeContext, Stage,
    indexed_pipeline_consumer_available,
)
from mindie_sim.schema.common import ModelingError
import mindie_sim.prediction.scheduling as scheduling

pytestmark = pytest.mark.tier0


def matmul_context(m, n, k, *, cores=2, groups=1, quant=None, epilogue=False):
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(), m=m, n=n, k=k,
                 weight_count=groups, quant=quant or QuantSpec("bf16"), category="dense_mlp", phase="prefill")
    if epilogue:
        rows, columns = int(op.local_shape[-3]), int(op.local_shape[-2])
        metadata = dict(op.metadata)
        metadata["matrix_epilogue"] = {"kind": "row_column_scale", "row_scale_bytes": rows * 4,
                                       "column_scale_bytes": columns * 8, "operations_per_output": 4}
        op = replace(op, metadata=tuple(metadata.items()), vector_ops=float(4 * rows * columns))
    rates = {"cube": 128, "hbm": 1024, "mte2": 16, "mte1": 1024, "mte3": 1024, "vector": 1024}
    pools = {"l1": 4096, "l0c": 4096}
    work = {"cube": op.flops, "hbm": op.hbm_bytes, "mte2": op.mte2_bytes, "mte3": op.mte3_bytes}
    if op.vector_ops:
        work["vector"] = op.vector_ops
    return RecipeContext(op, work, {name: name for name in rates}, rates, cores,
                         pool_names={name: name for name in pools}, pool_capacities=pools,
                         pool_units={name: cores for name in pools},
                         resource_units={name: 1 if name == "hbm" else cores for name in rates})


def recipe():
    return TiledMatmul(32, 32, 32, slots=2)


def indexed_consumer(engine, template, name, ready, *, owner="", quiescent=False):
    """Bounded reader using only frozen public fields (tiles/index_of/stages_for/bindings)."""
    assert template.indexed_panels is not None
    bindings = {row["name"]: row["scope"] for row in template.bindings}
    complete = engine.future(name + ":complete")
    current = ready
    if template.setup_ms:
        current = engine.submit(name + ":setup_ms", after=(current,), latency=template.setup_ms, owner=owner)

    def scoped_for(index, wave):
        prefixes = {"invocation": name, "output": f"{name}:output{wave}", "panel": f"{name}:tile{index}"}

        def scoped(items):
            out = []
            for item in items:
                scope = bindings[item.name]
                prefix = prefixes.get(scope) or f"{name}:slot{item.units[0] if item.units else 0}"
                assert scope in prefixes or scope == "slot"
                out.append(Allocation(f"{prefix}:{item.name}", item.pool, item.size, item.units))
            return tuple(out)
        return scoped

    def submit_group(stages, label, tail, scoped):
        local = {}
        for stage in stages:
            deps = [local[key] for key in stage.after] + [tail[key] for key in stage.previous if key in tail]
            if label != "drain" and stage.name in tail:
                deps.append(tail[stage.name])
            local[stage.name] = stage.submit(engine, f"{name}:{label}:{stage.name}", tuple(deps) or (current,),
                                             scoped, owner)
        return local

    setup = submit_group(template.setup, "setup", {}, scoped_for(0, 0))
    if setup:
        current = engine.submit(name + ":setup_complete", after=tuple(setup.values()), owner=owner)
    state = {"next": 0, "finished": 0}
    stage_tail = {}

    def issue():
        index = state["next"]
        if index >= template.tiles:
            return
        state["next"] += 1
        phase_id, end, wave, _panel, _slot = template.index_of(index)
        assert 0 <= phase_id < len(template.phases) and index < end <= template.tiles
        local = submit_group(template.phases[phase_id].stages_for(index), f"tile{index}", dict(stage_tail),
                             scoped_for(index, wave))
        stage_tail.update(local)
        engine.submit(f"{name}:tile{index}:retire", after=tuple(local.values()), on_complete=retire, owner=owner)

    def drain():
        local = submit_group(template.drain, "drain", stage_tail, scoped_for(0, 0))
        if not local:
            engine.resolve(complete)
            return
        engine.submit(name + ":drain_complete", after=tuple(local.values()),
                      on_complete=lambda _event: engine.resolve(complete), owner=owner)

    def retire(_event):
        state["finished"] += 1
        drain() if state["finished"] == template.tiles else issue()

    def begin(_event):
        if not template.tiles:
            drain()
        for _ in range(min(template.slots, template.tiles)):
            issue()

    engine.submit(name + ":begin", after=(current,), on_complete=begin, owner=owner)
    return engine.submit(name + ":complete_outer", after=(complete,), owner=owner)


indexed_consumer.indexed_panels_abi = 1


def old_reader(*args, **kwargs):
    raise AssertionError("a reader without the ABI marker must not execute an indexed catalog")


def execute(matmul, context):
    engine = Engine([Resource(name, float(rate), units=1 if name == "hbm" else context.core_count)
                     for name, rate in context.rates.items()],
                    pools=dict(context.pool_capacities),
                    pool_units={name: context.core_count for name in context.pool_capacities})
    done = matmul.lower(context).start(engine, "m", engine.submit("ready"))
    engine.advance(until=(done,))
    summary = engine.summary()
    return done.time, dict(summary["serviced"]), dict(summary["peak_allocated"]), dict(engine.peak_unit_allocated)


def lowered_templates(program):
    cores = next(stage.program for stage in program.stages if stage.name == "cores")
    return tuple(item.template for item in cores.programs)


BODY = (Stage("work", {"r": 1.0}),)
BOUND = (Stage("work", acquire=(Allocation("x", "p", 1, (0,)),)),)


def mapping(index):
    return (0, 1, 0, 0, 0)


@pytest.mark.parametrize("phase,kwargs,match", [
    (PipelinePhase("body", BODY, 1), {"indexed_panels": 1}, "requires_both_fields"),
    (PipelinePhase("body", BODY, 1), {"index_of": mapping}, "requires_both_fields"),
    (PipelinePhase("body", BODY, 1), {"indexed_panels": True, "index_of": mapping}, "indexed_panels_invalid"),
    (PipelinePhase("body", BODY, 1), {"indexed_panels": -1, "index_of": mapping}, "indexed_panels_invalid"),
    (PipelinePhase("body", BODY, 1), {"indexed_panels": 1.0, "index_of": mapping}, "indexed_panels_invalid"),
    (PipelinePhase("body", BODY, 2), {"indexed_panels": 2, "index_of": mapping}, "phase_count_must_be_one"),
    (PipelinePhase("body", BODY, 1, end=2), {"indexed_panels": 1, "index_of": mapping}, "phase_end_must_be_none"),
    (PipelinePhase("body", BOUND, 1), {"indexed_panels": 1, "index_of": mapping}, "allocation_symbol_unbound:x"),
    (PipelinePhase("body", BOUND, 1), {"indexed_panels": 1, "index_of": mapping, "bindings": (
        {"name": "x", "scope": "panel", "owner": 0}, {"name": "x", "scope": "output", "owner": 0})},
     "binding_scope_conflict:x"),
    (PipelinePhase("body", BOUND, 1), {"indexed_panels": 1, "index_of": mapping, "bindings": (
        {"name": "x", "scope": "panel", "owner": 1},)}, "binding_owner_mismatch:x"),
])
def test_indexed_fields_are_both_present_and_validated(phase, kwargs, match):
    with pytest.raises(ValueError, match=match):
        PipelineTemplate(0, 1, (phase,), **kwargs)


def test_valid_indexed_template_selects_its_single_phase():
    valid = PipelineTemplate(0, 1, (PipelinePhase("body", BOUND, 1),), indexed_panels=1, index_of=mapping,
                             bindings=({"name": "x", "scope": "panel", "owner": 0},))
    assert valid.tiles == 1 and valid.stage_for(0)[0].name == "work"


@pytest.mark.parametrize("m,n,expected_tiles", [(64, 64, 20), (128, 128, 80)])
def test_finite_catalog_construction_is_six_phase_eighteen_stage_six_builder(monkeypatch, m, n, expected_tiles):
    counts = {"stage": 0, "phase": 0}
    for cls, key in ((Stage, "stage"), (PipelinePhase, "phase")):
        def counted(self, *args, _original=cls.__init__, _key=key, **kwargs):
            counts[_key] += 1
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(cls, "__init__", counted)
    templates = recipe().finite_templates(matmul_context(m, n, 160))
    monkeypatch.undo()
    # Real constructor observations, not self-reported fields.
    assert counts == {"stage": 18, "phase": 6}
    assert len(templates) == 2
    for field in ("tiles", "indexed_panels", "dynamic_instances"):
        assert sum(getattr(template, field) for template in templates) == expected_tiles
    for field in ("phase_objects", "tile_factory_calls", "static_phase_classes"):
        assert sum(getattr(template, field) for template in templates) == 6
    assert sum(len(phase.stages) for template in templates for phase in template.phases) == 18
    for owner, template in enumerate(templates):
        assert template.motif_phases == () and template.motif_repeats == 1
        assert template.owner == owner
        assert [phase.count for phase in template.phases] == [1, 1, 1]
        assert all(phase.end is None for phase in template.phases)
        assert {(row["name"], row["scope"]) for row in template.bindings} == {
            ("panel", "panel"), ("accumulator", "output")}
        assert all(row["owner"] == owner for row in template.bindings)


def test_absolute_index_controls_for_small_and_asymmetric_tail():
    small = recipe().finite_templates(matmul_context(64, 64, 160))[0]
    expected = {  # index -> (phase key, end, wave, panel, slot)
        0: ((32, 32, 32, True, False), 1, 0, 0, 0),
        1: ((32, 32, 32, False, False), 4, 0, 1, 1),
        3: ((32, 32, 32, False, False), 4, 0, 3, 1),
        4: ((32, 32, 32, False, True), 5, 0, 4, 0),
        5: ((32, 32, 32, True, False), 6, 1, 0, 1),
    }
    for index, (geometry, *position) in expected.items():
        phase_id, *actual = small.index_of(index)
        assert small.phases[phase_id].key == geometry and actual == position
    for bad, error in ((-1, IndexError), (small.indexed_panels, IndexError), (True, TypeError)):
        with pytest.raises(error):
            small.index_of(bad)

    templates = recipe().finite_templates(matmul_context(196, 65, 70, groups=2))
    assert [template.tiles for template in templates] == [36, 36]
    owner = templates[1]
    for index, geometry, position in ((17, (2, 1, 6, False, True), [18, 5, 2, 1]),
                                      (18, (32, 32, 32, True, False), [19, 6, 0, 0])):
        phase_id, *actual = owner.index_of(index)
        assert owner.phases[phase_id].key == geometry and actual == position
    # The same absolute index maps to a different owner-local class because the
    # strided owner placement is real, not a shared nominal sequence.
    assert templates[0].phases[templates[0].index_of(17)[0]].key == (2, 32, 6, False, True)


def test_single_k_panel_is_simultaneously_first_and_last():
    templates = recipe().finite_templates(matmul_context(70, 65, 17))
    assert sum(template.tiles for template in templates) == 9
    for template in templates:
        assert len(template.phases) == 4
        for index in range(template.tiles):
            phase_id, end, _wave, panel, _slot = template.index_of(index)
            key = template.phases[phase_id].key
            assert key[2:] == (17, True, True) and panel == 0 and end == index + 1
            stages = {stage.name: stage for stage in template.stage_for(index)}
            assert stages["compute"].acquire and stages["store"].free


def test_zero_k_and_zero_owner_geometry_never_divide_or_emit_panels(monkeypatch):
    monkeypatch.setattr(scheduling, "start_pipeline_template", indexed_consumer)
    base = matmul_context(64, 64, 32)
    zero_k = replace(base, op=replace(base.op, local_shape=(64, 64, 0)))
    templates = recipe().finite_templates(zero_k)
    assert len(templates) == 2
    assert all(t.tiles == 0 and t.phases == () and t.indexed_panels == 0 for t in templates)
    with pytest.raises(IndexError):
        templates[0].index_of(0)
    lowered = lowered_templates(recipe().lower(zero_k))
    assert len(lowered) == 2 and all(template.tiles == 0 for template in lowered)
    for total_m, n in ((0, 64), (64, 0)):
        context = replace(base, op=replace(base.op, local_shape=(total_m, n, 32)))
        assert recipe().finite_templates(context) == ()
        assert lowered_templates(recipe().lower(context)) == ()


@pytest.mark.parametrize("m,n,k,groups,epilogue,expected_tiles", [
    (64, 64, 160, 1, False, 20),
    (128, 128, 160, 1, False, 80),
    (196, 65, 70, 2, False, 72),
    (70, 33, 17, 1, False, 6),
    (70, 33, 17, 1, True, 6),
    (67, 35, 129, 1, False, 30),
])
def test_indexed_catalog_matches_detailed_work_timing_and_peaks(monkeypatch, m, n, k, groups, epilogue,
                                                                expected_tiles):
    context = matmul_context(m, n, k, groups=groups, epilogue=epilogue,
                             quant=quant_preset("w8a8_int8") if epilogue else None)
    detailed = execute(TiledMatmul(32, 32, 32, slots=2, template_execution=False), context)
    calls = []

    def consumer(engine, template, name, ready, *, owner="", quiescent=False):
        calls.append(template.tiles)
        return indexed_consumer(engine, template, name, ready, owner=owner, quiescent=quiescent)

    consumer.indexed_panels_abi = 1
    monkeypatch.setattr(scheduling, "start_pipeline_template", consumer)
    assert execute(recipe(), context) == detailed
    # The default lower really entered the indexed producer; owner counts add up.
    assert sum(calls) == expected_tiles and all(tiles > 0 for tiles in calls)


def test_lower_pipeline_templates_and_reference_stages_share_one_catalog(monkeypatch):
    monkeypatch.setattr(scheduling, "start_pipeline_template", indexed_consumer)
    context = matmul_context(128, 128, 160)
    templates = recipe().pipeline_templates(context)
    lowered = lowered_templates(recipe().lower(context))
    assert len(templates) == len(lowered) == 2
    for left, right in zip(templates, lowered):
        assert (left.tiles, left.indexed_panels, left.bindings) == (right.tiles, right.indexed_panels, right.bindings)
        assert left.phase_objects == right.phase_objects == 3
        assert left.stage_objects == right.stage_objects == 9
        assert [phase.key for phase in left.phases] == [phase.key for phase in right.phases]
        for index in range(left.tiles):
            assert left.index_of(index) == right.index_of(index)
            assert left.stage_for(index)[0].work == right.stage_for(index)[0].work
    # reference_stages selects catalog objects by absolute index, never copies.
    template = recipe().finite_templates(matmul_context(64, 64, 160))[0]
    references = list(template.reference_stages())
    assert len(references) == template.tiles == 10
    for index, stages in enumerate(references):
        catalog = template.phases[template.index_of(index)[0]].stages_for(index)
        assert len(stages) == len(catalog) and all(a is b for a, b in zip(stages, catalog))


def test_no_marker_keeps_exact_ordered_template_and_refuses_explicit_indexed(monkeypatch):
    monkeypatch.setattr(scheduling, "start_pipeline_template", old_reader)
    context = matmul_context(64, 64, 160)
    templates = recipe().pipeline_templates(context)
    assert all(template.indexed_panels is None for template in templates)
    assert sum(template.tiles for template in templates) == 20
    for field, expected in (("phase_objects", 12), ("stage_objects", 36), ("tile_factory_calls", 12)):
        assert sum(getattr(template, field) for template in templates) == expected
    lowered = lowered_templates(recipe().lower(context))
    assert all(template.indexed_panels is None for template in lowered)
    assert sum(template.tiles for template in lowered) == 20
    explicit = recipe().finite_templates(context)
    assert sum(template.tiles for template in explicit) == 20
    engine = Engine([Resource("r", 1.0)])
    with pytest.raises(ModelingError, match="pipeline_indexed_consumer_unavailable"):
        PipelineTemplateProgram(explicit[0]).start(engine, "m", engine.submit("ready"))


def test_default_mode_tracks_the_actual_consumer_marker():
    context = matmul_context(64, 64, 160)
    available = indexed_pipeline_consumer_available()
    templates = recipe().pipeline_templates(context)
    phases, stages = (6, 18) if available else (12, 36)
    assert sum(template.tiles for template in templates) == 20
    assert sum(template.phase_objects for template in templates) == phases
    assert sum(template.stage_objects for template in templates) == stages
    for template in templates + lowered_templates(recipe().lower(context)):
        assert (template.indexed_panels is not None) == available


def test_two_owners_and_two_invocations_build_independent_catalogs():
    context = matmul_context(196, 65, 70, groups=2)
    first, second = recipe().finite_templates(context), recipe().finite_templates(context)
    assert [template.owner for template in first] == [0, 1]
    assert all(template.bindings[0]["owner"] == template.owner for template in first)
    assert first[0].index_of(17) != first[1].index_of(17)
    for left, right in zip(first, second):
        assert left is not right and left.index_of(0) == right.index_of(0)
        assert left.phases[0].stages[0] is not right.phases[0].stages[0]
