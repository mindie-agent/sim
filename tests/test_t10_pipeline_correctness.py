"""Detailed PipelineProgram reference for r4 contention, buffer and construction.

[Role]
  Preserve the existing causal PipelineProgram as the correctness reference after
  withdrawing the unproven compact body: per-tile stages, real fair arbitration,
  real allocation grant/release and one stage construction per issued tile.

[Invariants]
  * Concurrent competitors keep the detailed completion order; an external
    successor becomes ready at its detailed completion, not an earlier average.
  * A pool slot held by another owner lets the pipeline run on the remaining
    capacity; the pipeline does not reserve all slots up front.
  * Mixed tail allocations reach the detailed peak (1+3=4), not max*slots.
  * ``tile`` is constructed once per issued tile, so construction grows linearly
    with K, not twice with K.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.topology import TransportOptions
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineProgram, PipelineTemplate,
    PipelineTemplateProgram, RecipeBook, RecipeContext, Stage,
    indexed_pipeline_consumer_available,
)
from mindie_sim.prediction.services import service_identity


def _context(m, n, k, *, cores=2):
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                 m=m, n=n, k=k, quant=QuantSpec("bf16"), category="dense_mlp",
                 phase="prefill")
    rates = {"cube": 128, "hbm": 1024, "mte2": 16, "mte1": 1024, "mte3": 1024}
    pools = {"l1": 4096, "l0c": 4096}
    return RecipeContext(
        op, {"cube": op.flops}, {name: name for name in rates}, rates, cores,
        pool_names={name: name for name in pools},
        pool_capacities=pools, pool_units={name: cores for name in pools},
        resource_units={name: 1 if name == "hbm" else cores for name in rates},
    )


def _uniform_tile(index):
    return (
        Stage("load", {"r": 1.0},
              acquire=(Allocation(f"buf{index}", "p", 1, (0,)),)),
        Stage("compute", {"r": 1.0}, after=("load",),
              free=(Allocation(f"buf{index}", "p", 1, (0,)),)),
    )


def _run_uniform(*, outside=0.0):
    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    done = PipelineProgram(8, 2, _uniform_tile).start(
        engine, "p", engine.submit("ready"))
    external = engine.submit("external", {"r": outside}, release=0)
    successor = engine.submit("successor", after=(external,), release=0)
    engine.advance(until=(done, external, successor), first=False)
    return engine, done.time, external.time, successor.time


def test_detailed_fair_competition_preserves_external_and_successor_ready():
    engine, done, external, successor = _run_uniform(outside=4.0)
    assert engine.serviced == {"r": 20.0}
    assert done == 20.0
    # The two live tile competitors at the start (load0/load1) leave the
    # external task at 11, and its successor is ready exactly then.  A single
    # averaged body would finish external at 8 and make successor early.
    assert external == 11.0
    assert successor == 11.0


def test_other_owner_holding_one_slot_does_not_deadlock_pipeline():
    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    held = Allocation("external", "p", 1, (0,))
    hold = engine.submit("external-hold", acquire=(held,), release=0)
    done = PipelineProgram(8, 2, _uniform_tile).start(
        engine, "p", engine.submit("ready"))
    release = engine.submit("external-release", after=(done,), free=(held,))
    engine.advance(until=(hold, done, release), first=False)
    assert done.done and release.done
    assert engine.now >= done.time
    assert not engine.allocated["p"]


def test_mixed_tail_allocations_keep_detailed_peak_four():
    def tile(index):
        size = 1 if index == 0 else 3
        name = f"tail{index}"
        return (
            Stage("load", {"r": 1.0}, acquire=(Allocation(name, "p", size, (0,)),)),
            Stage("compute", {"r": 0.0}, after=("load",),
                  free=(Allocation(name, "p", size, (0,)),)),
        )

    engine = Engine([Resource("r", 1.0)], pools={"p": 10}, pool_units={"p": 1})
    done = PipelineProgram(2, 2, tile).start(
        engine, "tail", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.peak_allocated["p"] == 4
    assert done.done


def test_panel_tile_is_constructed_once_per_issued_tile():
    def run(count):
        constructed = []

        def tile(index):
            constructed.append(index)
            return (Stage("work", {"r": 1.0}),)

        engine = Engine([Resource("r", 1.0)])
        done = PipelineProgram(count, 2, tile).start(
            engine, "p", engine.submit("ready"))
        engine.advance(until=(done,))
        return constructed

    assert run(8) == list(range(8))
    assert run(16) == list(range(16))


def test_zero_work_join_identity_has_empty_precision_and_no_model_lookup():
    base = gemm_op(op_id="join", deps=("a", "b"), stage=0, layer=0,
                   requests=(), m=1, n=1, k=1, quant=QuantSpec("bf16"),
                   category="dense_mlp", phase="primitive")
    join = replace(base, family="barrier", recipe="join", dtype="", layout="",
                   flops=0.0, matrix_flops=(), hbm_bytes=0.0, mte2_bytes=0.0,
                   mte3_bytes=0.0, vector_ops=0.0, link_bytes=0.0,
                   deps=("a", "b"), stream="aux")
    case = SimpleNamespace(
        model=None, machine=None,
        runtime=SimpleNamespace(vllm_impl="vllm", vllm_version="1",
                                kernel_format=""),
    )
    identity = service_identity(case, join, scope="step")
    assert identity["dtype"] == ""
    assert identity["layout"] == ""
    assert identity["deps"] == ["a", "b"]
    assert identity["stream"] == "aux"
    assert identity["replay_scope"] == "step"
    assert identity["hardware"] == {}


def test_pipeline_template_constructs_one_phase_class_per_output_tile():
    constructed = []

    def tile(index):
        constructed.append(index)
        return (Stage("work", {"r": 1.0},
                      acquire=(Allocation(f"b{index}", "p", 1, (0,)),)),)

    program = PipelineProgram(6, 2, tile, homogeneous=True)
    template = program.template(owner=3, panels_per_output=3, k_tail=True)
    # Two output tiles; first/middle/last K panels are distinct because the
    # accumulator is acquired at the first and released at the last.
    assert constructed == [0, 1, 2, 3, 4, 5]
    assert [phase.kind for phase in template.phases] == [
        "first", "middle", "last", "first", "middle", "last"]
    assert [phase.count for phase in template.phases] == [1, 1, 1, 1, 1, 1]
    assert template.tiles == 6
    assert len(list(template.reference_stages())) == 6
    assert template.phases[0].stages[0].acquire


def _template_engine(context, *, trace_limit=0):
    return Engine(
        [Resource(name, float(rate),
                  units=1 if name == "hbm" else context.core_count)
         for name, rate in context.rates.items()],
        pools=dict(context.pool_capacities),
        pool_units={name: context.core_count
                    for name in context.pool_capacities},
        trace_limit=trace_limit,
    )


def _accumulator_edges(template):
    first, middle, last = template.phases[:3]
    acquired = [allocation.name for stage in first.stages
                for allocation in stage.acquire
                if allocation.name.startswith("accumulator")]
    middle_acquired = [allocation.name for stage in middle.stages
                       for allocation in stage.acquire
                       if allocation.name.startswith("accumulator")]
    released = [allocation.name for stage in last.stages
                for allocation in stage.free
                if allocation.name.startswith("accumulator")]
    return acquired, middle_acquired, released


def test_tiled_matmul_template_owner_counts_match_detailed_program():
    import mindie_sim.prediction.scheduling as scheduling
    from mindie_sim.schema.common import ModelingError

    context = _context(67, 35, 129)
    detailed = TiledMatmul(32, 32, 32, slots=2,
                            template_execution=False).lower(context)
    cores = next(stage.program for stage in detailed.stages if stage.name == "cores")
    expected_counts = [program.count for program in cores.programs]

    # The actual T04 callable, not a producer self-report, owns the marker.
    marker = getattr(scheduling.start_pipeline_template,
                     "indexed_panels_abi", None)
    assert type(marker) is int
    assert marker == 1
    assert indexed_pipeline_consumer_available()

    templates = TiledMatmul(32, 32, 32, slots=2).pipeline_templates(context)
    assert len(templates) == len(cores.programs)
    assert [item.tiles for item in templates] == expected_counts
    assert sum(item.tiles for item in templates) == sum(expected_counts)
    assert all(item.phases for item in templates)
    assert all(item.indexed_panels is not None and item.index_of is not None
               for item in templates)
    assert [item.owner for item in templates] == [0, 1]

    # Indexed symbols are stable names, not per-wave digit strings.
    acquired, middle_acquired, released = _accumulator_edges(templates[0])
    assert acquired == released == ["accumulator"]
    assert middle_acquired == []
    assert [phase.kind for phase in templates[0].phases[:3]] == [
        "first", "middle", "last"]

    # Absolute index selects a real first/middle/last class and exclusive end.
    template = templates[0]
    assert template.index_of(0) == (0, 1, 0, 0, 0)
    assert template.index_of(1) == (1, 4, 0, 1, 1)
    assert template.index_of(4) == (2, 5, 0, 4, 0)
    assert template.index_of(5) == (0, 6, 1, 0, 1)

    # Physical scope: the accumulator spans one output's K panels, while a
    # panel buffer stays tile-local.  The bindings use those same symbols.
    first, last = template.phases[0], template.phases[2]
    accumulator = first.stages[1].acquire[0]
    assert accumulator == last.stages[-1].free[0]
    assert accumulator.name == "accumulator"
    panel = first.stages[0].acquire[0]
    assert panel == first.stages[1].free[0]
    assert panel.name == "panel"
    bindings = template.allocation_bindings
    assert {row["name"]: row["scope"] for row in bindings} == {
        "accumulator": "output", "panel": "panel"}
    assert all(row["owner"] == 0 for row in bindings)
    declared = {row["name"] for row in bindings}
    assert all(allocation.name in declared
               for phase in template.phases
               for stage in phase.stages
               for allocation in (*stage.acquire, *stage.free))

    # Run the actual reader under the real marker: physical instance keys carry
    # owner and output/panel scope, repeated K panels share the accumulator and
    # every absolute panel keeps its own buffer.
    engine = _template_engine(context, trace_limit=256)
    done = PipelineTemplateProgram(template).start(
        engine, "m", engine.submit("ready"))
    engine.advance(until=(done,))
    trace = list(engine.allocation_trace)
    granted = {row["name"] for row in trace if row["action"] == "acquire"}
    freed = {row["name"] for row in trace if row["action"] == "release"}
    assert granted == freed
    assert {f"m:owner0:output{wave}:accumulator"
            for wave in range(3)} <= granted
    assert {f"m:owner0:panel{index}:panel"
            for index in range(15)} <= granted
    assert not any(name.endswith(":accumulator0") for name in granted)
    assert engine.allocated == {"l1": 0.0, "l0c": 0.0}
    assert engine.peak_allocated == {"l1": 4096.0, "l0c": 4096}

    # Remove only the real callable's marker as an explicit no-marker control.
    # The same producer must return the exact legacy expanded encoding, and the
    # explicit indexed-only handoff must refuse before any execution mutation.
    del scheduling.start_pipeline_template.indexed_panels_abi
    try:
        assert not indexed_pipeline_consumer_available()
        legacy = TiledMatmul(32, 32, 32, slots=2).pipeline_templates(context)
        assert len(legacy) == len(cores.programs)
        assert [item.tiles for item in legacy] == expected_counts
        assert sum(item.tiles for item in legacy) == sum(expected_counts)
        assert all(item.indexed_panels is None and item.index_of is None
                   for item in legacy)
        legacy_acquired, legacy_middle, legacy_released = \
            _accumulator_edges(legacy[0])
        assert legacy_acquired == legacy_released == ["accumulator0"]
        assert legacy_middle == []
        assert [phase.kind for phase in legacy[0].phases[:3]] == [
            "first", "middle", "last"]
        assert legacy[0].motif_repeats == 3
        legacy_bindings = legacy[0].allocation_bindings
        assert ("accumulator0", "output") in {
            (row["name"], row["scope"]) for row in legacy_bindings}
        assert ("panel0", "panel") in {
            (row["name"], row["scope"]) for row in legacy_bindings}
        assert all(row["owner"] == 0 for row in legacy_bindings)

        ready = engine.submit("blocked")
        before = engine.summary()
        with pytest.raises(ModelingError,
                           match="pipeline_indexed_consumer_unavailable"):
            PipelineTemplateProgram(template).start(engine, "blocked", ready)
        assert engine.summary() == before
    finally:
        scheduling.start_pipeline_template.indexed_panels_abi = marker
    assert type(getattr(scheduling.start_pipeline_template,
                        "indexed_panels_abi", None)) is int
    assert scheduling.start_pipeline_template.indexed_panels_abi == 1
    assert indexed_pipeline_consumer_available()


def test_tiled_template_panel_construction_does_not_grow_with_k_panels():
    calls = []
    original = PipelineProgram.template

    def counting_template(self, **kwargs):
        tile = self.tile

        def wrapped(index):
            calls.append(index)
            return tile(index)

        object.__setattr__(self, "tile", wrapped)
        try:
            return original(self, **kwargs)
        finally:
            object.__setattr__(self, "tile", tile)

    short = _context(67, 35, 129)
    long = _context(67, 35, 257)
    calls.clear()
    with patch.object(
            PipelineProgram, "template", counting_template):
        short_templates = TiledMatmul(32, 32, 32, slots=2).pipeline_templates(short)
        short_calls = len(calls)
        calls.clear()
        long_templates = TiledMatmul(32, 32, 32, slots=2).pipeline_templates(long)
        long_calls = len(calls)
    assert short_calls == long_calls
    if indexed_pipeline_consumer_available():
        # The indexed catalog is built directly from geometry; no wave tile
        # callback is invoked at all, and K grows only the logical count.
        assert short_calls == 0
        assert all(item.indexed_panels is not None
                   for item in (*short_templates, *long_templates))
        assert sum(item.tiles for item in short_templates) > 0
    else:
        assert short_calls < sum(item.tiles for item in short_templates)
        assert short_calls == sum(len(item.phases)
                                  for item in short_templates)
    assert sum(item.tiles for item in long_templates) > sum(
        item.tiles for item in short_templates)


def test_paged_gqa_decode_planned_recipe_makes_declared_and_planned_memory_equal():
    import json
    from pathlib import Path

    raw = json.loads(Path("examples/cases/qwen38-a5.json").read_text())
    raw["runtime"]["attention_policy"] = {"kv_storage": "paged",
                                         "fused_batch": False}
    ops = target_operators(raw, phase="decode", tokens=1, context=1028)
    rows = operator_costs(raw, ops, input_ready_ms=7)
    row = next(item for item in rows["operators"]
               if item["operator"]["family"] == "attention")
    assert row["program"] == "StagedGQADecode"
    memory = {key: value for key, value in row["declared_service"].items()
              if key.startswith("memory@")}
    serviced = {key: value for key, value in row["serviced"].items()
                if key.startswith("memory@")}
    assert memory == pytest.approx(serviced, rel=1e-12, abs=1e-6)
    assert next(iter(memory.values())) == pytest.approx(4_239_504.0)
    execution = dict(row["operator"]["metadata"])["program_execution"]
    assert execution["kv_cache_access"]["storage"] == "paged"
    assert execution["kv_cache_access"]["page_size"] == 128
    physical = execution["decode_physical_contract"]
    assert physical["conversion"]["scope"].startswith("on_chip")
    assert physical["workspace"]["owner_slots"] > 0
    assert physical["workspace"]["capacity_bytes"]["total"] > 0
    assert {"q_l1", "scores", "k_l0b", "v_l0b"} <= {
        row["name"] for row in physical["buffer_lifecycle"]}


def test_zero_floor_programs_have_no_synthetic_observation_or_span():
    engine = Engine([Resource("r", 1.0)])
    done = KernelProgram((Stage("work", {"r": 1.0}),)).start(
        engine, "kernel", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 1.0

    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    done = PipelineProgram(2, 1, _uniform_tile).start(
        engine, "pipeline", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 4.0


def test_explicit_measurement_floor_remains_observable():
    engine = Engine([Resource("r", 1.0)])
    done = KernelProgram((Stage("work", {"r": 1.0}),), minimum_ms=10.0).start(
        engine, "kernel", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 10.0


def _run_uniform_pipeline(template):
    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    ready = engine.submit("ready")
    program = PipelineProgram(8, 2, _uniform_tile)
    bindings = tuple({"name": f"buf{index}", "scope": "panel",
                      "owner": 0, "ordinal": index, "pool": "p",
                      "size": 1, "units": (0,), "action": "acquire",
                      "phase": index, "repeat": 1} for index in range(8))
    done = (program.start_template(engine, "p", ready, panels_per_output=1,
                                   bindings=bindings)
            if template else program.start(engine, "p", ready))
    external = engine.submit("external", {"r": 4.0}, release=0)
    successor = engine.submit("successor", after=(external,), release=0)
    engine.advance(until=(done, external, successor), first=False)
    return done.time, external.time, successor.time, engine.summary()


def test_template_matches_detailed_fair_external_and_successor():
    detailed = _run_uniform_pipeline(False)
    template = _run_uniform_pipeline(True)
    assert detailed[:3] == (20.0, 11.0, 11.0)
    assert template[:3] == detailed[:3]
    assert template[3]["serviced"] == detailed[3]["serviced"]
    assert template[3]["peak_allocated"] == detailed[3]["peak_allocated"]


def _tail_tile(index):
    last = index == 64
    return (
        Stage("a", {"ra": 1.0}),
        Stage("b", {"rb": 1.0}, after=("a",) if last else ()),
    )


def test_template_matches_detailed_65_tile_tail_dependency():
    rates = {"ra": 1.0, "rb": 1.0}
    def run(template):
        engine = Engine([Resource(name, rate) for name, rate in rates.items()])
        program = PipelineProgram(65, 1, _tail_tile, homogeneous=True)
        done = (program.start_template(engine, "p", engine.submit("ready"),
                                       panels_per_output=1)
                if template else program.start(engine, "p", engine.submit("ready")))
        engine.advance(until=(done,))
        return done.time, engine.summary()["serviced"]
    detailed = run(False)
    templated = run(True)
    assert detailed == (66.0, {"ra": 65.0, "rb": 65.0})
    assert templated == detailed


def test_tiled_matmul_template_matches_detailed_work_and_peaks():
    context = _context(67, 35, 129)

    def run(template):
        engine = Engine(
            [Resource("cube", 128.0, units=2), Resource("hbm", 1024.0),
             Resource("mte2", 16.0, units=2), Resource("mte1", 1024.0, units=2),
             Resource("mte3", 1024.0, units=2), Resource("vector", 1024.0, units=2)],
            pools={"l1": 4096, "l0c": 4096},
            pool_units={"l1": 2, "l0c": 2})
        program = TiledMatmul(32, 32, 32, slots=2,
                              template_execution=template).lower(context)
        done = program.start(engine, "m", engine.submit("ready"))
        engine.advance(until=(done,))
        summary = engine.summary()
        return (done.time, summary["serviced"], summary["peak_allocated"])

    detailed = run(False)
    templated = run(True)
    assert templated == detailed  # zero-observation event count may differ


def test_pipeline_template_carries_periodic_key_metadata():
    program = PipelineProgram(4, 2, lambda index: (
        Stage("work", {"r": 1.0},
              acquire=(Allocation(f"b{index}", "p", 1, (0,)),)),),
        homogeneous=True)
    template = program.template(panels_per_output=2,
                                key_of=lambda index: (index, index % 2))
    assert [(phase.kind, phase.key) for phase in template.phases] == [
        ("first", (0, 0)), ("last", (1, 1)),
        ("first", (2, 0)), ("last", (3, 1))]


def test_arbitrary_tile_callback_uses_correct_detailed_window():
    def tile(index):
        return (Stage("work", {"r": float(index + 1)}),)

    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    program = PipelineProgram(5, 1, tile)
    done = program.start_template(engine, "p", engine.submit("ready"),
                                  panels_per_output=5)
    engine.advance(until=(done,))
    assert done.time == 15.0
    assert engine.serviced["r"] == 15.0
    with pytest.raises(Exception, match="declared_homogeneity"):
        program.template(panels_per_output=5)


def test_prefill_factory_runs_selected_plan_once_via_recipe_book():
    import json
    from pathlib import Path

    raw = json.loads(Path("examples/cases/qwen38-a5.json").read_text())
    ops = target_operators(raw, phase="prefill", tokens=256,
                           context=512, sequences=2)
    index = next(i for i, op in enumerate(ops) if op.family == "attention")
    recipes = RecipeBook.from_mapping(
        {"recipes": {"flash_attention": {"kind": "staged_gqa_prefill"}}})
    rows = operator_costs(raw, ops[index:index + 1], recipes=recipes)
    row = rows["operators"][0]
    assert row["program"] == "StagedGQAPrefill"
    memory = {key: value for key, value in row["declared_service"].items()
              if key.startswith("memory@")}
    serviced = {key: value for key, value in row["serviced"].items()
                if key.startswith("memory@")}
    assert memory == pytest.approx(serviced, rel=1e-12, abs=1e-3)
    assert dict(row["operator"]["metadata"])["program_execution"]


def test_pipeline_phase_slot_residue_and_explicit_end():
    first = (Stage("work", {"r": 1.0}),)
    second = (Stage("work", {"r": 2.0}),)
    phase = PipelinePhase("full", first, 10,
                          residue_stages=(first, second), end=42)
    assert phase.stages_for(0) is first
    assert phase.stages_for(1) is second
    assert phase.stages_for(3) is second
    assert phase.end == 42


def test_template_setup_and_drain_share_explicit_invocation_binding():
    reserve = Allocation("w", "p", 1, (0,))
    template = PipelineTemplate(
        0, 1,
        (PipelinePhase("body", (Stage("work", {"r": 1.0}),), 1),),
        setup=(Stage("reserve", acquire=(reserve,)),),
        drain=(Stage("release", free=(reserve,)),),
        bindings=({"name": "w", "scope": "invocation", "owner": 0,
                   "ordinal": 0, "pool": "p", "size": 1, "units": (0,),
                   "action": "acquire", "phase": 0, "repeat": 1},),
    )
    engine = Engine([Resource("r", 1.0)], pools={"p": 1}, pool_units={"p": 1})
    done = PipelineTemplateProgram(template).start(
        engine, "p", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 1.0
    assert engine.serviced["r"] == 1.0
    assert not engine.allocated["p"]


def test_tiled_legacy_expanded_template_remains_exact_without_consumer():
    from unittest.mock import patch
    import mindie_sim.prediction.scheduling as scheduling

    def old_reader(*args, **kwargs):
        raise AssertionError("construction must not execute the old reader")

    def run(m, n):
        context = _context(m, n, 160)
        return TiledMatmul(32, 32, 32, slots=2).pipeline_templates(context)

    with patch.object(scheduling, "start_pipeline_template", old_reader):
        small = run(64, 64)
        large = run(128, 128)
    assert all(t.indexed_panels is None for t in small + large)
    assert all(len(t.motif_phases) == 3 for t in small + large)
    assert [t.motif_repeats for t in small] == [2, 2]
    assert [t.motif_repeats for t in large] == [8, 8]
    assert sum(t.tiles for t in small) == 20
    assert sum(t.tiles for t in large) == 80
    assert sum(t.static_phase_classes for t in small) == 6
    assert sum(t.static_phase_classes for t in large) == 6
    # The old exact path still constructs one phase per output/K role:
    # 12 -> 48 Phase and 36 -> 144 body Stage objects.
    assert sum(t.tile_factory_calls for t in small) == 12
    assert sum(t.tile_factory_calls for t in large) == 48
    assert sum(t.phase_objects for t in small) == 12
    assert sum(t.phase_objects for t in large) == 48
    assert sum(t.stage_objects for t in small) == 36
    assert sum(t.stage_objects for t in large) == 144


def test_uniform_five_tile_template_allocation_identities():
    bindings = tuple({
        "name": f"buf{index}", "scope": "panel", "owner": 0,
        "ordinal": index, "pool": "p", "size": 1, "units": (0,),
        "action": "acquire", "phase": index, "repeat": 1,
    } for index in range(5))
    engine = Engine([Resource("r", 1.0)], pools={"p": 2},
                    pool_units={"p": 1})
    program = PipelineProgram(5, 2, _uniform_tile, homogeneous=True)
    done = program.start_template(engine, "p", engine.submit("ready"),
                                  panels_per_output=5, bindings=bindings)
    engine.advance(until=(done,), first=False)
    assert done.time == 10.0
    assert engine.serviced["r"] == 10.0
    assert engine.peak_allocated["p"] == 2
    assert not engine.allocated["p"]


def test_for_phase_keeps_typed_transport_and_serializes_identity_fields():
    options = TransportOptions(
        chunk_bytes=None, hop_startup_us=2.5,
        receive_capacity_bytes=((1, 10.0), (0, 20.0)),
        receive_buffer_ids=((1, "rx1"), (0, "rx0")),
        receive_capacity_bytes_per_peer=None,
        receive_buffer_id="rx", max_inflight_per_rank=3,
        max_physical_chunks=77,
    )
    context = _context(8, 8, 8)
    context = replace(context, service_identity={"transport_options": options})
    phase = context.for_phase(
        Work(resources={"hbm": 1000.0}), kernel="copy",
        dtype="bf16", layout="contiguous")
    assert phase.transport_options is options
    projected = phase.transport_identity
    assert projected["chunk_bytes"] is None
    assert projected["hop_startup_us"] == 2.5
    assert projected["receive_capacity_bytes"] == [[0, 20.0], [1, 10.0]]
    assert projected["receive_buffer_ids"] == [[0, "rx0"], [1, "rx1"]]
    assert projected["receive_capacity_bytes_per_peer"] is None
    assert projected["max_inflight_per_rank"] == 3
    assert projected["max_physical_chunks"] == 77
    assert projected["receive_buffer_id"] == "rx"
    import json
    json.dumps(phase.service_identity)
