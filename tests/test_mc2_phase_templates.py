"""Physical MC2 templates share source IR, never invocation execution state."""
from dataclasses import replace
import json
import math

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mc2_phase import build
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.operators import OpSpec


def inputs():
    raw = dict(kind='combine', phase='weighted_reduce', width=5120, dtype='bf16',
               source_tokens=8, top_k=6, source_assignments=48,
               received_assignments=48, experts=384, ep=8)
    op = OpSpec('layer0.combine', 'mc2_protocol', 'decode', 0, 0, ('r0',),
                (), (), (), (), metadata=(('mc2_phase', raw),))
    roles = dict(hbm=1, vector=64, mte2_aiv=64, mte3=64,
                 scalar_ub_access=64, scalar_alu_issue=64, pipe_barrier=64)
    return raw, op, Geometry(32, roles)


def execute(plan, op, geometry, *, namespace, rate, release):
    names = {role: namespace + role for role in geometry.resource_units}
    rates = {names[role]: rate * units for role, units in geometry.resource_units.items()}
    context = RecipeContext(op, {}, names, rates, geometry.compute_units,
                            resource_units=geometry.resource_units)
    engine = Engine([Resource(names[role], rates[names[role]], units=units)
                     for role, units in geometry.resource_units.items()], trace_limit=0)
    ready = engine.submit(namespace + 'input', release=release)
    done = plan.lower(context).start(engine, namespace + op.op_id, ready)
    engine.advance(until=(done,))
    assert engine.summary()['live_tasks'] == engine.summary()['live_programs'] == 0
    return done.time - release, {role: engine.serviced.get(name, 0) for role, name in names.items()}


def test_repeated_layers_share_templates_with_independent_engine_rates_and_readiness():
    _, first_op, geometry = inputs()
    second_op = replace(first_op, op_id='layer39.combine', layer=39, requests=('r7',))
    first = build(first_op, geometry)
    second = build(second_op, geometry)
    assert first is second
    slow, slow_work = execute(first, first_op, geometry, namespace='r0:', rate=1000, release=7)
    fast, fast_work = execute(second, second_op, geometry, namespace='r7:', rate=2000, release=19)
    assert slow == pytest.approx(2 * fast, rel=1e-12)
    assert slow_work == pytest.approx(fast_work)
    assert slow_work['scalar_ub_access'] == 48
    # The return row reads whole 480-byte data regions, including its tail.
    assert slow_work['hbm'] == 8 * 6 * (4 + math.ceil(5120 * 2 / 480) * 480)
    assert slow_work['vector'] == 8 * (3 * 6 + 1) * 5120


def test_template_snapshot_and_shared_diagnostics_cannot_change_later_invocations():
    raw, op, geometry = inputs()
    plan = build(op, geometry)
    diagnostic = json.dumps(plan.metadata, sort_keys=True)
    raw.update(source_tokens=16, source_assignments=96, received_assignments=96)
    larger = build(op, geometry)
    assert larger is not plan
    assert larger.work.resources['hbm'] == 2 * plan.work.resources['hbm']
    owners = plan.metadata['primitive_services']['owners']
    first_insert = next(iter(owners[0].values()))
    with pytest.raises(TypeError):
        first_insert['counts']['scalar_ub_access'] = 999
    assert json.dumps(plan.metadata, sort_keys=True) == diagnostic
    _, work = execute(plan, op, geometry, namespace='snapshot:', rate=1000, release=3)
    assert work['scalar_ub_access'] == 48


def test_owner_topology_and_declared_primitive_roles_select_different_templates():
    _, op, geometry = inputs()
    ordinary = build(op, geometry)
    fewer_roles = {**geometry.resource_units, 'vector': 4, 'mte2_aiv': 4, 'mte3': 4}
    fewer_roles.pop('scalar_ub_access')
    fewer = build(op, Geometry(32, fewer_roles))
    assert fewer is not ordinary
    assert len(fewer.metadata['active_owner_domain']) == 4
    assert len(ordinary.metadata['active_owner_domain']) == 8
    assert 'scalar_ub_access' not in fewer.work.resources
    assert sum(row['unpriced_service_work'].get('scalar_ub_access', 0)
               for owner in fewer.metadata['primitive_services']['owners'].values()
               for row in owner.values()) == 48
    for role in ('hbm', 'mte2_aiv', 'mte3', 'vector'):
        assert fewer.work.resources.get(role, 0) == ordinary.work.resources.get(role, 0)


def test_million_token_owner_program_reuses_full_rows_and_keeps_fractional_tail_work():
    raw, op, original = inputs()
    tokens, units = 1_000_000.25, 4
    raw.update(source_tokens=tokens, source_assignments=tokens * raw['top_k'],
               received_assignments=tokens * raw['top_k'])
    roles = {role: 1 if role == 'hbm' else units for role in original.resource_units}
    geometry = Geometry(units, roles)
    plan = build(op, geometry)
    context = RecipeContext(op, {}, {role: role for role in roles},
                            {role: 1000 * count for role, count in roles.items()},
                            units, resource_units=roles)
    program = plan.program(context)
    total, tails = {}, 0

    def add(stages, repeats=1):
        for stage in stages:
            for role, value in stage.work.items():
                total[role] = total.get(role, 0) + repeats * value

    for stage in program.stages:
        if stage.program is None:
            add((stage,))
            continue
        rows = stage.program
        first, last = rows.tile(0), rows.tile(rows.count - 1)
        # Construction/binding is bounded by owners and two row geometries,
        # while the original token count and per-token instruction DAG remain.
        assert rows.tile(rows.count // 2) is first
        add(first, rows.count - 1)
        add(last)
        tails += last is not first
    assert tails == 1
    assert total == pytest.approx(plan.work.resources, rel=1e-12)
    assert total['hbm'] == tokens * raw['top_k'] * (
        4 + math.ceil(raw['width'] * 2 / 480) * 480)
    assert total['vector'] == tokens * (3 * raw['top_k'] + 1) * raw['width']
