"""Real Engine controls for buffers shared across a native kernel's phase DAG."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan
from test_device_kernel_groups import phase
from test_graph_analysis import op
from test_mechanism_contract import hardware_case


BUFFER = 'shared_scratch'


def raw_case(*, startup=0, minimum=0):
    raw = hardware_case()
    raw['machine']['device'].update(
        buffers={'ub': 1024}, buffer_units={'ub': 2},
        resource_units={'vector': 2}, rates={'memory': 1e6, 'vector': 2e6},
        kernel_invocations={'group_test': {'startup_ms': startup, 'minimum_ms': minimum}})
    return raw


def buffer_recipe(operation, geometry):
    owner = dict(operation.metadata).get('owner', 0)
    work = Work(resources={'vector': operation.vector_ops})

    def program(context):
        # The same phase remains a finite-buffer program in isolation.
        allocations = (() if BUFFER in context.group_buffers else
                       (context.allocate('local', 'ub', 1024, (owner,)),))
        return KernelProgram((Stage(
            'service', context.bind(work),
            placement=context.placement('vector', (owner,)),
            acquire=allocations, free=allocations),))

    return OperatorPlan(work, program)


def group(name='group'):
    entry = phase(name + '.entry', group=name, entry=True)
    declaration = {**dict(entry.metadata)['device_kernel_group'],
                   'buffers': [{'id': BUFFER, 'pool': 'ub', 'bytes': 1024, 'units': [0]}]}
    entry = replace(entry, metadata=(('device_kernel_group', declaration),
                                    ('invocation_class', 'group_test')))
    first = replace(phase(name + '.first', group=name, deps=(name + '.entry',), stream=name+'.first'),
                    vector_ops=1000, recipe='buffered')
    last = replace(phase(name + '.last', group=name, deps=(name + '.first',), stream=name+'.last'),
                   vector_ops=2000, recipe='buffered')
    return (entry, first, last,
            phase(name + '.exit', group=name, exit=True, deps=(name + '.last',)))


def recipes():
    return RecipeBook({'buffered': PlannedRecipe(buffer_recipe)})


def competitor(owner, *, deps=('late',)):
    return replace(op('competitor', deps=deps, stream='independent'), vector_ops=1000,
                   recipe='buffered', metadata=(('owner', owner),))


@pytest.mark.parametrize('owner,start,finish', [(0, 3, 4), (1, .5, 1.5)])
def test_phase_gap_does_not_lend_the_whole_kernel_buffer_to_another_stream(owner,start,finish):
    result = graph_costs(raw_case(), (*group(), competitor(owner)), recipes=recipes(),
                         external_inputs={'late': .5}, isolated=False, trace_limit=1000)
    rows = {row['task']: row for row in result['events']}
    service = next(row for name,row in rows.items() if 'competitor' in name and name.endswith(':service'))
    assert service['start_ms'] == start
    assert service['end_ms'] == finish
    acquire = next(row for name,row in rows.items() if name.endswith(':buffers_acquire'))
    retire = next(row for name,row in rows.items() if name.endswith(':retire'))
    assert acquire['start_ms'] == 0 and retire['end_ms'] == 3
    assert result['execution']['serviced']['vector@r0:s0'] == 4000
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert not any(result['execution']['live_allocated'].values())


def test_buffer_admission_precedes_startup_and_floor_after_a_prior_user_releases_it():
    raw = raw_case(startup=2, minimum=8)
    prior = replace(competitor(0, deps=()), op_id='prior', vector_ops=5000)
    ops = list(group())
    ops[0] = replace(ops[0], deps=('begin',))
    result = graph_costs(raw, (prior,*ops), recipes=recipes(), external_inputs={'begin': 1},
                         isolated=False, trace_limit=1000)
    acquire = next(row for row in result['events'] if row['task'].endswith(':buffers_acquire'))
    starts = [row for row in result['events'] if row['task'].endswith(':service') and '.first:' in row['task']]
    assert acquire['start_ms'] == 5
    assert starts[0]['start_ms'] == 7
    assert result['duration_ms'] == 13  # 5 wait + max(2 startup + 3 work, 8 floor)
    assert result['execution']['serviced']['vector@r0:s0'] == 8000
    assert not any(result['execution']['live_allocated'].values())


def test_all_external_inputs_hold_reservation_and_phase_analysis_keeps_its_own_buffer():
    ops = list(group())
    ops[2] = replace(ops[2], deps=(*ops[2].deps, 'late'))
    result = graph_costs(raw_case(), ops, recipes=recipes(), external_inputs={'late': 11},
                         isolated=False, trace_limit=1000)
    acquire = next(row for row in result['events'] if row['task'].endswith(':buffers_acquire'))
    assert acquire['start_ms'] == 11
    assert result['complete_ms'] == 14
    # The public isolated API has no enclosing invocation to reserve a buffer.
    standalone = operator_costs(raw_case(), (group()[1],), recipes=recipes())
    assert standalone['operators'][0]['duration_ms'] == 1
    assert standalone['operators'][0]['peak_buffers']['ub@r0:s0'] == 1024


def test_same_group_ids_in_later_iterations_release_every_reservation():
    parsed = parse_case(raw_case())
    execution = Execution(parsed, CostModel(parsed), recipes=recipes(), include_submission=False)
    work = IterationWork(group(), IterationPlan(()), 1, 0, 0, None, 'decode', 'group')
    first,second = execution.add(work,0,5),execution.add(work,0,5)
    execution.engine.advance(until=(first.completion,second.completion),first=False)
    assert (first.completion.time,second.completion.time) == (8,11)
    assert execution.engine.summary()['live_tasks'] == execution.engine.summary()['live_programs'] == 0
    assert not execution.engine.allocations and not any(execution.engine.allocated.values())


def test_declared_idle_owner_still_reserves_its_buffer_and_floor_keeps_it_live():
    raw = raw_case(minimum=8)
    ops = list(group())
    entry_meta = dict(ops[0].metadata)
    entry_meta['device_kernel_group'] = {**entry_meta['device_kernel_group'],
        'buffers': [{'id': BUFFER, 'pool': 'ub', 'bytes': 1024, 'units': [0, 1]}]}
    ops[0] = replace(ops[0], metadata=tuple(entry_meta.items()))
    result = graph_costs(raw, (*ops,competitor(1)), recipes=recipes(),
                        external_inputs={'late': 1}, isolated=False, trace_limit=1000)
    service = next(row for row in result['events']
                   if 'competitor' in row['task'] and row['task'].endswith(':service'))
    assert (service['start_ms'],service['end_ms']) == (8,9)
    assert result['duration_ms'] == 9
    assert not any(result['execution']['live_allocated'].values())


def test_group_requires_capacity_for_each_declared_owner_before_any_phase_runs():
    raw = raw_case()
    prior = replace(competitor(1, deps=()), op_id='prior', vector_ops=5000)
    ops = list(group())
    entry_meta = dict(ops[0].metadata)
    entry_meta['device_kernel_group'] = {**entry_meta['device_kernel_group'],
        'buffers': [{'id': BUFFER, 'pool': 'ub', 'bytes': 1024, 'units': [0, 1]}]}
    ops[0] = replace(ops[0], deps=('begin',), metadata=tuple(entry_meta.items()))
    result = graph_costs(raw, (prior,*ops), recipes=recipes(),
                        external_inputs={'begin': 1}, isolated=False, trace_limit=1000)
    acquire = next(row for row in result['events'] if row['task'].endswith(':buffers_acquire'))
    first = next(row for row in result['events']
                 if '.first:' in row['task'] and row['task'].endswith(':service'))
    assert acquire['start_ms'] == first['start_ms'] == 5
    assert result['duration_ms'] == 8
    assert not any(result['execution']['live_allocated'].values())
