"""Ordinary HCCL calls retain source copy storage across network/local phases."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs
from mindie_sim.prediction.hccl_aiv import UB_RESIDENCY
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from test_graph_analysis import op
from test_hccl_source_graph import declared, isolated_calls


def first_call(raw):
    ops = isolated_calls(raw)[0]
    group = dict(ops[0].metadata)['hccl_source_group']
    return tuple(o for o in ops if dict(o.metadata)['hccl_source_group'] == group)


def competing_owner(operation, geometry):
    work = Work(resources={'vector': 15625})

    def program(context):
        allocation = context.allocate('competing', 'ub', 100*1024, (2,))
        return KernelProgram((Stage('service', context.bind(work),
            placement=context.placement('vector', (2,)),
            acquire=(allocation,), free=(allocation,)),))

    return OperatorPlan(work,program)


def test_ordinary_oneshot_reducer_buffer_is_unavailable_while_producers_are_running():
    raw = declared(); raw['model']['hidden_size'] = 1680
    operations = first_call(raw)
    books = RecipeBook.from_mapping(raw['runtime']['kernels'])
    books.recipes['competitor'] = PlannedRecipe(competing_owner)
    baseline = graph_costs(raw, operations, recipes=books, isolated=False)
    external = replace(op('competing',deps=('arrival',),stream='other'),recipe='competitor')
    report = graph_costs(raw,(*operations,external),recipes=books,
                         external_inputs={'arrival': .1}, isolated=False,trace_limit=1000)
    service = next(row for row in report['events']
                   if 'competing:' in row['task'] and row['task'].endswith(':service'))
    assert service['start_ms'] == pytest.approx(baseline['duration_ms'])
    assert service['end_ms'] == pytest.approx(baseline['duration_ms']+1)
    for role,value in baseline['execution']['serviced'].items():
        if not role.startswith('vector@'):
            assert report['execution']['serviced'][role] == pytest.approx(value)
    assert report['execution']['peak_allocated_per_unit']['ub@r0:s0:unit2'] == UB_RESIDENCY
    assert not any(report['execution']['live_allocated'].values())
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


def test_twoshot_whole_call_and_isolated_phase_use_distinct_lifetime_scopes():
    raw = declared(template='mesh_twoshot'); raw['model']['hidden_size'] = 1680
    operations = first_call(raw)
    report = graph_costs(raw,operations,isolated=False)
    # Host launches63; the conservative whole-call declaration holds every
    # launched owner's allocation, including the early-return block62.
    peaks = report['execution']['peak_allocated_per_unit']
    assert len(peaks) == 63
    assert all(value == UB_RESIDENCY for value in peaks.values())
    assert not any(report['execution']['live_allocated'].values())
    phase = next(o for o in operations if o.family == 'hccl_local'
                 and dict(o.metadata)['hccl_phase']['phase'] == 'precopy')
    isolated = operator_costs(raw,(phase,))['operators'][0]
    assert isolated['peak_buffers']['ub@r0:s0'] == UB_RESIDENCY
    assert isolated['duration_ms'] > 0
    assert not isolated['service_delta']
