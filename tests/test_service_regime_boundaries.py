"""Independent physical boundary controls for observable native-call responses."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan
from test_communication_topology import exchange
from test_device_kernel_groups import fork, phase
from test_graph_analysis import op
from test_joint_collective_execution import cohort_case
from test_kernel_group_buffers import raw_case, group, competitor, recipes
from test_mechanism_contract import hardware_case
from test_resource_services import copy_op


def install(raw, rule):
    raw.setdefault('runtime', {}).setdefault('kernels', {})['service_regimes'] = [rule]
    return raw


def response(us, **kwargs):
    return {'id': 'native-response', 'match': {'kernel': 'NativeDispatch'},
            'response': {'fixed_us': us}, **kwargs}


def clean(result):
    execution = result['execution']
    assert execution['live_tasks'] == execution['live_programs'] == 0
    assert not any(execution['live_allocated'].values())


@pytest.mark.parametrize('response_us,complete', [(9000, 26), (15000, 29)])
def test_json_shape_axis_matches_immutable_source_and_inclusive_hardware_minimum(response_us, complete):
    raw = hardware_case()
    raw['machine']['device'].update(launch_us=3000, kernel_invocations={
        'vector': {'startup_ms': 2, 'minimum_ms': 12}})
    operation = replace(copy_op(4000), metadata=(
        ('invocation_class', 'vector'), ('geometry', {'tiles': ((1, 2), (3, 4))})))
    rule = response(response_us, match={'local_shape': [2000],
        'metadata.geometry': {'tiles': [[1, 2], [3, 4]]}})
    install(raw, json.loads(json.dumps(rule)))
    result = graph_costs(raw, (operation,), input_ready_ms=11,
                         include_submission=True, isolated=False, trace_limit=100)
    assert result['complete_ms'] == complete
    assert result['parameters']['service_regimes'][0]['queries'] == {'observable_response': 1}
    assert result['execution']['submission']['host_service_ms'] == 3
    assert result['execution']['serviced']['memory@r0:s0'] == 4000
    device_events = [event for event in result['events']
                     if event['demand'].get('memory@r0:s0')]
    assert min(event['start_ms'] for event in device_events) == 16 # release11 + host3 + startup2
    clean(result)


@pytest.mark.parametrize('scope,hosts,host_ms', [('none', 2, 4), ('full', 1, 3)])
def test_two_native_groups_keep_one_response_and_one_host_submission_boundary(scope, hosts, host_ms):
    raw = install(hardware_case(), response(10000))
    raw['machine']['device'].update(launch_us=2000, kernel_invocations={
        'vector': {'startup_ms': 3, 'minimum_ms': 8}})
    raw['runtime']['submission'] = {'graph_replay_ms': 3}
    operators = fork('first', ('late',)) + fork('second', ('first.exit',))
    result = graph_costs(raw, operators, replay_scope=scope, include_submission=True,
                         external_inputs={'late': 11}, isolated=False, trace_limit=300)
    assert result['complete_ms'] == 31 # 11 + max(3+6,10) twice
    assert result['parameters']['service_regimes'][0]['queries'] == {'observable_response': 2}
    submission = result['execution']['submission']
    assert submission['eager_invocations' if scope == 'none' else 'graph_replays'] == hosts
    assert submission['host_service_ms'] == host_ms
    assert result['execution']['serviced']['memory@r0:s0'] == 10000
    assert result['execution']['serviced']['vector@r0:s0'] == 6000
    clean(result)


def test_regime_waits_for_finite_group_buffer_admission_and_is_not_added_to_startup():
    raw = install(raw_case(startup=2, minimum=8), response(12000))
    prior = replace(competitor(0, deps=()), op_id='prior', vector_ops=5000)
    operations = list(group()); operations[0] = replace(operations[0], deps=('begin',))
    result = graph_costs(raw, (prior, *operations), recipes=recipes(),
                         external_inputs={'begin': 1}, isolated=False, trace_limit=200)
    admission = next(e for e in result['events'] if e['task'].endswith(':buffers_acquire'))
    assert admission['start_ms'] == 5
    assert result['complete_ms'] == 17 # max(2 startup+3 work,12 response) after admission
    assert result['parameters']['service_regimes'][0]['queries'] == {'observable_response': 1}
    assert result['execution']['serviced']['vector@r0:s0'] == 8000
    clean(result)


@pytest.mark.parametrize('bandwidth,complete', [(.5, 20), (1, 19), (2, 19)])
def test_cross_dp_native_group_keeps_late_endpoint_wire_once_and_physical_bound(bandwidth, complete):
    raw = install(cohort_case(2), response(10000))
    raw['machine']['links'][0]['capacity_bytes_per_ms'] = bandwidth
    raw['machine']['device'].update(launch_us=2000, kernel_invocations={
        'vector': {'startup_ms': 3, 'minimum_ms': 8}})
    raw['runtime']['kernels']['families']['alltoall'] = {
        'kind': 'direct_peer_collective', 'max_inflight_per_rank': 1}
    graphs = {}
    for rank, volume in [(0, 0), (1, 4)]:
        entry = replace(phase('entry', entry=True, deps=('input'+str(rank),)),
                        vector_ops=0, metadata=phase('entry', entry=True).metadata +
                        (('invocation_class', 'vector'),))
        transfer = replace(exchange(volume, axis='ep'), op_id='wire', deps=('entry',),
                           metadata=exchange(volume, axis='ep').metadata + phase('wire').metadata)
        graphs[rank] = (entry, transfer, phase('exit', exit=True, deps=('wire',)))
    result = graph_costs(raw, graphs, external_inputs={'input0': 1, 'input1': 9},
                         include_submission=True, isolated=False, trace_limit=300)
    # Receiver has no outgoing bytes but must await rank1. Rank1 starts at12;
    # transfer is 8/4/2ms. Its own inclusive call floor ends19.
    assert result['complete_ms'] == complete
    assert sum(v for k,v in result['execution']['serviced'].items() if k.startswith('link:')) == 4
    assert result['parameters']['service_regimes'][0]['queries'] == {'observable_response': 2}
    assert result['execution']['submission']['eager_invocations'] == 2
    assert result['execution']['submission']['host_service_ms'] == 4
    clean(result)


def test_cube_scale_response_does_not_override_the_unscaled_fp32_physical_bound():
    operation = replace(op('mixed'), recipe='mixed-cube', source_boundary='kernel')
    def planner(operation, geometry):
        work = Work(matrix={'bf16': 2000, 'fp32': 2000})
        return OperatorPlan(work, lambda context: KernelProgram((
            Stage('matrix', context.bind(work)),)))
    book = RecipeBook({'mixed-cube': PlannedRecipe(planner)})
    rule = response(0, match={'recipe': 'mixed-cube'}, coordinates={
        'flops': {'path': 'work.cube'}}, response={'terms': [{
            'coordinate': 'flops', 'us_per_unit': 1,
            'scale_with': {'resource': 'cube', 'dtype': 'bf16', 'reference_rate': 2e6}}]})
    durations = []
    for factor in (.5, 1, 2):
        raw = install(hardware_case(), rule)
        raw['machine']['device']['compute_peaks']['bf16'] *= factor
        result = graph_costs(raw, (operation,), recipes=book, isolated=False)
        durations.append(result['duration_ms'])
        assert result['execution']['work']['flops'] == 4000
        clean(result)
    assert durations == [8, 4, 2.5]


def test_domain_miss_preserves_analytic_program_and_missing_coordinate_is_auditable():
    rule = response(9000, match={'recipe': copy_op(4000).recipe}, coordinates={
        'bytes': {'path': 'work.hbm'}}, domain={'bytes': {'min': 1000, 'max': 2000}})
    raw = install(hardware_case(), rule)
    report = graph_costs(raw, (copy_op(4000),), isolated=False)
    assert report['duration_ms'] == 4
    assert report['parameters']['service_regimes'][0]['queries'] == {'analytic_outside_domain': 1}
    raw['runtime']['kernels']['service_regimes'][0]['coordinates']['bytes']['path'] = 'metadata.not_declared'
    missing = graph_costs(raw, (copy_op(4000),), isolated=False)
    assert missing['duration_ms'] == 4
    assert missing['parameters']['service_regimes'][0]['queries'] == {'missing_coordinate': 1}
    clean(report); clean(missing)


def test_repeated_iteration_group_ids_have_independent_responses_and_release_all_state():
    parsed = parse_case(install(hardware_case(), response(12000)))
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    work = IterationWork(fork(), IterationPlan(()), 1, 0, 0, None, 'decode', 'group')
    first, second = execution.add(work, 0, 5), execution.add(work, 0, 5)
    execution.engine.advance(until=(first.completion, second.completion), first=False)
    assert (first.completion.time, second.completion.time) == (17, 29)
    assert next(iter(execution.cost.regimes.values()))['queries'] == {'observable_response': 2}
    assert execution.engine.summary()['live_tasks'] == execution.engine.summary()['live_programs'] == 0


def test_disjoint_physical_domains_require_distinct_receipt_identities():
    raw = hardware_case()
    base = response(3000, match={'recipe': copy_op(2000).recipe}, coordinates={
        'bytes': {'path': 'work.hbm'}}, domain={'bytes': {'min': 1000, 'max': 2000}})
    second = deepcopy(base); second.update(id='larger', domain={'bytes': {'min': 3000, 'max': 5000}},
                                         response={'fixed_us': 9000})
    raw['runtime']['kernels'] = {'service_regimes': [base, second]}
    operations = (copy_op(2000), replace(copy_op(4000), op_id='larger'))
    result = graph_costs(raw, operations, isolated=False)
    assert result['duration_ms'] == 12
    assert {r['id']: r['queries'] for r in result['parameters']['service_regimes']} == {
        'native-response': {'observable_response': 1, 'analytic_outside_domain': 1},
        'larger': {'analytic_outside_domain': 1, 'observable_response': 1}}
    second['id'] = base['id']
    with pytest.raises(ModelingError, match='service_regime_requires_unique_id'):
        graph_costs(raw, operations, isolated=False)
    clean(result)


def test_existing_primitive_minimum_and_regime_are_inclusive_competing_constraints():
    from mindie_sim.prediction.cost import primitive_key
    from mindie_sim.prediction.primitives import PrimitiveStore
    raw = install(hardware_case(), response(12000, match={'recipe': copy_op(4000).recipe}))
    raw['runtime']['kernels']['invocations'] = [{'match': {'recipe': copy_op(4000).recipe},
                                                'startup_ms': 3, 'minimum_ms': 8}]
    operation = copy_op(4000)
    with PrimitiveStore(':memory:') as store:
        store.record(primitive_key(parse_case(raw), operation, 'none'), duration_ms=20,
                     run_id='independent-control', repeat_index=0, source='test-known-bound')
        result = graph_costs(raw, (operation,), primitives=store, input_ready_ms=7, isolated=False)
    assert result['complete_ms'] == 27 # max(20 primitive,12 regime,3+4 physical), not 39
    assert result['execution']['serviced']['memory@r0:s0'] == 4000
    clean(result)


def test_group_link_scaling_uses_real_route_capacity_and_keeps_startup_on_the_critical_path():
    rule = response(0, coordinates={'bytes': {'path': 'work.link'}}, response={'terms': [{
        'coordinate': 'bytes', 'us_per_unit': 2000,
        'scale_with': {'resource': 'link', 'reference_rate': 1000}}]})
    spans = []
    for factor in (.5, 1, 2):
        raw = install(cohort_case(2), rule)
        raw['machine']['links'][0]['capacity_bytes_per_ms'] = factor
        raw['runtime']['kernels']['families']['alltoall'] = {
            'kind': 'direct_peer_collective', 'max_inflight_per_rank': 1}
        raw['runtime']['kernels']['invocations'] = [{'match': {'kernel':'NativeDispatch'}, 'startup_ms':3}]
        graphs = {}
        for rank in range(2):
            move = replace(exchange(4, axis='ep'), op_id='wire', deps=('entry',),
                           metadata=exchange(4, axis='ep').metadata + phase('wire').metadata)
            graphs[rank] = (phase('entry', entry=True), move, phase('exit', exit=True, deps=('wire',)))
        result = graph_costs(raw, graphs, isolated=False)
        spans.append(result['duration_ms'])
        assert sum(v for k,v in result['execution']['serviced'].items() if k.startswith('link:')) == 8
        assert result['parameters']['service_regimes'][0]['queries'] == {'observable_response':2}
        clean(result)
    assert spans == [16, 8, 5] # at2x, 3ms startup+2ms actual wire beats4ms response


def test_directional_service_mode_scaling_uses_the_declared_mode_rate_not_base_pipe():
    operation = replace(op('operand'), recipe='operand', source_boundary='kernel')
    def planner(operation, geometry):
        work = Work(resources={'mte1_l0a': 1000})
        return OperatorPlan(work, lambda context: KernelProgram((Stage('read', context.bind(work)),)))
    book = RecipeBook({'operand': PlannedRecipe(planner)})
    rule = response(0, match={'recipe':'operand'}, coordinates={
        'bytes': {'path':'work.mte1_l0a'}}, response={'terms':[{
            'coordinate':'bytes','us_per_unit':1,
            'scale_with':{'resource':'mte1_l0a','reference_rate':2e6}}]})
    spans = []
    for factor in (.5,1,2):
        raw = install(hardware_case(),rule)
        raw['machine']['device']['rates']['mte1'] = 4e6
        raw['machine']['device']['service_modes'] = {'mte1_l0a':{'resource':'mte1','rate':2e6*factor}}
        result=graph_costs(raw,(operation,),recipes=book,isolated=False)
        spans.append(result['duration_ms'])
        # Demand is normalized to the one real MTE1 pipe; changing the mode
        # changes occupancy, without inventing another physical engine.
        assert result['execution']['serviced']['mte1@r0:s0'] == 2000/factor
        clean(result)
    assert spans == [2,1,.5]


def test_inclusive_resource_curve_keeps_real_competing_supply_under_outer_response():
    from mindie_sim.prediction.services import ResourceService
    from test_resource_services import curve
    raw = install(hardware_case(), response(5000, match={'recipe':'copy'}))
    parsed = parse_case(raw)
    book = RecipeBook({'copy': ResourceService(curve(parsed, copy_op(2000)))})
    operations = tuple(replace(copy_op(2000), op_id=f'copy{i}', stream=f'copy{i}') for i in range(3))
    result = graph_costs(raw, operations, recipes=book, input_ready_ms=7, isolated=False)
    # Three 1ms setup phases overlap; actual6000B/1000Bms service takes6ms.
    # The observed5ms outer response does not erase shared-resource contention.
    assert result['complete_ms'] == 14
    assert result['execution']['serviced']['memory@r0:s0'] == 6000
    assert result['parameters']['service_regimes'][0]['queries'] == {'observable_response':3}
    clean(result)
