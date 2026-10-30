"""A native HCCL response cannot transfer across changed physical transport."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.services import service_identity
from mindie_sim.prediction.structure import parse_case
from test_hccl_source_graph import declared, isolated_calls


def source_call():
    raw = declared('AICPU_TS', 'mesh_oneshot')
    ops = isolated_calls(raw)[0]
    group = dict(ops[0].metadata)['hccl_source_group']
    selected = tuple(op for op in ops if dict(op.metadata)['hccl_source_group'] == group)
    keys = {op.op_id for op in selected}
    selected = tuple(replace(op, deps=tuple(d for d in op.deps if d in keys)) for op in selected)
    metadata = dict(selected[0].metadata)
    identity = service_identity(parse_case(raw), selected[0])
    raw['runtime']['kernels']['service_regimes'] = [{
        'id': 'captured-native-communication-domain',
        'match': {'kernel': metadata['device_kernel_group']['kernel'],
                  'transport': identity['transport']},
        'response': {'fixed_us': 5000},
        'provenance': {'kind': 'causal_control', 'boundary': 'one_native_collective_invocation'},
    }]
    return raw, selected


@pytest.mark.parametrize('field,value', [('capacity_bytes_per_ms', 5e5), ('latency_us', 10)])
def test_changed_link_exits_native_response_and_retains_exact_analytic_program(field, value):
    raw, ops = source_call()
    matched = graph_costs(raw, ops, input_ready_ms=7, isolated=False, trace_limit=10000)
    assert matched['duration_ms'] == 5
    assert matched['parameters']['service_regimes'][0]['queries'] == {'observable_response': 1}
    assert sum(event['task'].endswith(':minimum') for event in matched['events']) == 1

    changed = deepcopy(raw)
    changed['machine']['links'][0][field] = value
    observed = graph_costs(changed, ops, input_ready_ms=7, isolated=False, trace_limit=10000)
    del changed['runtime']['kernels']['service_regimes']
    analytic = graph_costs(changed, ops, input_ready_ms=7, isolated=False, trace_limit=10000)
    assert observed['parameters']['service_regimes'][0]['queries'] == {'physical_identity_mismatch': 1}
    assert observed['duration_ms'] == analytic['duration_ms'] < matched['duration_ms']
    assert observed['events'] == analytic['events']
    assert observed['execution'] == analytic['execution']
    assert observed['execution']['serviced'] == matched['execution']['serviced']
    assert observed['execution']['live_tasks'] == observed['execution']['live_programs'] == 0
