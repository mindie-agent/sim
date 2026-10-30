"""Explicit rope-free analytic geometry retains finite sparse pipeline service.

[Role] Check the new geometry on the real shared Engine and public frontend.
[Invariants] The extension neither claims a native ABI nor changes defaults.
[Timeouts & Locks] Bounded component graphs only; no subprocesses or locks.
[Side-Effect Policy] In-memory geometry and hardware declarations only.
"""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.prediction.indexed_attention import IMPLEMENTATION, build
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


def case(*, compact=True, explicit=True):
    kernels = {}
    if explicit:
        kernels['programs'] = {'indexed_attention': {
            'kind': 'indexed_attention_pipeline', 'scratch_policy': 'offchip',
            'operand_modes': False, 'compact': compact, 'rope0_analytic': True}}
    return {
        'model': {'catalog_key': 'glm-5.3', 'layers': 1,
                  'layer_overrides': {'0': {'qk_rope_head_dim': 0}}},
        'machine': {'backend': 'ascend', 'sku': 'Ascend950DT_9582', 'num_devices': 8},
        'runtime': {'tp': 8, 'kernels': kernels},
        'workload': {'requests': [{'id': 'r', 'prompt_tokens': 1, 'output_tokens': 1}]},
    }


def operators(raw, *, batch=2, query=4, context=511):
    return [op for op in target_graphs(raw, scope='body', phase='decode', tokens=query,
                                      sequences=batch, context=context)[0]
            if dict(op.metadata).get('program_kind') == 'indexed_attention']


def execute(raw, ops, *, trace=0):
    return graph_costs(raw, ops, input_ready_ms=5,
                       external_inputs={dep: 7 for op in ops for dep in op.deps},
                       isolated=False, trace_limit=trace)


@pytest.mark.parametrize('batch,query,context', [(1, 1, 0), (2, 4, 511), (3, 1, 1023)])
def test_rope0_compact_and_detailed_preserve_payload_and_ready_owners(batch, query, context):
    raw = case()
    ops = operators(raw, batch=batch, query=query, context=context)
    assert len(ops) == 1
    shape = dict(ops[0].metadata)['indexed_attention_geometry']
    assert shape['rope'] == 0 and shape['latent'] == 512
    plan = build(ops[0], Geometry.from_devices(parse_case(raw).machine.dies),
                 scratch_policy='offchip', rope0_analytic=True)
    assert plan.metadata['coverage'] == 'analytic_geometry_extension'
    compact = execute(raw, ops, trace=5000)
    detailed = execute(case(compact=False), ops, trace=5000)
    assert compact['execution']['serviced'] == pytest.approx(detailed['execution']['serviced'])
    assert compact['execution']['work'] == detailed['execution']['work']
    gathers = [row for row in detailed['events'] if ':gather' in row['task']]
    # Cache K/V alias each selected BF16 latent row once, plus two INT32 indices.
    # The detailed control exposes each gather on the real shared Engine.
    if query == 1 and context == 0:
        assert sum(row['demand'].get('memory@r0:s0', 0) for row in gathers) == 1024 + 8
    assert all(row['start_ms'] >= 7 for row in compact['events'] if row['demand'])
    for report in (compact, detailed):
        assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
        assert not any(report['execution']['live_allocated'].values())
    # No extra vector owner is manufactured for a single query/softmax row.
    if batch == query == 1:
        softmax = [row for row in detailed['events'] if ':softmax' in row['task']]
        assert softmax and all(tuple(row['placement']['vector@r0:s0']) == (0,) for row in softmax)


def test_rope0_is_explicit_and_native_selector_keeps_its_true_geometry_domain():
    raw = case(explicit=False)
    ops = operators(raw, batch=1)
    report = execute(raw, ops)
    assert report['timeline'][0]['program'] == 'ResourceEnvelope'
    native = deepcopy(raw)
    native['runtime']['kernels'] = {'lowerings': {'indexed_attention': {
        'implementation': IMPLEMENTATION, 'options': {'scratch_policy': 'offchip'}}}}
    native_ops = operators(native, batch=1)
    with pytest.raises(ModelingError, match='source_geometry_outside_reviewed_domain'):
        execute(native, native_ops)
    explicit = case()
    explicit_report = operator_costs(explicit, operators(explicit))
    metadata = explicit_report['operators'][0]['operator']['metadata']['program_execution']
    assert metadata['coverage'] == 'analytic_geometry_extension'
    assert metadata['kernel_dispatch'] == 'not_established'


def test_rope0_shared_memory_rate_remains_a_real_physical_supply():
    values = []
    for rate in (2e10, 4e10, 8e10):
        raw = case()
        raw['machine']['hbm_bw_bytes_per_s'] = rate
        ops = operators(raw, batch=3, query=4, context=2048)
        result = execute(raw, ops)
        values.append(result['complete_ms'] - 7)
        if len(values) == 1:
            reference = result['execution']['work']
        else:
            assert result['execution']['work'] == reference
    assert values[0] > values[1] > values[2]
