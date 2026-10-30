"""Native AI_CPU boundary, batch multiplicity and attention dependencies."""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.sparse_mla_metadata import IMPLEMENTATION, NATIVE, SERVICE
from test_indexed_attention_rope0 import case


def raw_case():
    raw = case()
    raw['runtime']['kernels']['lowerings'] = {'sparse_mla_metadata': {
        'implementation': IMPLEMENTATION, 'options': {'layer_groups': [[0]]}}}
    return raw


def selected_graph(raw, batch):
    graph = target_graphs(raw, phase='decode', tokens=4, sequences=batch,
                          context=4096, scope='body')[0]
    metadata = next(op for op in graph if op.recipe == IMPLEMENTATION)
    publish = next(op for op in graph if op.recipe == 'sparse_mla_metadata_publish')
    attention = next(op for op in graph if op.family == 'attention')
    assert len([op for op in graph if op.recipe == IMPLEMENTATION]) == 1
    assert publish.deps == (metadata.op_id,)
    assert publish.op_id in attention.deps
    return metadata, publish, attention


@pytest.mark.parametrize('batch', (1, 8))
def test_one_cpu_invocation_per_batch_is_a_real_body_attention_dependency(batch):
    raw = raw_case()
    nodes = selected_graph(raw, batch)
    meta = dict(nodes[0].metadata)
    assert meta['execution_domain'] == 'AI_CPU'
    assert meta['metadata_call']['rows'] == batch * 4
    assert not any(op.vector_ops or op.flops for op in nodes[:2])
    raw['runtime']['kernels']['service_regimes'] = [{
        'id': 'typed-native-control', 'match': {'kernel': NATIVE},
        'response': {'fixed_us': 100}, 'provenance': {'kind': 'synthetic_control'}}]
    keys = {op.op_id for op in nodes}
    report = graph_costs(raw, nodes, isolated=False, input_ready_ms=7,
        external_inputs={d: 7 for op in nodes for d in op.deps if d not in keys})
    rows = {row['op_id']: row for row in report['timeline']}
    assert rows[nodes[0].op_id]['elapsed_ms'] == pytest.approx(.1)
    assert rows[nodes[2].op_id]['ready_ms'] >= rows[nodes[1].op_id]['complete_ms']
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert report['parameters']['service_regimes'][0]['queries'] == {'observable_response': 1}


def test_controller_capacity_scales_once_without_borrowing_vector_or_host_supply():
    base = raw_case()
    node = selected_graph(base, 2)[0]
    spans = []
    for factor in (.5, 1, 2):
        raw = deepcopy(base)
        raw['machine']['device'] = {'rates': {SERVICE: 1e4 * factor},
                                    'resource_units': {SERVICE: 1},
                                    'work_units': {SERVICE: 'calls'}}
        report = graph_costs(raw, (node,), isolated=False,
                            external_inputs={dep: 7 for dep in node.deps})
        spans.append(report['duration_ms'])
        work = report['execution']['serviced']
        assert work[SERVICE + '@r0:s0'] == 1
        assert not work.get('vector@r0:s0', 0) and not work.get('host@r0:s0', 0)
        assert work['memory@r0:s0'] == 4 * (2 * 2 + 1 + 8) + 4096
    assert spans[0] > spans[1] > spans[2]
    assert spans[0] - spans[1] == pytest.approx(.1)
    assert spans[1] - spans[2] == pytest.approx(.05)


def test_multiple_layers_share_only_the_declared_metadata_group_and_keep_padded_input():
    raw = raw_case()
    raw['model']['layers'] = 3
    raw['model']['layer_overrides'] = {str(i): {'qk_rope_head_dim': 0} for i in range(3)}
    raw['runtime'].update(graph_mode='auto', graph_buckets=[16])
    options = raw['runtime']['kernels']['lowerings']['sparse_mla_metadata']['options']
    for groups, calls in (([[0, 1, 2]], 1), ([[0], [1], [2]], 3)):
        options['layer_groups'] = groups
        graph = target_graphs(raw, phase='decode', tokens=4, sequences=3,
                              context=4096, scope='body')[0]
        nodes = [op for op in graph if op.recipe == IMPLEMENTATION]
        assert len(nodes) == calls
        assert all(dict(op.metadata)['metadata_call']['rows'] == 16 for op in nodes)
        assert all(dict(op.metadata)['metadata_call']['valid_rows'] == 12 for op in nodes)
        publications = {op.op_id for op in graph if op.recipe == 'sparse_mla_metadata_publish'}
        attention = [op for op in graph if op.family == 'attention']
        assert len(attention) == 3
        assert all(set(op.deps) & publications for op in attention)
        assert all(op.recipe in {IMPLEMENTATION, 'sparse_mla_metadata_publish'}
                   for op in graph[:2 * calls])
        assert all(op.deps == ('analysis_input',) for op in nodes)
        first_layer_input = next(op for op in graph[2 * calls:]
                                 if 'analysis_input' in op.deps)
        assert publications <= set(first_layer_input.deps)
        ids = {op.op_id for op in graph}
        available = {dep for op in graph for dep in op.deps if dep not in ids}
        for op in graph:
            assert all(dep in available for dep in op.deps)
            available.add(op.op_id)


def test_captured_layer_ordered_metadata_stays_after_selection_before_attention():
    raw = raw_case()
    raw['runtime']['kernels']['lowerings']['sparse_mla_metadata']['options']['schedule'] = 'layer_ordered'
    graph = target_graphs(raw, phase='decode', tokens=4, sequences=2,
                          context=4096, scope='body')[0]
    node = next(op for op in graph if op.recipe == IMPLEMENTATION)
    positions = {op.op_id: i for i, op in enumerate(graph)}
    selector = next(op for op in graph if op.op_id == node.deps[0])
    assert selector.family == 'topk'
    attention = next(op for op in graph if op.family == 'attention')
    assert positions[selector.op_id] < positions[node.op_id] < positions[attention.op_id]
    assert dict(node.metadata)['metadata_schedule'] == 'layer_ordered'
