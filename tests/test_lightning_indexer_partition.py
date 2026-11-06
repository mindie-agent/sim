"""Native indexer domains preserve sequence partitions at equal total work."""
from mindie_sim import operator_costs, target_operators
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def indexed_case():
    raw = case(tp=8)
    raw['model'] = {'catalog_key': 'GLM-5.3-W4A8', 'layers': 1}
    raw['runtime'].update(graph_mode='eager', kernels={'lowerings': {
        'native_fusion': {'implementation': 'declared_native_fusion',
                          'options': {'roles': ['lightning_indexer']}}}})
    device = raw['machine']['device']
    device['rates']['native_call_service'] = 1e9
    device['resource_units'] = {'native_call_service': 1}
    device['work_units'] = {'native_call_service': 'busy_ns'}
    return raw


def indexer(ops):
    return next(op for op in ops if dict(op.metadata).get('fusion_role') == 'lightning_indexer')


def test_equal_tokens_and_flops_do_not_match_another_sequence_partition():
    raw = indexed_case()
    one = indexer(target_operators(raw, phase='prefill', tokens=8, context=0))
    two = indexer(target_operators(raw, phase='prefill', tokens=4, context=2, sequences=2))
    four = indexer(target_operators(raw, phase='prefill', tokens=2, context=3, sequences=4))
    assert one.local_shape == two.local_shape == four.local_shape
    view = dict(one.metadata)['fusion_tensor_view']
    assert one.flops == two.flops == four.flops == 2 * 36 * view['heads'] * view['width']
    partition = view['sequence_partition']
    assert partition == ({'query': 8, 'context': 0, 'kv_length': 8},)
    raw['runtime']['kernels']['service_regimes'] = [{
        'id': 'one-sequence-kernel', 'match': {
            'recipe': 'declared_native_fusion',
            'metadata.fusion_tensor_view.sequence_partition': partition},
        'execution': {'kind': 'joint_native', 'resource': 'native_call_service',
                      'reference_rate': 1e9, 'native_kernel': 'LightningIndexer'},
        'response': {'fixed_us': 123},
    }]
    rows = [operator_costs(raw, (op,))['operators'][0] for op in (one, two, four)]
    assert rows[0]['program'] == 'JointNativeService'
    assert rows[0]['duration_ms'] == .123
    assert all(row['program'] != 'JointNativeService' for row in rows[1:])


def test_partition_order_is_physical_but_request_ids_are_not():
    parsed = parse_case(indexed_case())

    def lower(queries):
        work = Frontend()(IterationPlan(queries), parsed.model, parsed.runtime, parsed.machine)
        return dict(indexer(work.ops).metadata)['fusion_tensor_view']

    views = (RequestView('first', 2, 7, False, 'prefill'),
             RequestView('second', 6, 1, False, 'prefill'))
    original = lower(views)
    renamed = lower(tuple(RequestView(str(i), v.q_len, v.context, False, 'prefill')
                          for i, v in enumerate(views)))
    reversed_order = lower(views[::-1])
    assert original == renamed
    assert original != reversed_order
    assert original['sequence_partition'] == (
        {'query': 2, 'context': 7, 'kv_length': 9},
        {'query': 6, 'context': 1, 'kv_length': 7})
    assert reversed_order['sequence_partition'] == original['sequence_partition'][::-1]


def test_graph_bucket_padding_is_separate_from_active_indexer_queries():
    raw = indexed_case()
    raw['runtime'].update(graph_mode='full', graph_buckets=[4], graph_capture_keys=[4])
    op = indexer(target_operators(raw, phase='decode', tokens=1, context=8, sequences=3))
    view = dict(op.metadata)['fusion_tensor_view']
    assert view['tokens'] == 3
    assert view['physical_tokens'] == 4
    assert view['sequence_partition'] == ({'query': 1, 'context': 8, 'kv_length': 9},) * 3
    assert op.local_shape[0] == 3
    assert op.flops == 2 * 3 * 9 * view['heads'] * view['width']
