"""Source caller contracts against retained B4/B8 native tensor geometry."""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction.index_pool_native import IMPLEMENTATION
from mindie_sim.prediction.structure import parse_case
from test_index_pool import case, layer


def native_case(tail_capacity=9):
    raw = case()
    raw['runtime']['kernels'].setdefault('lowerings', {})['pool_key_indexer'] = {
        'implementation': IMPLEMENTATION,
        'options': {'tail_capacity': tail_capacity, 'block_table_columns': 8195}}
    return raw


@pytest.mark.parametrize('batch', (4, 8))
def test_native_q6_call_shapes_capacity_and_separate_projection_formats(batch):
    ops = {op.recipe: op for op in layer(native_case(), batch)}
    pool, indexer = ops['cann_key_pool'], ops['cann_pool_key_indexer']
    p, i = dict(pool.metadata), dict(indexer.metadata)
    assert pool.local_shape == (batch * 6, 128, 4096)
    assert pool.dtype == 'bf16'
    assert pool.flops == 4 * batch * 6 * 128 * 4096
    assert p['projection_formats'] == {'hidden': 'bf16', 'wk': 'bf16', 'gate': 'bf16'}
    assert p['output_shape'] == (batch * 6 // 4 + batch, 128)
    assert p['valid_output_rows'] == batch
    state = dict(ops['cann_keypool_state_gather'].metadata)
    assert state['state_shape'] == (1 + 2 * batch, 4, 256)
    assert indexer.local_shape == (batch * 6 + 1, 32, 128)
    assert i['output_shape'] == (batch * 6 + 1, 2051)
    assert dict(ops['cann_keypool_nonempty_inputs'].metadata)['block_table_shape'] == (batch + 1, 8195)
    assert ops['index_weights'].local_shape == (batch * 6, 32, 4096)
    assert ops['index_weights'].dtype == 'fp32'
    assert ops['index_weights'].deps == (ops['index_weights_input_cast'].op_id,)
    assert ops['index_weights_cast'].mte2_bytes == batch * 6 * 32 * 4
    assert ops['index_weights_scale'].op_id in ops['cann_keypool_state_gather'].deps
    assert list(ops).index('index_weights_scale') < list(ops).index('cann_key_pool')
    assert dict(ops['index_topk'].metadata)['selected_token_rows'] == batch * (6 * 2048 + 11)
    assert not dict(ops['index_topk'].metadata)['pool_id_expansion_in_adapter']
    attention = next(op for op in ops.values() if op.family == 'attention')
    assert dict(attention.metadata)['indexed_gather_rows'] == batch * (6 * 2048 + 11)


def test_graph_padding_belongs_to_dummy_request_and_does_not_create_real_candidates():
    ops = {op.recipe: op for op in layer(native_case(), 3)}
    meta = dict(ops['cann_pool_key_indexer'].metadata)
    assert meta['valid_query_rows'] == 18
    assert meta['padded_query_rows'] == 24
    assert meta['native_query_rows'] == 25
    assert meta['dummy_request_queries'] == 7
    assert meta['scanned_pools'] == 3 * sum((131201 + i) // 4 for i in range(1, 7))
    assert ops['index_topk'].local_shape == (24, 2051)
    assert dict(ops['index_topk'].metadata)['selected_token_rows'] == 3 * (6 * 2048 + 11)


def test_lookahead_reprojection_preserves_retained_window_and_cache_residency():
    ordinary = {op.recipe: op for op in layer(native_case(4), 4)}
    rollback = {op.recipe: op for op in layer(native_case(9), 4)}
    write = dict(rollback['cann_keypool_cache_write'].metadata)
    assert write['tail_launch_rows'] == write['valid_tail_rows'] == 24
    assert write['masked_pooled_rows'] == 10 - 4
    assert rollback['cann_keypool_rollback_key'].local_shape == (24, 128, 4096)
    assert rollback['cann_keypool_rollback_gate'].local_shape == (24, 128, 4096)
    assert rollback['cann_keypool_rollback_key_fp32'].mte2_bytes == 24 * 128 * 2
    assert rollback['index_norm'].deps == (rollback['cann_keypool_rollback_key_fp32'].op_id,)
    assert dict(ordinary['cann_keypool_cache_write'].metadata)['tail_mode'] == 'cann_state_tail'
    assert dict(ordinary['cann_keypool_cache_write'].metadata)['valid_tail_rows'] == 4 * 3
    parsed = parse_case(native_case())
    plan = parsed.model.layer_plans[3]
    cache = layer_cache_components(parsed.model, plan, parsed.runtime.parallel, 131201)
    assert cache['index_pool_tail'] == 9 * 2 * 128 * 4
    terms = {t.name: t for t in layer_weight_terms(plan, parsed.model)}
    assert term_bytes(terms['indexer_weights_prepared_fp32'], parsed.model.quant,
                      parsed.runtime.parallel) == 32 * 4096 * 4
    assert not any(name.startswith(('indexer_wk_prepared_', 'indexer_pool_gate_prepared_')) for name in terms)


def test_native_boundaries_accept_explicit_joint_service_without_fitting_body_or_adapters():
    raw = native_case()
    ops = {op.recipe: op for op in layer(raw, 4)}
    names = ('cann_key_pool', 'cann_pool_key_indexer')
    kernels = tuple(ops[name] for name in names)
    configured = deepcopy(raw)
    configured['machine']['device'] = {
        'rates': {'observed_pool': 1e9}, 'work_units': {'observed_pool': 'busy_ns'},
        'resource_units': {'observed_pool': 1}}
    configured['runtime']['kernels']['service_regimes'] = [
        {'id': name, 'match': {'recipe': name}, 'response': {'fixed_us': us},
         'execution': {'kind': 'joint_native', 'resource': 'observed_pool',
                       'reference_rate': 1e9, 'native_kernel': dict(ops[name].metadata)['kernel']},
         'provenance': {'kind': 'synthetic_control; not_capture'}}
        for name, us in zip(names, (100, 200))]
    report = graph_costs(configured, kernels, isolated=False, input_ready_ms=7,
                         external_inputs={dep: 7 for op in kernels for dep in op.deps})
    assert report['duration_ms'] == pytest.approx(.3)
    assert report['execution']['serviced']['observed_pool@r0:s0'] == pytest.approx(300000)
    assert all(row['program'] == 'JointNativeService' for row in report['timeline'])
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    # Invocation service is independent of the source adapter bytes and graph.
    assert ops['cann_keypool_nonempty_inputs'].hbm_bytes > 0
    assert ops['cann_keypool_cache_write'].hbm_bytes > 0


def test_complete_native_caller_executes_after_inputs_and_keeps_cache_to_selection_order():
    raw = native_case()
    ops = tuple(op for op in layer(raw, 4) if op.recipe.startswith(('cann_key', 'cann_pool', 'index_')))
    ids = {op.op_id for op in ops}
    report = graph_costs(raw, ops, isolated=False, input_ready_ms=7,
                         external_inputs={dep: 7 for op in ops for dep in op.deps if dep not in ids})
    rows = {row['op_id'].rsplit('.', 1)[-1]: row for row in report['timeline']}
    assert rows['cann_key_pool']['ready_ms'] >= rows['index_weights_scale']['complete_ms']
    assert rows['cann_pool_key_indexer']['ready_ms'] >= rows['cann_keypool_cache_write']['complete_ms']
    assert rows['index_topk']['ready_ms'] >= rows['cann_pool_key_indexer']['complete_ms']
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
