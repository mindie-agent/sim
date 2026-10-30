"""Causal pool conservation and real GLM q6/131201 tensor geometry controls."""
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.cache import index_selection_cost, layer_cache_components
from mindie_sim.catalog.index_pool import pool_selection
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction.structure import parse_case


ROOT = Path(__file__).resolve().parents[1]


def case():
    raw = json.loads((ROOT / 'examples/cases/high-concurrency-20261030/matrix-extension/'
                     'a3_glm53flash_tp16_aclgraph_c32.json').read_text())
    raw['model']['layers'] = 4
    raw['runtime']['kernels']['programs']['indexed_attention'] = {
        'kind': 'indexed_attention_packed_int8', 'scratch_policy': 'offchip'}
    return raw


def layer(raw, batch=1, query=6, context=131201):
    return [op for op in target_graphs(raw, scope='body', phase='decode', tokens=query,
                                     context=context, sequences=batch)[0] if op.layer == 3]


def test_pool_scan_and_expansion_match_independent_causal_sets():
    for context in (0, 1, 3, 7, 2046, 4096, 131201):
        for query in (0, 1, 3, 6, 9):
            result = pool_selection(query, context, 2048, 4, always_select_tail=True)
            visible = [n for n in range(context + 1, context + query + 1)]
            complete = [n // 4 for n in visible]
            selected = [min(n, 512) for n in complete]
            tails = [n % 4 for n in visible]
            assert result.scanned_pools == sum(complete)
            assert result.selected_pools == sum(selected)
            assert result.selected_tokens == sum(n * 4 + t for n, t in zip(selected, tails))
            assert result.causal_tail_tokens == sum(tails)
            assert result.completed_pools == len([n for n in visible if n % 4 == 0])


def test_catalog_pool_cache_keeps_full_token_kv_and_real_gate_parameters():
    parsed = parse_case(case()); plan = parsed.model.layer_plans[3]
    assert (plan.index_kpool, plan.index_kpool_compress, plan.index_kpool_always_select_tail) == (4, True, True)
    cache = layer_cache_components(parsed.model, plan, parsed.runtime.parallel, 131201,
                                   capacity_padded=False)
    assert cache['index_k'] == (131201 // 4) * 128 * 2
    assert cache['index_pool_tail'] == 4 * 2 * 128 * 4
    assert cache['mla_kv'] > cache['index_k'] * 4
    selection = index_selection_cost(parsed.model, plan, 131201)
    assert selection.scored_rows == 131201 // 4
    terms = {term.name: term for term in layer_weight_terms(plan, parsed.model)}
    assert terms['indexer_pool_gate'].rows == 128
    assert terms['indexer_pool_gate'].cols == parsed.model.hidden_size
    assert terms['indexer_pool_ape'].rows == 4
    for name, width in (('indexer_wk', 128), ('indexer_weights', 32), ('indexer_pool_gate', 128)):
        original, prepared = terms[name], terms[name + '_prepared_fp32']
        assert term_bytes(original, parsed.model.quant, parsed.runtime.parallel) == width * 4096 * 2
        assert term_bytes(prepared, parsed.model.quant, parsed.runtime.parallel) == width * 4096 * 4
        assert term_bytes(prepared, parsed.model.quant, parsed.runtime.parallel, source=True) == 0


@pytest.mark.parametrize('batch', [1, 4, 8])
def test_real_q6_buckets_scan_all_pools_and_keep_2051_capacity(batch):
    ops = {op.recipe: op for op in layer(case(), batch)}
    scanned = batch * sum((131201 + i) // 4 for i in range(1, 7))
    scores = ops['index_pool_scores']; meta = dict(scores.metadata)
    assert meta['scanned_pools'] == scanned
    assert scores.flops == 0
    assert scores.vector_ops == 2 * scanned * 128
    assert meta['score_mode'] == 'linear_weighted_query'
    assert ops['index_pool_topk'].local_shape == (batch * 6, 512)
    assert ops['index_topk'].local_shape == (batch * 6, 2051)
    assert dict(ops['index_topk'].metadata)['selected_token_rows'] == batch * (6 * 2048 + 11)
    attention = next(op for op in ops.values() if op.family == 'attention')
    attn = dict(attention.metadata)
    assert attn['indexed_gather_rows'] == batch * (6 * 2048 + 11)
    assert attn['logical_pairs'] == attn['indexed_gather_rows']
    assert ops['index_pool_tail_store'].deps[0].endswith('.index_pool_compress')


def test_4k_to_131201_changes_scan_work_but_not_the_2048_token_history_budget():
    values = []
    for context in (4096, 131201):
        ops = {op.recipe: op for op in layer(case(), context=context)}
        values.append(ops)
    lo, hi = values
    assert hi['index_pool_scores'].vector_ops / lo['index_pool_scores'].vector_ops == pytest.approx(
        sum((131201+i)//4 for i in range(1,7)) / sum((4096+i)//4 for i in range(1,7)))
    assert hi['index_topk'].local_shape == lo['index_topk'].local_shape == (6, 2051)
    assert hi['index_pool_scores'].vector_ops > 31 * lo['index_pool_scores'].vector_ops


def test_source_fp32_projections_share_hidden_cast_and_fused_key_weight_output():
    ops = {op.recipe: op for op in layer(case())}
    projected = ops['index_k_weights']; gate = ops['index_pool_gate']
    assert projected.local_shape == (6, 160, 4096)
    assert projected.dtype == gate.dtype == 'fp32'
    assert projected.deps == gate.deps == (ops['index_pool_hidden_fp32'].op_id,)
    assert dict(projected.metadata)['packed_weight_bytes'] == 160 * 4096 * 4
    assert ops['index_norm'].deps == (projected.op_id,)
    assert ops['index_norm'].mte2_bytes == (6 + 2) * 128 * 4
    assert 'index_norm_input_fp32' not in ops
    assert 'index_weights_input_cast' not in ops
    assert 'index_weights' not in ops
    weights = ops['index_weights_cast']
    assert weights.deps == (projected.op_id,)
    assert weights.mte2_bytes == 6 * 32 * 4
    assert dict(weights.metadata)['input_column_slice'] == (128, 160)
    assert ops['index_pool_weighted_query'].deps == (
        ops['index_pool_query_fp32'].op_id, ops['index_pool_weights_fp32'].op_id)


def test_declared_per_head_alternative_keeps_pool_scan_and_token_selection():
    raw = case(); reference = {op.recipe: op for op in layer(raw)}
    raw['runtime']['kernels'].setdefault('lowerings', {})['pool_key_indexer'] = {
        'implementation': 'causal_pool_indexer_analytic',
        'options': {'score_mode': 'per_head_relu'}}
    other = {op.recipe: op for op in layer(raw)}
    assert other['index_pool_scores'].flops == dict(reference['index_pool_scores'].metadata)['scanned_pools'] * 32 * 256
    assert reference['index_pool_weighted_query'].vector_ops == 6 * 32 * 128 * 2
    assert 'index_pool_weighted_query' not in other
    assert other['index_topk'].local_shape == reference['index_topk'].local_shape


def test_graph_padding_preserves_projection_capacity_without_fake_pool_writes():
    ops = {op.recipe: op for op in layer(case(), batch=3)}
    # The retained graph buckets pad 18 valid target queries to 24 rows.
    for name in ('index_pool_hidden_fp32', 'index_k_weights', 'index_pool_gate',
                 'index_norm', 'index_weights_cast', 'index_pool_query_fp32',
                 'index_pool_weighted_query', 'index_topk'):
        assert ops[name].local_shape[0] == 24
    metadata = dict(ops['index_pool_scores'].metadata)
    assert metadata['valid_query_rows'] == 18
    assert metadata['padded_query_rows'] == 24
    assert metadata['scanned_pools'] == 3 * sum((131201 + i) // 4 for i in range(1, 7))
    assert dict(ops['index_pool_compress'].metadata)['completed_pools'] == 3
    assert dict(ops['index_pool_tail_store'].metadata)['retained_rows'] == 12
    assert dict(ops['index_topk'].metadata)['selected_token_rows'] == 3 * (6 * 2048 + 11)


def test_compact_and_detailed_sparse_pipelines_conserve_pool_tail_work_and_readiness():
    reports = []
    for compact in (False, True):
        raw = case(); raw['runtime']['kernels']['programs']['indexed_attention']['compact'] = compact
        ops = [op for op in layer(raw, 2, 3, 2047) if op.family == 'attention']
        result = graph_costs(raw, ops, input_ready_ms=5, isolated=False,
                             external_inputs={dep: 7 for op in ops for dep in op.deps}, trace_limit=1000)
        assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
        assert all(row['start_ms'] >= 7 for row in result['events'] if row['demand'])
        reports.append(result)
    assert reports[0]['execution']['serviced'] == pytest.approx(reports[1]['execution']['serviced'])
