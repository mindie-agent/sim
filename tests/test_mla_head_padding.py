"""Real logical/physical head and mixed router-output precision contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from test_prediction_system import case

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CASE = ROOT / 'examples/cases/high-concurrency-20261030/kimi-variants/a5_kimi_tp8_e16_bf16_c32_p4k.json'


def control(heads=12, selected=True):
    raw = case(prompt=1, output=2)
    raw['model'].update(layers=1, attention='mla', num_heads=heads, num_kv_heads=1,
                        head_dim=192, mla={'q_lora_rank': 32, 'kv_lora_rank': 512,
                        'qk_nope_head_dim': 128, 'qk_rope_head_dim': 64, 'v_head_dim': 128})
    if selected:
        raw['runtime']['kernels'] = {'lowerings': {
            'mla_head_padding': 'vllm_ascend_mla_head_padding'}}
    return raw


def ops(raw, batch=2, phase='decode'):
    return {op.recipe: op for op in target_graphs(
        raw, phase=phase, tokens=1, context=16, sequences=batch)[0]}


@pytest.mark.parametrize('logical,physical', [(3, 4), (12, 16), (16, 16), (20, 32)])
def test_projection_and_fia_pad_only_the_physical_head_axis(logical, physical):
    graph = ops(control(logical))
    assert graph['q_b'].local_shape == (2, physical * 192, 32)
    assert dict(graph['q_absorb'].metadata)['matrix_groups'] == physical
    assert graph['latent_attention0'].local_shape[2] == physical
    assert dict(graph['v_expand'].metadata)['matrix_groups'] == logical
    assert graph['output'].local_shape == (2, 32, logical * 128)
    assert graph['kv_store'].mte3_bytes == 2 * (512 + 64) * 2


@pytest.mark.parametrize('batch', [1, 2, 32])
def test_views_are_free_but_stride_copy_and_pad_traffic_are_real(batch):
    graph = ops(control(), batch=batch)
    assert ('mla_query_contiguous' in graph) == (batch > 1)
    if batch > 1:
        copy = graph['mla_query_contiguous']
        assert copy.mte2_bytes == copy.mte3_bytes == batch * 12 * 512 * 2
        assert graph['mla_query_pad_nope'].deps == (copy.op_id,)
    else:
        assert graph['mla_query_pad_nope'].deps == (graph['q_absorb'].op_id,)
    for part, width in [('nope', 512), ('rope', 64)]:
        pad = graph['mla_query_pad_' + part]
        assert pad.mte2_bytes == batch * 12 * width * 2
        assert pad.mte3_bytes == batch * 16 * width * 2
        assert dict(pad.metadata)['zero_fill_bytes'] == batch * 4 * width * 2
    assert graph['v_expand'].deps == tuple(graph[f'latent_attention{i}'].op_id for i in range(batch))
    assert not any('slice' in recipe for recipe in graph)
    assert not any(op.mte2_bytes or op.mte3_bytes for name, op in graph.items() if name.endswith('_view'))


def test_ordinary_prefill_and_unselected_mla_keep_logical_geometry():
    for raw, phase in ((control(selected=False), 'decode'), (control(), 'prefill')):
        graph = ops(raw, phase=phase)
        assert graph['q_b'].local_shape[1] == 12 * 192
        assert dict(graph['q_absorb'].metadata)['matrix_groups'] == 12
        assert not any(name.startswith('mla_query_') for name in graph)
    parsed = parse_case(control())
    unbound = replace(parsed, runtime=replace(parsed.runtime, kernels={}))
    assert ops(unbound)['q_b'].local_shape[1] == 12 * 192
    assert 'mla_head_padding' not in dict(parse_case(unbound).model.metadata)
    selected = control()
    selected['runtime']['kernels']['lowerings']['mla_decode'] = 'vllm_triton_mla_decode_tuple'
    with pytest.raises(ModelingError, match='conflicts_with_explicit_mla'):
        target_graphs(selected, phase='decode', tokens=1, context=16)


def test_prepared_weights_are_runtime_copies_not_cache_or_checkpoint_parameters():
    base, padded = map(parse_case, (control(selected=False), control()))
    plan = padded.model.layer_plans[0]
    plain_terms = layer_weight_terms(base.model.layer_plans[0], base.model)
    terms = layer_weight_terms(plan, padded.model)
    added = [term for term in terms if term.name.startswith('mlapo_prepared_')]
    assert {term.name for term in added} == {
        'mlapo_prepared_q_a', 'mlapo_prepared_kv_a', 'mlapo_prepared_q_b', 'mlapo_prepared_uk',
        'mlapo_prepared_uk_unpadded', 'mlapo_prepared_uv'}
    assert sum(term_bytes(t, padded.model.quant, padded.runtime.parallel) for t in added) == (
        32 * 32 + (512 + 64) * 32 + 16 * 192 * 32 + 16 * 128 * 512 + 12 * 256 * 512) * 2
    assert all(term_bytes(t, padded.model.quant, padded.runtime.parallel, source=True) == 0 for t in added)
    assert sum(t.logical_parameters for t in terms if t.source_owned) == sum(t.logical_parameters for t in plain_terms)
    assert layer_cache_components(base.model, plan, base.runtime.parallel, 4096) == layer_cache_components(
        padded.model, plan, padded.runtime.parallel, 4096)
    assert parse_case(padded).model == padded.model


def test_actual_frontend_keeps_bf16_router_compute_and_fp32_topk_payload():
    raw = json.loads(SOURCE_CASE.read_text())
    raw['model']['layers'] = 4
    graph = target_graphs(raw, phase='decode', tokens=1, sequences=32, context=4096)[0]
    by_id = {op.op_id: op for op in graph}
    router, topk = by_id['s0.l3.router'], by_id['s0.l3.topk']
    meta = dict(router.metadata)
    assert router.local_shape == (4, 16, 7168)
    assert meta['weight_format'] == meta['activation_format'] == meta['compute_format'] == 'bf16'
    assert router.dtype == 'bf16' and meta['output_format'] == 'fp32'
    assert router.mte3_bytes == topk.mte2_bytes == 4 * 16 * 4
    assert by_id['s0.l3.output'].local_shape == (32, 7168, 1536)
    assert by_id['s0.l3.output_gate'].local_shape == (32, 1536, 7168)
    assert by_id['s0.l3.q_b'].local_shape == (32, 3072, 1536)
    plain = deepcopy(raw)
    plain['model']['layer_overrides']['3']['router_output_dtype'] = 'bf16'
    other = {op.op_id: op for op in target_graphs(plain, phase='decode', tokens=1, sequences=32, context=4096)[0]}
    assert other['s0.l3.router'].flops == router.flops
    assert other['s0.l3.router'].mte3_bytes * 2 == router.mte3_bytes
    assert other['s0.l3.topk'].mte2_bytes * 2 == topk.mte2_bytes


def test_real_engine_respects_input_readiness_and_releases_all_resources():
    raw = control()
    graph = target_graphs(raw, phase='decode', tokens=1, context=16, sequences=2)
    result = graph_costs(raw, graph, input_ready_ms=7, trace_limit=10000)
    assert all(row['start_ms'] >= 7 for row in result['events'] if row['demand'])
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert not any(result['execution']['live_allocated'].values())
    timeline = {row['op_id']: row for row in result['timeline']}
    assert timeline['s0.l0.mla_query_contiguous']['ready_ms'] >= timeline['s0.l0.q_absorb']['complete_ms']
    assert timeline['s0.l0.v_expand']['ready_ms'] >= max(
        timeline[f's0.l0.latent_attention{i}']['complete_ms'] for i in range(2))


def test_all_twelve_new_cases_parse_with_independent_logical_and_physical_axes():
    paths = sorted(SOURCE_CASE.parent.glob('a5_kimi_tp8_e16_bf16_*.json'))
    assert len(paths) == 12
    for path in paths:
        parsed = parse_case(json.loads(path.read_text()))
        assert parsed.model.layers == 93
        assert {p.num_heads for p in parsed.model.layer_plans if p.attention == 'mla'} == {96}
        assert all(p.router_dtype == 'bf16' and p.router_output_dtype == 'fp32' for p in parsed.model.layer_plans)
        assert all(p['physical_heads'] == 16 for p in dict(parsed.model.metadata)['mla_head_padding']['layers'].values())


def test_single_head_layout_unknown_keeps_simulation_with_explicit_capacity_bound():
    raw = control(heads=1)
    parsed = parse_case(raw)
    metadata = dict(parsed.model.metadata)
    assert metadata['model_coverage'] == 'partial'
    assert metadata['capacity_coverage'] == 'upper_bound_for_single_head_mla_prepared_nd_parent'
    assert 'prepared_weight_aliasing:single_head_original_kv_b_layout_unspecified' in metadata['unmodeled_features']
    terms = layer_weight_terms(parsed.model.layer_plans[0], parsed.model)
    parent = next(term for term in terms if term.name == 'mlapo_prepared_uk_unpadded')
    # Conservative NZ->ND conversion retains the entire parent, even though
    # UK's single-head slice itself is a view. Original ND can alias instead.
    assert term_bytes(parent, parsed.model.quant, parsed.runtime.parallel) == (128 + 128) * 512 * 2
    graph = ops(raw)
    assert dict(graph['v_expand'].metadata)['matrix_groups'] == 1
    assert not any(name.startswith('mla_query_pad') for name in graph)
