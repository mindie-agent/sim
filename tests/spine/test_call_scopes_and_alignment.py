"""Call-scope routing and alignment contracts.
[Role] Exercise model and verification rows, cache alignment and indexer selection.
[Invariants] Conserve row ownership, cache traffic and work across scopes.
[Timeouts & Locks] Finite simulation and bounded local fixtures.
[Side-Effect Policy] Repository fixtures are read-only.
"""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from _support.linear_cases import case as kimi_case
from _support.memory_cases import draft_case
from _support.moe_hc_mc2_fixtures import mxfp8_case as glm_case
from _support.operator_cases import declared_case as case, sparse_mla_metadata_raw_case as sparse_case
from mindie_sim import compose_case, graph_costs, predict, target_graphs
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.context_parallel import owned_positions
from mindie_sim.catalog.draft import draft_quant
from mindie_sim.catalog.draft_geometry import draft_layer_plans
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weights import stage_weight_bytes
from mindie_sim.feature_mechanisms import MECHANISM_CATALOG, MechanismContext
from mindie_sim.prediction import local_projection_tp, operator_analysis as oa
from mindie_sim.prediction.analysis_inputs import analysis_view
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.operators.sparse_indexer.batch_prefix_indexer import apply_indexer_prefix_reads
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.frontend_features import _runtime_metadata
from mindie_sim.prediction.operators.linear_attention._gdn_decode_programs import CONV
from mindie_sim.prediction.operators.mla.generic_latent_attention import index_key_window, unique_index_key_rows
from mindie_sim.prediction.implementation_selection import graph_selections
from mindie_sim.prediction.operators.mc2.moe_comm_selection import MoeCommConfig
from mindie_sim.prediction.operators.moe_routing.moe_init_mxfp8 import RECIPE
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.operators.attention import attention_online; from mindie_sim.prediction.operators.vector import cuda_norm
from mindie_sim.prediction.prefix_reads import CacheAccess, plan_prefix_reads
from mindie_sim.prediction.rewrite_indexer_rules import online_indexer_topk
from mindie_sim.prediction.speculation import draft_program
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.stack_tracks import RUNTIME_STACKS, resolve_stack_track
from tools.kimi_call_scopes import SITU_DOWN, SITU_INPUT, SITU_QUANT, situ_reference

pytestmark = pytest.mark.tier0
predict_quiet = partial(predict, event_limit=0, trace_limit=0)
def assert_fields(record, **expected):
    assert {key: record[key] for key in expected} == expected
def frontend(case, *views):
    return Frontend()(IterationPlan(views), case.model, case.runtime, case.machine)
def selection(report):
    return report['execution']['step_summary']['implementation_selections']
def with_runtime(case, **changes):
    return replace(case, runtime=replace(case.runtime, **changes))
ROOT = Path(__file__).resolve().parents[2]
NORM = 'vllm_cuda_rms_norm'
ATTENTION_BINDING = {'implementation': attention_online.IMPLEMENTATION_ID, 'options': {'compute_capability': [8, 9]}}
NORM_BINDING = {'implementation': cuda_norm.IMPLEMENTATION, 'recipes': ['rmsnorm']}
QUANT = QuantSpec('descriptor_conditioned_mxfp8', weight_format='fp8_e4m3', activation_format='fp8_e4m3', expert_weight_format='mxfp4', weight_block=(1, 32), expert_block=32, scale_format='fp8_e8m0', scale_bytes=1, activation_group_size=32, activation_scale_format='fp8_e8m0', activation_scale_pack=2)
K, GATE_N, SITU_WIDTH, EXPERTS, GROUP = (3584, 6144, 3072, 28, 32)

def declared_case(*, output=2, **lowerings):
    raw = case(output=output)
    raw['model'].update(layers=1, activation_dtype='bf16', weight_dtype='bf16', compute_dtype='bf16', kv_dtype='bf16')
    device = raw['machine']['device']
    roles = {**attention_online.SERVICE_ROLES, **cuda_norm.SERVICE_ROLES}
    device['rates']['shared_memory'] = device['compute_peaks']['fp16'] = 1e9
    device['service_modes'] = {role: {'resource': resource, 'rate': 1e9} for role, resource in roles.items()}
    raw['runtime']['kernels'] = {'lowerings': lowerings}
    return raw
def work_for(parsed, views=None):
    return Frontend(attention_policy=AttentionPolicy(**parsed.runtime.attention_policy))(IterationPlan(views or (RequestView('r', 4, 0, True, 'prefill'),)), parsed.model, parsed.runtime, parsed.machine)
def test_runtime_declaration_reaches_real_model_and_shared_execution():
    parsed = parse_case(declared_case(rmsnorm=NORM_BINDING, gqa_attention=ATTENTION_BINDING))
    work = work_for(parsed)
    source = [op for op in work.ops if op.plan_required]
    attention = next((op for op in source if op.family == 'attention'))
    metadata = dict(attention.metadata)
    assert metadata['semantic_pairs'] == 4 * (1 + 2 + 3 + 4)
    assert attention.flops > 4 * 8 * metadata['semantic_pairs']
    assert metadata['source_params']['page_block_size'] == parsed.runtime.kv_block_size
    assert attention.deps == ('s0.l0.cache', 's0.l0.rope')
    assert {op.family for op in source} == {'attention', 'rmsnorm'}
    isolated = operator_costs(parsed, work.ops, input_ready_ms=3)
    execution = Execution(parsed, CostModel(parsed))
    submission = execution.add(work, 0, 0)
    execution.engine.advance(until=(submission.completion,))
    expected = {}
    for row in isolated['operators']:
        assert not row['service_before_input']
        for resource, amount in row['serviced'].items():
            expected[resource] = expected.get(resource, 0) + amount * row['occurrences']
    assert {key: value for key, value in execution.engine.serviced.items() if value} == pytest.approx(expected)
    assert not any(execution.engine.allocated.values())
    result = predict(parsed)
    assert result['requests'][0]['generated_tokens'] == 2
    assert result['parameters']['operator_occurrences']['measured_primitive'] == 0
def test_source_attention_uses_declared_call_batching_without_fabricating_launches():
    parsed = parse_case(declared_case(gqa_attention=ATTENTION_BINDING))
    views = (RequestView('a', 3, 2, False, 'prefill'), RequestView('b', 5, 0, False, 'prefill'))
    with pytest.raises(ModelingError, match='one_sequence_launch'):
        work_for(parsed, views)
    parsed = with_runtime(parsed, attention_policy={'fused_batch': False})
    work = work_for(parsed, views)
    calls = [op for op in work.ops if op.family == 'attention']
    assert len(calls) == 2
    assert [dict(op.metadata)['source_params']['query'] for op in calls] == [3, 5]
    assert [op.requests for op in calls] == [('a',), ('b',)]
    assert work.ops[-1].deps
    report = operator_costs(parsed, calls)
    assert report['operator_occurrences'] == 2
    assert all((row['program'] == 'OperatorPlan' for row in report['operators']))
def policy(binding, domain=None, *, otherwise='error'):
    return {'cases': [{'domain': domain or {'phases': ['decode']}, 'binding': binding}], 'otherwise': otherwise}
def test_serving_and_both_analysis_scopes_share_padding_aware_selection():
    domain = {'phases': ['decode'], 'valid_tokens': {'max': 1}, 'padded_tokens': {'min': 4, 'max': 4}}
    raw = declared_case(output=3, rmsnorm=policy(NORM, domain, otherwise='analytical'))
    raw['runtime'].update(graph_mode='auto', graph_buckets=[4])
    original = deepcopy(raw)
    result = predict_quiet(raw)
    decisions = selection(result)
    assert {(d['case'], d['steps']) for d in decisions} == {('otherwise', 1), (0, 2)}
    assert result['requests'][0]['generated_tokens'] == 3
    assert not result['execution']['live_tasks']
    for scope in ('body', 'model'):
        graph = target_graphs(raw, phase='decode', tokens=1, context=4, scope=scope)[0]
        assert any((op.recipe.startswith('cuda_norm.') for op in graph))
        choice, = graph_selections(graph)
        assert choice['case'] == 0
        assert_fields(choice['geometry'], valid_tokens=1, padded_tokens=4)
        prefill = target_graphs(raw, phase='prefill', tokens=4, scope=scope)[0]
        assert not any((op.recipe.startswith('cuda_norm.') for op in prefill))
        assert graph_selections(prefill)[0]['coverage'] == 'analytical_semantic_work'
    assert raw == original
def test_mixed_batch_requires_its_whole_phase_domain_and_unmatched_is_visible():
    raw = case()
    raw['runtime']['kernels'] = {'lowerings': {'rmsnorm': policy(NORM)}}
    parsed = parse_case(raw)
    views = (RequestView('decode', 1, 8, True, 'decode'), RequestView('prefill', 4, 0, True, 'prefill'))
    with pytest.raises(ModelingError, match='implementation_domain_unmatched:rmsnorm'):
        frontend(parsed, *views)
    raw['runtime']['kernels']['lowerings']['rmsnorm']['cases'].append({'domain': {'phases': ['decode', 'prefill']}, 'binding': 'analytical'})
    parsed = parse_case(raw)
    work = frontend(parsed, *views)
    choice, = work.implementation_selections
    assert choice['case'] == 1 and choice['geometry']['valid_tokens'] == 5
    assert not any((op.recipe.startswith('cuda_norm.') for op in work.ops))
    assert set(work.ops[-1].requests) == {'decode', 'prefill'}
def test_sparse_decode_metadata_is_explicitly_excluded_from_prefill():
    raw = sparse_case()
    bindings = raw['runtime']['kernels']['lowerings']
    bindings['sparse_mla_metadata'] = policy(bindings['sparse_mla_metadata'], otherwise='analytical')
    for scope in ('body', 'model'):
        prefill = target_graphs(raw, phase='prefill', tokens=8, scope=scope)[0]
        decode = target_graphs(raw, phase='decode', tokens=4, context=128, scope=scope)[0]
        assert not any((op.family == 'attention_metadata' for op in prefill))
        metadata, = [op for op in decode if op.family == 'attention_metadata']
        assert dict(metadata.metadata)['metadata_call']['rows'] == 4
        assert graph_selections(prefill)[0]['implementation'] == 'analytical'
    raw['model']['layer_overrides']['0']['qk_rope_head_dim'] = 64
    with pytest.raises(ModelingError, match='sparse_mla_metadata_requires_nope_bf16_paged_decode'):
        target_graphs(raw, phase='decode', tokens=4, context=128)
def first_four_layers(case):
    model = replace(case.model, layers=4, body_layers=4, layer_plans=case.model.layer_plans[:4])
    return replace(with_runtime(case, pp_layer_partition=(4,)), model=model)
def test_optional_prepared_weights_stay_resident_when_mixed_calls_select_analytical():
    original = first_four_layers(parse_case(kimi_case(1)))
    resident = stage_weight_bytes(original.model, original.runtime.parallel)
    kernels = deepcopy(original.runtime.kernels)
    for selector in ('mla_head_padding', 'causal_conv_update', 'kda', 'paged_mla_attention'):
        kernels['lowerings'][selector] = policy(kernels['lowerings'][selector], {'phases': ['decode'], 'max_query_tokens': {'max': 1}}, otherwise='analytical')
    parsed = parse_case(with_runtime(original, kernels=kernels))
    assert stage_weight_bytes(parsed.model, parsed.runtime.parallel) == resident
    views = (RequestView('d', 1, 4096, False, 'decode', draft_advanced_tokens=1), RequestView('p', 8, 0, False, 'prefill'))
    mixed = frontend(parsed, *views)
    assert all((row['implementation'] == 'analytical' for row in mixed.implementation_selections))
    assert any((op.recipe == 'short_convolution' for op in mixed.ops))
    assert not any((op.recipe in {CONV, 'mla_query_pad_nope', 'mla_query_pad_rope'} for op in mixed.ops))
    decode = frontend(parsed, views[0])
    assert any((op.recipe == CONV for op in decode.ops))
    assert any((op.recipe == 'mla_query_pad_nope' for op in decode.ops))
    assert stage_weight_bytes(parsed.model, parsed.runtime.parallel) == resident
def test_draft_calls_select_their_own_phase_and_q1_geometry_before_target_verification():
    parsed = replace(draft_case(), machine=parse_case(declared_case()).machine)
    policy_cases = {'cases': [{'domain': {'model_scopes': ['target'], 'phases': ['prefill']}, 'binding': NORM}, {'domain': {'model_scopes': ['draft'], 'phases': ['decode'], 'max_query_tokens': {'max': 1}}, 'binding': NORM}], 'otherwise': 'analytical'}
    parsed = with_runtime(parsed, kernels={'lowerings': {'rmsnorm': policy_cases}})
    view = RequestView('r', 1, 4, True, 'decode', draft_proposal_tokens=2)
    work = frontend(parsed, view)
    target, = [record for record in work.implementation_selections if record['model_scope'] == 'target']
    drafts = [record for record in work.implementation_selections if record['model_scope'] == 'draft']
    assert target['implementation'] == 'analytical'
    assert target['geometry']['max_query_tokens'] == 3
    assert len(drafts) == 2 and all((record['case'] == 1 for record in drafts))
    assert [record['geometry']['max_context_tokens'] for record in drafts] == [4, 5]
    assert all((record['geometry']['max_query_tokens'] == 1 for record in drafts))
    selected = [op for op in work.ops if op.recipe.startswith('cuda_norm.')]
    assert selected and all((op.phase == 'draft' for op in selected))
    result = predict_quiet(parsed)
    decisions = selection(result)
    assert {(row['model_scope'], row['case']) for row in decisions} == {('target', 0), ('target', 'otherwise'), ('draft', 1), ('draft', 'otherwise')}
    draft_decodes, = [row for row in decisions if row['model_scope'] == 'draft' and row['case'] == 1]
    assert draft_decodes['calls'] > draft_decodes['steps']
    assert result['requests'][0]['generated_tokens'] == 6
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
def test_target_mxfp8_routing_does_not_rewrite_explicit_analytical_bf16_draft():
    parsed = first_four_layers(parse_case(glm_case(16)))
    view = RequestView('r', 4, 0, True, 'prefill', draft_prepare=True)
    with pytest.raises(ModelingError, match='moe_init_mxfp8_requires_complete_BF16_to_E4M3'):
        frontend(parsed, view)
    kernels = {**parsed.runtime.kernels, 'lowerings': {'moe_init_routing': policy(RECIPE, {'model_scopes': ['target']}, otherwise='analytical')}}
    parsed = with_runtime(parsed, kernels=kernels)
    work = frontend(parsed, view)
    native = [op for op in work.ops if op.recipe == RECIPE]
    assert len(native) == 1 and native[0].phase == 'prefill'
    draft = [op for op in work.ops if op.recipe in {'grouped_gate_up', 'balanced_grouped_gate_up'} and op.phase == 'draft_prefill']
    assert len(draft) == 1 and dict(draft[0].metadata)['activation_format'] == 'bf16'
    assert draft[0].local_shape[0] == 4 * parsed.model.auxiliary.mtp.top_k
    probe = target_graphs(parsed, phase='prefill', tokens=4, scope='model', draft_prepare=True)[0]
    assert {(record['model_scope'], record['implementation']) for record in graph_selections(probe)} == {('target', RECIPE), ('draft', 'analytical')}
    kernels['lowerings']['moe_init_routing']['otherwise'] = RECIPE
    with pytest.raises(ModelingError, match='moe_init_mxfp8_requires_complete_BF16_to_E4M3'):
        frontend(parsed, view)

def _mtp_case():
    return parse_case(compose_case('deepseek-v4.1-flash', hardware='a3', tp=1, ep=1, num_devices=1, layers=2, mtp_proposal_tokens=3, mtp_acceptance_length=2.5))
@pytest.mark.parametrize('tokens,query,proposals', [(1, 1, 0), (4, 1, 3), (2, 2, 0), (3, 3, 0), (5, 5, 0), (8, 8, 0)])
def test_analysis_view_preserves_verification_and_explicit_query_rows(tokens, query, proposals):
    view = analysis_view(_mtp_case(), 'r', tokens, 128, 'model', 'decode')
    assert (view.q_len, view.draft_proposal_tokens) == (query, proposals)
@pytest.mark.parametrize('tokens', [1, 4, 8])
def test_model_scope_routes_verification_rows_without_double_expansion(tokens):
    graph = target_graphs(_mtp_case(), phase='decode', tokens=tokens, context=128, sequences=1, scope='model')
    assert graph[0][0].local_shape[0] == tokens
@pytest.mark.parametrize("override,query,capacity", [({}, 4, 8), ({'uniform_decode_query_len': 1}, 1, 2)])
def test_transport_capacity_uses_the_declared_verification_width(override, query, capacity):
    metadata = dict(moe_comm_mode='dynamic', max_num_seqs=2, draft_proposal_tokens=3) | override
    features = SimpleNamespace(tp=1, graph_buckets=(), metadata=tuple(metadata.items()))
    config = MoeCommConfig.from_features(features, ep=8, top_k=4, experts=32)
    assert config is not None
    assert (config.uniform_decode_query_len, config.capacity) == (query, capacity)
@pytest.mark.parametrize("metadata,expected", [((), 3), ((('draft_proposal_tokens', 5),), 5)])
def test_runtime_metadata_forwards_the_declared_dspark_width(metadata, expected):
    runtime = SimpleNamespace(metadata=metadata, dspark=DsparkSpec(draft_len=3, acceptance=(0.6, 0.4, 0.2)))
    assert dict(_runtime_metadata(runtime))['draft_proposal_tokens'] == expected
@pytest.mark.parametrize('dp', (32, 64))
def test_model_scope_uniform_replicas_match_unrolled_lowering(dp, monkeypatch):
    raw = compose_case('qwen3-30b-a3b', hardware='a3', tp=1, ep=dp, dp=dp, num_devices=dp, layers=1, prompt_tokens=8, output_tokens=2, concurrency=1, max_num_seqs=1)
    case = parse_case(raw)
    real_ops, calls = oa._target_ops, []
    def recording(case_, views_, scope_, **kwargs):
        calls.append((case_, views_, scope_, dict(kwargs)))
        return real_ops(case_, views_, scope_, **kwargs)
    monkeypatch.setattr(oa, '_target_ops', recording)
    monkeypatch.setattr(oa, 'bind_cohort_traffic', lambda _case, graphs: graphs)
    monkeypatch.setattr(local_projection_tp, 'lower_local_projection_tp_cohort', lambda graphs, **_kwargs: graphs)
    graphs = oa.target_graphs(case, phase='decode', tokens=1, context=128, sequences=1, scope='model')
    (first_case, _first_views, scope, kwargs), = calls
    assert scope == 'model' and len(graphs.programs) == dp
    for replica in sorted(graphs.programs)[1:]:
        target_views = tuple((view for view in kwargs['global_views'] if view.request_id.startswith(f'analysis{replica}:')))
        unrolled_kwargs = kwargs | dict(replica=replica, physical_rank=oa._physical_rank_for(case, replica), model_programs={})
        unrolled = real_ops(first_case, target_views, 'model', **unrolled_kwargs)
        assert [op.to_dict() for op in unrolled] == [op.to_dict() for op in graphs[replica]]
        assert unrolled_kwargs['model_programs'][replica].to_dict(include_ops=True) == graphs.programs[replica].to_dict(include_ops=True)

def situ_case(*, beta=4.0, linear_beta=25.0):
    raw = case()
    raw['model'].update(layers=1, intermediate_size=SITU_WIDTH)
    raw['model']['layer_plans'] = [{'index': 0, 'ffn': 'dense', 'intermediate_size': SITU_WIDTH, 'ffn_activation': 'situ', 'activation_situ_beta': beta, 'activation_situ_linear_beta': linear_beta}]
    return parse_case(raw)
def situ_descriptors(m):
    return {'input_shapes': [[m, K], [m, K // GROUP // 2, 2], [EXPERTS, GATE_N, K // 2], [EXPERTS, GATE_N, K // GROUP // 2, 2], [EXPERTS]], 'output_shapes': [[m, SITU_WIDTH], [m, SITU_WIDTH // GROUP // 2, 2]], 'input_dtypes': ['DT_FLOAT8_E4M3FN', 'DT_FLOAT8_E8M0', 'DT_FLOAT4_E2M1', 'DT_FLOAT8_E8M0', 'INT64'], 'output_dtypes': ['DT_FLOAT8_E4M3FN', 'DT_FLOAT8_E8M0'], 'input_formats': ['ND', 'ND', 'FRACTAL_NZ_C0_16', 'ND', 'ND'], 'output_formats': ['ND', 'ND']}
@pytest.mark.parametrize('m,expected_flops', [(2875, 126615552000), (3286, 144716070912)])
def test_situ_work_and_traffic_match_an_independent_integer_oracle(m, expected_flops):
    result = situ_reference(situ_case(), situ_descriptors(m), QUANT)
    gate, activation, quantize = result['ops']
    ledger = result['ledger']
    groups = m * (SITU_WIDTH // GROUP)
    flops = 2 * m * GATE_N * K
    input_payload, input_scales = (m * K, m * (K // GROUP))
    weight_payload = EXPERTS * GATE_N * (K // 2)
    weight_scales = EXPERTS * GATE_N * (K // GROUP)
    gate_bf16, situ_bf16 = (m * GATE_N * 2, m * SITU_WIDTH * 2)
    output_payload, output_scales = (m * SITU_WIDTH, m * (SITU_WIDTH // GROUP))
    situ_ops = 12 * m * SITU_WIDTH
    quant_ops = 4 * m * SITU_WIDTH - groups + 3 * groups
    assert flops == expected_flops
    assert gate.flops == flops
    assert gate.mte2_bytes == input_payload + input_scales + weight_payload + weight_scales
    assert gate.mte3_bytes == gate_bf16
    assert activation.vector_ops == situ_ops
    assert (activation.mte2_bytes, activation.mte3_bytes) == (gate_bf16, situ_bf16)
    assert quantize.vector_ops == quant_ops
    assert (quantize.mte2_bytes, quantize.mte3_bytes) == (situ_bf16, output_payload + output_scales)
    assert ledger['traffic']['weight_payload_bytes'] == weight_payload
    assert ledger['traffic']['weight_scale_bytes'] == weight_scales
    assert ledger['traffic']['total_op_hbm_bytes'] == gate.hbm_bytes + activation.hbm_bytes + quantize.hbm_bytes
    assert ledger['workspaces']['intermediate_bf16_read_write_bytes'] == 2 * gate_bf16 + 2 * situ_bf16
    assert_fields(ledger['work'], situ_scalar_ops=situ_ops, output_quant_vector_ops=quant_ops)
    if m == 2875:
        assert [weight_payload, weight_scales, situ_ops, quant_ops] == [308281344, 19267584, 105984000, 35880000]
def test_situ_graph_has_three_ops_no_down_and_no_input_quantizer():
    result = situ_reference(situ_case(), situ_descriptors(2875), QUANT)
    gate, activation, quantize = result['ops']
    assert gate.recipe == 'matrix_envelope'
    assert type(result['recipes'].select(gate)).__name__ == 'TiledMatmul'
    assert [op.op_id for op in result['ops']] == ['situ.gate_up', 'situ.activation', SITU_QUANT]
    assert SITU_DOWN not in [op.op_id for op in result['ops']]
    assert not any((op.family == 'down_projection' for op in result['ops']))
    assert gate.deps == (SITU_INPUT,)
    assert activation.deps == (gate.op_id,)
    assert quantize.deps == (activation.op_id,)
    assert [op.op_id for op in result['ops'] if op.op_id.endswith('.quantize')] == [SITU_QUANT]
    graph = graph_costs(situ_case(), result['ops'], recipes=result['recipes'], external_inputs={SITU_INPUT: 0}, isolated=False)
    assert [row['op_id'] for row in graph['timeline']] == ['situ.gate_up', 'situ.activation', SITU_QUANT]
    assert graph['timeline'][1]['predecessors'] == [graph['timeline'][0]['id']]
    assert graph['timeline'][2]['predecessors'] == [graph['timeline'][1]['id']]
    assert graph['duration_ms'] > 0
    assert_fields(graph['execution'], live_tasks=0, live_programs=0)
    assert not any(graph['execution']['live_allocated'].values())
def test_situ_beta_changes_metadata_not_the_declared_selection():
    first = situ_reference(situ_case(beta=4.0), situ_descriptors(3286), QUANT)
    changed = situ_reference(situ_case(beta=5.0), situ_descriptors(3286), QUANT)
    assert [op.op_id for op in first['ops']] == [op.op_id for op in changed['ops']]
    assert [op.recipe for op in first['ops']] == [op.recipe for op in changed['ops']]
    assert [(op.flops, op.vector_ops) for op in first['ops']] == [(op.flops, op.vector_ops) for op in changed['ops']]
    assert dict(first['ops'][1].metadata)['activation_beta'] == 4.0
    assert dict(changed['ops'][1].metadata)['activation_beta'] == 5.0
    assert first['ledger']['activation']['beta'] == 4.0
    assert changed['ledger']['activation']['beta'] == 5.0
    assert first['ledger']['matrix_recipe'] == changed['ledger']['matrix_recipe']
def test_situ_rejects_geometry_that_does_not_match_the_descriptor():
    bad = situ_descriptors(2875)
    bad['output_shapes'][0][1] = 4096
    with pytest.raises(ModelingError, match='output_must_be_half_gate_up_width'):
        situ_reference(situ_case(), bad, QUANT)
    bad = situ_descriptors(2875)
    bad['input_shapes'][1] = [2875, 56]
    with pytest.raises(ModelingError, match='input_scale_geometry_mismatch'):
        situ_reference(situ_case(), bad, QUANT)
    with pytest.raises(ModelingError, match='case_plan_ffn_activation_situ'):
        raw = situ_case().to_dict()
        raw['model']['layer_plans'][0]['ffn_activation'] = 'silu'
        situ_reference(parse_case(raw), situ_descriptors(2875), QUANT)
@pytest.mark.parametrize('changes,reason', [({'exclusions': ('routed_expert',)}, 'quant_must_declare_fp4_weights'), ({'activation_scale_format': 'fp32'}, 'input_scale_declaration_mismatch'), ({'activation_scale_pack': 3}, 'input_scale_declaration_mismatch')])
def test_situ_cannot_execute_a_different_operand_representation_than_its_ledger(changes, reason):
    with pytest.raises(ModelingError, match=reason):
        situ_reference(situ_case(), situ_descriptors(2875), replace(QUANT, **changes))

def native_config(*, dsa_layers=None):
    config = json.loads((ROOT / 'mindie_sim/model_configs/kimi-k3.json').read_text(encoding='utf-8'))
    text = config['text_config']
    text['dsa_config'] = {'dsa_layers': text['linear_attn_config']['full_attn_layers'] if dsa_layers is None else dsa_layers, 'index_num_heads': 32, 'index_head_dim': 128, 'index_topk': 2048, 'index_rope_dim': 0}
    text.update(model_type='kimi_k31', dspark_attention_kind='mla', dspark_block_size=5)
    return config
def sparse_alignment_case(cp=8, proposals=3):
    return parse_case(compose_case(native_config(), hardware='a5', tp=8, dp=4, ep=32, dcp=cp, num_devices=32, block_size=1536, max_model_len=140000, quant={'name': 'k31_fp8_cache', 'weight_format': 'bf16', 'activation_format': 'bf16', 'expert_weight_format': 'bf16', 'kv_dtype': 'fp8_e4m3', 'index_dtype': 'fp8_e4m3', 'index_scale_dtype': 'fp16'}, mtp_proposal_tokens=proposals, mtp_acceptance_length=min(2, proposals + 1), graph_capture_mode='none', prompt_tokens=131072, output_tokens=1024))
def test_native_sparse_configuration_and_independent_draft_formats():
    c = sparse_alignment_case()
    sparse = [p for p in c.model.layer_plans if p.indexer]
    assert len(sparse) == 24
    assert sum((p.attention == 'kda' for p in c.model.layer_plans)) == 69
    assert sparse[0].index == 3 and sparse[-1].index == 92
    assert all(((p.index_heads, p.index_dim, p.index_topk) == (32, 128, 2048) for p in sparse))
    assert all((p.owns_index_k and p.index_query_per_layer for p in sparse))
    assert cache_format(c.model, 'mla_kv').row_bytes(576) == 656
    assert cache_format(c.model, 'index_k').row_bytes(128) == 130
    draft = c.model.auxiliary.dspark
    assert draft_quant(c.model, draft).kv_dtype == 'bf16'
    assert all((not p.indexer and (not p.mla_output_gate) for p in draft_layer_plans(c.model, 'dspark')))
@pytest.mark.parametrize('cp', [4, 8])
@pytest.mark.parametrize('proposals', [0, 3])
def test_shared_pool_quote_and_allocation_report_conserve_bytes(cp, proposals):
    c = sparse_alignment_case(cp, proposals)
    memory = Memory(c)
    pool_unit = 1536 * (24 * 656 + 24 * 130 + (5 * 1152 if proposals else 0))
    assert memory.pages.groups[0].bytes_per_group_page == pool_unit
    states = 3 * (2 + proposals)
    for tokens in (1, 1536 * cp, 1536 * cp + 1, 131588):
        history = (tokens + 1536 * cp - 1) // (1536 * cp)
        expected = (history + states) * pool_unit
        assert memory.cache_bytes(tokens) == (expected,)
        report = memory.cache_allocation(tokens)
        assert report['allocated']['total_bytes_per_stage_rank'] == [expected]
        assert report['allocated']['reserve_bytes_per_stage_rank'][0] >= 0
    assert memory.cache_bytes(0) == (0,)
def test_index_history_is_sharded_but_recurrent_heads_are_not():
    c = sparse_alignment_case()
    sparse = next((p for p in c.model.layer_plans if p.indexer))
    components = layer_cache_components(c.model, sparse, c.runtime.parallel, 131072, capacity_padded=False)
    assert components['index_k'] == 131072 * 130 / 8
    graph = target_graphs(c, phase='decode', tokens=4, context=131072, sequences=2, scope='body', resource_fusion=False)[0]
    qsfa = [op for op in graph if op.recipe == 'paired_fp8_qsfa_analytic']
    assert len(qsfa) == 24
    assert all((dict(op.metadata)['gathered_q_heads'] == 96 for op in qsfa))
    assert len([op for op in graph if op.recipe == 'dcp_output_alltoall']) == 24
    assert not any((op.family == 'dcp_allgather' for op in graph))
    scan = next((op for op in graph if op.recipe == 'quant_lightning_indexer'))
    assert not any((op.recipe in {'index_scores', 'index_topk'} for op in graph))
    assert dict(scan.metadata)['native_kind'] == 'aclnnQuantLightningIndexerV2'
    assert dict(scan.metadata)['selection_output_dtype'] == 'int32'
    owned = [sum((owned_positions(0, 131072 + offset, 8, 1536, rank) for _ in range(2) for offset in range(1, 5))) for rank in range(8)]
    assert scan.flops == 2 * max(owned) * 32 * 128
    tile_rows = [2 * owned_positions(0, 131072 + 4 - 1, 8, 1536, rank) for rank in range(8)]
    by_id = {op.op_id: op for op in graph}
    operand_bytes = sum((by_id[dep].mte3_bytes for dep in scan.deps[:2]))
    assert scan.mte2_bytes == operand_bytes + max(tile_rows) * 130
    assert scan.mte3_bytes == 4 * 2 * 2048 * 4
    candidate = next((op for op in graph if op.recipe == 'candidate_gather'))
    assert candidate.link_bytes == 8 * 2048 * 8 * 7
    kda = [op for op in graph if op.family in {'recurrent_vector', 'recurrent_scan'}]
    assert kda
    assert all((dict(op.metadata)['heads'] == 12 for op in kda))
def test_invalid_native_sparse_layer_is_visible():
    config = native_config(dsa_layers=[1])
    with pytest.raises(ValueError, match='dsa_layers'):
        compose_case(config, hardware='a5')
def test_external_draft_context_rows_are_distinct_from_query_rows():
    c = sparse_alignment_case()
    views = (RequestView('request', 4, 131072, True, 'decode', draft_proposal_tokens=3),)
    regions, _ = draft_program(Frontend(resource_fusion=False), c.model, c.runtime, views, 'ready', machine=c.machine)
    ops = [op for region in regions for op in region.ops]
    features = next((op for op in ops if op.recipe == 'draft_features'))
    assert features.local_shape[0] == 4
    context = [op for op in ops if op.recipe == 'draft_context_mla_prolog_v3']
    assert len(context) == 5
    assert all((op.local_shape == (4, 1536 + 576) for op in context))
    assert all((dict(op.metadata)['native_kind'] == 'aclnnMlaPrologV3WeightNz' for op in context))
    stores = [op for op in ops if op.recipe == 'draft_context_kv_scatter']
    cp = max(1, c.runtime.parallel.cp)
    block = max(1, c.runtime.kv_block_size)
    start = 131072
    owned = max((owned_positions(start, start + 3, cp, block, rank) for rank in range(cp)))
    assert len(stores) == 5
    assert all((op.local_shape[0] == owned for op in stores))
    assert all((op.mte3_bytes == owned * 1152 for op in stores))
    assert all((dict(op.metadata)['native_kind'] == 'aclnnScatterPaKvCache' for op in stores))
    embedding = next((op for op in ops if op.recipe == 'draft_query_embedding'))
    assert embedding.mte3_bytes == 3 * 7168 * 2
    assert not any((op.family == 'indexer' for op in ops))
    assert all((op.dtype == 'bf16' for op in context))
def test_paired_pool_is_used_by_live_admission_and_release():
    c = sparse_alignment_case()
    quote = Memory(c).cache_bytes(131588)[0]
    budget = Memory(c, MemoryPolicy(derive_workspace=False)).fixed_per_rank[0] + 2 * quote
    machine = replace(c.machine, dies=tuple((replace(device, memory_bytes=budget) for device in c.machine.dies)))
    c = replace(with_runtime(c, gpu_memory_utilization=1), machine=machine)
    memory = Memory(c, MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, 'first', 131588)
    assert memory.resize(0, 'second', 131588)
    assert not memory.resize(0, 'third', 131588)
    memory.release(0, 'first')
    assert memory.resize(0, 'third', 131588)
def test_fp8_call_has_an_executable_analytic_recipe_in_the_shared_engine():
    c = sparse_alignment_case().to_dict()
    plan = c['model']['layer_plans'][3]
    plan.update(index=0, kv_owner_layer_id=0, index_k_owner_layer_id=0, topk_owner_layer_id=0, ffn='dense', intermediate_size=128)
    c['model'].update(layers=1, body_layers=1, layer_plans=[plan])
    c['runtime'].update(pp_layer_partition=[1], mtp_acceptance_length=1)
    c['runtime']['metadata'].update(draft_proposal_tokens=0)
    c['runtime']['metadata'].pop('draft_mode', None)
    result = predict(c, scope='body', phase='decode', tokens=4, context=4096, sequences=1)
    assert result['duration_ms'] > 0
    assert result['operator_occurrences'] > 0
@pytest.mark.parametrize('cp', [1, 2, 4, 8])
def test_query_head_gather_and_output_collective_scale_with_dcp(cp):
    graph = target_graphs(sparse_alignment_case(cp), phase='decode', tokens=1, context=4096, scope='body', resource_fusion=False)[0]
    attention = [op for op in graph if op.recipe == 'paired_fp8_qsfa_analytic']
    assert len(attention) == 24
    assert all((dict(op.metadata)['gathered_q_heads'] == 12 * cp for op in attention))
    assert len([op for op in graph if op.recipe == 'dcp_output_alltoall']) == (24 if cp > 1 else 0)
def test_explicit_int8_template_is_not_silently_rebound_to_fp8_dcp():
    c = with_runtime(sparse_alignment_case(), kernels={'programs': {'indexed_attention': {'kind': 'indexed_attention_packed_int8'}}})
    with pytest.raises(ValueError, match='conflicts_with'):
        target_graphs(c, phase='decode', tokens=1, context=4096, scope='body')
def test_indexer_precision_is_independent_of_main_kv_precision():
    c = sparse_alignment_case()
    quant = replace(c.model.quant, index_dtype='bf16', index_scale_dtype='none')
    metadata = dict(c.model.metadata)
    metadata['cache_representations'] = {key: value for key, value in metadata['cache_representations'].items() if key != 'index_k'}
    memory = Memory(replace(c, model=replace(c.model, quant=quant, metadata=tuple(metadata.items()))))
    assert memory.pages.groups[0].bytes_per_group_page == 1536 * (24 * 656 + 24 * 128 * 2 + 5 * 1152)

def test_index_key_window_counts_physical_compressed_rows():
    rows, refills = index_key_window(((4, 128),), compression=32)
    assert (rows, refills) == (132 // 32, 132 // 32)
    logical, _ = index_key_window(((4, 128),))
    assert logical == 132
    assert rows == logical // 32
def test_unique_index_key_rows_deduplicates_only_the_shared_prefix():
    views = tuple((RequestView(f'r{i}', 1, 128, False, 'decode', prefix_tokens=128, prefix_cache_key='shared-prefix') for i in range(2)))
    union, refills = unique_index_key_rows(views, compression=32)
    assert union == (128 + 1) // 32
    assert refills == 2 * union
    private, private_refills = unique_index_key_rows(views, compression=32, share_prefix=False)
    assert private == 2 * union
    assert private_refills == refills
def test_shared_prefix_planner_separates_component_and_row_bytes():
    access = partial(CacheAccess, context_tokens=4096, q_len=1, prefix_tokens=512, prefix_key='p', layer=0,
                     logical_rows=512, row_bytes=128.0, component='index_k')
    by_row = plan_prefix_reads((access('a'), access('b'), access('c', row_bytes=256.0)), mode='batch_shared', kernel_reuse='tile')
    assert by_row.read_reuse_applied is True
    assert by_row.unique_hbm_bytes == pytest.approx(512 * 128.0 + 512 * 256.0)
    by_component = plan_prefix_reads((access('a'), access('b'), access('c', component='key'), access('d', component='key')), mode='batch_shared', kernel_reuse='tile')
    assert by_component.read_reuse_applied is True
    assert by_component.unique_hbm_bytes == pytest.approx(2 * 512 * 128.0)
def decode_op(op_id, family, recipe, shape, *, requests=('r',), dtype='bf16', **work):
    return OpSpec(op_id, family, 'decode', 0, 0, requests, shape, shape, shape, shape, recipe=recipe, dtype=dtype, **work)
@pytest.mark.parametrize("rows,shared_rows,component,match", [
    ((128, 128), (128, 128), 1000, None),
    ((128, 128), (128, 128), 2000, 'prefix_read_component_exceeds'),
    ((128,), (1, 2), None, 'index_prefix_access_lengths'),
])
def test_indexer_prefix_read_accounting_and_input_validation(rows, shared_rows, component, match):
    op = decode_op('s0.l0.index_scores', 'indexer', 'index_scores', (128, 1, 1), requests=('r0', 'r1'),
                   dtype='int8', hbm_bytes=1000, mte2_bytes=900)
    runtime = SimpleNamespace(prefix_cache_mode='shared', prefix_read_mode='auto')
    views = tuple(RequestView(f'r{i}', 1, 4096, False, 'decode', prefix_tokens=640, prefix_cache_key='key') for i in range(2))
    kwargs = dict(views=views, rows_by_view=rows, shared_rows_by_view=shared_rows, row_bytes=16, layer_index=0, runtime=runtime)
    if component is not None:
        kwargs.update(hbm_component=component, mte2_component=900)
    if match:
        with pytest.raises(ModelingError, match=match):
            apply_indexer_prefix_reads(op, **kwargs)
        return
    updated = apply_indexer_prefix_reads(op, **kwargs, compression_ratio=32, allow_recipe_rebind=False)
    metadata = dict(updated.metadata)
    assert metadata['prefix_read_reuse_applied'] is True
    prefix_rows = 640 // 32
    expected_hbm = (2 * (128 - prefix_rows) + prefix_rows) * 16
    assert metadata['prefix_read_hbm_bytes'] == pytest.approx(expected_hbm)
    assert updated.mte2_bytes == pytest.approx(900)
    assert updated.hbm_bytes == pytest.approx(op.hbm_bytes - component + expected_hbm)
def test_native_quant_indexer_ignores_shared_prefix_pool():
    def rows(mode):
        parsed = parse_case(compose_case('deepseek-v4-10t', hardware='a5', tp=1, ep=1, num_devices=1, layers=4, prompt_tokens=640, output_tokens=2, concurrency=1))
        kernels = {**dict(parsed.runtime.kernels), 'lowerings': {**dict(parsed.runtime.kernels.get('lowerings', {})), 'quant_lightning_indexer': 'cann_quant_lightning_indexer_v2_complete'}}
        case = with_runtime(parsed, prefix_cache_mode=mode, prefix_read_mode='auto', kernels=kernels)
        ops = target_operators(case, phase='decode', tokens=1, context=4096, sequences=1, scope='body')
        return [op for op in ops if op.recipe == 'quant_lightning_indexer_v2']
    private, shared = (rows('private'), rows('shared'))
    assert private and len(private) == len(shared)
    for left, right in zip(private, shared):
        assert (left.hbm_bytes, left.mte2_bytes, left.mte3_bytes) == (right.hbm_bytes, right.mte2_bytes, right.mte3_bytes)
        assert dict(right.metadata).get('prefix_read_reuse_applied') is None
        assert dict(right.metadata).get('prefix_read_mode') is None
def _indexer_pair(metadata=None):
    score_contract = dict(index_score_shards=1, index_context_partitions=1, index_candidate_filter=False,
                          index_head_reduction='complete_local', graph_segment='capture')
    score = decode_op('l0.index_scores', 'indexer', 'index_scores', (4, 2, 8), dtype='int8', deps=('q',),
                      flops=100, vector_ops=10, hbm_bytes=100, mte2_bytes=40, mte3_bytes=16, metadata=tuple((score_contract | (metadata or {})).items()))
    topk = decode_op('l0.index_topk', 'topk', 'index_topk', (4, 2), dtype='fp32', deps=(score.op_id,),
                     flops=20, vector_ops=5, hbm_bytes=16, mte2_bytes=16, mte3_bytes=32,
                     metadata=(('selection_output_dtype', 'int32'), ('graph_segment', 'capture')))
    return score, topk
def test_online_indexer_topk_conserves_work_and_removes_roundtrip():
    score, topk = _indexer_pair()
    op, = online_indexer_topk([score, topk], 'bf16')
    assert op.recipe == 'index_online_topk' and op.deps == ('q',)
    for field in ('flops', 'vector_ops'):
        assert getattr(op, field) == getattr(score, field) + getattr(topk, field)
    assert dict(op.matrix_flops) == {'int8': score.flops, 'fp32': topk.flops}
    assert sum(dict(op.matrix_flops).values()) == pytest.approx(op.flops)
    assert (op.mte2_bytes, op.mte3_bytes) == (score.mte2_bytes, topk.mte3_bytes)
    assert op.hbm_bytes == score.hbm_bytes + topk.hbm_bytes - score.mte3_bytes - topk.mte2_bytes
    metadata = dict(op.metadata)
    assert metadata['score_materialization'] == 'analytical_online_ub'
    assert metadata['native_binary_claimed'] is False
    assert metadata['removed_score_roundtrip_bytes'] == 32.0
@pytest.mark.parametrize('override', [{'index_score_shards': 2}, {'index_context_partitions': 2}, {'index_candidate_filter': True}, {'index_head_reduction': 'partial'}])
def test_online_indexer_topk_preserves_incomplete_reductions(override):
    score, topk = _indexer_pair(override)
    assert [op.recipe for op in online_indexer_topk([score, topk], 'bf16')] == ['index_scores', 'index_topk']
def test_online_indexer_topk_requires_single_consumer_and_segment():
    score, topk = _indexer_pair()
    extra = replace(topk, op_id='l0.other', recipe='other')
    assert [op.recipe for op in online_indexer_topk([score, topk, extra], 'bf16')] == ['index_scores', 'index_topk', 'other']
    other_segment = replace(topk, metadata=(('graph_segment', 'eager'),))
    assert [op.recipe for op in online_indexer_topk([score, other_segment], 'bf16')] == ['index_scores', 'index_topk']
def test_indexer_online_mechanism_is_ideal_only():
    raw = compose_case('deepseek-v4.1-flash', hardware='a3', tp=8, ep=8, num_devices=8, layers=2)
    model = parse_case(raw).model
    context = MechanismContext(model=model, sku='a3', arch='a3', parallel={'tp': 8, 'ep': 8}, caller_explicit=frozenset(), options={})
    mechanism = MECHANISM_CATALOG['indexer_online_ub_topk']
    assert mechanism.eligible(context) is True
    runtime = {}
    mechanism.apply(runtime, context)
    assert 'indexer_online_topk' in runtime['lowering_passes']
    for track, selected in (('merged', False), ('unmerged_pr', False), ('expected_ideal', True)):
        result = resolve_stack_track(RUNTIME_STACKS['vllm-ascend'], track=track, model_key='deepseek-v4.1-flash', ctx=context)
        assert any(item['id'] == 'indexer_online_ub_topk' for item in result['active_mechanisms']) is selected
