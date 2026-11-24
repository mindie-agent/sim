"""Implementation and native mechanism selection.
[Role] Exercise operator bindings, software controls, recommended execution and native mechanisms.
[Invariants] Preserve work, dependencies and declared selection domains.
[Timeouts & Locks] Finite simulation and bounded local fixtures.
[Side-Effect Policy] Repository fixtures are read-only.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from functools import partial
import math
from types import SimpleNamespace
import pytest
from _support.linear_cases import caller
from _support.moe_hc_mc2_fixtures import raw_case
from _support.operator_cases import declared_case as case
from mindie_sim import compose_case, graph_costs, predict, study, target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.compose import apply_va_recommended, va_recommended_config
from mindie_sim.feature_mechanisms import MechanismContext, _apply_basic_norm_quant
from mindie_sim.hardware_derivation import derive_hardware_sku
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from mindie_sim.prediction.operators.attention.attention_pipeline import TiledFlashAndPagedAttentionRecipe
from mindie_sim.prediction.operators.kda.family_kda_recurrent_membase import IMPLEMENTATION as DECODE
from mindie_sim.prediction.operators.kda.family_kda_state import IMPLEMENTATION as PREFILL
from mindie_sim.prediction.operators.paged_mla.family_mqsmla import query_shape, unsupported_reason
from mindie_sim.prediction.operators.paged_mla.mqsmla import query_rows
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.implementation_selection import graph_selections
from mindie_sim.prediction.operators.mc2.mc2_graph import lower, select_common
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.operators.attention import attention_online; from mindie_sim.prediction.operators.vector import cuda_norm
from mindie_sim.prediction.program import HostStaging, RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from tools.kimi_call_scopes import kda_stage_ledger

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
def recommended_kda(hardware, **request):
    return compose_case('kimi-k3', hardware=hardware, tp=8, ep=8, layers=4, va_recommended=True, **request)
KDA_NATIVE_SKUS = [('a3', 'vllm_ascend_recurrent_kda_arch22'), ('a5', 'vllm_ascend_recurrent_kda_arch35')]
GDN_SKUS = ('a3', 'a5')
NORM = 'vllm_cuda_rms_norm'
ATTENTION_BINDING = {'implementation': attention_online.IMPLEMENTATION_ID, 'options': {'compute_capability': [8, 9]}}
NORM_BINDING = {'implementation': cuda_norm.IMPLEMENTATION, 'recipes': ['rmsnorm']}
FUSIONS = ('activation_quantization', 'norm_quant', 'residual_norm')
hc_ops = partial(target_operators, phase='decode', tokens=1, context=2048, sequences=2, scope='body')
WIDTH, TOKENS, TOPK, EP, RECEIVED = 256, 8, 4, 8, 16.0

def declared_case(*, dtype='bf16', tp=1, output=2, passes=(), **lowerings):
    raw = case(tp=tp, output=output)
    raw['model'].update(layers=1, activation_dtype=dtype, weight_dtype=dtype, compute_dtype=dtype, kv_dtype=dtype)
    device = raw['machine']['device']
    roles = {**attention_online.SERVICE_ROLES, **cuda_norm.SERVICE_ROLES}
    device['rates']['shared_memory'] = device['compute_peaks']['fp16'] = 1e9
    device['service_modes'] = {role: {'resource': resource, 'rate': 1e9} for role, resource in roles.items()}
    raw['runtime'].update({'lowering_passes': passes} if passes else {}, kernels={'lowerings': deepcopy(lowerings)})
    return raw
def with_first_layer(case, **changes):
    return replace(case, model=replace(case.model, layer_plans=(replace(case.model.layer_plans[0], **changes),)))
def work_for(parsed, views=None):
    return Frontend(attention_policy=AttentionPolicy(**parsed.runtime.attention_policy))(IterationPlan(views or (RequestView('r', 4, 0, True, 'prefill'),)), parsed.model, parsed.runtime, parsed.machine)
def test_source_selection_keeps_unselected_analytic_model_usable():
    raw = declared_case()
    parsed = parse_case(raw)
    attention = next((op for op in work_for(parsed).ops if op.family == 'attention'))
    assert isinstance(RecipeBook.from_mapping(parsed.runtime.kernels).select(attention), TiledFlashAndPagedAttentionRecipe)
    assert attention.flops == 4 * 8 * 4 * (1 + 2 + 3 + 4)
    raw['machine']['name'], raw['model']['name'] = 'unseen-device-label', 'unseen-model-label'
    assert predict(raw)['metrics'] == predict(parsed)['metrics']
def test_unknown_binding_is_a_visible_modeling_failure_in_both_public_paths():
    raw = declared_case(rmsnorm='unregistered_norm')
    error = 'implementation_not_registered:rmsnorm:unregistered_norm'
    with pytest.raises(ModelingError, match=error):
        target_operators(raw, phase='prefill', tokens=3)
    with pytest.raises(ModelingError, match=error):
        predict(raw)
def test_source_option_typo_cannot_be_ignored_or_abort_other_study_candidates():
    raw = declared_case(rmsnorm=NORM_BINDING)
    result = study([{'id': 'aligned', 'case': raw}, {'id': 'typo', 'case': raw, 'overrides': {'runtime': {'kernels': {'lowerings': {'rmsnorm': {'options': {'aligend': False}}}}}}}], [{'id': 'prefill', 'phase': 'prefill', 'tokens': 3}])
    assert result['status'] == 'partial'
    accepted, rejected = result['evaluations']
    assert accepted['status'] == 'modeled' and accepted['duration_ms'] > 0
    assert rejected['error'] == 'implementation_options_unknown:rmsnorm:aligend'
    assert 'duration_ms' not in rejected
    assert [row['candidate'] for row in result['rankings'][0]['candidates']] == ['aligned']
def test_source_attention_rates_change_time_without_rewriting_work():
    raw = declared_case(gqa_attention=ATTENTION_BINDING)
    parsed = parse_case(raw)
    ops = [op for op in target_operators(parsed, phase='prefill', tokens=7) if op.family == 'attention']
    fast = operator_costs(parsed, ops)['operators'][0]
    raw['machine']['device']['service_modes']['fp32_exp']['rate'] = 1000.0
    slow = operator_costs(parse_case(raw), ops)['operators'][0]
    assert slow['duration_ms'] > fast['duration_ms']
    assert slow['operator']['metadata']['resource_work'] == fast['operator']['metadata']['resource_work']
    assert slow['operator']['flops'] == fast['operator']['flops']
def test_attention_binding_uses_local_heads_and_declared_source_tile_dispatch():
    raw = declared_case(tp=2, gqa_attention=ATTENTION_BINDING)
    raw['model'].update(hidden_size=1024, head_dim=256)
    first = next((op for op in target_operators(raw, phase='prefill', tokens=7) if op.family == 'attention'))
    raw['runtime']['kernels']['lowerings']['gqa_attention']['options']['compute_capability'] = [10, 0]
    second = next((op for op in target_operators(raw, phase='prefill', tokens=7) if op.family == 'attention'))
    a, b = (dict(first.metadata), dict(second.metadata))
    assert_fields(a['source_params'], heads=2, key_heads=1)
    assert a['semantic_pairs'] == b['semantic_pairs'] == 2 * 7 * 8 // 2
    assert (a['dispatch']['tile_size'], b['dispatch']['tile_size']) == (32, 128)
    assert first.flops < second.flops
def test_source_attention_domain_does_not_silently_erase_window_or_kv_quantization():
    parsed = parse_case(declared_case(gqa_attention=ATTENTION_BINDING))
    windowed = with_first_layer(parsed, window=2)
    with pytest.raises(ModelingError, match='unwindowed_unquantized'):
        work_for(windowed)
    analytic = with_runtime(windowed, kernels={})
    attention = next((op for op in work_for(analytic).ops if op.family == 'attention'))
    assert attention.flops == 4 * 8 * 4 * (1 + 2 + 2 + 2)
    quantized = replace(parsed, model=replace(parsed.model, quant=replace(parsed.model.quant, kv_dtype='int8')))
    with pytest.raises(ModelingError, match='unwindowed_unquantized'):
        work_for(quantized)
@pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp32'])
def test_model_norm_preserves_precision_and_source_traffic(dtype):
    parsed = parse_case(declared_case(dtype=dtype, rmsnorm=NORM_BINDING))
    ops = target_operators(parsed, phase='prefill', tokens=3)
    norm = next((op for op in ops if op.op_id == 's0.l0.norm1'))
    assert norm.dtype == dtype
    row = operator_costs(parsed, (norm,))['operators'][0]
    size = 4 if dtype == 'fp32' else 2
    assert row['operator']['hbm_bytes'] == 3 * 32 * size * 4
    assert row['operator']['metadata']['dispatch']['width'] == 32
    assert row['program'] == 'OperatorPlan'
def test_explicit_norm_binding_survives_generic_fusion_and_keeps_quantization_separate():
    parsed = parse_case(declared_case(passes=FUSIONS, rmsnorm=NORM_BINDING))
    parsed = replace(parsed, model=replace(parsed.model, quant=quant_preset('w8a8_int8')))
    ops = target_operators(parsed, phase='prefill', tokens=3)
    norm = next((op for op in ops if op.op_id == 's0.l0.norm1'))
    quant = next((op for op in ops if op.family == 'quantize' and op.deps == (norm.op_id,)))
    assert norm.plan_required and norm.dtype == 'bf16'
    assert quant.mte3_bytes > 0 and (not quant.plan_required)
    assert norm.returns == ()
    assert dict(quant.metadata)['activation_scale_bytes'] > 0
def test_fused_norm_binding_requires_actual_fusion_and_preserves_residual_consumers():
    parsed = parse_case(declared_case(passes=['activation_quantization', 'residual_norm'], add_rmsnorm=cuda_norm.IMPLEMENTATION))
    ops = target_operators(parsed, phase='prefill', tokens=3)
    fused = next((op for op in ops if op.family == 'add_rmsnorm'))
    assert fused.plan_required and fused.returns == ('residual',)
    assert len(fused.deps) == 2
    assert dict(fused.metadata)['dispatch']['kernel'] == 'fused_add_rms_norm'
    assert any((fused.op_id in op.deps and op.recipe == 'residual_add' for op in ops))
    row = operator_costs(parsed, (fused,))['operators'][0]
    assert row['operator']['hbm_bytes'] == 3 * 32 * 2 * 6
def test_recipe_filter_leaves_distinct_qk_norm_contract_analytic():
    parsed = parse_case(declared_case(rmsnorm=NORM_BINDING))
    parsed = with_first_layer(parsed, qk_norm=True)
    ops = target_operators(parsed, phase='prefill', tokens=3)
    assert next((op for op in ops if op.recipe == 'qk_norm')).source_boundary == ''
    assert next((op for op in ops if op.op_id == 's0.l0.norm1')).plan_required
    all_norms = replace(parsed.runtime, kernels={'lowerings': {'rmsnorm': cuda_norm.IMPLEMENTATION}})
    with pytest.raises(ModelingError, match='normalization_semantic_contract_required:qk_norm'):
        target_operators(replace(parsed, runtime=all_norms), phase='prefill', tokens=3)
def policy(binding, domain=None, *extra_cases, otherwise='error'):
    return {'cases': [{'domain': domain or {'phases': ['decode']}, 'binding': binding}, *extra_cases], 'otherwise': otherwise}
def test_overlapping_geometry_does_not_silently_give_first_case_priority():
    raw = case()
    overlap = {'domain': {'valid_tokens': {'min': 1}}, 'binding': 'analytical'}
    raw['runtime']['kernels'] = {'lowerings': {'rmsnorm': policy(NORM, None, overlap)}}
    with pytest.raises(ModelingError, match='implementation_domains_overlap:rmsnorm'):
        target_graphs(raw, phase='decode', tokens=1)
def test_prefill_kda_chooses_its_chunk_source_without_decode_source_in_mixed_batch():
    parsed, _ = caller((1,))
    kernels = {'lowerings': {'kda': policy(DECODE, otherwise=PREFILL)}}
    parsed = with_first_layer(with_runtime(parsed, kernels=kernels), linear_key_heads=2)
    views = (RequestView('d', 1, 32, True, 'decode'), RequestView('p', 16, 0, True, 'prefill'))
    work = frontend(parsed, *views)
    choice, = work.implementation_selections
    assert choice['implementation'] == PREFILL
    assert not any((op.recipe == DECODE for op in work.ops))
    assert any((op.source_boundary for op in work.ops))
    decode = target_graphs(parsed, phase='decode', tokens=1, scope='body')[0]
    assert any((op.recipe == DECODE for op in decode))
    native = tuple((op for op in decode if op.recipe == DECODE))
    result = graph_costs(parsed, native, isolated=False, external_inputs={dep: 0 for op in native for dep in op.deps})
    assert result['duration_ms'] > 0
    assert result['execution']['live_tasks'] == 0
def test_hc_prefill_geometry_selects_source_only_inside_its_declared_row_domain():
    parsed = parse_case(raw_case())
    parsed = replace(parsed, model=replace(parsed.model, layers=1, body_layers=1, layer_plans=parsed.model.layer_plans[:1]))
    binding = parsed.runtime.kernels['lowerings']['native_fusion']
    kernels = {**parsed.runtime.kernels, 'lowerings': {'native_fusion': policy(binding, {'padded_tokens': {'min': 1, 'max': 256}}, otherwise='analytical')}}
    parsed = replace(parsed, runtime=replace(parsed.runtime, kernels=kernels, graph_mode='none', pp_layer_partition=(1,), parallel=replace(parsed.runtime.parallel, dp=1, ep=1)))
    small = target_graphs(parsed, phase='prefill', tokens=256, scope='body')[0]
    large = target_graphs(parsed, phase='prefill', tokens=257, scope='body')[0]
    assert any((dict(op.metadata).get('fusion_role') == 'hc_pre' for op in small))
    assert not any((dict(op.metadata).get('fusion_role') == 'hc_pre' for op in large))
    assert graph_selections(small)[0]['implementation'] == 'declared_native_fusion'
    assert graph_selections(large)[0]['implementation'] == 'analytical'
    projection = next((op for op in large if op.recipe == 'hc_projection'))
    assert projection.local_shape[0] == 257 and projection.flops > 0
def maybe_pairs(tokens, chunk):
    return tokens // chunk * chunk * chunk + (tokens % chunk) ** 2
def kda_descriptors(tokens=8192, chunk=64):
    chunks = math.ceil(tokens / chunk)
    return {'input_shapes': [[1, tokens, 12, 128]] * 4 + [[1, 12, tokens], [12], [1536], [1, 12, 128, 128], [2], [2 * chunks]], 'input_dtypes': ['DT_BF16'] * 4 + ['FLOAT'] * 4 + ['INT64', 'INT64'], 'output_shapes': [[tokens * 12 * 128], [1, 12, 128, 128], [1, 12, tokens, 128], [tokens * 12 * chunk], [tokens * 12 * chunk]] + [[1, 12, tokens, 128]] * 5 + [[1, chunks, 12, 128, 128], [1, 12, tokens, 128], [1, 12, tokens, 128]], 'output_dtypes': ['DT_BF16', 'FLOAT', 'FLOAT'] + ['DT_BF16'] * 10, 'input_formats': ['NCHW'] * 4 + ['ND'] * 4 + ['ND'] * 2, 'output_formats': ['NCHW', 'ND', 'ND'] + ['NCHW'] + ['ND'] * 9}
def test_kda_source_stage_products_and_storage_are_independent_oracles():
    ledger = kda_stage_ledger(kda_descriptors())
    tokens, heads, dk, dv, chunk, pairs = (8192, 12, 128, 128, 64, 524288)
    stages = {stage['name']: stage for stage in ledger['stage_ledger']}
    source_prepare = 4 * pairs * heads * dk
    source_finalize = 2 * tokens * heads * dk * dv + 2 * pairs * heads * dv
    assert ledger['geometry']['pairs'] == maybe_pairs(tokens, chunk) == pairs
    assert ledger['source_condition']['split_stages'] is True
    assert ledger['source_condition']['launches'] == 4
    assert ledger['source_condition']['stage_scalar_present'] is False
    assert stages['GATE_PREPARE']['recognized_product_flops'] == source_prepare == 3221225472
    assert stages['POST_WU']['recognized_product_flops'] == 0
    assert stages['FWD_H']['recognized_product_flops'] == 0
    assert stages['FINALIZE']['recognized_product_flops'] == source_finalize == 4831838208
    assert ledger['recognized_product_work'] == {'source_prepare_flops': 3221225472, 'source_finalize_flops': 4831838208, 'source_finalize_qg_h_flops': 3221225472, 'source_finalize_aqk_v_flops': 1610612736}
    storage_bytes = sum((math.prod(tensor['shape']) * (2 if tensor['dtype'] == 'bf16' else 4) for tensor in ledger['storage']['output_contract']['tensors']))
    assert len(ledger['storage']['output_contract']['tensors']) == 13
    assert ledger['storage']['output_contract']['unique_output_footprint_bytes'] == storage_bytes == 327942144
    assert ledger['storage']['workspace']['known_workspace_bytes'] == 2 * tokens * heads * chunk * 4 == 50331648
    assert ledger['storage']['traffic']['total_bytes'] is None
    assert ledger['storage']['traffic']['status'] == 'unresolved_source_stage_traffic'
def test_kda_corrected_phase_work_retains_prior_shift_without_claiming_native_timing():
    ledger = kda_stage_ledger(kda_descriptors())
    shift = ledger['semantic_phase_mismatch']
    assert shift['flops_shifted_from_prepare_to_output'] == 0
    assert shift['generic_chunk_products_flops'] == shift['source_prepare_flops']
    assert shift['generic_recurrent_output_flops'] == shift['source_finalize_flops']
    assert shift['prior_to_revision']['flops_previously_shifted_from_prepare_to_output'] == 1610612736
    assert shift['combined_flops_equal'] is True
    assert shift['combined_flops'] == shift['source_prepare_flops'] + shift['source_finalize_flops']
    assert [stage['name'] for stage in ledger['stage_ledger']] == ['GATE_PREPARE', 'POST_WU', 'FWD_H', 'FINALIZE']
    assert all((stage['mapping_status'] == 'semantic_projection_only_not_source_phase_timing' for stage in ledger['stage_ledger']))
    assert all((stage['evidence'] for stage in ledger['stage_ledger']))
    def numbers(value):
        if isinstance(value, (int, float)) and (not isinstance(value, bool)):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from numbers(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from numbers(item)
    def keys(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from keys(item)
    assert not {'duration_ms', 'mean_ms', 'p50_ms', 'activity_sum_ms', 'native_launch_count'} & set(keys(ledger))
    assert 1104 not in set(numbers(ledger))
def test_kda_source_branch_conditions_select_dense_full_or_split_stages():
    split = kda_stage_ledger(kda_descriptors())
    assert split['source_condition']['dense_fastpath']['selected'] is False
    assert split['source_condition']['launches'] == 4
    dense = kda_descriptors()
    for field in ('input_shapes', 'input_dtypes', 'input_formats'):
        del dense[field][8]
    ledger = kda_stage_ledger(dense)
    assert ledger['source_condition']['dense_fastpath']['selected'] is True
    assert ledger['source_condition']['split_stages'] is False
    assert ledger['source_condition']['launches'] == 1
    assert [stage['name'] for stage in ledger['stage_ledger']] == ['FULL']
    ledger = kda_stage_ledger(kda_descriptors(tokens=64), chunk_size=64)
    assert ledger['geometry']['total_chunks'] == 1
    assert ledger['source_condition']['split_stages'] is False
    assert ledger['source_condition']['launches'] == 1
    assert ledger['storage']['output_contract']['tensors'][0]['bytes'] == 64 * 12 * 128 * 2
    wider = kda_stage_ledger(kda_descriptors(tokens=256, chunk=128), chunk_size=128)
    assert wider['geometry']['total_chunks'] == 2
    assert wider['source_condition']['launches'] == 4
    assert wider['geometry']['pairs'] == 2 * 128 ** 2
    with pytest.raises(ModelingError, match='source_chunk_size_must_be_64_or_128'):
        kda_stage_ledger(kda_descriptors(chunk=8192), chunk_size=8192)

def test_hardware_rate_change_alters_time_and_conserves_logical_work():
    raw = case(prompt=8)
    before = predict(raw)
    raw['machine']['device']['compute_peaks']['bf16'] /= 4
    after = predict(raw)
    assert after['metrics']['makespan_ms'] > before['metrics']['makespan_ms']
    assert after['execution']['work'] == before['execution']['work']
    raw = case(prompt=8)
    raw['machine']['device']['rates']['memory'] /= 4
    slower_memory = predict(raw)
    assert slower_memory['metrics']['makespan_ms'] > before['metrics']['makespan_ms']
def test_full_graph_replay_reduces_submissions_and_preserves_padding_work():
    raw = case()
    raw['machine']['device']['launch_us'] = 1000
    eager = predict(raw)
    raw['runtime'].update(graph_capture_keys=[1, 2, 4], graph_mode='auto')
    graph = predict(raw)
    assert graph['execution']['work'] == eager['execution']['work']
    assert graph['requests'][0]['ttft_ms'] == eager['requests'][0]['ttft_ms']
    assert graph['requests'][0]['tpot_ms'] < eager['requests'][0]['tpot_ms']
    assert graph['execution']['serviced']['host@r0:s0'] < eager['execution']['serviced']['host@r0:s0']
@pytest.mark.parametrize('hardware,source', KDA_NATIVE_SKUS)
def test_kda_recommendation_selects_native_only_inside_decode_domain(hardware, source):
    raw = recommended_kda(hardware)
    for phase, tokens, expected in (('decode', 1, source), ('decode', 8, source), ('decode', 9, 'analytical'), ('prefill', 4, 'analytical')):
        graph = target_graphs(raw, phase=phase, tokens=tokens, context=4)[0]
        decision, = [row for row in graph_selections(graph) if row['selector'] == 'kda']
        assert decision['implementation'] == expected
        native = [op for op in graph if 'recurrent_kda' in op.recipe]
        assert bool(native) == (expected == source)
        assert any((op.family in {'recurrent_scan', 'kda_recurrent'} for op in graph))
    views = (RequestView('decode', 1, 4, True, 'decode'), RequestView('prefill', 4, 0, True, 'prefill'))
    work = frontend(parse_case(raw), *views)
    decision, = [row for row in work.implementation_selections if row['selector'] == 'kda']
    assert decision['coverage'] == 'analytical_semantic_work'
    assert not any(('recurrent_kda' in op.recipe for op in work.ops))
    assert {request for op in work.ops for request in op.requests} == {'decode', 'prefill'}
def test_recommended_kda_completes_chunked_mixed_serving():
    result = predict_quiet(recommended_kda('a3', prompt_tokens=4, output_tokens=3, concurrency=2, max_num_batched_tokens=5))
    assert [request['generated_tokens'] for request in result['requests']] == [3, 3]
    decisions = selection(result)
    assert {row['implementation'] for row in decisions if row['selector'] == 'kda'} == {'analytical', 'vllm_ascend_recurrent_kda_arch22'}
    assert result['execution']['live_tasks'] == 0
def test_recommended_native_kda_propagates_source_domain_failures():
    raw = recommended_kda('a5')
    plan = next((plan for plan in raw['model']['layer_plans'] if plan['attention'] == 'kda'))
    plan['linear_head_dim'] = plan['linear_key_head_dim'] = 64
    with pytest.raises(ModelingError, match='recurrent_kda_source_head_geometry_or_precision_unsupported'):
        target_graphs(raw, phase='decode', tokens=1, context=4)
@pytest.mark.parametrize('hardware', GDN_SKUS)
def test_gdn_recommendation_preserves_heterogeneous_query_work(hardware):
    case = parse_case(compose_case('qwen3.5-35b-a3b', hardware=hardware, tp=8, ep=8, layers=4, va_recommended=True))
    for lengths, native in (((1, 1), True), ((2, 3), True), ((1, 2), False), ((9, 9), hardware == 'a3')):
        views = tuple((RequestView(str(i), q, 16, True, 'decode') for i, q in enumerate(lengths)))
        work = frontend(case, *views)
        choice, = [row for row in work.implementation_selections if row['selector'] == 'gdn_decode']
        assert choice['implementation'] == ('vllm_ascend_gdn_decode' if native else 'analytical')
        assert_fields(choice['geometry'], min_query_tokens=min(lengths), max_query_tokens=max(lengths))
        assert {request for op in work.ops for request in op.requests} == {'0', '1'}
def test_recommended_gdn_completes_requests_with_different_mtp_tail_lengths():
    raw = compose_case('qwen3.5-35b-a3b', hardware='a3', tp=8, ep=8, layers=4, prompt_tokens=4, output_tokens=4, concurrency=2, mtp_proposal_tokens=1, mtp_acceptance_length=1, va_recommended=True)
    raw['workload']['requests'][0]['output_tokens'] = 2
    result = predict_quiet(raw)
    assert [request['generated_tokens'] for request in result['requests']] == [2, 4]
    choices = selection(result)
    assert {choice['implementation'] for choice in choices if choice['selector'] == 'gdn_decode'} == {'analytical', 'vllm_ascend_gdn_decode'}
    assert result['execution']['live_tasks'] == 0
def test_va_recommendation_preserves_hybrid_normalization_in_gdn_only_subset():
    raw = compose_case('qwen3-next-80b-a3b', hardware='a3', tp=4, ep=4, layers=3)
    raw['runtime']['lowering_passes'] = ['activation_quantization', 'residual_norm', 'norm_quant']
    original = deepcopy(raw)
    enriched = apply_va_recommended(raw)
    graph = target_graphs(enriched, phase='decode', tokens=1, context=128)[0]
    norms = [op for op in graph if op.family in {'rmsnorm', 'add_rmsnorm', 'add_rmsnorm_quant'}]
    assert norms
    assert any((dict(op.metadata).get('norm_weight_offset') == 1.0 for op in norms))
    assert not any((dict(op.metadata).get('native_norm_kind') for op in norms))
    assert raw == original
@pytest.mark.parametrize('alias,canonical,quant_name,expert_weight_format', [('deepseek-v4-flash', 'deepseek-v4-flash-0731', 'w4a8', 'int4'), ('deepseek-v4-pro', 'deepseek-v4-pro-0813', 'w4a8', 'int4')])
def test_sparse_series_alias_and_checkpoint_share_recommendations(alias, canonical, quant_name, expert_weight_format):
    config = va_recommended_config(alias, 'a3')
    assert config == va_recommended_config(canonical, 'a3')
    assert_fields(config, quant=quant_name, mtp_proposal_tokens=1 if 'flash' in canonical else 0)
    raw = compose_case(alias, hardware='a3', tp=8, layers=4, va_recommended=True)
    quant = raw['model']['quant']
    assert_fields(quant, weight_format='int8', expert_weight_format=expert_weight_format, activation_format='int8')
def test_documented_recommendations_replace_catalog_fp8_but_preserve_explicit_quantization():
    options = dict(hardware='a5', tp=8, layers=4)
    raw_model = compose_case('qwen3.5-397b-a17b', **options)['model']
    raw_model['quant'] = compose_case('deepseek-v3', **options)['model']['quant']
    baseline = compose_case(raw_model, **options, va_recommended=False)
    assert baseline['model']['quant']['weight_block'] == [128, 128]
    recommended = compose_case(raw_model, **options, va_recommended=True)
    quant = recommended['model']['quant']
    assert_fields(quant, weight_block=[1, 32], scale_format='fp8_e8m0', scale_bytes=1)
    for explicit in ('bf16', baseline['model']['quant']):
        expected = compose_case(raw_model, **options, quant=explicit)['model']['quant']
        actual = compose_case(raw_model, **options, quant=explicit, va_recommended=True)
        assert actual['model']['quant'] == expected
        assert actual['runtime']['metadata']['va_profile']['quantization_action'] == 'explicit_quant_preserved'
@pytest.mark.parametrize('model', ['deepseek-v3', 'deepseek-v4-pro', 'kimi-k3'])
def test_unconfirmed_fp8_profile_keeps_catalog_quantization(model):
    options = dict(hardware='a5', tp=8, layers=4)
    baseline = compose_case(model, **options, va_recommended=False)
    recommended = compose_case(model, **options, va_recommended=True)
    assert recommended['model']['quant'] == baseline['model']['quant']
    profile = recommended['runtime']['metadata']['va_profile']
    assert profile['quant'] is None
    assert profile['quantization_action'] == 'no_profile_override'
    assert profile['quantization_basis'] == 'preserve_caller_no_documented_hardware_quantization'
    assert profile['coverage'] == 'supported_simulation_subset'
@pytest.mark.parametrize('model', ['qwen3.5-397b-a17b', 'glm-5.3-flash'])
def test_fp8_documented_mxfp8_alternative_retains_provenance(model):
    raw = compose_case(model, hardware='a5', tp=8, layers=4, va_recommended=True)
    quant = raw['model']['quant']
    assert_fields(quant, weight_block=[1, 32], activation_format='fp8_e4m3', scale_format='fp8_e8m0')
    profile = raw['runtime']['metadata']['va_profile']
    assert profile['quantization_action'] == 'documented_alternative_applied'
    assert profile['source'] == va_recommended_config(model, 'a5')['source']
    assert profile['source']['repository'] == 'vllm-project/vllm-ascend'
    assert raw['runtime']['metadata'].get('draft_proposal_tokens', 0) == 0

def _v41_case(*, hardware='a3', layers=2):
    return parse_case(compose_case('deepseek-v4.1-flash', hardware=hardware, tp=8, ep=8, num_devices=8, layers=layers))
def _with_passes(case, *extra, kernels=None):
    current = tuple(name for name in case.runtime.lowering_passes if name not in {'mhc_boundary_pr', 'mhc_boundary_ideal'})
    updated = with_runtime(case, lowering_passes=(*current, *extra))
    if kernels is None:
        return updated
    bindings = dict(case.runtime.kernels)
    return with_runtime(updated, kernels={**bindings, 'lowerings': {**bindings.get('lowerings', {}), **kernels}})
def test_hc_projection_declares_fp32_input_and_output():
    ops = hc_ops(_v41_case())
    projections = [op for op in ops if op.recipe == 'hc_projection']
    assert projections
    for op in projections:
        metadata = dict(op.metadata)
        assert_fields(metadata, input_format='fp32', output_format='fp32')
def test_mhc_ideal_fuses_the_closed_coefficient_chain():
    case = _v41_case()
    plain, ops = hc_ops(case), hc_ops(_with_passes(case, 'mhc_boundary_ideal'))
    fused = [op for op in ops if op.recipe == 'hc_coefficients_fused']
    assert fused
    assert not [op for op in ops if op.recipe in {'hc_values_fp32', 'hc_projection', 'hc_inverse_rms', 'hc_sinkhorn'}]
    chain = tuple((op for op in plain if op.recipe in {'hc_values_fp32', 'hc_projection', 'hc_inverse_rms', 'hc_normalize_scale_bias_gate', 'hc_sinkhorn'}))
    assert len(fused) == 4 and len(chain) == 20
    first = fused[0]
    assert (first.family, first.dtype, first.returns, first.stream) == ('gemm', 'fp32', ('post', 'comb'), 'mhc_aux')
    for field in ('flops', 'vector_ops'):
        assert getattr(first, field) == pytest.approx(sum(getattr(op, field) for op in chain[:5]))
    values, projection, inv_rms, affine, sinkhorn = chain[:5]
    expected_read = values.mte2_bytes + max(0.0, projection.mte2_bytes - values.mte3_bytes) + max(0.0, affine.mte2_bytes - projection.mte3_bytes - inv_rms.mte3_bytes)
    assert first.mte2_bytes == pytest.approx(expected_read)
    assert first.mte3_bytes == pytest.approx(sinkhorn.mte3_bytes)
    assert first.hbm_bytes == pytest.approx(first.mte2_bytes + first.mte3_bytes)
    eliminated = 3 * values.mte3_bytes + 2 * projection.mte3_bytes + 2 * inv_rms.mte3_bytes + 2 * affine.mte3_bytes
    assert sum((op.hbm_bytes for op in chain[:5])) - first.hbm_bytes == pytest.approx(eliminated, rel=1e-09)
def test_mhc_staged_boundary_keeps_the_declared_coefficient_chain():
    ops = hc_ops(_with_passes(_v41_case(), 'mhc_boundary_pr'))
    assert not [op for op in ops if op.recipe == 'hc_coefficients_fused']
    assert [op for op in ops if op.recipe == 'hc_sinkhorn']
def test_native_fusion_protects_the_source_hc_pre_subgraph():
    binding = {'implementation': 'declared_native_fusion', 'options': {'roles': ['hc_pre'], 'traffic_profile': {'hc_pre': 'hc_pre_arch35_mk_175ef4d4'}}}
    ops = hc_ops(_with_passes(_v41_case(), 'mhc_boundary_ideal', kernels={'native_fusion': binding}))
    assert not [op for op in ops if op.recipe == 'hc_coefficients_fused']
    assert [op for op in ops if op.recipe == 'hc_pre.native'] or [op for op in ops if op.family == 'native_fusion']
    other = _with_passes(_v41_case(), 'mhc_boundary_ideal', kernels={'native_fusion': {'implementation': 'declared_native_fusion', 'options': {'roles': ['hc_post']}}})
    assert [op for op in hc_ops(other) if op.recipe == 'hc_coefficients_fused']
def test_qk_norm_does_not_block_residual_normalization():
    case = parse_case(compose_case('qwen3-32b', hardware='a3', tp=1, ep=1, num_devices=1, layers=2))
    assert any((plan.qk_norm for plan in case.model.layer_plans))
    assert case.model.activation_dtype == 'bf16'
    assert not (dict(case.model.catalog_facts).get('normalization') or {}).get('norm_weight_offset', 0.0)
    assert 'residual_norm' in case.runtime.lowering_passes
    assert dict(case.runtime.kernels['lowerings'])['add_rmsnorm'] == 'vllm_ascend_native_norm'
    context = MechanismContext(model=case.model, sku='a3', arch='a3', parallel={'tp': 1}, caller_explicit=frozenset(), options={})
    runtime = {}
    _apply_basic_norm_quant(runtime, context)
    assert 'residual_norm' in runtime['lowering_passes']

def decode_op(op_id, family, recipe, shape, **work):
    return OpSpec(op_id, family, 'decode', 0, 0, ('r',), shape, shape, shape, shape, recipe=recipe, dtype='bf16', **work)
def _mc2_graph():
    prefix = 's0.l0.ffn'
    shape = (TOKENS, TOPK, WIDTH)
    chain = (('pack', 'routing', 'route_pack', ()), ('dispatch', 'collective', 'expert_dispatch', ('pack',)),
             ('combine', 'collective', 'expert_combine', ('dispatch',)), ('unpack', 'elementwise', 'route_unpack', ('combine',)))
    ops = [decode_op(f'{prefix}.{name}', family, recipe, shape, deps=tuple(f'{prefix}.{dep}' for dep in deps))
           for name, family, recipe, deps in chain]
    rows = TOKENS * TOPK
    scale = dict(activation_group_size=WIDTH, activation_scale_format='fp32')
    quant = dict(scale, input_format='bf16', output_format='int8', activation_scale_shape=(rows,),
                 activation_scale_bytes=rows * 4.0, source_rows=float(rows), quantization_scope='moe_dispatch_expert_input')
    gate = dict(scale, activation_format='int8', activation_scale_shape=(math.ceil(RECEIVED),), activation_scale_bytes=RECEIVED * 4.0)
    pack_quantizer = decode_op(prefix + '.pack.quantize', 'quantize', 'dynamic_quantize', (rows, WIDTH),
                               deps=(prefix + '.pack',), mte2_bytes=800, mte3_bytes=200, vector_ops=4000, metadata=tuple(quant.items()))
    expert_up = decode_op(prefix + '.expert_up', 'gemm', 'expert_up', (TOKENS, 2 * WIDTH, WIDTH),
                          deps=(prefix + '.dispatch',), metadata=tuple(gate.items()))
    return prefix, [*ops, pack_quantizer, expert_up]
def _mc2_lower(ops, prefix, **options):
    runtime = SimpleNamespace(kernels={'native_moe_protocol': {'kind': 'arch35_fullmesh', **options.pop('protocol', {})}}, metadata={}, moe_recipe='alltoall')
    return lower(ops, runtime=runtime, prefix=prefix, tokens=TOKENS, top_k=TOPK, width=WIDTH, experts=32, ep=EP, source_partitions=1, received_assignments=RECEIVED, dtype='bf16', **options)
@pytest.mark.parametrize("transport", ['float', 'per_token_int8'])
def test_mc2_moves_quantization_to_the_receiving_boundary(transport):
    prefix, ops = _mc2_graph()
    options = {} if transport == 'float' else {'dispatch_quant': transport}
    result = {op.op_id.removeprefix(prefix + '.'): op for op in _mc2_lower(ops, prefix, protocol=options)}
    assert 'pack.quantize' not in result
    gate = result['expert_up']
    if transport == 'per_token_int8':
        assert 'dispatch.received_quantize' not in result
        assert dict(gate.metadata)['activation_producer'] == 'explicit_transport_quantization'
        assert gate.deps == (prefix + '.dispatch',)
        return
    quantizer = result['dispatch.received_quantize']
    assert gate.deps == (quantizer.op_id,)
    assert_fields(dict(quantizer.metadata), quantization_scope='moe_received_expert_input', source_rows=RECEIVED)
    scale = RECEIVED / (TOKENS * TOPK)
    assert (quantizer.mte2_bytes, quantizer.mte3_bytes, quantizer.vector_ops) == pytest.approx(
        (800 * scale, 200 * scale, 4000 * scale))
    assert_fields(dict(gate.metadata), activation_payload_producer=quantizer.op_id, activation_scale_producer=quantizer.op_id)
def test_mc2_rejects_an_external_pack_quantizer_consumer():
    prefix, ops = _mc2_graph()
    consumer = decode_op('other.consumer', 'elementwise', 'scale_read', (TOKENS, TOPK, WIDTH), deps=(prefix + '.pack.quantize',))
    with pytest.raises(ModelingError, match='mc2_cannot_absorb_external'):
        _mc2_lower([*ops, consumer], prefix)
def test_mc2_non_mc2_selection_falls_through_without_raising():
    runtime = SimpleNamespace(kernels={}, metadata={'moe_comm_mode': 'dynamic'}, moe_recipe='alltoall')
    features = SimpleNamespace(tp=1, graph_buckets=(), metadata=(('moe_comm_mode', 'dynamic'), ('max_num_seqs', 1)))
    assert select_common(runtime, features, tokens=1, top_k=4, experts=32, ep=1) is None

def test_mqsmla_pure_swa_has_zero_compressed_rows():
    case = _v41_case(hardware='a5', layers=4)
    views = (RequestView('r', 128, 131583, True, 'decode'),)
    pure = next((plan for plan in case.model.layer_plans if plan.attention == 'mla' and plan.compress_ratio == 0 and plan.index_topk))
    shape = query_shape(views, pure, topk_present=True)
    assert shape['ratio'] == 0
    assert set(query_rows(shape)) == {(128, 0)}
    sparse = next((plan for plan in case.model.layer_plans if plan.attention == 'mla' and plan.compress_ratio > 0))
    sparse_shape = query_shape(views, sparse, topk_present=True)
    assert set(query_rows(sparse_shape)) == {(128, 512)}
    no_topk = query_shape(views, replace(pure, compress_ratio=4), topk_present=False)
    assert no_topk['ratio'] == 0 and no_topk['topk'] == 0
    assert set(query_rows(no_topk)) == {(128, 0)}
def test_mqsmla_rejects_padded_views_and_keeps_padded_tokens():
    case = _v41_case(hardware='a5')
    plan, parallel = next(p for p in case.model.layer_plans if p.attention == 'mla' and p.compress_ratio == 0), case.runtime.parallel
    padded = (RequestView('r', 4, 128, False, 'decode', padding=True),)
    assert unsupported_reason(views=padded, plan=plan, parallel=parallel, model=case.model, padded_tokens=4) == 'mqsmla_binding_requires_contiguous_unpadded_decode_or_draft_rows'
    split = (RequestView('r', 4, 128, False, 'decode', query_ranges=((0, 2), (2, 4))),)
    assert unsupported_reason(views=split, plan=plan, parallel=parallel, model=case.model, padded_tokens=4) == 'mqsmla_binding_requires_contiguous_unpadded_decode_or_draft_rows'
    contiguous = (RequestView('r', 4, 128, True, 'decode'),)
    assert unsupported_reason(views=contiguous, plan=plan, parallel=parallel, model=case.model, padded_tokens=8) != 'mqsmla_binding_requires_positive_padded_rows'
    assert unsupported_reason(views=contiguous, plan=plan, parallel=parallel, model=case.model, padded_tokens=2) == 'mqsmla_binding_requires_positive_padded_rows'
def test_hbm_service_modes_scale_with_the_derived_bandwidth():
    machine = compose_case('qwen3-32b', hardware='a3', tp=1, ep=1, num_devices=1, layers=1)['machine']
    for device in machine.get('devices') or [machine['device']]:
        device['service_modes'] = {'hbm_read': {'resource': 'hbm', 'rate': float(device['rates']['hbm'])}}
    base = Hardware.from_mapping(machine)
    peak = float(dict(base.dies[0].rates)['hbm'])
    mode_rate = float(dict(base.dies[0].service_modes)['hbm_read'][1])
    cases = (({'hbm_peak_bytes_per_s': peak * 0.5}, 0.5), ({'hbm_efficiency': 0.7}, 0.7), ({'hbm_peak_bytes_per_s': peak * 0.5, 'hbm_efficiency': 0.6}, 0.3), ({'aic_count': base.dies[0].compute_units}, 1.0))
    for overrides, factor in cases:
        derived = derive_hardware_sku(base, sku='derived-hbm', **overrides)
        die = derived.dies[0]
        assert float(dict(die.rates)['hbm']) == pytest.approx(peak * factor)
        assert float(dict(die.service_modes)['hbm_read'][1]) == pytest.approx(mode_rate * factor)
        rederived = derive_hardware_sku(derived, sku='derived-hbm')
        assert float(dict(rederived.dies[0].service_modes)['hbm_read'][1]) == pytest.approx(mode_rate * factor)
def test_dram_bound_service_modes_scale_with_each_derived_hbm_rate():
    """ACLNN DRAM streams follow the declared HBM bandwidth, not a fixed rate.
    ``mte2_aic_dram`` / ``mte2_aiv_dram`` declare an HBM-serviced read stream
    while retaining their MTE2 resource name.  Re-deriving from the scaled
    profile must not apply the ratio a second time.
    """
    base = Hardware.from_mapping({'backend': 'ascend', 'sku': 'Ascend950DT_9582'})
    original_hbm = float(dict(base.dies[0].rates)['hbm'])
    original = {name: float(dict(base.dies[0].service_modes)[name][1]) for name in ('mte2_aic_dram', 'mte2_aiv_dram')}
    for peak in (1400000000000.0, 1200000000000.0):
        factor = peak / original_hbm
        derived = derive_hardware_sku(base, sku='derived-dram', hbm_peak_bytes_per_s=peak)
        die = derived.dies[0]
        assert float(dict(die.rates)['hbm']) == pytest.approx(peak)
        for name, rate in original.items():
            assert float(dict(die.service_modes)[name][1]) == pytest.approx(rate * factor)
        rederived = derive_hardware_sku(derived, sku='derived-dram')
        for name, rate in original.items():
            assert float(dict(rederived.dies[0].service_modes)[name][1]) == pytest.approx(rate * factor)
def _engram_case():
    return parse_case(compose_case('qwen3.8-flash-next', hardware='a5', tp=2, ep=1, num_devices=2, layers=2, engram_cpu_offload=True, prompt_tokens=8, output_tokens=2, concurrency=1))
def test_ple_host_lookup_has_zero_duplicate_device_work():
    ops = target_operators(_engram_case(), phase='prefill', tokens=8, context=0, sequences=1, scope='body')
    lookups = [op for op in ops if op.recipe == 'ple_host_uva_lookup']
    assert lookups
    for op in lookups:
        assert (op.vector_ops, op.mte2_bytes, op.mte3_bytes, op.hbm_bytes) == (0.0, 0.0, 0.0, 0.0)
        work = dict(op.metadata)['resource_work']
        assert set(work) == {'host_memory', 'host_vector', 'h2d'}
        assert work['host_memory'] == work['h2d'] > 0.0
        assert work['host_vector'] > 0.0
def test_host_staging_requires_qualified_bound_keys():
    def context(work, names=None):
        return SimpleNamespace(resource_names=names if names is not None else {'host_memory': 'host_memory@r0:s0', 'host_vector': 'host_vector@r0:s0', 'h2d': 'h2d@r0:s0'}, work=work, minimum_ms=0.0)
    program = HostStaging().lower(context({'host_memory@r0:s0': 81920.0, 'host_vector@r0:s0': 512.0, 'h2d@r0:s0': 81920.0}))
    stages = {stage.name: dict(stage.work) for stage in program.stages}
    assert stages['lookup'] == {'host_memory@r0:s0': 81920.0}
    assert stages['decode'] == {'host_vector@r0:s0': 512.0}
    assert stages['transfer'] == {'h2d@r0:s0': 81920.0}
    with pytest.raises(ModelingError, match='host_staging_requires_bound'):
        HostStaging().lower(context({'host_memory@r0:s0': 1.0}))
    with pytest.raises(ModelingError, match='host_staging_requires_host_memory'):
        HostStaging().lower(context({}, names={}))
