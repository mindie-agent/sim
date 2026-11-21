"""Resource conservation and causality at fusion boundaries.

[Role] Exercise native and generic fusion, placement, cache ownership and typed ports.
[Invariants] Conserve explicit work, dependencies and declared domains.
[Timeouts & Locks] Finite simulation and bounded local fixtures.
[Side-Effect Policy] Repository fixtures are read-only.
"""
import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from pathlib import Path
import pytest
from mindie_sim import graph_costs
from _support.operator_cases import hardware_case
from mindie_sim.catalog.cache import index_row_bytes
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy, _memory_hierarchy, gqa
from mindie_sim.prediction.blocks import norm
from mindie_sim.prediction.operators.hyper_connections.family_hc import pre_phase_work
from mindie_sim.prediction.operators.mla.generic_latent_attention import owned_completed_rows, semantic_index_selection, unique_index_key_rows, window_tile_rows
from mindie_sim.prediction.implementation_selection import resolve
from mindie_sim.prediction.implementations import consumer_index, input_ports, produced_values, select
from mindie_sim.prediction.operators.sparse_indexer.index_pool import row_shard_tp
from mindie_sim.prediction.operators.sparse_indexer.index_pool_native import POOL_PROJECTION_CATEGORIES, prepare_model, supported
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.hyper_connections.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operators.hyper_connections.native_fusion_mla_membase import validate_declared_view
from mindie_sim.prediction.operators.hyper_connections.native_fusion_traffic import apply_traffic_profile
from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs, target_operators
from mindie_sim.prediction.prefix_reads import CacheAccess, plan_prefix_reads
from mindie_sim.prediction.program import KernelProgram, PipelinePhase, PipelineProgram, PipelineTemplate, RecipeBook, Stage, coalesce_execution_program
from mindie_sim.prediction.program_catalog import compile_pipeline_template, compile_program, submit_program, submit_program_catalog
from mindie_sim.prediction.operators.recurrent.recurrent import linear
from mindie_sim.prediction.runtime import accel_bridge, accel_build
from mindie_sim.prediction.runtime.contracts import CatalogGeneration
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization, lower_implementations, norm_quant
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure_physical import node_physical

pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
HC_PROFILE = 'hc_pre_m_k_split_175ef4d4'
MLA_PROFILE = 'mla_prolog_arch35_split_m_n_ae923905'
GMM_PROFILE = 'grouped_swiglu_quant_int4_post_ae923905'
INDEXER_ARCHES = ('A2', 'A5')
prefill_semantic = partial(target_operators, phase='prefill', tokens=8, resource_fusion=False)
def assert_fields(actual, **expected):
    assert {key: actual[key] for key in expected} == expected
def example_case(name):
    return json.loads((ROOT / 'examples/cases' / (name + '.json')).read_text())

@pytest.mark.parametrize('rows,width', ((2, 8), (3, 65)))
def test_fused_public_graph_eliminates_only_private_intermediates(rows, width):
    raw = hardware_case()
    raw['machine']['device']['compute_peaks']['fp8_e4m3'] = 2e6
    parsed, quant = parse_case(raw), quant_preset('mxfp8')
    parsed = replace(parsed, model=replace(parsed.model, quant=quant))
    payload = rows * width * 2
    residual = vector_op(op_id='residual', deps=('input', 'saved'), stage=0, layer=0, requests=('r',), elements=rows * width, ops_per_element=1, dtype='bf16', family='elementwise', phase='prefill', recipe='residual_add', read_bytes_override=2 * payload, write_bytes_override=payload)
    normalized = norm(name='norm', deps=('residual',), stage=0, layer=0, requests=('r',), tokens=rows, width=width, model=parsed.model, mode='prefill')
    matrix = gemm_op(op_id='projection', deps=('norm',), stage=0, layer=0, requests=('r',), m=rows, n=width, k=width, quant=quant, category='dense_mlp', phase='prefill')
    consumer = replace(residual, op_id='consumer', deps=('projection', 'residual'))
    reports, versions = [], []
    for passes in (('activation_quantization',), ('activation_quantization', 'norm_quant', 'residual_norm')):
        runtime = replace(parsed.runtime, lowering_passes=passes)
        ops = lower_implementations((residual, normalized, matrix, consumer), runtime, 'bf16')
        versions.append(ops)
        reports.append(graph_costs(replace(parsed, runtime=runtime), ops, external_inputs={'input': 0, 'saved': 0}, isolated=False))
    before, after = (report['execution']['serviced'] for report in reports)
    for resource, saved in (('memory', 3 * payload), ('matrix', 0), ('vector', 0)):
        assert before[f'{resource}@r0:s0'] - after[f'{resource}@r0:s0'] == pytest.approx(saved, abs=1e-9)
    fused = find(versions[1], family='add_rmsnorm_quant')
    assert fused.returns == ('activation_scales', 'residual') and versions[1][-1].deps == ('projection', fused.op_id)
    assert len(versions[0]) == len(versions[1]) + 2 and all(r['execution']['live_tasks'] == 0 for r in reports)
def bind_native(case, roles, profile='preserve_workspaces', **lowerings):
    lowerings['native_fusion'] = {
        'implementation': 'declared_native_fusion', 'options': {'roles': roles, 'traffic_profile': profile},
    }
    kernels = {**case.runtime.kernels, 'lowerings': {**case.runtime.kernels.get('lowerings', {}), **lowerings}}
    return replace(case, runtime=replace(case.runtime, kernels=kernels))
def _fixture_case(name, roles, profile='preserve_workspaces', peaks=None, **source_lowerings):
    raw = example_case(name)
    if peaks:
        raw['machine'].setdefault('device', {}).setdefault('compute_peaks', {}).update(peaks)
    return bind_native(parse_case(raw), roles, profile, **source_lowerings)
def hc_group(profile='preserve_workspaces'):
    peaks = {'hf32': 2e14} if profile == HC_PROFILE else None
    case = _fixture_case('v4f-a3-w8a8-tp8', ['hc_pre'], profile, peaks=peaks)
    declaration, members = _group(prefill_semantic(case), 'hc_pre')
    return case, declaration, members
def _group(ops, role):
    return next(((declaration, members) for declaration, members in _groups(ops) if declaration['role'] == role))
def evaluate(case, ops, members, **kwargs):
    ids = {op.op_id for op in members}
    inputs = {dep: 0 for op in members for dep in op.deps if dep not in ids}
    return graph_costs(case, ops, external_inputs=inputs, isolated=False, **kwargs)
def find(ops, **attrs):
    return next(op for op in ops if all(getattr(op, key) == val for key, val in attrs.items()))
def sum_attr(ops, field):
    return sum(getattr(op, field) for op in ops)
def test_receive_only_sparse_rank_preserves_expert_fusion_without_local_situ():
    raw = example_case('kimi-k3-a5-sp-right-pad-conditional')
    raw['model']['layers'] = 2
    graphs = target_graphs(raw, phase='prefill', tokens=64, scope='body', sequences=[1, 0, 0, 0])
    for replica, ops in graphs.items():
        roles = {dict(op.metadata).get('fusion_role') for op in ops}
        assert 'grouped_gate_up_situ_quant' in roles
        assert ('situ_quant' in roles) == (replica == 0)
        assert any((op.family == 'alltoall' and op.link_bytes > 0 for op in ops))
    result = graph_costs(raw, graphs, external_inputs={'analysis_input': 0}, isolated=False)
    assert result['duration_ms'] > 0 and result['execution']['serviced']['matrix@r1:s0'] > 0
    assert_fields(result['execution'], live_tasks=0, live_programs=0)
def test_hc_fusion_runs_one_launch_with_source_gram_and_external_outputs():
    case, declaration, members = hc_group(HC_PROFILE)
    physical = lower_native_fusions(members, case.runtime)
    fused, = physical
    assert len(members) == 6
    phases = {phase['recipe']: phase for phase in dict(fused.metadata)['fusion_phases']}
    tokens, mixed, hc = (declaration['tensor_view'][key] for key in ('tokens', 'mixed', 'hc'))
    assert fused.hbm_bytes == sum_attr(members, 'hbm_bytes') - tokens * (2 * mixed + hc) * 4
    for field in ('flops', 'vector_ops'):
        assert getattr(fused, field) == sum_attr(members, field)
    unchanged = {'hbm_bytes': ('hc_values_fp32', 'hc_projection', 'hc_inverse_rms'),
                 'mte3_bytes': ('hc_sinkhorn', 'hc_pre_mix')}
    for field, recipes in unchanged.items():
        assert [phases[recipe][field] for recipe in recipes] == [
            getattr(find(members, recipe=recipe), field) for recipe in recipes]
    reference, selected = [evaluate(case, ops, members, include_submission=True) for ops in (members, physical)]
    before, after = (report['execution']['serviced'] for report in (reference, selected))
    assert_fields(selected['timeline'][0], fusion_role='hc_pre', fusion_group_key=dict(fused.metadata)['fusion_group_key'],
                  physical_source_program=HC_PROFILE, native_implementation=declaration['implementation'])
    assert before['host@r0:s0'] == pytest.approx(6 * after['host@r0:s0'])
    device = case.machine.dies[0]
    source = pre_phase_work(declaration['tensor_view'], Geometry(device.compute_units, {'vector': device.units('vector')}))
    for resource, key in (('memory', 'hbm'), ('vector', 'vector')):
        assert after[f'{resource}@r0:s0'] == pytest.approx(sum(work.resources.get(key, 0) for work in source.values()))
    assert after['matrix@r0:s0'] == pytest.approx(sum((1000 * amount / device.peak(dtype) for work in source.values() for dtype, amount in work.matrix.items())))
    assert after['matrix@r0:s0'] > before['matrix@r0:s0']
    assert_fields(selected['execution'], live_tasks=0, live_programs=0)
    affine = find(members, recipe='hc_normalize_scale_bias_gate')
    adjusted, ledger = apply_traffic_profile(members, declaration, HC_PROFILE, external_outputs=(affine.op_id,))
    assert find(adjusted, op_id=affine.op_id).mte3_bytes == affine.mte3_bytes
    assert ledger['eliminated_writes'] == 0 and ledger['eliminated_reads'] > 0
@pytest.mark.parametrize('extended', [False, True])
def test_default_fusion_conserves_semantic_and_extended_work(extended):
    case, _, members = hc_group()
    if extended:
        members[0] = replace(members[0], metadata=(*members[0].metadata, ('resource_work', {'vector': 1000})))
        physical = lower_native_fusions(members, case.runtime)
        before, after = (evaluate(case, ops, members)['execution']['serviced'] for ops in (members, physical))
        assert after['vector@r0:s0'] == before['vector@r0:s0']
        assert physical[0].vector_ops == sum_attr(members, 'vector_ops') + 1000
        return
    lowerings = {key: value for key, value in case.runtime.kernels['lowerings'].items() if key != 'native_fusion'}
    baseline = replace(case.runtime, kernels={**case.runtime.kernels, 'lowerings': lowerings})
    assert lower_native_fusions(members, baseline) == tuple(members)
    fused = lower_native_fusions(members, case.runtime)[0]
    for field in ('flops', 'hbm_bytes', 'mte2_bytes', 'mte3_bytes', 'vector_ops'):
        assert getattr(fused, field) == sum((getattr(op, field) for op in members))
    boundary = dict(fused.metadata)['fusion_boundary']
    assert boundary['traffic']['reused_edges'] == []
    assert boundary['evidence']['capture_binary_identity'] == 'not_established'

def test_mla_and_indexer_alias_live_outputs_without_deleting_query_workspaces():
    case = _fixture_case('glm53-a3-w4a8', ['mla_prolog', 'lightning_indexer'], {'mla_prolog': MLA_PROFILE}, mla_prolog='vllm_ascend_mla_prolog_v3')
    semantic = prefill_semantic(case)
    physical = lower_native_fusions(semantic, case.runtime)
    declaration, members = _group(semantic, 'mla_prolog')
    fused = next((op for op in physical if dict(op.metadata).get('fusion_group_key') == declaration['key']))
    after = {phase['recipe']: phase for phase in dict(fused.metadata)['fusion_phases']}
    for recipe in ('q_norm', 'dynamic_quantize', 'rotary', 'kv_store', 'q_b'):
        assert after[recipe]['mte3_bytes'] == find(members, recipe=recipe).mte3_bytes
    assert after['kv_norm']['mte3_bytes'] == after['kv_store']['mte2_bytes'] == 0
    assert fused.op_id in find(physical, op_id='s0.l0.index_q').deps
    assert fused.op_id in find(physical, layer=0, family='attention').deps
    indexer = next(op for op in physical if dict(op.metadata).get('fusion_role') == 'lightning_indexer')
    assert indexer.hbm_bytes == sum_attr(_group(semantic, 'lightning_indexer')[1], 'hbm_bytes')
    positions = {op.op_id: index for index, op in enumerate(physical)}
    assert all((positions[dep] < positions[op.op_id] for op in physical for dep in op.deps if dep in positions))
@pytest.mark.parametrize('name,profile', [('glm53-a3-w4a8', GMM_PROFILE), ('v4f-a3-w8a8-tp8', 'grouped_swiglu_quant_int8_split_workspace_ae923905')])
def test_grouped_postprocess_eliminates_only_swiglu_quant_intermediate(name, profile):
    case = _fixture_case(name, ['grouped_gate_up_activation_quant'], profile)
    semantic = prefill_semantic(case)
    _, members = _group(semantic, 'grouped_gate_up_activation_quant')
    fused = lower_native_fusions(members, case.runtime)[0]
    phases = dict(fused.metadata)['fusion_phases']
    matrix_writes = next(phase['mte3_bytes'] for phase in phases if phase['family'] == 'grouped_gemm')
    assert matrix_writes == find(members, family='grouped_gemm').mte3_bytes
    assert fused.hbm_bytes == sum_attr(members, 'hbm_bytes') - 2 * find(members, family='swiglu').mte3_bytes
    assert not any((op['recipe'] == 'expert_down' for op in phases))
    assert fused.vector_ops == sum((op.vector_ops for op in members))
def test_resource_cohorts_ignore_layer_identity_but_retain_selected_traffic_recipe():
    case = _fixture_case('v4f-a3-w8a8-tp8', ['hc_pre'], HC_PROFILE, peaks={'hf32': 2e14})
    semantic = prefill_semantic(case)
    fused = [op for op in lower_native_fusions(semantic, case.runtime) if op.family == 'native_fusion']
    signatures = {node_physical(op)['physical_signature'] for op in fused}
    assert len({op.layer for op in fused}) > 1 and len(signatures) == 1
    for phases in (dict(op.metadata)['fusion_phases'] for op in fused):
        assert [phase['op_id'] for phase in phases] == [str(i) for i in range(len(phases))]
    other = lower_native_fusions(semantic, _fixture_case('v4f-a3-w8a8-tp8', ['hc_pre']).runtime)
    assert node_physical(find(other, family='native_fusion'))['physical_signature'] not in signatures

def test_unknown_chunk_stage_cannot_be_priced_as_the_complete_native_algorithm():
    case = _fixture_case('kimi-k3-a5', ['kda_chunk'])
    with pytest.raises(ValueError, match='native_fusion_requires_complete_single_kernel'):
        lower_native_fusions(prefill_semantic(case), case.runtime)
    with pytest.raises(ValueError, match='native_fusion_selected_roles_unmatched:mla_prolog'):
        target_operators(_fixture_case('v4f-a3-w8a8-tp8', ['mla_prolog']), phase='prefill', tokens=8)
def test_local_reuse_cannot_silently_disagree_with_shape_generated_memory_program():
    case = _fixture_case('v4f-a3-w8a8-tp8', ['hc_pre'], HC_PROFILE, peaks={'hf32': 2e14})
    kernels = {**case.runtime.kernels, 'recipes': {'hc_normalize_scale_bias_gate': {'kind': 'tiled_pointwise', 'input_bytes_per_element': 4, 'outputs': [{'bytes_per_element': 4}]}}}
    case = replace(case, runtime=replace(case.runtime, kernels=kernels))
    with pytest.raises(ValueError, match='local_reuse_requires_work_preserving_phase_recipe'):
        target_operators(case, phase='prefill', tokens=8)
def test_resource_projection_preserves_declared_capture_boundaries():
    case = _fixture_case('v4f-a3-w8a8-tp8', ['hc_pre'])
    case = replace(case, runtime=replace(case.runtime, graph_mode='auto', graph_capture_mode='piecewise', piecewise_capture_keys=(8,), mixed_mode='mixed'))
    semantic = target_graphs(case, phase='prefill', tokens=8, resource_fusion=False)
    physical = target_graphs(case, phase='prefill', tokens=8)
    assert physical == {rank: lower_native_fusions(ops, replace(case.runtime, kernels=semantic.effective_kernels))
                        for rank, ops in semantic.items()}
    _, members = _group(next(iter(semantic.values())), 'hc_pre')
    assert all((dict(op.metadata)['graph_captured'] for op in members))
    fused_metadata = dict(lower_native_fusions(members, case.runtime)[0].metadata)
    assert fused_metadata['graph_segment'] == dict(members[0].metadata)['graph_segment']
    split = replace(case, runtime=replace(case.runtime, splitting_ops=('hc_inverse_rms',)))
    with pytest.raises(ValueError, match='native_fusion_crosses_graph_capture_boundary'):
        target_graphs(split, phase='prefill', tokens=8)

SELECTORS = ({'recipe': 'dense_gate_up'},)
domain_graphs = partial(target_graphs, scope='body', phase='decode', tokens=6, sequences=2,
                        context=4096, apply_body_defaults=False)
def query_conversions(ops):
    return {op.op_id for op in ops if op.family == 'quantize' and '.q_a.quantize' in op.op_id}
def test_norm_quant_selects_only_dense_ffn_inputs_in_the_full_body():
    original = example_case('high-concurrency-20261030/glm78-capture-rows/a3_glm53_w4a8_tp16_tpot30_c2')
    original['runtime']['lowering_passes'] = ['activation_quantization', 'norm_quant', 'residual_norm']
    raw = deepcopy(original)
    raw['runtime']['kernels']['fusion_domains'] = {'norm_quant': {'consumers': list(SELECTORS)}}
    generic, selected = (domain_graphs(value)[0] for value in (original, raw))
    before, after = [Counter(op.family for op in ops) for ops in (generic, selected)]
    assert_fields(before, add_rmsnorm_quant=60)
    assert_fields(after, add_rmsnorm_quant=3, add_rmsnorm=152, quantize=before['quantize'] + 57)
    for op in (op for op in selected if op.family == 'add_rmsnorm_quant'):
        consumer, = [o for o in selected if op.op_id in o.deps and o.recipe == 'dense_gate_up']
        assert_fields(dict(consumer.metadata), structure_block='ffn', structure_role='body')
        assert dict(op.metadata)['fusion_consumer_domain']['consumers'] == SELECTORS
    ordinary_before, ordinary_after = [query_conversions(ops) for ops in (generic, selected)]
    assert (len(ordinary_before), len(ordinary_after), len(ordinary_after - ordinary_before)) == (21, 78, 57)
    assert sum((dict(o.metadata).get('fusion_role') == 'mla_prolog' for o in selected)) == 78
    for field, added in (('flops', 0), ('vector_ops', 0), ('hbm_bytes', 57 * 2 * 12 * 6144 * 2)):
        assert sum_attr(selected, field) - sum_attr(generic, field) == added
    assert domain_graphs(original)[0] == generic
def shared_conversion():
    norm = vector_op(op_id='n', deps=('input',), stage=0, layer=0, requests=(), elements=32, ops_per_element=5, dtype='bf16', family='rmsnorm', phase='decode', recipe='rmsnorm', shape=(1, 32))
    quant = QuantSpec(name='control', weight_format='int8', activation_format='int8')
    def matrix(name, recipe, block):
        return gemm_op(op_id=name, deps=('n',), stage=0, layer=0, requests=(), m=1, n=64, k=32, quant=quant, category='dense_mlp', phase='decode', recipe=recipe, metadata=(('structure_block', block), ('structure_role', 'body')))
    return activation_quantization((norm, matrix('a', 'dense_gate_up', 'ffn'), matrix('b', 'q_projection', 'attention')), 'bf16')
def test_one_allowed_consumer_cannot_fuse_a_shared_conversion_for_other_domains():
    ops = shared_conversion()
    assert_fields(Counter(op.family for op in ops), quantize=1)
    selected = norm_quant(ops, 'bf16', consumers=SELECTORS)
    assert_fields(Counter(op.family for op in selected), rmsnorm=1, quantize=1, rmsnorm_quant=0)
    both = norm_quant(ops, 'bf16', consumers=(*SELECTORS, {'recipe': 'q_projection'}))
    assert_fields(Counter(op.family for op in both), rmsnorm_quant=1)
    assert {o.deps for o in both if o.recipe in {'dense_gate_up', 'q_projection'}} == {('a.quantize',)}
    assert norm_quant(ops, 'bf16') == norm_quant(ops, 'bf16', consumers=None)
def test_selector_structure_axes_are_conjunctive_and_empty_domain_disables_fusion():
    both = [replace(o, recipe='dense_gate_up') if o.op_id == 'b' else o for o in shared_conversion()]
    restricted = ({'recipe': 'dense_gate_up', 'structure_block': 'ffn'},)
    for selectors, expected in ((SELECTORS, True), (restricted, False), ((), False)):
        assert any(op.family == 'rmsnorm_quant' for op in norm_quant(both, 'bf16', consumers=selectors)) is expected

GENERATION = CatalogGeneration(registry_generation='pipeline', implementation_generation='catalog')
def _case(model, **kwargs):
    defaults = dict(hardware='a3', tp=8, ep=8, num_devices=8, layers=2)
    return parse_case(compose_case(model, **(defaults | kwargs)))
_ops = partial(target_operators, scope='body')
def _with_passes(case, selected):
    passes = tuple(name for name in case.runtime.lowering_passes
                   if not name.startswith('mhc_boundary') and name != 'kda_in_kernel_l2norm')
    return replace(case, runtime=replace(case.runtime, lowering_passes=(*passes, selected)))
def _placed_fixture(backend, peer_count):
    resources = [Resource('cube', 4.0, units=4), Resource('hbm', 4.0, units=4)]
    engine = Engine(resources, backend=backend)
    work = {'cube': 4.0, 'hbm': 4.0}
    pinned = engine.submit('pinned', work, coupled=True, placement={'cube': (0,), 'hbm': (0,)})
    peers = [engine.submit(f'peer{i}', work, coupled=True) for i in range(peer_count)]
    timer = engine.submit('timer', {}, after=(peers[0],), latency=10.0)
    for _ in range(100):
        if all((task.done for task in (*peers, pinned))) and timer.done:
            break
        engine.advance(until=(timer,), max_steps=1000)
    return (engine, pinned, peers, timer)
@pytest.mark.parametrize('peer_count,expected_timer,expected_pinned', [(8, 19, 12), (3, 14, 7)])
def test_c11_and_python_normalize_mixed_placements_identically(peer_count, expected_timer, expected_pinned):
    status = accel_build.build_accel()
    assert status['ok'], status
    accel_bridge.load_library(status)
    python, native = [_placed_fixture(backend, peer_count) for backend in ('python', 'c')]
    assert native[0].accel_status()['native_steps'] > 0
    for left, right in zip((python[1], *python[2]), (native[1], *native[2])):
        assert left.time == pytest.approx(right.time, abs=1e-09)
    assert (python[3].time, native[3].time, python[1].time) == pytest.approx(
        (expected_timer, expected_timer, expected_pinned))
SERIAL_RESOURCES = (Resource('hbm', 1.0), Resource('cube', 1.0))
TILE_STAGES = (Stage('load', {'hbm': 10.0}), Stage('compute', {'cube': 10.0}, previous=('load',)))
@pytest.mark.parametrize('minimum,coalesce', [(0, False), (0.01, False), (0, True)])
def test_tile_zero_ignores_previous_in_both_fast_paths(minimum, coalesce):
    engine = Engine(SERIAL_RESOURCES, backend='python')
    program = PipelineProgram(1, 2, lambda _: TILE_STAGES, minimum_ms=minimum)
    if coalesce:
        program = coalesce_execution_program(program)
    done = program.start(engine, 'p', engine.submit('ready'))
    engine.advance(until=(done,), max_steps=100)
    assert done.time == pytest.approx(10)
def test_catalog_resolves_iteration_offset_against_the_prior_tile():
    engine = Engine(SERIAL_RESOURCES, backend='python')
    template = PipelineTemplate(owner=0, slots=2, phases=(PipelinePhase('tile', TILE_STAGES, 2),))
    catalog = compile_pipeline_template(template, generation=GENERATION)
    submission = submit_program_catalog(engine, catalog, name='k', after=(engine.submit('ready'),))
    engine.advance(until=(submission.done,), max_steps=100)
    assert submission.done.time == pytest.approx(20.0)
def test_catalog_carries_stage_cohort_size_and_matches_direct_execution():
    group_work, peer_work = ({'hbm': units, 'cube': units} for units in (4, 1))
    group = Stage('group', group_work, coupled=True, cohort_size=4)
    kernel = KernelProgram((group, Stage('peer', peer_work, coupled=True)))
    catalog = compile_program(kernel, generation=GENERATION)
    assert find(catalog.stages, label='group').cohort_size == 4
    engine = Engine(SERIAL_RESOURCES, backend='python')
    submission = submit_program(engine, kernel, name='k', ready=engine.submit('ready'))
    engine.advance(until=(submission.done,), max_steps=100)
    direct = Engine(SERIAL_RESOURCES, backend='python')
    peer = direct.submit('peer', peer_work, coupled=True)
    direct.submit('group', group_work, coupled=True, cohort_size=4)
    direct.advance(until=(peer,), max_steps=1000)
    assert peer.done
    assert (peer.time, submission.done.time) == pytest.approx((5, peer.time))

@pytest.mark.parametrize('ideal', [False, True])
def test_mhc_norm_waits_for_the_correct_pre_gate_producer(ideal):
    case = _case('deepseek-v4.1-flash', feature_track='expected_ideal')
    case = _with_passes(case, 'mhc_boundary_ideal' if ideal else 'mhc_boundary_pr')
    layer = {op.op_id.removeprefix('s0.l0.'): op for op in _ops(case, phase='decode', tokens=1, context=2048, sequences=2)}
    for block, norm_id in (('attention', 'norm1'), ('ffn', 'norm2')):
        assert (layer[f'{block}.hc_sinkhorn'].op_id in layer[norm_id].deps) is ideal
        if not ideal:
            assert layer[f'{block}.hc_affine'].op_id in layer[norm_id].deps
    if not ideal:
        assert layer['ffn.hc_post'].recipe == 'hc_post_mix'
        assert 's0.l0.attention.hc_post' in layer['norm2'].deps
def test_lowrank_mhc_preserves_declared_injection_and_gate_slice_reads():
    case = _with_passes(_case('qwen3.8-flash-next', layers=1, quant='w8a8'), 'mhc_boundary_pr')
    block = {op.op_id.removeprefix('s0.l0.attention.'): op for op in _ops(case, phase='decode', tokens=1, context=2048)}
    for target, source, excluded, byte_count in (
        ('hc_gate_mix', 'hc_up', 'hc_down_inject', 40_960),
        ('hc_post', 'hc_down_inject', 'hc_up', 25_608),
    ):
        assert block[source].op_id in block[target].deps and block[excluded].op_id not in block[target].deps
        assert block[target].mte2_bytes == pytest.approx(byte_count)
def test_qkv_norm_rope_writes_every_declared_output_and_names_ports():
    case = _with_passes(_case('qwen3-next-80b-a3b', layers=4), 'qkv_norm_rope')
    nodes = by_id(_ops(case, phase='decode', tokens=1, context=128))
    fused = nodes['s0.l3.rope']
    shapes = dict(dict(fused.metadata)['qkv_output_shapes'])
    assert set(shapes) == outputs(produced_values(fused)) == {'query', 'key', 'value', 'gate'}
    byte_count = sum(math.prod(shape) * 2 for shape in shapes.values())
    assert (fused.mte3_bytes, fused.mte3_bytes) == pytest.approx((byte_count, 3072))
    assert fused.local_shape == shapes['query']
    for node, expected in (('cache', {'key', 'value'}), ('output_gate', {'gate'})):
        assert expected <= outputs(input_ports(nodes['s0.l3.' + node]))
    attention = nodes['s0.l3.attention']
    def reads_query(op):
        return 'query' in outputs(entry for entry in input_ports(op) if entry.get('op') == fused.op_id)
    assert fused.op_id in attention.deps and reads_query(attention)
    consumers = consumer_index(tuple(nodes.values()))[1][fused.op_id]
    assert any(consumer.op_id == attention.op_id and reads_query(consumer) for _, consumer in consumers)
def test_kda_l2norm_is_retained_for_a_second_sequence_or_phase():
    case = _with_passes(_case('kimi-k3', layers=1, tp=1, ep=1, num_devices=1), 'kda_in_kernel_l2norm')
    for sequences, absorbed in ((1, True), (2, False)):
        ops = _ops(case, phase='prefill', tokens=64, context=0, sequences=sequences)
        assert any(op.recipe == 'qk_l2norm' for op in ops) is not absorbed
        assert any(dict(op.metadata).get('in_kernel_qk_l2norm') for op in ops) is absorbed
    views = (RequestView('p', 64, 0, True, 'prefill'), RequestView('d', 1, 128, True, 'decode'))
    resolved_rt, _ = resolve(case.runtime, views, padded_tokens=65, replay_scope='none')
    res = linear(prefix='s0.l0', plan=case.model.layer_plans[0], views=views, stage=0, deps='in', model=case.model, parallel=resolved_rt.parallel, quant=case.model.quant, mode='prefill', policy=AttentionPolicy(recurrent_chunk=64), runtime=resolved_rt, padded_tokens=65)
    mixed = lower_implementations(res.ops, resolved_rt, case.model.activation_dtype)
    normalizer, = [op for op in mixed if op.recipe == 'qk_l2norm']
    assert normalizer.op_id in next(op.deps for op in mixed if op.op_id.endswith('recurrent_scan1'))

def _kimi_situ_case():
    raw = example_case('kimi-k3-a5')
    raw['model']['layers'] = 1
    raw['runtime']['lowering_passes'] = ['activation_quantization', 'norm_quant', 'activation_quant']
    raw['runtime'].setdefault('kernels', {}).setdefault('lowerings', {})['rmsnorm_quant'] = 'vllm_ascend_native_norm'
    case = parse_case(raw)
    quant = replace(case.model.quant, weight_format='fp8_e4m3', expert_weight_format='fp8_e4m3', weight_block=(32, 32), activation_format='fp8_e4m3', activation_group_size=32, activation_scale_format='fp8_e8m0', activation_scale_pack=2, exclusions=())
    return bind_native(replace(case, model=replace(case.model, quant=quant)), ['situ_quant'])
@pytest.mark.parametrize("role,recipe,context", [
    ('situ_quant', None, 16), ('grouped_gate_up_activation_quant', 'gmm_swiglu_quant', 128),
])
def test_generic_quantization_preserves_selected_native_groups(role, recipe, context):
    if recipe is None:
        ops = _ops(_kimi_situ_case(), phase='decode', tokens=1, context=context, resource_fusion=False)
        sizes = [len(members) for declaration, members in _groups(ops) if declaration['role'] == role]
        assert sizes and set(sizes) == {2}
    else:
        profile = {role: 'grouped_swiglu_quant_int8_split_workspace_ae923905'}
        case = _fixture_case('v4f-a3-w8a8-tp8', [role], profile)
        passes = tuple(dict.fromkeys((*case.runtime.lowering_passes, recipe)))
        ops = _ops(replace(case, runtime=replace(case.runtime, lowering_passes=passes)),
                   phase='decode', tokens=1, context=context)
        assert any(dict(op.metadata).get('fusion_role') == role for op in ops)
def test_w4a16_dequant_work_follows_active_packed_weight_bytes():
    case = _case('kimi-k2-thinking', tp=1, dp=16, ep=16, num_devices=16)
    expert = next(op for op in _ops(case, phase='decode', tokens=1, context=128) if op.layer == 1 and op.op_id.endswith('expert_up'))
    metadata = dict(expert.metadata)
    expected = float(metadata['dequant_weight_elements']) * float(metadata['dequant_ops_per_weight'])
    assert (expert.vector_ops, expert.vector_ops) == pytest.approx((expected, 44_040_192))
def by_id(ops):
    return {op.op_id: op for op in ops}
def outputs(ports_or_values):
    return {entry.get('output') for entry in ports_or_values}
def test_quantized_ports_attach_only_to_the_matrix_activation_operand():
    ops = by_id(_ops(_case('deepseek-v3.2', ep=1, quant='w8a8'), phase='decode', tokens=1, context=128))
    assert outputs(input_ports(ops['s0.l1.q_a.quantize'])) == {'residual'}
    assert {'payload', 'activation_scales'} <= outputs(input_ports(ops['s0.l1.q_a']))
def test_moe_routing_accounts_for_the_inverse_permutation():
    case = _case('kimi-k3', tp=1, dp=2, ep=2, num_devices=4, quant='w8a8')
    ops = by_id(_ops(case, phase='prefill', tokens=64, context=0))
    pack, unpack = ops['s0.l1.pack'], ops['s0.l1.unpack']
    inverse_bytes = dict(pack.metadata)['inverse_index_bytes']
    assert (inverse_bytes, inverse_bytes) == pytest.approx((math.prod(pack.local_shape[:2]) * 4, 4096))
    assert 'inverse_indices' in pack.returns and pack.op_id in unpack.deps
    assert pack.mte3_bytes == pytest.approx(3_674_112 + 4096)
def pool_case(quant=None, tail_capacity=9, **kwargs):
    raw = compose_case('glm-5.3-flash', hardware='a3', tp=8, ep=8, num_devices=8, layers=4, quant=quant, **kwargs)
    raw['runtime'].setdefault('kernels', {}).setdefault('lowerings', {})['pool_key_indexer'] = {
        'implementation': 'vllm_ascend_cann_keypool_caller',
        'options': {'tail_capacity': tail_capacity, 'block_table_columns': 8195}}
    return parse_case(raw)
pool_prefill = partial(_ops, phase='prefill', tokens=8, context=0)
def _native_pool_case(scale):
    return pool_case({'name': 'C8', 'kv_dtype': 'fp8_e4m3', 'index_dtype': 'fp8_e4m3', 'index_scale_dtype': scale})
def test_native_keypool_binds_bf16_pool_projections_and_requires_fp32_scales():
    case = _native_pool_case('fp32')
    overrides = dict(case.model.quant.category_overrides)
    assert {dict(overrides[key])['weight_format'] for key in POOL_PROJECTION_CATEGORIES} == {'bf16'}
    pooled = find(pool_prefill(case), layer=3, recipe='cann_key_pool')
    rows, width, hidden = pooled.local_shape
    meta = dict(pooled.metadata)
    assert meta['projection_formats']['gate'] == 'bf16'
    assert meta['cann_key_pool_work']['weight_bytes'] == 2 * width * hidden * 2
    assert dict(pooled.matrix_flops)['bf16'] == 4 * rows * width * hidden
    fp16 = _native_pool_case('fp16')
    gate = find(pool_prefill(fp16), layer=3, recipe='index_pool_gate')
    assert dict(gate.metadata)['weight_format'] == 'bf16'
    assert not supported(views=(), plan=case.model.layer_plans[0], model=fp16.model, quant=fp16.model.quant)
def test_native_keypool_preserves_an_explicit_caller_category_override():
    case = _native_pool_case('fp32')
    override = ('indexer.pool_gate', (('weight_format', 'fp8_e4m3'), ('activation_format', 'fp8_e4m3')))
    model = replace(case.model, quant=replace(case.model.quant, category_overrides=(override,)))
    binding = select(case.runtime.kernels.get('lowerings', {}), 'pool_key_indexer')
    overrides = dict(prepare_model(model, case.runtime, binding).quant.category_overrides)
    assert dict(overrides['indexer.pool_gate'])['weight_format'] == 'fp8_e4m3'
def test_shared_prefix_keypool_plan_matches_the_published_op_bytes():
    case = pool_case(tail_capacity=25, concurrency=4, prompt_tokens=131_201, prefix_tokens=131_072,
                     prefix_cache_mode='shared', prefix_read_mode='batch_shared', prefix_cache_key='shared_doc')
    ops = _ops(case, phase='decode', tokens=20, context=131_201, sequences=4)
    scored = find(ops, recipe='cann_pool_key_indexer')
    metadata = dict(scored.metadata)
    ledger = dict(metadata['pool_key_indexer_work'])
    assert scored.hbm_bytes > 0 and metadata['prefix_read_reuse_applied'] is True
    assert metadata['prefix_read_hbm_bytes'] < ledger['waves'] * ledger['wave_cache_bytes']
    book = RecipeBook.from_mapping(case.runtime.kernels)
    prepared, plan = book.prepare(scored, Geometry.from_devices(case.machine.dies[:8]))
    assert (prepared.hbm_bytes, plan.work.resources['hbm']) == pytest.approx((scored.hbm_bytes,) * 2)
    serviced = operator_costs(case, (scored,))['operators'][0]['serviced']
    assert (scored.mte2_bytes, serviced['memory@r0:s0']) == pytest.approx((prepared.mte2_bytes, prepared.hbm_bytes))
    assert prepared.mte2_bytes == pytest.approx(ledger['waves'] * ledger['wave_cache_bytes'] + ledger['waves'] * ledger['wave_descale_bytes'] + ledger['query_rows'] * ledger['heads'] * ledger['width'])
def test_row_shard_option_accepts_the_generic_alias():
    runtime = type('Runtime', (), {'metadata': {'row_shard_tp': 4}, 'parallel': type('Parallel', (), {'o_proj_tp': 8, 'linear_proj_tp': 1})()})()
    assert [row_shard_tp(options=options, runtime=runtime) for options in ({'row_shard_tp': 2}, {})] == [2, 4]
def test_mtp_shared_selection_consumes_the_producer_geometry():
    case = _case('glm-5.3-flash', hardware='a5', tp=1, dp=8, o_proj_tp=8, linear_proj_tp=8, lm_head_tp=8,
                 layers=4, concurrency=1, mtp_proposal_tokens=5, graph_capture_mode='full', max_num_seqs=8)
    graph = target_graphs(case, phase='decode', tokens=6, context=4, scope='model')[0]
    attention = [op for op in graph if op.recipe.startswith('latent_attention')]
    draft0, draft4 = [next(op for op in attention if op.op_id.startswith(prefix)) for prefix in ('draft0.', 'draft4.')]
    assert find(graph, recipe='index_topk').op_id == 'draft0.s0.l0.index_topk'
    for op in (draft0, draft4):
        assert_fields(dict(op.metadata), indexed_gather_rows=5, selection_index_read_bytes=5 * 4)
    assert draft4.flops == pytest.approx(draft0.flops)
def test_packed_compressor_feeds_the_indexer_from_its_latent_port():
    case = _case('deepseek-v4.1-flash', hardware='a5', layers=3)
    ops = by_id(_ops(case, phase='decode', tokens=2, context=2048))
    assert outputs(produced_values(ops['s0.l2.compress_compress'])) == {'output', 'latent'}
    assert 'latent' in outputs(input_ports(ops['s0.l2.index_k.quantize']))
    assert 'latent' not in outputs(input_ports(ops['s0.l2.compressed_store']))
@pytest.mark.parametrize('ranges,completed', [(((0, 1), (3, 4)), 1), (((0, 2),), 0)])
def test_pcp_group_completions_follow_owned_query_ranges(ranges, completed):
    assert owned_completed_rows(SimpleNamespace(context=0, q_len=2, query_ranges=ranges), 4) == completed
PCP_CASES = (
    ('deepseek-v4-flash-0731', 3, 4, {}, 'compressed_store', 'pcp_compressed_kv_gather', 1024),
    ('glm-5.3-flash', 4, 8, {'quant': {'name': 'C8', 'kv_dtype': 'fp8_e4m3'}}, 'kv_store', 'pcp_kv_gather', 2080),
)
@pytest.mark.parametrize('model,layers,tokens,options,store,gather,expected', PCP_CASES)
def test_pcp_store_and_gather_account_for_completed_rows(model, layers, tokens, options, store, gather, expected):
    case = _case(model, hardware='a5', tp=1, dp=2, pcp=2, ep=2, num_devices=4, layers=layers, **options)
    ops = by_id(_ops(case, phase='prefill', tokens=tokens, context=0))
    prefix = f's0.l{layers - 1}.'
    writes, wire = ops[prefix + store].mte3_bytes, ops[prefix + gather].link_bytes
    assert writes == pytest.approx(expected)
    if store == 'compressed_store':
        assert ops[prefix + 'index_store'].mte3_bytes == pytest.approx(130) and wire > 0
    else:
        assert wire == pytest.approx(writes)
@pytest.mark.parametrize('override', [{'do_rope': False}, {'parallel_cp': 2}])
def test_membase_prolog_requires_rope_and_unsharded_cache(override):
    view = dict(token_dtype='int8', projection_weight_dtypes=('int8',) * 3, absorbed_weight_dtype='bf16',
                norm_dtype='bf16', cache_dtype='bf16', query_norm_dtype='int8', query_norm_output=False,
                hidden=4096, query_rank=1536, latent_rank=512, rope_width=64, nope_width=128,
                heads=8, tokens=1, do_rope=True, parallel_cp=1)
    assert validate_declared_view(view)['rope'] == 64
    with pytest.raises(ModelingError, match='requires_rope_and_unsharded'):
        validate_declared_view(view | override)

def test_shared_prefix_union_merges_absolute_disjoint_swa_intervals():
    accesses = tuple(CacheAccess(name, end, q_len=1, prefix_tokens=1023, prefix_key='doc', row_bytes=2048,
                                 logical_rows=1024) for name, end in (('a', 1024), ('b', 3072)))
    plan = plan_prefix_reads(accesses, mode='batch_shared', kernel_reuse='tile')
    assert (plan.unique_hbm_bytes, plan.l2_to_core_bytes) == pytest.approx((4 * 1024**2, 2 * 1024 * 2048))
    case = _case('gemma-4-31b', ep=1, layers=1, prefix_cache_mode='shared', prefix_read_mode='batch_shared')
    views = tuple((RequestView(f'r{i}', 256, 2048, True, 'prefill', prefix_tokens=2048, prefix_cache_key='doc') for i in range(2)))
    res = gqa(prefix='s0.l0', plan=case.model.layer_plans[0], views=views, stage=0, deps='in', model=case.model, parallel=case.runtime.parallel, quant=case.model.quant, mode='prefill', policy=AttentionPolicy(), runtime=case.runtime, hierarchy=_memory_hierarchy(case.machine), padded_tokens=512)
    meta = dict(find(res.ops, family='attention').metadata)
    assert (meta['prefix_read_hbm_bytes'], meta['prefix_read_logical_bytes']) == pytest.approx((3_143_680, 9_428_992))
@pytest.mark.parametrize('noncausal,rows', [(False, 255), (True, 128)])
def test_window_tile_span_covers_every_query_in_the_tile(noncausal, rows):
    plan = SimpleNamespace(window=128, noncausal_attention=noncausal)
    assert window_tile_rows(plan, context=128, query=128, start=0, query_tile=128, kv_len=256) == rows
def test_windowed_mla_charges_the_full_tile_window_span():
    case = _case('deepseek-v4.1-flash', hardware='a5', layers=1)
    attention = find(_ops(case, phase='prefill', tokens=128, context=128), layer=0, family='attention')
    traffic = dict(attention.metadata)['attention_traffic_plan']
    window_row = cache_format(case.model, 'window').row_bytes(case.model.layer_plans[0].cache_row_elements)
    assert traffic['logical_read_bytes'] == pytest.approx(255 * window_row) and traffic['logical_read_bytes'] > 128 * window_row

def test_lightning_indexer_charges_query_and_head_weight_operands():
    model = load_model('deepseek-v3.2')
    model = replace(model, quant=replace(model.quant, index_dtype='fp8_e4m3', index_scale_dtype='fp32'))
    plan = next((p for p in model.layer_plans if p.indexer))
    views = tuple(RequestView(name, 2, 128, False, 'prefill') for name in ('0', '1'))
    parallel = ParallelSpec()
    row = index_row_bytes(model, plan.index_dim)
    hbm_rows, refill_rows = unique_index_key_rows(views, share_prefix=False, noncausal=False, cp=1, interleave=128, compression=max(1, plan.compress_ratio))
    def build(raw_eager, kernels=None, arch='A5'):
        runtime = type('Runtime', (), {'kernels': dict(kernels or {}), 'metadata': {'arch_family': arch, 'raw_eager': raw_eager}, 'prefix_cache_mode': 'private', 'prefix_read_mode': 'auto', 'parallel': parallel})()
        builder = Builder('s0', plan, 0, views, model, 'prefill')
        query = builder.work('q_operand', (), (4, 128), read=8, write=1024)
        weights = builder.work('w_operand', (), (128, 128), read=8, write=512)
        semantic_index_selection(builder, deps=(query, weights), views=views, plan=plan, model=model, quant=model.quant, parallel=parallel, shard=1, prefix='s0', padded_tokens=4, runtime=runtime)
        return builder, sum(op.mte3_bytes for op in builder.ops if op.op_id in (query, weights))
    for raw_eager, recipe in ((False, 'quant_lightning_indexer'), (True, 'index_scores')):
        builder, operand_bytes = build(raw_eager)
        op = next(op for op in builder.ops if op.recipe == recipe)
        assert op.mte2_bytes == pytest.approx(operand_bytes + refill_rows * row)
        assert op.hbm_bytes == pytest.approx(operand_bytes + hbm_rows * row + op.mte3_bytes)
    claimed_kernels = {'lowerings': {'native_fusion': {'implementation': 'declared_native_fusion', 'options': {'roles': ['lightning_indexer']}}}}
    for arch in INDEXER_ARCHES:
        claimed, _ = build(False, kernels=claimed_kernels, arch=arch)
        recipes = {op.recipe for op in claimed.ops}
        assert {'index_scores', 'index_topk'} <= recipes and 'quant_lightning_indexer' not in recipes
def test_sparse_gqa_topk_writes_int32_ids_and_attention_reads_them():
    case = _case('minimax-m3', hardware='a5', layers=4)
    nodes = by_id(_ops(case, phase='decode', tokens=1, context=8192))
    topk, attention = nodes['s0.l3.index_topk'], nodes['s0.l3.attention']
    plan = case.model.layer_plans[3]
    heads, kv_heads = (max(1, n // case.runtime.parallel.tp) for n in (plan.num_heads, plan.num_kv_heads))
    valid, selected = topk.local_shape
    width = plan.head_dim
    assert (topk.mte2_bytes, topk.mte3_bytes) == pytest.approx((8193 * 4, valid * selected * 4))
    kv_sweeps = dict(attention.metadata)['kv_sweeps']
    assert attention.mte2_bytes == pytest.approx(
        kv_sweeps * kv_heads * width * 2 * 2 + valid * heads * width * 2 + valid * selected * 4)
