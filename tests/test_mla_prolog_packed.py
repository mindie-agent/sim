"""Native INT8 prolog boundaries retain mixed cache bytes and optional query scales."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.prediction.mla_prolog_packed import LAYOUT, PROFILE
from mindie_sim.prediction.native_fusion_traffic import apply_traffic_profile
from mindie_sim.prediction.mla_prolog_arch22 import PROFILE as BF16_PROFILE
from mindie_sim.structure_subgraphs import subgraph_contract

ROOT = Path(__file__).resolve().parents[1]


def case(layers=78):
    path = ROOT / 'examples/cases/high-concurrency-20261030/glm78/a3_glm53_w4a8_tp16_tpot88_c8.json'
    raw = json.loads(path.read_text()); raw['model']['layers'] = layers
    raw['runtime']['kernels']['lowerings'] = {
        'mla_prolog': {'implementation': 'vllm_ascend_mla_prolog_v3',
                       'options': {'cache_layout': LAYOUT, 'cache_group_size': 128}},
        'native_fusion': {'implementation': 'declared_native_fusion',
                          'options': {'roles': ['mla_prolog'], 'traffic_profile': PROFILE}}}
    return raw


def groups(raw, tokens=6, sequences=8):
    ops = target_graphs(raw, scope='body', phase='decode', tokens=tokens,
                        sequences=sequences, context=4104, apply_body_defaults=False)[0]
    return tuple(op for op in ops if dict(op.metadata).get('fusion_role') == 'mla_prolog')


def test_full_model_native_group_has_57_ordinary_and_21_query_norm_outputs():
    selected = groups(case())
    assert len(selected) == 78
    assert Counter(dict(op.metadata)['fusion_tensor_view']['query_norm_output'] for op in selected) == {False:57, True:21}
    for op in selected:
        meta = dict(op.metadata); view = meta['fusion_tensor_view']
        assert view['tokens'] == 48
        assert view['query_norm_scale_shape'] == ((48,1) if view['query_norm_output'] else ())
        assert view['query_norm_scale_dtype'] == 'fp32'
        assert view['token_scale_shape'] == (48,1)
        assert view['token_dtype'] == 'int8'
        assert view['projection_weight_dtypes'] == ('int8',)*3
        assert view['absorbed_weight_dtype'] == 'bf16'
        assert view['packed_cache']['component_offsets'] == {'latent':0,'rope':512,'scales':640}
        assert view['packed_cache']['row_bytes'] == 512 + 64*2 + 4*4
        phases = meta['fusion_phases']
        store = next(p for p in phases if p['recipe'] == 'kv_store')
        assert store['mte3_bytes'] == 48*656
        assert store['mte2_bytes'] == 48*(576*2+8)+4
        assert store['vector_ops'] > 48*512
        assert meta['fusion_boundary']['submitted_kernels'] == 1
        assert meta['fusion_boundary']['traffic']['profile'] == PROFILE
        assert meta['fusion_boundary']['traffic']['eliminated_writes'] == 0
        canonical = subgraph_contract({'role':'mla_prolog','tensor_view':view})
        assert canonical[4] == (((1,656),'int8','ND'), ((),None,None))
        assert canonical[5][:2] == (((48,4,512),'bf16','ND'),((48,4,64),'bf16','ND'))
        assert canonical[5][2] == (((48,2048),'int8','ND') if view['query_norm_output'] else ((),None,None))


def test_selected_packed_cache_cannot_reuse_bf16_arch22_program():
    op = groups(case(1))[0]; view = dict(op.metadata)['fusion_tensor_view']
    with pytest.raises(ValueError, match='bf16_uk_norm_cache'):
        apply_traffic_profile((), {'role':'mla_prolog','tensor_view':view}, BF16_PROFILE)


def test_whole_native_observed_service_is_once_and_preserves_resource_cost():
    raw = case(4); selected = groups(raw); reference = operator_costs(raw, selected, input_ready_ms=7)
    calibrated = deepcopy(raw)
    calibrated['runtime']['kernels']['service_regimes'] = [{
        'id':'explicit-native-boundary-control',
        'match':{'metadata.fusion_role':'mla_prolog',
                 'metadata.fusion_tensor_view.cache_dtype':'int8'},
        'response':{'fixed_us':1000,'terms':[]},
        'provenance':{'basis':'synthetic boundary test; not hardware calibration'}}]
    changed = operator_costs(calibrated, selected, input_ready_ms=7)
    for before, after in zip(reference['operators'], changed['operators']):
        assert after['duration_ms'] == pytest.approx(1)
        assert before['serviced'] == pytest.approx(after['serviced'])
        assert not after['service_before_input']
        assert after['live_tasks'] == 0
    # One complete kernel dependency is preserved across late external input.
    external = {d:7 for d in selected[0].deps}
    graph = graph_costs(raw, selected[:1], external_inputs=external, input_ready_ms=7)
    assert graph['execution']['live_tasks'] == graph['execution']['live_programs'] == 0
    assert min(row['ready_ms'] for row in graph['timeline']) >= 7


def test_packed_cache_work_scales_with_valid_rows_and_resource_rate():
    raw = case(1); selected = groups(raw,tokens=1,sequences=1)
    from mindie_sim.prediction.structure import parse_case
    parsed = parse_case(raw); times=[]; work=[]
    for scale in (.5,1,2):
        devices = tuple(replace(d,rates=tuple({**dict(d.rates),'hbm':1e9*scale}.items())) for d in parsed.machine.dies)
        modified = replace(parsed,machine=replace(parsed.machine,dies=devices))
        result=operator_costs(modified,selected)['operators'][0]
        times.append(result['duration_ms']);work.append(result['serviced'])
    assert times[0] > times[1] > times[2]
    assert work[0] == pytest.approx(work[1])
    assert work[1] == pytest.approx(work[2])


def test_native_cache_capacity_retains_rope_and_scale_bytes_per_valid_row():
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.catalog.cache_format import cache_format
    from mindie_sim.catalog.cache import layer_cache_components
    parsed = parse_case(case(1))
    format = cache_format(parsed.model, 'mla_kv')
    assert format.row_bytes(576) == 656
    plan = parsed.model.layer_plans[0]
    # The cache's persistent footprint grows with context, independently of
    # the prolog's twelve freshly written rows in this decode invocation.
    # Invoke the public geometry API with its ordinary model/plan/parallel axes.
    components = layer_cache_components(parsed.model, plan, parsed.runtime.parallel, 1024)
    assert components['mla_kv'] == 1024 * 656
