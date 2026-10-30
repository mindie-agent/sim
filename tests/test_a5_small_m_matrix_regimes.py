"""Small-M native-call service binds physical matrix contracts, not model roles."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
from mindie_sim import graph_costs,target_graphs

ROOT=Path(__file__).resolve().parents[1]


def inputs(batch):
    raw=json.loads((ROOT/'examples/cases/high-concurrency-20261030/kimi-source-complete/a5_kimi_tp8_e16_bf16_c32_p4k_source_complete.json').read_text())
    raw['model']['layers']=2
    raw['runtime']['kernels'].update(json.loads((ROOT/'examples/service-regimes/a5-small-m-bf16-matmul-20261030.json').read_text()))
    ops=target_graphs(raw,scope='body',phase='decode',tokens=1,context=4104,sequences=batch,apply_body_defaults=True)[0]
    selected=next(o for o in ops if o.op_id=='s0.l1.latent_down')
    return raw,replace(selected,op_id='independent_matrix',deps=(),layer=-1)


def test_observable_small_m_matmul_is_native_once_and_retains_hbm():
    raw,op=inputs(32)
    result=graph_costs(raw,(op,),input_ready_ms=7)
    assert result['duration_ms']==pytest.approx(.027136)
    assert result['complete_ms']==pytest.approx(7.027136)
    assert result['execution']['work']['hbm_bytes']==op.hbm_bytes
    selected=next(x for x in result['parameters']['service_regimes'] if x['queries'].get('observable_response'))
    assert selected['queries']['observable_response']==1
    assert result['execution']['live_tasks']==result['execution']['live_programs']==0


def test_unobserved_m_keeps_analytic_source_and_slow_memory_can_exceed_response():
    raw,op=inputs(40) # local M=5, outside calibrated M1..4
    result=graph_costs(raw,(op,))
    control=deepcopy(raw);control['runtime']['kernels'].pop('service_regimes')
    assert result['duration_ms']==graph_costs(control,(op,))['duration_ms']
    assert any(row['queries'].get('analytic_outside_domain') for row in result['parameters']['service_regimes'])
    raw,op=inputs(32);raw['machine']['device']={'hbm_bw_bytes_per_s':1e9}
    assert graph_costs(raw,(op,))['duration_ms']>.027136


def test_effective_small_matrix_weight_service_tracks_hbm_without_changing_work():
    raw,op=inputs(32)
    spans, work = [], []
    for factor in (.5, 1, 2):
        case=deepcopy(raw)
        case['machine']['device']={'hbm_bw_bytes_per_s':4e12*factor}
        result=graph_costs(case,(op,),input_ready_ms=7)
        spans.append(result['duration_ms'])
        work.append(result['execution']['serviced'])
        assert result['complete_ms']==pytest.approx(7+result['duration_ms'])
    assert spans[0]>spans[1]>spans[2]
    assert spans[0]==pytest.approx(.054272)
    assert work[0]==pytest.approx(work[1],rel=1e-12,abs=1e-12)
    assert work[0]==pytest.approx(work[2],rel=1e-12,abs=1e-12)
