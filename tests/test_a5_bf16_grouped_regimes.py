"""Two-bank grouped services preserve valid work and reject ambiguous low rows."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
from mindie_sim import graph_costs,target_graphs

ROOT=Path(__file__).resolve().parents[1]


def inputs(batch):
    raw=json.loads((ROOT/f'examples/cases/high-concurrency-20261030/kimi-source-complete/a5_kimi_tp8_e16_bf16_c{batch}_p4k_source_complete.json').read_text())
    raw['model']['layers']=2
    raw['runtime']['kernels'].update(json.loads((ROOT/'examples/service-regimes/a5-bf16-grouped-two-bank-20261030.json').read_text()))
    ops=target_graphs(raw,scope='body',phase='decode',tokens=1,context=4104,sequences=batch,apply_body_defaults=True)[0]
    return raw,tuple(replace(op,deps=()) for op in ops if op.family=='grouped_gemm')


@pytest.mark.parametrize('batch',[16,32])
def test_grouped_call_service_retains_two_real_weight_banks_and_valid_rows(batch):
    raw,ops=inputs(batch)
    control=deepcopy(raw);control['runtime']['kernels'].pop('service_regimes')
    for op,expected in zip(ops,(.047053,.02211)):
        result=graph_costs(raw,(op,),input_ready_ms=7)
        assert result['duration_ms']==pytest.approx(expected)
        analytic=graph_costs(control,(op,),input_ready_ms=7)
        assert result['execution']['work']==analytic['execution']['work']
        assert result['execution']['work']['hbm_bytes']>=dict(op.metadata)['packed_weight_bytes']
        assert dict(op.metadata)['matrix_group_rows']==(batch//2,)*2
        assert result['execution']['live_tasks']==result['execution']['live_programs']==0
        assert not any(result['execution']['live_allocated'].values())
        assert any(row['queries'].get('observable_response')==1 for row in result['parameters']['service_regimes'])


def test_same_capacity_low_row_case_does_not_silently_borrow_high_row_service():
    raw,ops=inputs(1)
    control=deepcopy(raw);control['runtime']['kernels'].pop('service_regimes')
    for op in ops:
        actual=graph_costs(raw,(op,));analytic=graph_costs(control,(op,))
        assert actual['duration_ms']==analytic['duration_ms']
        assert actual['execution']['work']==analytic['execution']['work']
        assert any(row['queries'].get('analytic_outside_domain')==1 for row in actual['parameters']['service_regimes'])
