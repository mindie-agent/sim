"""The full SP caller owns each calibrated HCCL group once, including final AG."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest
from mindie_sim import predict, target_graphs

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/'examples/cases/high-concurrency-20261030/kimi-hccl-source-conditional'


def declared(label='p4k',layers=1):
    raw=json.loads((CASES/f'a5_kimi_tp8_e16_bf16_c32_{label}_hccl_observed.json').read_text())
    raw['model']['layers']=layers
    return raw


def probe(label='p4k'):
    return dict(scope='body',phase='decode',tokens=1,context=4104 if label=='p4k' else 16392,sequences=32)


def groups(raw):
    ops=target_graphs(raw,**probe(),apply_body_defaults=True)[0]
    entries={};exits={}
    for op in ops:
        meta=dict(op.metadata)
        if 'hccl_source_group' not in meta:continue
        key=meta['hccl_source_group'];info=meta['device_kernel_group']
        if info.get('entry'):entries[key]=(op,meta['hccl_selection'])
        if info.get('exit'):exits[key]=op
    assert entries.keys()==exits.keys()
    return ops,entries,exits


def spans(raw,result):
    _,entries,exits=groups(raw)
    timeline={row['op_id']:row for row in result['timeline']}
    return [(selection['collective'],
             (timeline[exits[key].op_id]['complete_ms']-timeline[entry.op_id]['ready_ms'])*1000)
            for key,(entry,selection) in entries.items()]


@pytest.mark.parametrize('label',['p4k','p16k'])
def test_complete_93_layer_sp_has_373_single_native_collectives(label):
    raw=declared(label,93)
    ops,entries,exits=groups(raw)
    assert Counter(s['collective'] for _,s in entries.values())=={'allgather':187,'reduce_scatter':186}
    assert not any(op.family in ('allgather','reduce_scatter') for op in ops)
    assert 'sp_output_gather.hccl' in entries
    by_id={op.op_id:op for op in ops}
    for key,(entry,s) in entries.items():
        assert s['payload_bytes']=={'allgather':4*7168*2,'reduce_scatter':32*7168*2}[s['collective']]
        assert s['ranks']==8 and entry.dtype=='bf16'
        pending=list(exits[key].deps);seen=set()
        while pending:
            current=pending.pop()
            if current in seen:continue
            seen.add(current)
            if current in by_id:pending.extend(by_id[current].deps)
        assert entry.op_id in seen


def test_call_service_applies_once_without_replacing_physical_work():
    calibrated=declared()
    source=deepcopy(calibrated);source['runtime']['kernels'].pop('service_regimes')
    before=predict(source,**probe());after=predict(calibrated,**probe())
    fixed={'allgather':10.7015,'reduce_scatter':15.8495}
    for family,elapsed in spans(calibrated,after):
        assert elapsed==pytest.approx(fixed[family],abs=1e-7)
    assert after['body_forward_ms']>before['body_forward_ms']
    assert after['execution']['work']==before['execution']['work']
    assert after['execution']['live_tasks']==after['execution']['live_programs']==0
    assert not any(after['execution']['live_allocated'].values())


def test_memory_sensitivity_remains_on_full_source_caller():
    reports=[]
    for factor in (.5,1,2):
        raw=declared();raw['machine']['device']={'hbm_bw_bytes_per_s':4e12*factor}
        reports.append(predict(raw,**probe()))
    durations=[r['body_forward_ms'] for r in reports]
    assert durations[0]>durations[1]>durations[2]
    assert reports[0]['execution']['work']==reports[1]['execution']['work']==reports[2]['execution']['work']
    # The observed call service is inclusive. Severe physical slowdown wins
    # over it instead of being replaced by an immutable timing lookup.
    slow=declared();slow['machine']['device']={'hbm_bw_bytes_per_s':1e8}
    result=predict(slow,**probe())
    assert all(elapsed>{'allgather':10.7015,'reduce_scatter':15.8495}[family]
               for family,elapsed in spans(slow,result))
