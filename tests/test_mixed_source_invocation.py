"""Physical mixed programs inherit class timing once without relabeling ABI."""
from dataclasses import replace
from functools import lru_cache

import pytest

from mindie_sim import graph_costs,operator_costs,target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.invocation import InvocationTiming


@lru_cache(None)
def control(name):
    if name=='swiglu':
        from test_mxfp8_swiglu_source import case,PROFILE
        raw=case(16)
        op=next(op for op in target_graphs(raw,scope='body',phase='decode',tokens=4,
            context=4096,sequences=16)[0] if op.recipe==PROFILE)
    elif name=='paged_mla':
        from test_paged_mla import case,attention
        raw=case();op=attention(raw,context=128)[0]
    elif name=='mqsmla':
        from test_mqsmla_source import case,attention
        raw=case();op=attention(target_graphs(raw,scope='body',phase='decode',tokens=1,
            context=128,sequences=1)[0])
    else:
        from test_indexed_quant_attention import case,ops
        raw=case();op=ops(raw,context=128)[0]
    return parse_case(raw),replace(op,deps=())


def hardware(case,**classes):
    return replace(case,machine=replace(case.machine,dies=tuple(replace(d,
        kernel_invocations=tuple((name,InvocationTiming(**timing)) for name,timing in classes.items()))
        for d in case.machine.dies)))


def cost(case,op):
    row=operator_costs(case,[op],input_ready_ms=7)['operators'][0]
    assert not row['service_before_input'] and row['live_tasks']==0
    assert not any(row['remaining_buffers'].values())
    return row


@pytest.mark.parametrize('name',('swiglu','paged_mla','mqsmla','packed_int8'))
def test_actual_mixed_source_program_pays_max_class_startup_and_floor_once(name):
    case,op=control(name)
    baseline=cost(hardware(case),op)
    # Deliberately visible physical timing controls, never production defaults.
    started=cost(hardware(case,cube={'startup_ms':.25},vector={'startup_ms':.5}),op)
    assert started['duration_ms']==pytest.approx(baseline['duration_ms']+.5)
    assert started['serviced']==pytest.approx(baseline['serviced'])
    floored=hardware(case,cube={'startup_ms':.25,'minimum_ms':2},
                          vector={'startup_ms':.5,'minimum_ms':3})
    report=graph_costs(floored,[op],input_ready_ms=7,isolated=False,trace_limit=100000)
    assert report['complete_ms']==pytest.approx(10)
    assert {k:v for k,v in report['execution']['serviced'].items() if v}==pytest.approx(baseline['serviced'])
    assert report['execution']['live_tasks']==report['execution']['live_programs']==0
    assert not any(report['execution']['live_allocated'].values())
    starts=[e for e in report['events'] if e['task'].endswith(':startup')]
    assert len(starts)==1 and starts[0]['start_ms']==7 and starts[0]['end_ms']==7.5
    assert min(e['start_ms'] for e in report['events'] if e['demand'])>=7.5
    assert len([e for e in report['events'] if e['task'].endswith(':minimum')])==1


def test_explicit_user_class_and_runtime_rule_still_override_inferred_mixed_classes():
    case,op=control('swiglu')
    explicit=replace(op,metadata=tuple({**dict(op.metadata),'invocation_class':'user_mixed'}.items()))
    case=hardware(case,cube={'minimum_ms':2},vector={'minimum_ms':3},user_mixed={'minimum_ms':4})
    assert cost(case,explicit)['duration_ms']==pytest.approx(4)
    case=replace(case,runtime=replace(case.runtime,kernels={**case.runtime.kernels,
        'invocations':[{'match':{},'startup_ms':.25,'minimum_ms':1}]}))
    assert cost(case,explicit)['duration_ms']==pytest.approx(1)


def test_nonempty_tiny_source_uses_existing_a5_inclusive_minimum_without_adding_work():
    from test_mxfp8_swiglu_source import changed
    case,op=control('swiglu')
    op=changed(op,(1,),64,128)
    physical=cost(hardware(case),op)
    ordinary=cost(case,op)
    assert physical['duration_ms']<.001578
    assert ordinary['duration_ms']==pytest.approx(.001578)
    assert ordinary['serviced']==pytest.approx(physical['serviced'])
