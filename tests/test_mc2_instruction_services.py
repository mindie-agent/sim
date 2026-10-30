"""Independent source-call arithmetic and physical MC2 issue-owner controls."""
from collections import defaultdict

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mc2_phase import build
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.operators import OpSpec


def execute(phase, *, kind='dispatch', tokens=4, top_k=8, units=8, experts=16,
            roles=('scalar_ub_access','scalar_alu_issue','pipe_barrier'),
            issue_units=None, supply=1, external=None):
    raw=dict(kind=kind,width=240,dtype='bf16',source_tokens=tokens,top_k=top_k,
             source_assignments=tokens*top_k,received_assignments=tokens*top_k,
             experts=experts,ep=4,phase=phase)
    op=OpSpec('p','mc2_protocol','decode',0,0,(),(),(),(),(),metadata=(('mc2_phase',raw),))
    rates={'hbm':1e12,'mte2_aiv':1e12,'mte3':1e12,'vector':1e12}
    physical={r:1 if r=='hbm' else units for r in rates}
    for role in roles:
        physical[role]=issue_units or units;rates[role]=supply*physical[role]
    plan=build(op,Geometry(units,physical))
    context=RecipeContext(op,{}, {r:r for r in rates},rates,units,resource_units=physical)
    engine=Engine([Resource(r,rate,units=physical[r]) for r,rate in rates.items()],trace_limit=10000)
    ready=engine.submit('input',release=7)
    if external:
        engine.submit('external',work={'scalar_ub_access':external['calls']},
                      placement={'scalar_ub_access':(external['unit'],)},release=7)
    done=plan.lower(context).start(engine,'p',ready)
    engine.advance(horizon=6);assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    if external:engine.advance()
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    expected=dict(plan.work.resources)
    if external:expected['scalar_ub_access']=expected.get('scalar_ub_access',0)+external['calls']
    assert engine.serviced==pytest.approx({r:expected.get(r,0) for r in rates})
    return done.time-7,engine,plan


@pytest.mark.parametrize('tokens,topk,units',[(1,2,4),(4,8,8),(17,3,16),(0,8,64)])
def test_sender_scalar_calls_follow_actual_strided_assignment_owner(tokens,topk,units):
    _,engine,plan=execute('send_read_quantize',tokens=tokens,top_k=topk,units=units)
    payload=units-min(max(16//16,1),units//2,8,16)
    expected=defaultdict(lambda:defaultdict(int))
    for index in range(tokens*topk):
        expected[index%payload]['scalar_ub_access']+=1
        if index>=topk:
            expected[index%payload]['scalar_ub_access']+=1
            expected[index%payload]['pipe_barrier']+=4
    for owner,inserts in plan.metadata['primitive_services']['owners'].items():
        actual=defaultdict(int)
        for insert in inserts.values():
            for role,n in insert['counts'].items():actual[role]+=n
        assert dict(actual)==dict(expected[owner])
    assert engine.serviced['scalar_ub_access']==pytest.approx(tokens*topk+max(0,tokens*topk-topk))
    assert engine.serviced['pipe_barrier']==pytest.approx(4*max(0,tokens*topk-topk))


@pytest.mark.parametrize('experts,units',[(16,8),(100,16),(256,64)])
def test_count_and_prefix_scalar_tail_work_includes_zero_experts(experts,units):
    # No source tokens: count status stores still cover every expert, but the
    # reduction-result reads and their four interior barriers do not execute.
    _,engine,plan=execute('count_compute',tokens=0,experts=experts,units=units)
    c=len(plan.metadata['active_owner_domain'])
    assert engine.serviced['scalar_ub_access']==experts
    assert engine.serviced['pipe_barrier']==c
    _,engine,plan=execute('prefix_counts',tokens=0,experts=experts,units=units)
    assert engine.serviced['scalar_alu_issue']==experts
    assert engine.serviced['scalar_ub_access']==pytest.approx(2*experts+4*c+2*(c-1))
    assert engine.serviced['pipe_barrier']==2*c-1


def test_successful_mask_and_weight_reads_do_not_consume_idle_aiv_capacity():
    # width240 -> one 512-byte frame; top8 -> one successful 64-bit mask.
    spans=[]
    for supply in (.5,1,2):
        elapsed,engine,plan=execute('receive_flags',kind='combine',tokens=1,top_k=8,
                                    units=64,supply=supply)
        assert elapsed==pytest.approx(2/supply,abs=1e-6)
        assert engine.serviced['scalar_ub_access']==1
        assert engine.serviced['scalar_alu_issue']==1
        assert engine.serviced['pipe_barrier']==1
        assert plan.metadata['active_owner_domain']==(0,)
        spans.append(elapsed)
    assert spans[0]>spans[1]>spans[2]
    _,engine,_=execute('weighted_reduce',kind='combine',tokens=5,top_k=3,units=4,
                       roles=('scalar_ub_access',))
    assert engine.serviced['scalar_ub_access']==15
    for owner in range(4):
        rows=[t for t in engine.trace if 'primitive' in t['task'] and f'owner{owner}:' in t['task']]
        assert len(rows)==(2 if owner==0 else 1)*3
        for row in rows:
            token_prefix=row['task'].split(':slot')[0]
            slot=row['task'].split(':slot')[1].split('_')[0]
            payload=next(t for t in engine.trace if t['task']==token_prefix+f':slot{slot}_read')
            assert payload['start_ms']>=row['end_ms']


def test_two_aiv_share_one_real_scalar_issue_pipe_and_contend_with_external_work():
    ordinary,_,_=execute('weighted_reduce',kind='combine',tokens=2,top_k=3,units=4,
                          issue_units=2,roles=('scalar_ub_access',))
    assert ordinary==pytest.approx(6,abs=1e-6)
    contended,engine,_=execute('weighted_reduce',kind='combine',tokens=2,top_k=3,units=4,
        issue_units=2,roles=('scalar_ub_access',),external={'calls':6,'unit':0})
    # Three concurrent owner-local clients share pipe0; each native task
    # has three one-call steps, so it retires at 9ms, before the 6-call rival.
    assert contended==pytest.approx(9,abs=1e-6)
    assert next(t for t in engine.trace if t['task']=='external')['end_ms']==pytest.approx(19,abs=1e-6)
    # The unused scalar pipe1 is not borrowed by either active AIV0/1.
    for row in engine.trace:
        if 'primitive' in row['task']:
            assert row['placement']['scalar_ub_access']==(0,)


def test_no_declared_service_keeps_physical_work_and_exposes_known_unpriced_calls():
    _,engine,plan=execute('send_read_quantize',roles=())
    assert not any('primitive' in row['task'] for row in engine.trace)
    inventory=plan.metadata['primitive_services']['owners']
    assert sum(v['unpriced_service_work'].get('scalar_ub_access',0)
               for owner in inventory.values() for v in owner.values())==56
    _,calibrated,priced=execute('send_read_quantize',roles=('scalar_ub_access',))
    for role in ('hbm','mte2_aiv','mte3','vector'):
        assert engine.serviced[role]==pytest.approx(calibrated.serviced[role])
    assert sum(v['unpriced_service_work'].get('pipe_barrier',0)
               for owner in priced.metadata['primitive_services']['owners'].values()
               for v in owner.values())==96


def test_public_device_services_reach_real_joint_groups_once_and_preserve_wire():
    from dataclasses import replace
    from mindie_sim import target_graphs,graph_costs
    from test_native_mc2_graph import case
    from test_joint_collective_execution import wire
    results=[]
    for factor in (.5,1,2):
        raw=case()
        raw['machine']['device']['compute_arch']={'instruction_services':{
            role:{'unit':'calls','owner_axis':'vector','issue_ns':1000/factor,
                  'provenance':{'kind':'synthetic_owner_capacity_control_not_hardware_calibration'}}
            for role in ('scalar_ub_access','scalar_alu_issue','pipe_barrier')}}
        graphs=target_graphs(raw,scope='body',phase='decode',tokens=1,context=128)
        selected={}
        for rank,ops in graphs.items():
            kept=[o for o in ops if dict(o.metadata).get('device_kernel_group')]
            ids={o.op_id for o in kept}
            selected[rank]=tuple(replace(o,deps=tuple(d for d in o.deps if d in ids)+
                ((f'late{rank}',) if dict(o.metadata)['device_kernel_group']['entry'] else ())) for o in kept)
        report=graph_costs(raw,selected,input_ready_ms=7,external_inputs={'late0':7,'late1':9},isolated=False)
        assert not report['execution']['live_tasks'] and not report['execution']['live_programs']
        assert sum(wire(report).values())==2176
        for row in report['timeline']:
            if row['replica']==1:assert row['ready_ms']>=9
        for role in ('scalar_ub_access','scalar_alu_issue','pipe_barrier'):
            assert sum(v for k,v in report['execution']['serviced'].items() if k.startswith(role+'@'))>0
        results.append(report)
    assert results[0]['duration_ms']>results[1]['duration_ms']>results[2]['duration_ms']
    assert results[0]['execution']['serviced']==pytest.approx(results[2]['execution']['serviced'])
