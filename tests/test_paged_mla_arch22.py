"""Source host oracle, GM conservation and actual finite arch22 MLA execution."""
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.paged_mla_arch22 import build, BUFFERS, IMPLEMENTATION
from mindie_sim.prediction.paged_mla_arch22_dispatch import Dispatch
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure_contracts import predicted_contract

ROOT = Path(__file__).resolve().parents[1]
ORACLE = json.loads((ROOT/'tests/fixtures/paged_mla_arch22_dispatch.json').read_text())['cases']


def declaration(heads=16, contexts=(1024,), cores=24):
    return next(row for row in ORACLE if row['heads']==heads and row['cores']==cores
                and tuple(row['contexts'])==contexts)


def op_for(raw, page=128):
    return OpSpec('mla','attention','decode',0,0,(),(),(),(),(),recipe=IMPLEMENTATION,
        source_boundary='kernel',metadata=(('paged_mla_geometry',{
            'contexts':tuple(raw['contexts']),'heads':raw['heads'],
            'page_size':page,'dispatch':raw}),))


def run(raw, *, supply=1, trace=0):
    cores=raw['cores']
    units={r:cores*(2 if r in ('vector','mte2_aiv','mte3') else 1)
           for r in ('hbm','vector','cube','mte1','mte2_aic','mte2_aiv','mte3','fixpipe')}
    units['hbm']=1
    rates={r:1e9*supply for r in units}
    rates['cube']=1
    pu={p:cores*(2 if p=='ub' else 1) for p in BUFFERS}
    plan=build(op_for(raw),Geometry(cores,units,BUFFERS,pu))
    engine=Engine([Resource(r,v,units=units[r]) for r,v in rates.items()],
                  pools=BUFFERS,pool_units=pu,trace_limit=trace)
    ctx=RecipeContext(plan.work.apply(op_for(raw)),{}, {r:r for r in rates},rates,cores,
        resource_units=units,compute_peaks={'bf16':1e12*supply},
        pool_names={p:p for p in BUFFERS},pool_capacities=BUFFERS,pool_units=pu)
    ready=engine.submit('input',release=7)
    done=plan.lower(ctx).start(engine,'mla',ready)
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    for role,amount in ctx.bind(plan.work).items():
        assert engine.serviced[role]==pytest.approx(amount,rel=1e-12)
    assert not any(engine.allocated.values())
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    return plan,engine,done.time-7


def test_retained_unmodified_host_endpoints_partition_kv_and_fd_heads():
    for raw in ORACLE:
        contexts=tuple(raw['contexts']); heads=raw['heads']
        dispatch=Dispatch.parse(raw,contexts,heads,raw['cores'])
        actual={}
        for owner,rows in enumerate(dispatch.owners):
            for seq,start,count,_ in rows:
                for tile in range(start,start+count):
                    assert (seq,tile) not in actual
                    actual[seq,tile]=owner
        assert set(actual)=={(seq,tile) for seq,c in enumerate(contexts)
                             for tile in range((c+512)//512)}
        merged={}
        for aiv,rows in enumerate(dispatch.merges):
            for seq,start,count,_ in rows:
                for head in range(start,start+count):
                    assert (seq,head) not in merged
                    merged[seq,head]=aiv
        split={seq for seq,*_ in raw['fd']}
        assert set(merged)=={(seq,head) for seq in split for head in range(heads)}
    # Independent source run uses 23 of 24 AICs here. A round-robin 24-way
    # equal split is a different dispatch despite having the same total work.
    d=Dispatch.parse(declaration(contexts=(126000,)),(126000,),16,24)
    assert len(d.owners)==23
    assert d.owners[0]==((0,0,11,23),) and d.owners[-1]==((0,236,11,23),)
    assert d.merges==(((0,0,8,23),),((0,8,8,23),))


@pytest.mark.parametrize('heads',[8,16])
def test_tail_hbm_conserves_gm_roundtrips_alias_rereads_and_atomic_updates(heads):
    plan,_,_=run(declaration(heads,(512,),1))
    # Two S2 tiles (512+1), M padded to 16 for Cube and atomic Nupdate,
    # vector score/P rows retain H and S aligned to 32 elements.
    expected=(2*heads*576*2 + 513*(576+512)*2 + 5*6*4
              + 16*528*4 + heads*544*4 + heads*544*2 + heads*513*2
              + 3*16*512*4 + 2*16*512*4 + heads*512*4 + heads*512*2)
    assert plan.work.resources['hbm']==expected
    assert plan.work.matrix['bf16']==2*16*528*(576+512)


def test_source_single_aiv_and_two_drains_keep_cube_and_vector_dependencies():
    plan,engine,_=run(declaration(8,(1024,),1),trace=10000)
    events=engine.trace
    commands={}
    for event in events:
        name=event['task']
        if ':producers:program:group0:tile' in name:
            iteration=int(name.split(':group0:tile')[1].split(':')[0])
            commands[iteration,name.rsplit(':',1)[1]]=event
    order=[commands[i,name] for i in range(4) for name in ('qk','pv')
           if (i,name) in commands and commands[i,name]['demand']]
    assert len(order)==6
    for before,after in zip(order,order[1:]):
        assert after['start_ms']>=before['end_ms']-1e-12
    assert commands[4,'output_read']['start_ms']>=commands[3,'partial']['end_ms']-1e-12
    assert len([e for e in events if e['task'].endswith(':output_write')])==1
    for event in events:
        for role in ('vector','mte2_aiv','mte3'):
            if role in event['demand']:
                assert event['placement'][role]==(0,)
    assert sum(e['serviced'].get('hbm',0) for e in events)==pytest.approx(plan.work.resources['hbm'])


def test_unequal_sequences_and_fd_share_real_resources_and_release_allocations():
    raw=declaration(16,(512,1024,1535,2060))
    base,_,slow=run(raw)
    _,_,fast=run(raw,supply=2)
    assert slow==pytest.approx(2*fast,rel=1e-10)
    dispatch=Dispatch.parse(raw,tuple(raw['contexts']),16,24)
    assert base.metadata['physical_segments']==sum(map(len,dispatch.owners))


def test_dispatch_cannot_be_reused_for_different_context_or_physical_core_geometry():
    raw=declaration()
    with pytest.raises(ModelingError,match='geometry_mismatch'):
        Dispatch.parse(raw,(1025,),16,24)
    with pytest.raises(ModelingError,match='geometry_mismatch'):
        Dispatch.parse(raw,(1024,),16,32)


@pytest.mark.parametrize('tp,heads',[(8,16),(16,8)])
def test_complete_frontend_uses_arch22_core_and_preserves_logical_output_heads(tp,heads):
    raw=json.loads((ROOT/'examples/cases/high-concurrency-20261030/kimi-source/'
                    'a3_kimi4_tp8_w4a8_c1_source.json').read_text())
    raw['machine']['num_devices']=tp
    raw['runtime'].update(tp=tp,ep=tp,max_model_len=131072)
    raw['runtime']['kernels']['lowerings']['paged_mla_attention']={
        'implementation':IMPLEMENTATION,'options':{'dispatch':declaration(heads,cores=20)}}
    graph=target_graphs(raw,scope='body',phase='decode',tokens=1,sequences=1,context=1024)[0]
    attention=next(o for o in graph if o.recipe==IMPLEMENTATION)
    assert dict(attention.metadata)['mla_head_contract']=={
        'logical_heads':96//tp,'physical_heads':heads}
    assert dict(next(o for o in graph if o.recipe=='v_expand').metadata)['matrix_groups']==96//tp
    assert predicted_contract(attention.to_dict())[0]=='attention_paged_mla_qk_rope'
    actual=operator_costs(raw,[attention],input_ready_ms=7)['operators'][0]
    assert not actual['service_before_input'] and actual['live_tasks']==0
    assert not any(actual['remaining_buffers'].values())
    parsed=parse_case(raw)
    planned=build(attention,Geometry.from_devices(parsed.machine.dies))
    assert actual['operator']['hbm_bytes']==planned.work.resources['hbm']
