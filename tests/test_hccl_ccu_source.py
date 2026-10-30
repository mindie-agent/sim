"""CCU buffer versus HBM byte equations and finite actual Engine programs."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Engine,Resource
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_ccu import build,contract,buffer_geometry,OPTIONAL
from mindie_sim.prediction.mechanism import Geometry,OperatorPlan,Work
from mindie_sim.prediction.program import RecipeContext,Stage,KernelProgram,ParallelProgram
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import ModelingError


def selected(kind='reduce',ranks=4,size=1024,version=2):
    return select({'expansion':'CCU_MS' if kind=='reduce' else 'CCU_SCHED',
                   'template':f'mesh_oneshot_v{version}' if kind=='reduce' else f'mesh_mem2mem_v{version}',
                   'dtype_bytes':2},collective='allreduce' if kind=='reduce' else 'allgather',
                  ranks=ranks,payload_bytes=size)


def total(plan,key):
    return sum(dict(p.work).get(key,0) for p in plan.phases)


def op_for(s,name):
    return OpSpec(name,'collective','decode',0,0,(),(),(),(),(),metadata=(
        ('hccl_phase',{'selection':s.to_dict(),'phase':name,'rank':0}),))


def whole(s,*,rates=None,ready=7,startup=0,minimum=0):
    rates=rates or {'hbm':1024,'ccu_copy':1024,'ccu_buffer_bytes':4096,
                    'ccu_reduce_elements':512,'ccu_control':1000,'link':1024}
    geometry=Geometry(1,{k:1 for k in rates})
    source=contract(s)
    plans={phase.name:build(op_for(s,phase.name),geometry) for phase in source.phases}
    work=sum((p.work for p in plans.values()),Work())
    wire=sum(sum(phase.exchange[0]) for phase in source.phases if phase.exchange)
    work+=Work(resources={'link':wire})
    ctx=RecipeContext(None,{}, {k:k for k in rates},rates,1,
                      resource_units=geometry.resource_units,setup_ms=startup,minimum_ms=minimum)
    def program(context):
        stages=[]
        for phase in source.phases:
            physical=plans[phase.name].program(context)
            if phase.exchange:
                physical=ParallelProgram((physical,KernelProgram((Stage('wire',{'link':sum(phase.exchange[0])}),))))
            stages.append(Stage(phase.name,after=phase.after,program=physical))
        return KernelProgram(tuple(stages))
    plan=OperatorPlan(work,program)
    engine=Engine([Resource(k,v) for k,v in rates.items()],trace_limit=3000)
    initial=engine.submit('input',release=ready)
    done=plan.lower(ctx).start(engine,'call',initial)
    engine.advance(horizon=ready-.01)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.serviced==pytest.approx({k:work.resources.get(k,0) for k in rates})
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    return done.time-ready,engine,plans


@pytest.mark.parametrize('version',(1,2))
@pytest.mark.parametrize('ranks',(2,4,8))
def test_reduce_direct_ccu_arrivals_never_become_hbm_scratch_writes(version,ranks):
    n=384
    source=contract(selected(ranks=ranks,size=n,version=version))
    phases={p.name:p for p in source.phases}
    assert total(source,'hbm')==(ranks+1)*n
    assert total(source,'ccu_copy')==2*n
    assert total(source,'ccu_reduce_elements')==(ranks-1)*n//2
    assert total(source,'ccu_buffer_bytes')==(2*ranks+2)*n
    assert phases['local_input_to_ccu'].after==phases['remote_input_to_ccu'].after==('pre_sync',)
    assert set(phases['load_wait'].after)=={'local_input_to_ccu','remote_input_to_ccu'}
    assert not dict(phases['local_reduce'].work).get('hbm',0)
    assert phases['ccu_to_output'].after==('reduce_wait',)
    assert sum(map(sum,phases['remote_input_to_ccu'].exchange))==ranks*(ranks-1)*n
    assert source.scratch_bytes==ranks*n  # template request; zero scratch traffic
    elapsed,_,_=whole(source.selection)
    assert elapsed>0


@pytest.mark.parametrize('size',(2,32768,32770,8*32768,9*32768+6))
def test_gather_equal_payload_and_copy_slots_keep_all_bytes_once(size):
    s=selected('gather',size=size)
    source=contract(s)
    assert total(source,'hbm')==2*s.ranks*size
    assert total(source,'ccu_copy')==total(source,'ccu_buffer_bytes')==2*size
    assert total(source,'ccu_reduce_elements')==0
    phases={p.name:p for p in source.phases}
    assert phases['peer_write'].after==phases['local_copy'].after==('pre_sync',)
    assert set(phases['write_wait'].after)=={'peer_write','local_copy'}
    elapsed,engine,_=whole(s)
    assert elapsed>0
    copies=[r for r in engine.trace if r['task'].endswith(':read') and ':local_copy:' in r['task']]
    assert len(copies)==(size+32767)//32768
    if size>8*32768:
        full=[r for r in engine.trace if ':full_groups:' in r['task']]
        tail=[r for r in engine.trace if ':tail:' in r['task']]
        assert min(r['start_ms'] for r in tail)>=max(r['end_ms'] for r in full)
    assert buffer_geometry(s)['copy_slots']==8


def test_copy_reduce_and_control_are_separate_declared_services():
    s=selected()
    rates={'hbm':1e9,'ccu_copy':1e9,'ccu_buffer_bytes':1e9,
           'ccu_reduce_elements':1e9,'ccu_control':1e9,'link':1e9}
    for role in ('hbm','ccu_copy','ccu_buffer_bytes','ccu_reduce_elements','ccu_control','link'):
        times=[];serviced=[]
        for factor in (.5,1,2):
            varied={**rates,role:100*factor}
            elapsed,engine,_=whole(s,rates=varied)
            times.append(elapsed);serviced.append(engine.serviced)
        assert times[0]>times[1]>times[2]
        assert serviced[0]==pytest.approx(serviced[1]) and serviced[1]==pytest.approx(serviced[2])


def test_unknown_ccu_hardware_rates_keep_physical_hbm_and_unpriced_inventory():
    s=selected()
    elapsed,_,plans=whole(s,rates={'hbm':1024,'link':1024})
    # Ownload1ms plus outgoingreads3ms contend on HBM, then output1ms.
    assert elapsed==pytest.approx(5)
    missing={}
    for plan in plans.values():
        for k,v in plan.metadata['unpriced_service_work'].items():missing[k]=missing.get(k,0)+v
        assert not any(k in plan.work.resources for k in ('vector','cube','mte2_aiv','hccl_reduce'))
    assert missing['ccu_copy']==2048 and missing['ccu_reduce_elements']==1536
    assert missing['ccu_buffer_bytes']==10240 and missing['ccu_control']==18


def test_one_outer_invocation_adds_startup_once_and_minimum_is_inclusive():
    s=selected('gather',size=32770)
    base,_,_=whole(s)
    with_startup,_,_=whole(s,startup=11)
    with_floor,_,_=whole(s,startup=11,minimum=base+40)
    assert with_startup==pytest.approx(base+11)
    assert with_floor==pytest.approx(base+40)


def test_expansion_version_and_large_reduction_domain_cannot_silently_fall_back():
    # These are alternate real executor branches, with different memory work.
    # An incompatible explicit declaration must never get the other's program.
    s=selected()
    for incompatible in (replace(s,expansion='CCU_SCHED'),replace(s,topology='TWO_DIE_REGULAR'),
                         replace(s,payload_bytes=4098),replace(s,ranks=16)):
        with pytest.raises(ModelingError):contract(incompatible)
    ag=selected('gather')
    with pytest.raises(ModelingError):contract(replace(ag,expansion='CCU_MS'))
    assert contract(replace(s,template='mesh_oneshot_v1')).phases==contract(s).phases
