"""Physical arbitration equivalence for a pure disjoint-partition cache."""
from random import Random
import math

from mindie_sim.prediction import engine as engine_module
from mindie_sim.prediction import ownership
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.program import PipelineProgram, Stage


def literal(rate, count, caps, placements):
    result=ownership._partition_rates(rate,count,caps,placements)
    return ownership.unit_rates(rate,count,caps,placements) if result is None else result


def test_ordinal_cache_preserves_exact_rates_order_and_rebinding():
    random=Random(31030)
    ownership._partition_template.cache_clear()
    for _ in range(500):
        units=random.choice((1,2,4,8,32,64));n=random.randrange(1,units*3+1)
        keys=random.sample(range(1000,10000),n)
        caps={key:random.choice((float('inf'),0,1/7,33.3,10000)) for key in keys}
        placements={key:(random.randrange(units),) for key in keys}
        rate=random.choice((1/3,1200.5,4224000000))
        expected=ownership._partition_rates(rate,units,caps,placements)
        actual=ownership._cached_partition_rates(rate,units,caps,placements)
        assert list(actual.items())==list(expected.items())
        # New IDs and names cannot retain a prior task or change tie ordering.
        names=[f'new:{key}' for key in keys]
        rebound=ownership._cached_partition_rates(rate,units,dict(zip(names,caps.values())),dict(zip(names,placements.values())))
        index={key:i for i,key in enumerate(keys)}
        assert list(rebound.items())==[(names[index[k]],v) for k,v in expected.items()]
    assert ownership._partition_template.cache_info().currsize<=512


def test_overlapping_or_reordered_groups_do_not_reuse_disjoint_topology():
    caps={91:30.1,55:float('inf'),872:float('inf')}
    first={91:(0,1),55:(0,1),872:(2,3)}
    original=ownership.unit_rates(99.1,4,caps,first)
    assert list(original.items())==list(ownership._partition_rates(99.1,4,caps,first).items())
    for placement in ({91:(0,1),55:(1,2),872:(2,3)},
                      {91:(0,1),55:(1,0),872:(2,3)},{}):
        if ownership._partition_rates(99.1,4,caps,placement) is None:
            assert ownership._cached_partition_rates(99.1,4,caps,placement) is None
    changed=ownership.unit_rates(99.1,4,{**caps,91:1.2},first)
    assert changed[91]==1.2 and changed!=original
    assert ownership.unit_rates(99.1,8,caps,first)!=original


def test_large_live_cohort_does_not_enter_retained_template_cache():
    ownership._partition_template.cache_clear()
    caps={index:float('inf') for index in range(1024)}
    placements={index:(index % 64,) for index in caps}
    actual=ownership.unit_rates(123.7,64,caps,placements)
    expected=ownership._partition_rates(123.7,64,caps,placements)
    assert list(actual.items())==list(expected.items())
    assert ownership._partition_template.cache_info().currsize==0


def test_float_scale_tail_service_and_zero_work_callbacks_are_conserved():
    # One-resource amounts span subnormal through large values; all work is
    # observable in service and dependent callbacks even when time rounds.
    for exponent in (-1074,-1070,-1022,-500,-20,0,20,500,1000):
        amount=math.ldexp(1.0,exponent)
        rate=amount if exponent < -1000 else amount/3
        engine=Engine([Resource('wire',rate)],trace_limit=10)
        first=engine.submit('first',{'wire':amount})
        callbacks=[]
        end=engine.submit('tail',after=(first,),on_complete=lambda e:callbacks.append(e.time))
        engine.advance(until=(end,))
        assert engine.serviced['wire']==amount
        assert callbacks==[end.time] and first.time==end.time
        assert engine.summary()['live_tasks']==0


def test_coupled_detection_after_ordinary_contenders_and_empty_coupled_task():
    from mindie_sim.prediction.coupled import coupled_rates
    engine=Engine([Resource('read',27.3,units=2),Resource('compute',19.1,units=2)])
    engine.submit('ordinary',{'read':7},placement={'read':(0,)})
    engine.submit('empty-coupled',latency=3,coupled=True)
    engine.submit('coupled',{'read':11,'compute':9},coupled=True,
                  placement={'read':(1,),'compute':(0,)})
    engine._settle()
    expected=coupled_rates(engine.resources,{key:engine._tasks[key] for key in engine._active})
    assert engine._rates()==expected
    engine.advance()
    assert engine.summary()['live_tasks']==0
    assert engine.serviced=={'read':18.0,'compute':9.0}


def execute(seed):
    r=Random(seed);e=Engine([Resource('copy',97.3,units=4),Resource('vector',300.1,units=4)],
        pools={'ub':64},pool_units={'ub':4},trace_limit=4000)
    ready=e.submit('late',latency=7);ends=[];callbacks=[]
    for owner in range(4):
        def tile(index,owner=owner):
            buf=Allocation(f'{owner}:{index}','ub',32,(owner,))
            return (Stage('read',{'copy':r.randrange(1,8)},acquire=(buf,),
                          placement={'copy':(owner,)},caps={'copy':13.7 if owner%2 else 1000}),
                    Stage('compute',{'vector':r.randrange(1,8)},after=('read',),
                          placement={'vector':(owner,)},free=(buf,)))
        end=PipelineProgram(7,2,tile).start(e,f'owner{owner}',ready)
        e.submit(f'callback{owner}',after=(end,),on_complete=lambda event,owner=owner:callbacks.append((owner,event.time)))
        ends.append(end)
    e.submit('overlap',{'copy':9},release=7.02,placement={'copy':(0,1)},caps={'copy':18.3})
    e.advance(until=ends,first=False)
    e.advance()
    return e.summary(),e.trace,callbacks


def test_actual_engine_schedule_service_buffers_and_callbacks_are_identical(monkeypatch):
    cached=[execute(seed) for seed in range(8)]
    monkeypatch.setattr(engine_module,'unit_rates',literal)
    assert [execute(seed) for seed in range(8)]==cached
    assert all(not result[0]['live_tasks'] and not result[0]['live_programs'] for result in cached)
