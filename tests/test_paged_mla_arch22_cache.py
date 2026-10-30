"""Declared adjacent-tile reuse preserves copy traffic and bounded cache lifetime."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.paged_mla_arch22 import BUFFERS, build, tile_work
from mindie_sim.prediction.paged_mla_arch22_dispatch import source_dispatch
from mindie_sim.prediction.paged_mla_arch22_matrix import matrix_panels, matrix_program
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def setup(retained, *, cache_rate=2e8, capacity=None, contexts=(1024,)):
    cores=1
    units={r:2 if r in ('vector','mte2_aiv','mte3') else 1
           for r in ('hbm','vector','cube','mte1','mte2_aic','mte2_aiv','mte3','fixpipe','l2')}
    rates={r:1e8 for r in units};rates.update(cube=1,l2=cache_rate)
    pools={**BUFFERS,'l2':2*512*576*2 if capacity is None else capacity}
    pu={p:2 if p=='ub' else 1 for p in pools}
    op=OpSpec('mla','attention','decode',0,0,(),(),(),(),(),metadata=(
        ('paged_mla_geometry',{'contexts':contexts,'heads':16,'page_size':128,
         'retained_kv_rows':retained,'dispatch':source_dispatch(contexts,16,cores)}),))
    plan=build(op,Geometry(cores,units,pools,pu))
    ctx=RecipeContext(plan.work.apply(op),{}, {r:r for r in rates},rates,cores,
        resource_units=units,compute_peaks={'bf16':1e12},
        pool_names={p:p for p in pools},pool_capacities=pools,pool_units=pu,
        service_modes={'mte2_aic_cached':('mte2_aic',2e11),
                       'l2_read':('l2',cache_rate*1000),'l2_write':('l2',cache_rate*1000)})
    engine=Engine([Resource(r,v,units=units[r]) for r,v in rates.items()],
                  pools=pools,pool_units=pu,trace_limit=10000)
    return plan,ctx,engine


def execute(retained, **kwargs):
    plan,ctx,engine=setup(retained,**kwargs)
    done=plan.lower(ctx).start(engine,'mla',engine.submit('input',release=7))
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    for role,amount in ctx.bind(plan.work).items():
        assert engine.serviced[role]==pytest.approx(amount,rel=1e-11)
    assert not any(engine.allocated.values())
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    return plan,ctx,engine,done.time-7


@pytest.mark.parametrize('retained',[128,512])
def test_alias_reread_moves_only_declared_pv_rows_to_paid_l2(retained):
    cold,_,_,_=execute(0)
    cached,ctx,engine,_=execute(retained)
    # Two complete S2 tiles plus one real row. QK is still cold, as are page
    # indices, probability, atomics and other workspace reads and writes.
    rows=2*retained+1
    saved=rows*512*2
    assert cold.work.resources['hbm']-cached.work.resources['hbm']==saved
    assert cold.work.resources['mte2_aic']-cached.work.resources['mte2_aic']==saved
    assert cached.work.resources['mte2_aic_cached']==saved
    assert cached.work.resources['l2_read']==saved
    assert cached.work.resources['l2_write']==rows*576*2
    assert cold.work.matrix==cached.work.matrix
    assert cached.metadata['cache_policy']['resident_bytes']==2*retained*576*2
    # Cached reads share AIC0's physical copy resource. L2 remains one shared
    # device resource, rather than a private cache bandwidth per AIC.
    events=[e for e in engine.trace if ':pv_' in e['task'] and e['demand'].get('l2',0)]
    assert events
    assert all(e['placement']['mte2_aic']==(0,) and 'l2' not in e['placement'] for e in events)


def test_resident_cache_capacity_covers_two_generations_and_is_released_between_calls():
    with pytest.raises(ModelingError,match='two_tiles_in_shared_l2'):
        setup(512,capacity=512*576*2)
    plan,ctx,engine=setup(512)
    ready=engine.submit('ready')
    first=plan.lower(ctx).start(engine,'first',ready)
    second=plan.lower(ctx).start(engine,'second',ready)
    engine.advance(until=(first,second),first=False)
    _,_,_,isolated=execute(512)
    assert max(first.time,second.time)==pytest.approx(2*isolated,rel=1e-10)
    assert not any(engine.allocated.values())


def test_cached_v_reads_have_finite_service_and_share_the_l2_capacity_rate():
    _,_,_,fast=execute(512)
    _,_,_,slow=execute(512,cache_rate=1e7)
    assert slow>fast
    plan,ctx,engine=setup(512)
    wrong=replace(ctx,service_modes={**ctx.service_modes,'mte2_aic_cached':('mte2_aiv',2e11)})
    with pytest.raises(ModelingError,match='share_aic_pipe'):
        plan.lower(wrong)


@pytest.mark.parametrize('kind',['qk','pv'])
@pytest.mark.parametrize('keys,retained',[(129,64),(512,512)])
def test_cached_matrix_compaction_preserves_finite_refill_service(kind,keys,retained):
    plan,ctx,_=setup(retained)
    total=Work()
    for load,steps in matrix_panels(kind,keys,16,128,retained_kv_rows=retained):
        total+=load
        for operand,cube in steps:total+=operand+cube
    ledger=tile_work(keys,16,first=True,last=True,splits=1,page_size=128,retained_kv_rows=retained)
    assert total==sum((ledger[kind+s] for s in ('_load','_operands','')),Work())
    durations=[]
    for compact in (False,True):
        _,_,engine=setup(retained)
        program=matrix_program(ctx,kind,keys,16,128,0,1,1,
                               compact=compact,retained_kv_rows=retained)
        done=program.start(engine,'matrix',engine.submit('input'))
        engine.advance(until=(done,))
        for role,amount in ctx.bind(total).items():
            assert engine.serviced[role]==pytest.approx(amount,rel=1e-12)
        durations.append(done.time)
    assert durations[0]==pytest.approx(durations[1],rel=1e-12)
