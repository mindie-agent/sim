"""Lazy native reduction versus the prior eager finite per-token Engine DAG."""
import math
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mc2_phase import build
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.schema.operators import OpSpec


def execute(tokens, units, lazy):
    width, topk = 240, 3
    raw = dict(kind='combine', width=width, dtype='bf16', source_tokens=tokens,
               top_k=topk, source_assignments=tokens*topk,
               received_assignments=tokens*topk, experts=16, ep=8, phase='weighted_reduce')
    op = OpSpec('phase','mc2_protocol','decode',0,0,(),(),(),(),(),metadata=(('mc2_phase',raw),))
    rates = {'hbm':100000, 'mte2_aiv':64000, 'mte3':64000, 'vector':64000}
    physical = {r:1 if r=='hbm' else units for r in rates}
    ctx = RecipeContext(op,{}, {r:r for r in rates},rates,units,resource_units=physical)
    plan = build(op,Geometry(units,physical))
    if lazy:
        program = plan.program(ctx)
    else:
        # Previous finite eager DAG. Build directly from per-token arithmetic,
        # not from the new tile factory, and retain owner0's fractional tail.
        active=min(units,math.ceil(tokens));base,tail=divmod(tokens,active)
        stages=[]
        for owner in range(active):
            count=base+min(1,max(0,tail-owner));previous=()
            def add(name,resources):
                nonlocal previous
                stage_id=f'owner{owner}:{name}'
                demand=Work(resources=resources)
                stages.append(Stage(stage_id,ctx.bind(demand),after=previous,
                                    placement=ctx.owned(owner,units,demand.resources)))
                previous=(stage_id,)
            add('weights',{'hbm':count*topk*4,'mte2_aiv':count*topk*4})
            for token in range(math.ceil(count)):
                scale=min(1,count-token)
                add(f't{token}:init',{'vector':scale*width})
                for slot in range(topk):
                    add(f't{token}:s{slot}:read',{'hbm':scale*480,'mte2_aiv':scale*480})
                    for operation in ('cast','multiply','accumulate'):
                        add(f't{token}:s{slot}:{operation}',{'vector':scale*width})
        program=KernelProgram(tuple(stages))
    engine=Engine([Resource(r,v,units=physical[r]) for r,v in rates.items()],trace_limit=0)
    done=program.start(engine,'run',engine.submit('ready',release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    assert not any(engine.allocated.values())
    return done.time, dict(engine.serviced), engine.summary()['peak_live_tasks']


@pytest.mark.parametrize('tokens,units',[(1,2),(1.25,2),(3,2),(8.75,4),(65,64),(65,3)])
def test_lazy_and_prior_eager_engine_have_equal_span_and_work(tokens,units):
    actual=execute(tokens,units,True)
    prior=execute(tokens,units,False)
    assert actual[0]==pytest.approx(prior[0],abs=1e-11)
    assert actual[1]==pytest.approx(prior[1],rel=1e-12,abs=1e-9)


def test_pending_tasks_are_bounded_by_owner_window_not_token_count():
    small=execute(5,2,True)
    large=execute(257,2,True)
    eager=execute(257,2,False)
    assert large[2]==small[2]
    assert large[2]<100
    assert eager[2]>1000
    assert large[0]==pytest.approx(eager[0],abs=1e-11)
    assert large[1]==pytest.approx(eager[1],rel=1e-12)
