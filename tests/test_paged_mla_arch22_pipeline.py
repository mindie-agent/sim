"""Compact source-sized refills retain the real finite control's work and time."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.paged_mla_arch22 import tile_work
from mindie_sim.prediction.paged_mla_arch22_matrix import matrix_panels, matrix_program
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.operators import OpSpec


def execute(kind,keys,heads,supply,compact):
    total=Work()
    for load,steps in matrix_panels(kind,keys,heads,128):
        total += load
        for operand,cube in steps:
            total += operand+cube
    ledger=tile_work(keys,heads,first=True,last=True,splits=1,page_size=128)
    selected=sum((ledger[f'{kind}_{part}'] for part in ('load','operands')),Work())+ledger[kind]
    assert total==selected
    rates={'hbm':1e10*supply,'mte2_aic':2e6*supply,'mte1':1e6,'cube':1}
    units={role:1 for role in rates}
    op=OpSpec('matrix','matmul','decode',0,0,(),(),(),(),())
    ctx=RecipeContext(total.apply(op),{}, {r:r for r in rates},rates,1,
                      resource_units=units,compute_peaks={'bf16':1e10})
    engine=Engine([Resource(r,v) for r,v in rates.items()],trace_limit=1000)
    ready=engine.submit('input',release=7)
    program=matrix_program(ctx,kind,keys,heads,128,0,1,1,compact=compact)
    done=program.start(engine,'matrix',ready)
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    for role,amount in ctx.bind(total).items():
        assert engine.serviced[role]==pytest.approx(amount,rel=1e-12)
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    assert not any(engine.allocated.values())
    return done.time-7,engine,total,rates


@pytest.mark.parametrize('kind',['qk','pv'])
@pytest.mark.parametrize('keys,heads',[(1,8),(129,16),(512,8),(512,16)])
@pytest.mark.parametrize('supply',[.1,1,10])
def test_compact_matches_finite_three_refill_two_consumer_control(kind,keys,heads,supply):
    exact,_,_,_=execute(kind,keys,heads,supply,False)
    compact,_,_,_=execute(kind,keys,heads,supply,True)
    assert compact==pytest.approx(exact,rel=1e-10)


@pytest.mark.parametrize('kind',['qk','pv'])
def test_source_l1_refills_and_l0_consumers_overlap_without_deleting_work(kind):
    duration,engine,total,rates=execute(kind,512,16,1,False)
    serial=(total.resources['mte2_aic']/rates['mte2_aic']
            +total.resources['mte1']/rates['mte1']+total.matrix['bf16']/1e7)
    assert duration < .7*serial
    events=list(engine.trace)
    loads=[e for e in events if e['task'].endswith(':load')]
    cubes=[e for e in events if e['task'].endswith(':cube')]
    operands=[e for e in events if e['task'].endswith(':operand')]
    def overlaps(left,right):
        return any(max(a['start_ms'],b['start_ms'])<min(a['end_ms'],b['end_ms'])
                   for a in left for b in right)
    assert overlaps(loads,cubes) and overlaps(operands,cubes)
