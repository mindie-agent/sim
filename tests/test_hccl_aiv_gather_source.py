"""Source owner arithmetic checked independently of real Engine service."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_aiv import GROUP_BUFFER, UB_RESIDENCY
from mindie_sim.prediction.hccl_aiv_gather import build,contract,jobs,DATA_LIMIT,FLAG_BYTES
from mindie_sim.prediction.mechanism import Geometry,OperatorPlan,Work
from mindie_sim.prediction.program import RecipeContext,KernelProgram,ParallelProgram,Stage
from mindie_sim.schema.operators import OpSpec


def selection(size, ranks=4, blocks=64, width=2):
    return select({'expansion':'AIV','template':'mesh','block_limit':blocks,'dtype_bytes':width},
                  collective='allgather',ranks=ranks,payload_bytes=size)


def whole(s, *, rates=None, ready=7, startup=0, minimum=0, borrow=False, after_call=None):
    units={'hbm':1,'mte2_aiv':64,'mte3':64,'vector':64,'link':1}
    rates=rates or dict.fromkeys(units,1e9)
    geometry=Geometry(32,units,buffers={'ub':256*1024},buffer_units={'ub':64})
    plan=contract(s)
    programs={}
    for phase in plan.phases:
        op=OpSpec(phase.name,'collective','decode',0,0,(),(),(),(),(),metadata=(
            ('hccl_phase',{'selection':s.to_dict(),'phase':phase.name,'rank':0}),))
        programs[phase.name]=build(op,geometry)
    work=sum((p.work for p in programs.values()),Work())
    wire=sum(sum(phase.exchange[0]) for phase in plan.phases if phase.exchange)
    work+=Work(resources={'link':wire})
    context=RecipeContext(None,{},dict(zip(units,units)),rates,32,
                          resource_units=units,pool_names={'ub':'ub'},pool_units={'ub':64},
                          setup_ms=startup,minimum_ms=minimum,
                          group_buffers=(GROUP_BUFFER,) if borrow else ())
    def lower(ctx):
        stages=[]
        for phase in plan.phases:
            local=programs[phase.name].program(ctx)
            if phase.exchange:
                local=ParallelProgram((local,KernelProgram((
                    Stage('wire',{'link':sum(phase.exchange[0])}),))))
            stages.append(Stage(phase.name,after=phase.after,program=local))
        return KernelProgram(tuple(stages))
    engine=Engine([Resource(k,rates[k],units=units[k]) for k in units],
                  pools={'ub':256*1024},pool_units={'ub':64},trace_limit=10000)
    initial=engine.submit('input',release=ready)
    allocation=context.allocate('outer','ub',UB_RESIDENCY,tuple(range(s.block_limit)))
    if borrow:
        initial=engine.submit('outer_acquire',after=(initial,),acquire=(allocation,))
    done=OperatorPlan(work,lower).lower(context).start(engine,'ag',initial)
    if borrow:
        done=engine.submit('outer_free',after=(done,),free=(allocation,))
    if after_call:after_call(engine,context,done)
    engine.advance(horizon=ready-.001)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.serviced==pytest.approx({k:work.resources.get(k,0) for k in units})
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    assert not any(engine.summary()['live_allocated'].values())
    return done.time-ready,engine,programs


@pytest.mark.parametrize('ranks,blocks,size',[(4,64,2),(4,64,30),(4,64,4096),(8,3,42),
    (4,64,DATA_LIMIT),(3,64,DATA_LIMIT+6),(8,64,2*DATA_LIMIT),(64,64,DATA_LIMIT+2),
    (2,4,2*DATA_LIMIT+6)])
def test_independent_owner_and_flag_byte_equations(ranks,blocks,size):
    s=selection(size,ranks,blocks)
    owner_jobs,launched,detail=jobs(s)
    assert launched==blocks
    assert sum(a.size for actions in owner_jobs['precopy'].values() for a in actions if a.kind=='copy')==size
    assert sum(a.size for actions in owner_jobs['gather'].values() for a in actions if a.kind=='copy')==ranks*size
    if size>=DATA_LIMIT and blocks>=2*ranks:
        usable=blocks//ranks*ranks
        assert set(owner_jobs['precopy'])==set(range(ranks))
        assert set(owner_jobs['gather'])==set(range(ranks,usable))
        cut=(usable-ranks)//ranks
        q,tail=divmod(size//2,ranks)
        pieces=sum(min(q+(i<tail),cut) for i in range(ranks))
        records,waits=pieces,ranks*pieces
    else:
        assert set(owner_jobs['gather'])==set(range(min(blocks,ranks)))
        records=waits=2*ranks
    if not detail['pingpong']:
        records+=2*ranks;waits+=ranks
    p=contract(s)
    totals={k:sum(dict(phase.work).get(k,0) for phase in p.phases)
            for k in ('hbm','mte2_aiv','mte3')}
    assert totals=={'hbm':2*(ranks+1)*size+(records+waits)*128,
                   'mte2_aiv':(ranks+1)*size+waits*128,
                   'mte3':(ranks+1)*size+records*128}
    assert sum(dict(p.control_counts)[k] for k in ('wait','record'))==records+waits
    elapsed,_,_=whole(s)
    assert elapsed>0


def test_small_four_rank_grid64_uses_only_four_owners_and_exact_tail():
    s=selection(30)
    owners,_,_=jobs(s)
    assert [actions[0].size for actions in owners['precopy'].values()]==[6,6,6,12]
    _,engine,_=whole(s,borrow=True)
    copy_tasks=[row for row in engine.trace if row['task'].endswith((':read',':write'))]
    assert copy_tasks and all(any(f':owner{i}:' in row['task'] for i in range(4)) for row in copy_tasks)
    # Buffer admission is full launched-grid while physical service has no
    # right to borrow idle owners. Borrowed phases must not allocate it again.
    assert len(engine.summary()['peak_allocated_per_unit'])==64
    assert set(engine.summary()['peak_allocated_per_unit'].values())=={UB_RESIDENCY}


@pytest.mark.parametrize('role',('hbm','mte2_aiv','mte3','link'))
def test_directional_rate_sensitivity_and_service_conservation(role):
    s=selection(4096)
    samples=[]
    for factor in (.5,1,2):
        rates=dict.fromkeys(('hbm','mte2_aiv','mte3','vector','link'),1e12)
        rates[role]=1e6*factor
        samples.append(whole(s,rates=rates))
    assert samples[0][0]>samples[1][0]>samples[2][0]
    assert samples[0][1].serviced==pytest.approx(samples[1][1].serviced)
    assert samples[1][1].serviced==pytest.approx(samples[2][1].serviced)


def test_copy_window_and_flag_order_with_single_outer_invocation():
    s=selection(DATA_LIMIT+6,ranks=3)
    base,engine,_=whole(s)
    start,_,_=whole(s,startup=2)
    floor,_,_=whole(s,startup=2,minimum=base+5)
    assert start==pytest.approx(base+2)
    assert floor==pytest.approx(base+5)
    owner0=[row for row in engine.trace if ':precopy:' in row['task'] and ':owner0:' in row['task']]
    work=[row for row in owner0 if row['task'].endswith((':read',':write',':record'))]
    assert all(b['start_ms']>=a['end_ms']-1e-12 for a,b in zip(work,work[1:]))
    assert engine.summary()['peak_live_tasks']<1000


def test_successful_flag_wire_is_separate_from_useful_payload():
    for s in (selection(4096),selection(DATA_LIMIT+6,ranks=3)):
        p=contract(s)
        detail=jobs(s)[2]
        flags=detail['positive_input_fragments'] if detail['branch']=='producer_consumer' else 2
        flags+=int(not detail['pingpong'])
        expected=(s.ranks-1)*(s.payload_bytes+flags*FLAG_BYTES)
        assert sum(sum(phase.exchange[0]) for phase in p.phases if phase.exchange)==expected
        # Rank affects traversal order only, not byte counts or peer topology.
        assert [phase.work for phase in contract(s,rank=s.ranks-1).phases]==[phase.work for phase in p.phases]
