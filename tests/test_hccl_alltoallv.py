"""Source byte/slot oracles and complete-rank physical Engine controls."""
from dataclasses import replace
from math import ceil
import random

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.hccl_alltoallv import AlltoAllVGroup, contract, peer_schedule, bind_local_work
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.program import RecipeContext
from test_collective_cohorts import per_pair_topology, ready_events, topology
from test_prediction_system import case


def selection(matrix, mode='write', rank=0):
    return select({'expansion':'AICPU_TS','template':'mesh_'+mode,'dtype_bytes':1},
                  collective='alltoallv', ranks=len(matrix), payload_bytes=sum(matrix[rank]))


def amount(plan, role):
    return sum(dict(phase.work).get(role,0) for phase in plan.phases)


def run(matrix, mode='write', *, ready=None, memory=1e15, copy=None, control=None,
        bandwidth=1, latency=0, framing=None, topo=None, ranks=None):
    count=len(matrix)
    ranks=tuple(range(count)) if ranks is None else ranks
    topo=topo or per_pair_topology(count,capacity=bandwidth,latency_us=latency,framing=framing)
    resources=list(topo.resources());contexts={}
    for rank in ranks:
        rates={'hbm':memory}
        if copy is not None:rates['hccl_copy']=copy
        if control is not None:rates['hccl_control']=control
        names={role:f'{role}:{rank}' for role in rates}
        resources.extend(Resource(names[role],rate) for role,rate in rates.items())
        contexts[rank]=RecipeContext(None,{},names,{names[k]:v for k,v in rates.items()},1,
                                    resource_units={name:1 for name in names.values()})
    engine=Engine(resources,trace_limit=30000)
    initial={rank:engine.submit(f'ready{rank}',release=(ready or [0]*count)[index])
             for index,rank in enumerate(ranks)}
    group=AlltoAllVGroup(selection(matrix,mode),tuple(map(tuple,matrix)),ranks,topo)
    done=group.start_participants(engine,'native',initial,contexts)
    engine.advance(until=tuple(done.values()),first=False)
    assert all(event.done for event in done.values())
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    for plan in group.plans():
        physical=ranks[plan.rank]
        for role in ('hbm','hccl_copy','hccl_control'):
            if role in contexts[physical].resource_names:
                assert engine.serviced.get(f'{role}:{physical}',0)==pytest.approx(amount(plan,role))
    return done,engine,group


def test_source_left_right_slots_are_unique_with_reciprocal_pair_groups():
    for count in (2,3,8,16,17,18,31,32,33,34,65):
        for rank in range(count):
            rows=peer_schedule(count,rank)
            assert {p for p,_,_ in rows}==set(range(count))-{rank}
            assert len(rows)==count-1
            for loop in range(ceil((count-1)/16)):
                slots=[slot for _,slot,l in rows if l==loop]
                assert len(set(slots))==len(slots)<=16
            for peer,slot,loop in rows:
                back=next(row for row in peer_schedule(count,peer) if row[0]==rank)
                assert back[2]==loop
                assert 0<=slot<min(16,count-1)


@pytest.mark.parametrize('mode',('read','write'))
def test_nonuniform_integer_matrix_keeps_diagonal_and_copy_direction(mode):
    rng=random.Random(0xA2A)
    for count in (2,3,8,19):
        matrix=tuple(tuple(rng.randrange(5)*16 for _ in range(count)) for _ in range(count))
        for rank in range(count):
            plan=contract(selection(matrix,mode,rank),matrix,rank)
            out=sum(matrix[rank])-matrix[rank][rank]
            incoming=sum(row[rank] for row in matrix)-matrix[rank][rank]
            diagonal=matrix[rank][rank]
            assert amount(plan,'hbm')==2*diagonal+(3*out+incoming if mode=='read' else out+3*incoming)
            assert amount(plan,'hccl_copy')==diagonal+(out if mode=='read' else incoming)
            assert sum(p.send_bytes for p in plan.peers)==out
            assert sum(p.recv_bytes for p in plan.peers)==incoming
            assert plan.thread_count==min(16,count-1)+1
            assert all(p.owner_domain==() or p.owner_domain[0]>0 for p in plan.phases)


@pytest.mark.parametrize('mode',('read','write'))
@pytest.mark.parametrize('count',(2,8,17,18,33,34))
def test_full_mesh_uses_sixteen_concurrent_slots_not_rank_minus_one_startups(mode,count):
    matrix=tuple(tuple(int(src!=dst) for dst in range(count)) for src in range(count))
    done,engine,_=run(matrix,mode,latency=5)
    assert [e.time for e in done.values()]==pytest.approx([ceil((count-1)/16)*1.005]*count,abs=1e-9)
    assert sum(v for k,v in engine.serviced.items() if k.startswith('link:'))==count*(count-1)
    # Exactly one wire service per directed message, never one for both source
    # and destination plans. No local control task secretly owns network bytes.
    services=[row for row in engine.trace if row['task'].endswith(':hop0:service')]
    assert len(services)==count*(count-1)


@pytest.mark.parametrize('mode',('read','write'))
def test_zero_peer_does_not_create_remote_barrier_and_receive_only_waits_for_input(mode):
    matrix=((4,0,0),(0,0,16),(0,0,0))
    done,engine,group=run(matrix,mode,ready=[1,2,9],memory=16,copy=8)
    assert done[0].time==pytest.approx(1.5)
    assert done[1].time>=9 and done[2].time>=9
    wire=next(row for row in engine.trace if row['task'].endswith(':hop0:service'))
    assert wire['start_ms']>=9
    assert sum(v for k,v in engine.serviced.items() if k.startswith('link:'))==16
    assert amount(group.plans()[2],'hccl_copy')==(0 if mode=='read' else 16)
    empty=((0,0),(0,0))
    done,_,_=run(empty,mode,ready=[3,17])
    assert done[0].time==3 and done[1].time==17


def test_read_first_pre_copy_join_differs_from_write_post_copy():
    matrix=((0,16),(0,0))
    read,er,_=run(matrix,'read',memory=16,copy=8,bandwidth=16)
    write,ew,_=run(matrix,'write',memory=16,copy=8,bandwidth=16)
    # Copy is 2ms. Read publishes scratch first, then wire1ms; write wire1ms
    # first then receiver's2ms copy. Both transfer identical total wire/HBM.
    assert max(e.time for e in read.values())==pytest.approx(3)
    assert max(e.time for e in write.values())==pytest.approx(3)
    rwire=next(row for row in er.trace if row['task'].endswith(':hop0:service'))
    wwire=next(row for row in ew.trace if row['task'].endswith(':hop0:service'))
    assert rwire['start_ms']==pytest.approx(2) and wwire['start_ms']==0
    rp=next(row for row in er.trace if row['task']=='native:rank0:precopy_1')
    wp=next(row for row in ew.trace if row['task']=='native:rank1:postcopy_0')
    assert rp['end_ms']<=rwire['start_ms'] and wp['start_ms']>=wwire['end_ms']


def test_later_slot_waits_for_postcopy_without_imposing_whole_round_barrier():
    count=18
    matrix=[[0]*count for _ in range(count)]
    # Rank0 left-distance8(slot0) then opposite-distance9(slot0).
    matrix[10][0]=100
    matrix[9][0]=1
    matrix[1][0]=1
    _,engine,_=run(matrix,'write',memory=100,copy=1,bandwidth=100)
    rows={r['task']:r for r in engine.trace}
    assert rows['native:9->0:hop0:service']['start_ms']>=rows['native:rank0:postcopy_10']['end_ms']
    assert rows['native:1->0:hop0:service']['start_ms']==0
    assert rows['native:rank0:postcopy_1']['end_ms']<rows['native:rank0:postcopy_10']['end_ms']


def test_return_transpose_and_physical_rank_mapping_preserve_framed_hops():
    raw=case(tp=4);raw['model'].update(num_heads=8)
    raw['machine']['links']=[
        {'name':'left','members':[1,0],'semantics':'per_pair','capacity_bytes_per_ms':16,
         'transaction_bytes':8,'protocol_overhead_bytes':2,'latency_us':5},
        {'name':'right','members':[0,3],'semantics':'per_pair','capacity_bytes_per_ms':32,
         'transaction_bytes':16,'protocol_overhead_bytes':4,'latency_us':7},
    ]
    topo=topology(raw)
    matrix=((7,9),(21,3))
    for cells in (matrix,tuple(zip(*matrix))):
        _,engine,_=run(cells,topo=topo,ranks=(1,3))
        # Each of9 and21bytes traverses two hops. Per-hop framing is retained.
        expected=ceil(9/8)*10+ceil(21/8)*10+ceil(9/16)*20+ceil(21/16)*20
        assert sum(v for k,v in engine.serviced.items() if k.startswith('link:'))==expected
        assert sum(v for k,v in engine.serviced.items() if k.startswith('hbm:'))==4*(9+21)+2*(7+3)


def test_declared_control_rate_changes_only_control_service_not_payload():
    matrix=((0,8),(8,0))
    values=[]
    for scale in (.5,1,2):
        done,engine,group=run(matrix,'read',memory=100,copy=100,control=10*scale,bandwidth=100)
        values.append(max(e.time for e in done.values()))
        assert dict(group.plans()[0].control_counts)['notify']==6
        assert dict(group.plans()[0].control_counts)['wait']==6
        assert sum(v for k,v in engine.serviced.items() if k.startswith('link:'))==16
    assert values[0]>values[1]>values[2]
    ctx=RecipeContext(None,{}, {'hbm':'hbm'}, {'hbm':100},1,resource_units={'hbm':1})
    demand,omitted=bind_local_work((('hbm',32),('hccl_copy',16),('hccl_control',4)),ctx)
    assert demand=={'hbm':32} and omitted=={'hccl_copy':16,'hccl_control':4}


@pytest.mark.parametrize('axis',('memory','copy','bandwidth'))
def test_independent_physical_rates_affect_real_copy_or_transport_not_payload(axis):
    matrix=((0,32),(64,0))
    times=[];work=[]
    for factor in (.5,1,2):
        rates={'memory':1e9,'copy':1e9,'bandwidth':1e9}
        rates[axis]=32*factor
        done,engine,_=run(matrix,'read',**rates)
        times.append(max(e.time for e in done.values()))
        work.append(dict(engine.serviced))
    assert times[0]>times[1]>times[2]
    assert work[0]==pytest.approx(work[1]) and work[1]==pytest.approx(work[2])
