"""Source hierarchy byte oracles and actual remote-notify Engine schedules."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_hierarchy import contract
from mindie_sim.prediction.hccl_signals import SignalGroup
from test_communication_topology import topology
from test_hccl_source_graph import declared, isolated_calls
from test_joint_collective_execution import wire
from test_prediction_system import case


def selection(local=2, remote=4, n=1024, **options):
    return select(dict(expansion='AICPU_TS', template='parallel_mesh_nhr_route0',
        topology_levels=2, level0_size=local, max_loop_bytes=n, **options),
        collective='allreduce', ranks=local*remote, payload_bytes=n)


@pytest.mark.parametrize('local,remote', [(2,2),(2,4),(4,2),(4,4),(8,4)])
def test_source_hierarchy_exact_directed_work_and_no_alias_copies(local,remote):
    r, n = local*remote, local*remote*128
    plan = contract(selection(local,remote,n))
    work = lambda role: sum(dict(p.work).get(role,0) for p in plan.phases)
    assert work('hbm') == 7*n+2*n/local-5*n/r
    assert work('hccl_reduce') == n*(r-1)/r
    assert work('hccl_copy') == 2*n/local
    assert sum(sum(map(sum,p.exchange)) for p in plan.phases if p.exchange)==2*n*(r-1)
    log = (remote-1).bit_length()
    assert sum(len(p.signals) for p in plan.phases)==r*(4*(local-1)+3*log)
    assert not any(p.name.startswith(('l1_rs.input_to','l1_rs.scratch_to',
                                      'l1_ag.input_to','l1_ag.scratch_to')) for p in plan.phases)
    assert not any(p.signals for p in plan.phases if '_executor_' in p.name)
    fused = [p for p in plan.phases if dict(p.control_counts).get('remote_notify_fused_with_payload')]
    assert len(fused)==log and all(not p.signals for p in fused)
    # Equal shards retain equal local work for every physical rank.
    for rank in (0,local-1,r-1):
        other=contract(plan.selection,rank)
        assert sum(dict(p.work).get('hbm',0) for p in other.phases)==work('hbm')
    for p in plan.phases:
        for src,row in enumerate(p.exchange):
            for dst,value in enumerate(row):
                if value:
                    assert (src//local==dst//local if p.name.startswith('l0_')
                            else src%local==dst%local)


def test_hierarchy_round_dependency_graph_and_controller_classification():
    plan=contract(selection())
    phases={p.name:p for p in plan.phases}
    for previous,next_stage in [('l0_rs','l1_rs'),('l1_rs','l1_ag'),('l1_ag','l0_ag')]:
        assert phases[next_stage+'_executor_pre'].after==(previous+'_executor_post',)
    for phase in plan.phases:
        if not phase.exchange:
            continue
        ack=phases[phase.after[0]]
        assert ack.name.endswith('_ack_wait')
        signal=phases[ack.after[0]]
        assert signal.signals
        complete=phases[phase.name+'_complete_wait']
        if phase.name.startswith('l1_rs'):
            assert complete.after==(phase.name,)  # Fused payload notification.
        else:
            assert phases[complete.after[0]].signals
    # Every event references earlier predecessors, not an invented global tail.
    seen=set()
    for phase in plan.phases:
        assert set(phase.after)<=seen
        seen.add(phase.name)


def test_zero_payload_notify_keeps_remote_propagation_and_late_endpoint_readiness():
    raw=case(tp=2)
    raw['machine']['links'][0].update(latency_us=5,capacity_bytes_per_ms=1)
    topo=topology(raw)
    eng=Engine(topo.resources(),trace_limit=100)
    ready={0:eng.submit('r0'),1:eng.submit('r1',release=7)}
    end=SignalGroup((0,1),topo,((0,1),(1,0))).start_participants(eng,'ack',ready)
    eng.advance(horizon=6)
    assert not any(eng.serviced.values())
    eng.advance(until=tuple(end.values()))
    assert [e.time for e in end.values()]==pytest.approx([7.005,7.005])
    assert not any(eng.serviced.values())
    assert eng.summary()['live_tasks']==0


def raw_case(latency=0, signal_bytes=None):
    raw=declared('AICPU_TS','parallel_mesh_nhr_route0',ranks=8)
    raw['model'].update(num_heads=8,num_kv_heads=8)
    raw['machine']['device']['rates']['memory']=1e15
    raw['machine']['links'][0].update(latency_us=latency,capacity_bytes_per_ms=1e12)
    raw['runtime']['kernels']['hccl']['allreduce'].update(
        topology_levels=2,level0_size=2,max_loop_bytes=1048576,signal_bytes=signal_bytes)
    return raw


def test_public_caller_one_invocation_remote_rounds_and_frame_bytes_once():
    raw=raw_case()
    ops=isolated_calls(raw)
    zero=graph_costs(raw,ops,isolated=False,input_ready_ms=7,include_submission=True)
    slow=deepcopy(raw);slow['machine']['links'][0]['latency_us']=5
    delayed=graph_costs(slow,ops,isolated=False,input_ready_ms=7,include_submission=True)
    # R=8,L=2,H=4: 6 payload + 10 notification propagation waves/call.
    # Four collectives in this two-layer ordinary body.
    assert delayed['duration_ms']-zero['duration_ms']==pytest.approx(4*16*.005,abs=1e-6)
    assert delayed['execution']['submission']['eager_invocations']==4
    assert delayed['execution']['live_tasks']==delayed['execution']['live_programs']==0
    assert sum(wire(delayed).values())==4*2*7*64
    assert all(row['ready_ms']>=7 for row in delayed['timeline'])
    framed=raw_case(signal_bytes=4)
    framed['machine']['links'][0].update(transaction_bytes=16,protocol_overhead_bytes=4)
    with_signals=graph_costs(framed,isolated_calls(framed),isolated=False)
    framed['runtime']['kernels']['hccl']['allreduce']['signal_bytes']=None
    payload_only=graph_costs(framed,isolated_calls(framed),isolated=False)
    assert sum(wire(with_signals).values())-sum(wire(payload_only).values())==4*80*20


def test_inter_group_bandwidth_scan_preserves_work_and_only_cross_group_service():
    raw=raw_case()
    # Consecutive pairs form level0; same position across pairs forms level1.
    raw['machine']['links']=[
        {'name':f'local{i}','members':[2*i,2*i+1], 'capacity_bytes_per_ms':1e9}
        for i in range(4)] + [
        {'name':'inter','members':list(range(8)), 'capacity_bytes_per_ms':1000,
         'semantics':'per_pair','latency_us':2}]
    ops=isolated_calls(raw)
    results=[]
    for factor in (.5,1,2):
        varied=deepcopy(raw)
        varied['machine']['links'][-1]['capacity_bytes_per_ms']*=factor
        results.append(graph_costs(varied,ops,isolated=False))
    assert results[0]['duration_ms']>results[1]['duration_ms']>results[2]['duration_ms']
    assert results[0]['execution']['serviced']==results[1]['execution']['serviced']==results[2]['execution']['serviced']


def test_remote_signals_cross_two_links_and_contend_at_receiver_once():
    raw=case(tp=4)
    raw['machine']['num_devices']=5
    raw['machine']['links']=[
        {'name':'uplink','members':[0,1,2,4], 'semantics':'per_port',
         'capacity_bytes_per_ms':1e6,'latency_us':1,
         'transaction_bytes':16,'protocol_overhead_bytes':4},
        {'name':'downlink','members':[4,3], 'semantics':'per_port',
         'capacity_bytes_per_ms':1000,'latency_us':2,
         'transaction_bytes':16,'protocol_overhead_bytes':4},
    ]
    topo=topology(raw)
    eng=Engine(topo.resources())
    initial={r:eng.submit(f'entry{r}',release=7 if r==3 else 0) for r in range(4)}
    done=SignalGroup((0,1,2,3),topo,((0,3),(1,3),(2,3)),4).start_participants(
        eng,'incast',initial)
    eng.advance(until=tuple(done.values()),first=False)
    assert done[3].time==pytest.approx(7+.060+.000060+.003)
    assert eng.serviced['link:downlink:rx:port3']==60
    assert eng.serviced['link:uplink:rx:port4']==60
    # Three 20B frames, two physical hops; Tx/Rx capacity demand isn't four hops.
    assert sum(v for k,v in eng.serviced.items() if ':tx:' in k)==120
    assert sum(v for k,v in eng.serviced.items() if ':rx:' in k)==120
    assert eng.summary()['live_tasks']==eng.summary()['live_programs']==0


def test_public_hierarchy_routes_inter_groups_across_two_physical_hops():
    raw=raw_case(signal_bytes=4)
    raw['machine']['num_devices']=5
    raw['runtime']['tp']=4
    raw['machine']['links']=[
        {'name':'local0','members':[0,1],'capacity_bytes_per_ms':1e6},
        {'name':'local1','members':[2,3],'capacity_bytes_per_ms':1e6},
        {'name':'left','members':[0,1,4],'capacity_bytes_per_ms':1000,
         'semantics':'per_port','latency_us':2},
        {'name':'right','members':[4,2,3],'capacity_bytes_per_ms':1000,
         'semantics':'per_port','latency_us':3},
    ]
    report=graph_costs(raw,isolated_calls(raw),isolated=False)
    service=report['execution']['serviced']
    # R4: each of four calls has 128 B data on each inter hop, and three
    # remote signal waves x four directions x 4 B = 48 B per inter hop.
    assert sum(v for k,v in service.items() if k.startswith('link:left:tx:'))==4*(128+48)
    assert sum(v for k,v in service.items() if k.startswith('link:right:rx:'))==4*(128+48)
    assert report['execution']['live_tasks']==report['execution']['live_programs']==0
