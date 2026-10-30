"""Source collective work, actual ordinary callers and finite owner controls."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs, predict
from mindie_sim.prediction.hccl_aiv import contract, jobs
from mindie_sim.prediction.hccl_contracts import select
from test_prediction_system import case
from test_joint_collective_execution import wire


def declared(expansion='AIV', template='mesh_oneshot', *, ranks=2, blocks=64):
    raw=case(tp=ranks)
    raw['machine']['device'].update(
        rates={'memory':1e9,'vector':1e9,'mte2_aiv':1e9,'mte2_aic':1e9,'mte3':1e9},
        buffers={'ub':256*1024},
        resource_units={'vector':64,'mte2_aiv':64,'mte3':64},
        buffer_units={'ub':64})
    raw['machine']['links'][0].update(capacity_bytes_per_ms=1e6,latency_us=5)
    raw['runtime']['kernels']={'hccl':{'allreduce':{'expansion':expansion,'template':template,
                                                  **({'block_limit':blocks} if expansion=='AIV' else {})}}}
    return raw


def isolated_calls(raw):
    graphs=target_graphs(raw,phase='decode',tokens=1,context=16,scope='body')
    selected={}
    for rank,ops in graphs.items():
        keep=[op for op in ops if 'hccl_source_group' in dict(op.metadata)]
        ids={op.op_id for op in keep}
        selected[rank]=tuple(replace(op,deps=tuple(d for d in op.deps if d in ids)) for op in keep)
    return selected


@pytest.mark.parametrize('expansion,template',[
    ('AIV','mesh_oneshot'),('AIV','mesh_twoshot'),
    ('AICPU_TS','mesh_oneshot'),('AICPU_TS','mesh_twoshot'),
    ('AICPU_TS','nhr'),('CCU_MS','mesh_oneshot_v1'),('CCU_MS','mesh_oneshot_v2')])
def test_ordinary_caller_preserves_wire_work_once_and_one_invocation(expansion,template):
    raw=declared(expansion,template)
    # N64 is divisible by R, but not the two-shot 60-slice group. Use 60 rows.
    raw['model']['hidden_size']=960
    raw['machine']['device']['launch_us']=3
    selected=isolated_calls(raw)
    groups={dict(op.metadata)['hccl_source_group'] for op in selected[0]}
    assert len(groups)==4  # attention+FFN in each of two layers
    report=graph_costs(raw,selected,include_submission=True,isolated=True,input_ready_ms=7)
    per_rank=1920 if template=='mesh_oneshot' else 1920
    assert sum(wire(report).values())==4*2*per_rank
    assert report['execution']['submission']['eager_invocations']==4
    assert report['execution']['live_tasks']==report['execution']['live_programs']==0
    assert all(row['ready_ms']>=7 for row in report['timeline'])
    assert report['bounds']['combined_ms']<=report['duration_ms']


@pytest.mark.parametrize('expansion,template', [
    ('AIV','mesh'),
    ('AICPU_TS','nhr'),('CCU_SCHED','mesh_mem2mem_v1'),('CCU_SCHED','mesh_mem2mem_v2')])
def test_gather_program_is_reached_through_the_selected_frontend(expansion, template):
    raw=declared(expansion, template)
    declaration={'expansion':expansion,'template':template,
                 **({'block_limit':64} if expansion=='AIV' else {})}
    raw['runtime']['kernels']['hccl']={'allgather':declaration}
    # Reuse the ordinary caller's exact communication tensor and dependencies.
    from mindie_sim.prediction.hccl_graph import lower
    from mindie_sim.prediction.structure import parse_case
    raw['runtime']['kernels'].pop('hccl')
    ops=target_graphs(raw,phase='decode',tokens=1,context=16,scope='body')[0]
    original=next(op for op in ops if op.family=='allreduce')
    gather=replace(original, family='allgather', recipe='allgather', deps=(),
                   metadata=tuple({**dict(original.metadata), 'collective':'allgather'}.items()))
    raw['runtime']['kernels']['hccl']={'allgather':declaration}
    parsed=parse_case(raw)
    selected=lower((gather,),parsed.runtime,parsed.model.activation_dtype)
    r=graph_costs(raw,selected,input_ready_ms=7,isolated=False)
    assert r['duration_ms']>0
    assert r['execution']['live_tasks']==r['execution']['live_programs']==0
    assert all(row['ready_ms']>=7 for row in r['timeline'])
    assert any(row['recipe'] in {'hccl_ccu_source_phase','hccl_aicpu_source_phase',
                                'hccl_aiv_gather_source_phase'}
               for row in r['timeline'])


def test_oneshot_r_plus_one_owners_and_rmw_oracle():
    for r in (2,4,8):
        s=select({'expansion':'AIV','template':'mesh_oneshot','block_limit':64},
                 collective='allreduce',ranks=r,payload_bytes=1024)
        p=contract(s)
        assert p.thread_count==r+1
        assert sum(dict(x.work).get('hbm',0) for x in p.phases)==(5*r-1)*1024
        assert sum(dict(x.work).get('mte2_aiv',0) for x in p.phases)==2*r*1024
        assert not any('vector' in dict(x.work) for x in p.phases)
        assert sum(sum(row) for row in p.phases[0].exchange)==r*(r-1)*1024


def test_twoshot_consumers_and_local_reduce_are_separate_owner_domains():
    s=select({'expansion':'AIV','template':'mesh_twoshot','block_limit':64},
             collective='allreduce',ranks=2,payload_bytes=3932160)
    owner_jobs,count,_=jobs(s,64)
    assert count==63
    assert len(owner_jobs['precopy'])==2
    assert len(owner_jobs['get_remote'])==60
    assert len(owner_jobs['reduce'])==30
    assert 62 not in set().union(*(set(x) for x in owner_jobs.values()))
    p=contract(s)
    assert sum(dict(x.work)['hbm'] for x in p.phases)==(6+3/2)*s.payload_bytes
    assert sum(sum(sum(row) for row in x.exchange) for x in p.phases)==2*s.payload_bytes


@pytest.mark.parametrize('payload', [2, 30, 65536, 1048576])
def test_twoshot_integer_tail_keeps_exact_transport_and_critical_local_shard(payload):
    from mindie_sim.prediction.hccl_graph import lower
    from mindie_sim.prediction.structure import parse_case
    from test_communication_topology import exchange
    raw=declared(template='mesh_twoshot',ranks=4,blocks=60)
    parsed=parse_case(raw)
    original=replace(exchange(2*3/4*payload, family='allreduce',axis='tp'),deps=())
    selected=lower((original,),parsed.runtime,parsed.model.activation_dtype)
    report=graph_costs(raw,selected,input_ready_ms=7,trace_limit=10000)
    assert sum(wire(report).values()) == pytest.approx(2*3*payload, rel=1e-12)
    assert report['execution']['live_tasks']==report['execution']['live_programs']==0
    assert not any(report['execution']['live_allocated'].values())
    assert all(row['start_ms'] >= 7 for row in report['events'] if row['demand'])
    phases=[op for op in selected if op.family=='alltoallv']
    for op in phases:
        rows=dict(op.metadata)['peer_traffic']['payload_bytes']
        assert op.link_bytes == sum(map(sum,rows))/4
    s=select({'expansion':'AIV','template':'mesh_twoshot','block_limit':60},
             collective='allreduce',ranks=4,payload_bytes=payload)
    local=[contract(s,rank=i) for i in range(4)]
    for index,p in enumerate(local[0].phases):
        assert dict(p.work)['hbm'] == max(dict(x.phases[index].work)['hbm'] for x in local)


def test_aiv_bandwidth_and_network_sensitivity_preserve_demand():
    raw=declared(ranks=2)
    selected=isolated_calls(raw)
    reports=[]
    for factor in (.5,1,2):
        varied=deepcopy(raw)
        for role in ('mte2_aiv','mte3'):
            varied['machine']['device']['rates'][role]*=factor
        reports.append(graph_costs(varied,selected,isolated=False))
    assert reports[0]['duration_ms']>reports[1]['duration_ms']>reports[2]['duration_ms']
    assert reports[0]['execution']['serviced']==reports[1]['execution']['serviced']==reports[2]['execution']['serviced']
    slow=deepcopy(raw);slow['machine']['links'][0]['capacity_bytes_per_ms']=1
    assert graph_costs(slow,selected,isolated=False)['duration_ms']>reports[1]['duration_ms']
    # Public body path installs the same mechanism, with no observed data input.
    result=predict(raw,scope='body',phase='decode',tokens=1,context=16,sequences=1)
    assert result['status']=='modeled'
    assert any(row['family']=='hccl_local' for row in result['timeline'])
