"""Dense latent cache traffic, batch ownership, tails and caller boundaries."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.paged_mla import build, segments, IMPLEMENTATION, BUFFERS
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'examples/cases/high-concurrency-20261030/kimi-source/a5_kimi_tp8_e16_bf16_c1_p4k_source.json'


def case(splits=1):
    raw = json.loads(CASE.read_text())
    raw['runtime']['kernels']['lowerings']['paged_mla_attention'] = {
        'implementation': IMPLEMENTATION, 'options': {'kv_splits': splits}}
    return raw


def attention(raw, batch=1, context=4096):
    graph = target_graphs(raw, scope='body', phase='decode', tokens=1,
                          sequences=batch, context=context)[0]
    return next(o for o in graph if o.recipe == IMPLEMENTATION), graph


def execute(raw, op):
    row = operator_costs(raw, [op], input_ready_ms=7)['operators'][0]
    assert not row['service_before_input'] and row['live_tasks'] == 0
    assert all(abs(n) < 1e-7 for n in row['remaining_buffers'].values())
    return row


@pytest.mark.parametrize('batch,context,splits', [(1,4096,1), (1,16384,4),
                                                (32,4096,1), (32,16384,4)])
def test_physical_latent_is_loaded_once_and_final_output_once(batch, context, splits):
    raw = case(splits)
    op, _ = attention(raw, batch, context)
    parsed = parse_case(raw)
    plan = build(op, Geometry.from_devices(parsed.machine.dies))
    physical = dict(op.metadata)['paged_mla_geometry']['heads']
    spans = segments((context,)*batch, splits)
    # Shared latent+rope cache: never a second V HBM read or query-head multiplier.
    kv = batch*(context+1)*576*2
    queries = len(spans)*physical*576*2
    pages = sum(((min((start+count)*128,context+1)+127)//128-start)*4
                for _, start, count, _ in spans)
    output = batch*physical*512*2
    partial_roundtrip = sum(2*physical*(512*4+8) for *_, count in spans if count > 1)
    assert plan.work.resources['hbm'] == kv+queries+pages+output+partial_roundtrip
    row = execute(raw, op)
    assert row['operator']['hbm_bytes'] == plan.work.resources['hbm']
    assert dict(row['operator']['metadata']['program_execution'])['query_sequences'] == batch


def test_batch_is_one_kernel_and_caller_head_materializations_are_not_duplicated():
    raw = case(4)
    op, graph = attention(raw, 32, 129)
    layer = [o for o in graph if o.layer == 3]
    assert len([o for o in layer if o.recipe == IMPLEMENTATION]) == 1
    assert not any(o.recipe.startswith('paged_mla_query_pad') for o in layer)
    assert sum(o.recipe == 'mla_query_pad_nope' for o in layer) == 1
    assert dict(op.metadata)['attention_tensor_view']['output_shape'] == (16,32,1,512)
    assert dict(next(o for o in layer if o.recipe == 'v_expand').metadata)['matrix_groups'] == 12
    base = execute(raw, attention(raw, 1, 4096)[0])['duration_ms']
    batched = execute(raw, attention(raw, 32, 4096)[0])['duration_ms']
    assert batched < 8*base  # Shared owners, not B serial launches.


def test_tail_sequence_horizons_and_physical_bandwidth_response():
    raw = case(4)
    op, _ = attention(raw, 3, 256)
    meta = dict(op.metadata)
    meta['paged_mla_geometry'] = {**meta['paged_mla_geometry'], 'contexts': (0,129,256)}
    op = replace(op, metadata=tuple(meta.items()))
    assert [sum(count for seq,_,count,_ in segments((0,129,256),4) if seq == i)
            for i in range(3)] == [1,2,3]
    times = []
    for scale in (.5, 1, 2):
        arm = deepcopy(raw)
        # Normalized declared physical resources make the tested axis explicit.
        machine = parse_case(arm).machine.to_dict()
        for device in machine['devices']:
            device['rates'] = {k: v*scale for k,v in device['rates'].items()}
            device['compute_peaks'] = {k: v*scale for k,v in device['compute_peaks'].items()}
        arm['machine'] = machine
        times.append(execute(arm, op)['duration_ms'])
    assert times[0] > times[1] > times[2]


def test_two_physical_owners_keep_merge_buffers_and_serialize_same_owner_sequences():
    raw = case(2)
    op, _ = attention(raw, 5, 256)
    parsed = parse_case(raw)
    dies = tuple(replace(d, compute_units=2,
                         resource_units=tuple((k, (4 if v == 2*d.compute_units else 2)
                                               if v in (d.compute_units,2*d.compute_units) else v)
                                              for k,v in d.resource_units),
                         buffer_units=tuple((k,4 if k == 'ub' else 2)
                                            for k,_ in d.buffer_units)) for d in parsed.machine.dies)
    parsed = replace(parsed, machine=replace(parsed.machine, dies=dies))
    result = graph_costs(parsed, [replace(op,deps=())], isolated=False,
                         input_ready_ms=7, trace_limit=10000)
    trace = result['events']
    release = [x for x in trace if x['task'].endswith(':release')]
    merge_writes = [x for x in trace if ':merge:' in x['task'] and x['task'].endswith(':write')]
    assert len(release) == 1 and len(merge_writes) == 5
    assert min(x['start_ms'] for x in release) >= max(x['end_ms'] for x in merge_writes)
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert not any(result['execution']['live_allocated'].values())
    for role in ('mte2_aic','mte1','fixpipe'):
        owners = {u for x in trace for k,p in x['placement'].items() if role in k for u in p}
        assert owners <= {0,1}


@pytest.mark.parametrize('heads', [1,2,16,31,32,33,48,63,64])
def test_vector_subblocks_cannot_borrow_idle_sibling_under_external_contention(heads):
    op=OpSpec('mla','attention','decode',0,0,(),(),(),(),(),metadata=(
        ('paged_mla_geometry',{'contexts':(0,), 'heads':heads,'kv_splits':1,'page_size':128}),))
    units={r:(2 if r in ('vector','mte2_aiv','mte3') else 1)
           for r in ('hbm','vector','cube','mte1','mte2_aic','mte2_aiv','mte3','fixpipe')}
    rates={r:(2 if r=='vector' else 1e20) for r in units}
    rates['cube']=1
    pools={p:max(n,1<<20) for p,n in BUFFERS.items()}
    pu={p:2 if p=='ub' else 1 for p in pools}
    plan=build(op,Geometry(1,units,pools,pu))
    engine=Engine([Resource(r,v,units=units[r]) for r,v in rates.items()],
                  pools=pools,pool_units=pu)
    ctx=RecipeContext(plan.work.apply(op),{}, {r:r for r in rates},rates,1,
        resource_units=units,compute_peaks={'bf16':1e20},pool_names={p:p for p in pools},
        pool_capacities=pools,pool_units=pu)
    ready=engine.submit('input',release=7)
    competitor=engine.submit('external',{'vector':plan.work.resources['vector']*3},
                            after=(ready,),placement={'vector':(0,)})
    done=plan.lower(ctx).start(engine,'mla',ready)
    engine.advance(until=(done,))
    # Source InitTask partitions actualM, not the capacity M64.
    expected=plan.work.resources['vector']*((heads+1)//2)/heads*2
    assert done.time-7 == pytest.approx(expected,rel=1e-12)
    engine.advance(until=(competitor,))
    assert not any(engine.allocated.values())


@pytest.mark.parametrize('batch', [1,4,32])
def test_retained_host_segments_preserve_cross_sequence_owners_and_fd_head_rows(batch):
    path = ROOT / ('examples/cases/high-concurrency-20261030/kimi-source-paged-host/'
                   f'a5_kimi_tp8_e16_bf16_c{batch}_p4k_source.json')
    raw=json.loads(path.read_text());op,_=attention(raw,batch,4096)
    declaration=dict(op.metadata)['paged_mla_geometry']['dispatch']
    owners=declaration['aic_owner_segments']
    coverage={}
    for owner,spans in enumerate(owners):
        for seq,start,count in spans:
            for tile in range(start,start+count):
                assert (seq,tile) not in coverage
                coverage[seq,tile]=owner
    assert set(coverage)=={(seq,tile) for seq in range(batch) for tile in range(33)}
    if batch==4:
        assert any(len(spans)>1 for spans in owners)  # Host crosses a sequence boundary.
    parsed=parse_case(raw);plan=build(op,Geometry.from_devices(parsed.machine.dies))
    result=execute(raw,op)
    assert result['operator']['hbm_bytes']==plan.work.resources['hbm']
    assert result['operator']['metadata']['program_execution']['active_aic_owners']==len(owners)
    if batch==1:
        report=graph_costs(raw,[replace(op,deps=())],isolated=False,input_ready_ms=7,trace_limit=15000)
        writes=[row for row in report['events'] if ':merge:' in row['task'] and row['task'].endswith(':write')]
        assert len(writes)==16
        actual={row['placement']['mte3@r0:s0'][0]:row['demand']['memory@r0:s0'] for row in writes}
        assert actual=={aiv:heads*512*2 for aiv,(_,_,heads) in enumerate(declaration['aiv_merge_segments'])}


def test_dispatch_rejects_reused_context_or_duplicated_physical_tile():
    path = ROOT / ('examples/cases/high-concurrency-20261030/kimi-source-paged-host/'
                   'a5_kimi_tp8_e16_bf16_c4_p4k_source.json')
    raw=json.loads(path.read_text());op,_=attention(raw,4,4096)
    parsed=parse_case(raw);geometry=Geometry.from_devices(parsed.machine.dies)
    meta=deepcopy(dict(op.metadata))
    meta['paged_mla_geometry']['dispatch']['aic_owner_segments'][1][0][1]-=1
    with pytest.raises(ModelingError,match='overlap_or_gap'):
        build(replace(op,metadata=tuple(meta.items())),geometry)
    meta=deepcopy(dict(op.metadata));meta['paged_mla_geometry']['contexts']=(4097,)*4
    with pytest.raises(ModelingError,match='geometry_mismatch'):
        build(replace(op,metadata=tuple(meta.items())),geometry)


@pytest.mark.parametrize('context', [0,128,256,383,519])
def test_source_preload_two_preserves_cube_command_order_and_tail_work(context):
    raw=case(1)
    op,_=attention(raw,1,context)
    parsed=parse_case(raw)
    dies=tuple(replace(d,compute_units=1,
        resource_units=tuple((k,2 if v==2*d.compute_units else 1)
                             for k,v in d.resource_units),
        buffer_units=tuple((k,2 if k=='ub' else 1) for k,_ in d.buffer_units))
        for d in parsed.machine.dies)
    parsed=replace(parsed,machine=replace(parsed.machine,dies=dies))
    result=graph_costs(parsed,[replace(op,deps=())],isolated=False,
                       input_ready_ms=7,trace_limit=10000)
    commands={}
    for event in result['events']:
        name=event['task']
        if ':tiles:program:tile' in name and name.endswith((':qk',':pv')) and event['demand']:
            iteration=int(name.split(':tiles:program:tile')[1].split(':')[0])
            commands[iteration,name.rsplit(':',1)[1]]=event
    tiles=(context+1+127)//128
    # Retained ExecuteTask does C1(i) then C2(i-1); this is not two free
    # streams sharing Cube bandwidth. The final iteration is a real drain.
    order=[]
    for iteration in range(tiles+1):
        if iteration<tiles: order.append(commands[iteration,'qk'])
        if iteration: order.append(commands[iteration,'pv'])
    assert len(order)==2*tiles
    for before,after in zip(order,order[1:]):
        assert after['start_ms']>=before['end_ms']-1e-12
    assert order[0]['start_ms']>=7
    plan=build(op,Geometry.from_devices(dies))
    assert sum(v for e in result['events'] for k,v in e['serviced'].items()
               if k=='memory@r0:s0')==pytest.approx(plan.work.resources['hbm'])
    assert result['execution']['live_tasks']==result['execution']['live_programs']==0
    assert not any(result['execution']['live_allocated'].values())


@pytest.mark.parametrize('context', [0,128,519])
@pytest.mark.parametrize('supply', [.5,1,2])
def test_source_cube_driver_waits_for_each_mma_before_next_macro_copy(context,supply):
    # In IterateBmm1/2 the M_FIX wait is inside the Cube driver call. With
    # all other services effectively free, copy + both MMAs are additive,
    # although the independent AIV and Fixpipe paths still overlap normally.
    op=OpSpec('mla','attention','decode',0,0,(),(),(),(),(),metadata=(
        ('paged_mla_geometry',{'contexts':(context,), 'heads':16,'kv_splits':1,'page_size':128}),))
    units={r:(2 if r in ('vector','mte2_aiv','mte3') else 1)
           for r in ('hbm','vector','cube','mte1','mte2_aic','mte2_aiv','mte3','fixpipe')}
    rates={r:1e25 for r in units};rates['cube']=1;rates['mte2_aic']=1e6*supply
    peak=1e11*supply
    pools={p:max(n,1<<20) for p,n in BUFFERS.items()};pu={p:2 if p=='ub' else 1 for p in pools}
    plan=build(op,Geometry(1,units,pools,pu))
    engine=Engine([Resource(r,v,units=units[r]) for r,v in rates.items()],pools=pools,pool_units=pu)
    ctx=RecipeContext(plan.work.apply(op),{}, {r:r for r in rates},rates,1,
        resource_units=units,compute_peaks={'bf16':peak},pool_names={p:p for p in pools},
        pool_capacities=pools,pool_units=pu)
    ready=engine.submit('input',release=7)
    done=plan.lower(ctx).start(engine,'mla',ready)
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    expected=plan.work.resources['mte2_aic']/rates['mte2_aic']+1000*plan.work.matrix['bf16']/peak
    assert done.time-7==pytest.approx(expected,rel=1e-10)
    for resource,amount in ctx.bind(plan.work).items():
        assert engine.serviced[resource]==pytest.approx(amount,rel=1e-12)
    assert not any(engine.allocated.values()) and engine.summary()['live_tasks']==0
