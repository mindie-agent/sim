"""Ordinary mixed FP8 gate/up boundaries, paired owners and actual source work."""
from copy import deepcopy
from dataclasses import replace
from collections import Counter
import json
from pathlib import Path

import pytest

from mindie_sim import target_graphs, operator_costs, graph_costs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.mxfp8_grouped import Mapping
from mindie_sim.prediction.mxfp8_swiglu import PROFILE, ROLE, build, owner_tiles, vector_work
from mindie_sim.prediction.structure import parse_case

ROOT=Path(__file__).resolve().parents[1]


def case(batch=32,compact=True):
    raw=json.loads((ROOT/f'examples/cases/high-concurrency-20261030/glm-flash-ws9/a5_glm53flash_tp8_c{batch}_p4k.json').read_text())
    raw['runtime']['kernels'].setdefault('lowerings',{})['native_fusion']={
        'implementation':'declared_native_fusion','options':{
            'roles':[ROLE],'program_profiles':{ROLE:PROFILE},
            'program_options':{ROLE:{'l1_k':512,'compact':compact}}}}
    return raw


@pytest.fixture(scope='module')
def sample():
    raw=case()
    ops=target_graphs(raw,scope='body',phase='decode',tokens=4,context=4096,sequences=32)[0]
    return raw,next(op for op in ops if op.recipe==PROFILE),ops


def changed(op,rows,width,k,compact=True):
    md=deepcopy(dict(op.metadata));md['mxfp8_swiglu_config'].update(rows=rows,width=width,k=k,compact=compact)
    return replace(op,local_shape=(sum(rows),width),metadata=tuple(md.items()))


def test_full_frontend_fuses_only_gate_up_activation_quant_and_keeps_down(sample):
    raw,first,ops=sample
    generic=deepcopy(raw);generic['runtime']['kernels']['lowerings'].pop('native_fusion')
    semantic=target_graphs(generic,scope='body',phase='decode',tokens=4,context=4096,sequences=32)[0]
    fused=[op for op in ops if op.recipe==PROFILE]
    assert len(fused)==42
    source_down=[op for op in ops if op.op_id.endswith('.expert_down')]
    ordinary_down=[op for op in semantic if op.op_id.endswith('.expert_down')]
    assert len(source_down)==len(ordinary_down)==42
    for source,original in zip(source_down,ordinary_down):
        assert replace(source,deps=original.deps)==original
        assert len(source.deps)==1 and source.deps[0] in {op.op_id for op in fused}
    ids={op.op_id for op in ops}
    for op in fused:
        md=dict(op.metadata)
        assert not ids.intersection(md['fusion_semantic_ids'])
        assert len(md['fusion_semantic_ids'])==3
        assert md['fusion_boundary']['submitted_kernels']==1
    rows=dict(first.metadata)['mxfp8_swiglu_config']['rows']
    assert Counter(rows)=={4:160,3:128}


def test_source_fixed_base_and_rolling_bank_owner_covers_all_output_elements():
    tile=Mapping(128,256,128,512,2)
    for rows in ((0,1,0,3),(129,257),(2,3,4,5)):
        for width in (64,256,320):
            owners=owner_tiles(rows,width,tile,3)
            cells=Counter((bank,y,x) for items in owners for bank,m,n,mo,no in items
                          for y in range(mo,mo+m) for x in range(no,no+n))
            assert len(cells)==sum(rows)*width and set(cells.values())=={1}
    owners=owner_tiles((1,1,1,1,1),256,tile,3)
    assert [[item[0] for item in values] for values in owners]==[[0,3],[1,4],[2]]


@pytest.mark.parametrize('rows,width,k',[((0,3,9),64,128),((1,),256,640),((129,2),320,256)])
def test_detailed_and_compact_conserve_both_matrices_local_fp32_and_paired_stores(sample,rows,width,k):
    raw,original,_=sample
    values=[]
    for compact in (False,True):
        op=changed(original,rows,width,k,compact)
        plan=build(op,Geometry.from_devices(parse_case(raw).machine.dies))
        row=operator_costs(raw,[op],input_ready_ms=11)['operators'][0]
        assert row['operator']['flops']==4*sum(rows)*width*k
        assert plan.work.resources['fixpipe']==8*sum(rows)*width
        assert plan.work.resources['mte3']==sum(rows)*(width+width//32)
        for role,amount in plan.work.resources.items():
            key=('memory' if role=='hbm' else role)+'@r0:s0'
            assert row['serviced'][key]==pytest.approx(amount)
        assert not row['service_before_input'] and not row['live_tasks']
        assert not any(row['remaining_buffers'].values())
        values.append(row)
    assert values[0]['serviced']==pytest.approx(values[1]['serviced'])
    assert abs(values[0]['duration_ms']-values[1]['duration_ms'])/values[0]['duration_ms']<.15


def test_two_aiv_subblocks_keep_unequal_rows_and_next_tile_waits_for_both(sample):
    raw,original,_=sample
    op=changed(original,(3,)*33,64,128)
    report=graph_costs(raw,[replace(op,deps=())],isolated=False,input_ready_ms=7,trace_limit=10000)
    events=report['events']
    vec=[e for e in events if e['task'].endswith(':swiglu_mx')]
    assert len(vec)==66
    observed=Counter()
    for event in vec:
        units=event['placement']['vector@r0:s0']
        assert len(units)==1
        observed[units[0]]+=1
    assert observed[0]==observed[1]==2 and all(observed[i]==1 for i in range(2,64))
    owner0=sorted((event for event in vec if tuple(event['placement']['vector@r0:s0'])==(0,)),key=lambda e:e['start_ms'])
    assert owner0[-1]['start_ms']>=owner0[0]['end_ms']
    assert report['execution']['live_tasks']==report['execution']['live_programs']==0


def test_real_batch_matrix_has_one_invocation_and_no_fp32_hbm_scratch(sample):
    raw,op,_=sample
    parsed=parse_case(raw);plan=build(op,Geometry.from_devices(parsed.machine.dies))
    row=operator_costs(raw,[op])['operators'][0]
    md=row['operator']['metadata']['program_execution']
    assert md['source_output_tiles']==288 and md['active_aic_owners']==32
    assert row['operator']['flops']==4*1024*256*4096
    assert row['operator']['mte3_bytes']==1024*(256+8)
    assert plan.work.resources['fixpipe']==1024*512*4
    assert max(value for key,value in row['peak_buffers_per_unit'].items() if key.startswith('ub@'))==184832


def test_real_resource_scaling_is_monotone_without_changing_work(sample):
    raw,op,_=sample
    original=parse_case(raw)
    for roles,compute in [({'hbm'},False),({'mte1','mte2_aic','fixpipe'},True),({'vector','mte3'},False)]:
        values=[];work=[]
        for factor in (.5,1,2):
            dies=tuple(replace(d,rates=tuple((k,v*factor if k in roles else v) for k,v in d.rates),
                compute_peaks=tuple((k,v*factor if compute else v) for k,v in d.compute_peaks)) for d in original.machine.dies)
            arm=replace(original,machine=replace(original.machine,dies=dies))
            row=operator_costs(arm,[op])['operators'][0]
            values.append(row['duration_ms']);work.append(row['operator']['hbm_bytes'])
        assert values[0]>=values[1]>=values[2] and work[0]==work[1]==work[2]


def test_extra_materialization_cannot_disappear_inside_selected_three_node_program(sample):
    from mindie_sim.prediction.family_mxfp8_swiglu import bind
    from mindie_sim.prediction.native_fusion import _groups
    from mindie_sim.prediction.implementations import select
    from mindie_sim.schema.common import ModelingError
    raw,_,_=sample
    parsed=parse_case(raw)
    ops=target_graphs(parsed,scope='body',phase='decode',tokens=4,context=4096,
                      sequences=32,resource_fusion=False)[0]
    declaration,members=next((d,m) for d,m in _groups(ops) if d['role']==ROLE)
    selection=select(parsed.runtime.kernels['lowerings'],'native_fusion')
    copy=replace(members[0],op_id='additional_copy',family='copy',metadata=(),flops=0)
    with pytest.raises(ModelingError,match='requires_gate_up_SwiGLU_quantizer'):
        bind(declaration,[*members,copy],[members[-1].op_id],selection,{'l1_k':512})
