"""Actual MX/NZ integer-bank source programs on the common Engine."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.mxfp8_grouped import Mapping, build, owner_tiles, panel_work, recipe
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def raw():
    return json.loads((ROOT / 'examples/cases/high-concurrency-20261030/glm-flash-ws9/'
                      'a5_glm53flash_tp8_c32_p4k.json').read_text())


@pytest.fixture(scope='module')
def actual(raw):
    return next(o for o in target_graphs(raw, phase='decode', tokens=4, context=4096,
                sequences=32, scope='body')[0] if o.op_id.endswith('.expert_down'))


def op(actual, rows, n, k):
    md = dict(actual.metadata)
    md.update(matrix_group_rows=rows, matrix_groups=len(rows), matrix_weight_bank_groups=len(rows),
              activation_scale_shape=(sum(rows), (k+63)//64, 2))
    return replace(actual, local_shape=(len(rows),sum(rows),n,k),
                   metadata=tuple(md.items()))


def price(raw, value, compact=True, ready=0, l1_k=None):
    return operator_costs(raw, (value,), input_ready_ms=ready,
        recipes=RecipeBook({value.recipe:recipe(compact=compact,l1_k=l1_k)}))['operators'][0]


def test_ordinary_full_layer_frontend_has_one_explicit_down_kernel(raw,actual):
    case=json.loads(json.dumps(raw))
    case['runtime']['kernels'].setdefault('recipes',{})['balanced_grouped_down']={
        'kind':'ascend_mxfp8_grouped_nz'}
    result=operator_costs(case,(actual,))['operators'][0]
    meta=result['operator']['metadata']['program_execution']
    assert meta['source_mapping']['output_tiles_by_owner']==(144,)*32
    assert Counter(meta['source_mapping']['group_rows'])=={4:160,3:128}
    assert result['operator']['flops']==2*1024*4096*256
    assert result['serviced']['fixpipe@r0:s0']==pytest.approx(2*1024*4096)
    assert not any('mte3' in key or 'vector' in key for key in result['serviced'])
    assert not result['live_tasks'] and not any(result['remaining_buffers'].values())


def test_last_declared_bank_split_and_zeros_preserve_every_output_element():
    # Independently paint spatial output ownership, including trailing zero banks
    # (the source does not tail-split an earlier nonzero bank).
    for rows in ((3,2),(257,5),(0,9,0),(35,0,17),(7,6,11),(0,0)):
        for n in (32,128,160,256,384,512,768):
            for cores in (3,7,16,32):
                tile=Mapping(256,256,128,256,1)
                owners=owner_tiles(rows,n,tile,cores)
                cells=Counter()
                for owner, items in enumerate(owners):
                    for bank,m,w,mo,no in items:
                        for y in range(mo,mo+m):
                            for x in range(no,no+w):
                                cells[bank,y,x]+=1
                assert len(cells)==sum(rows)*n
                assert all(v==1 for v in cells.values())


def test_rolling_owner_cursor_does_not_restart_at_each_expert():
    owners=owner_tiles((1,1,1,1),4096,Mapping(256,256,128,256,1),32)
    assert all([t[0] for t in owner]==([0,2] if i<16 else [1,3])
               for i,owner in enumerate(owners))


@pytest.mark.parametrize('rows,n,k', [((0,3,0,17),256,256),((1,3),160,193),
                                     ((257,7),384,320),((0,0),128,64)])
def test_source_work_scale_pairs_and_late_input(raw,actual,rows,n,k):
    value=op(actual,rows,n,k)
    result=price(raw,value,ready=7,l1_k=256)
    plan=build(value,Geometry.from_devices(parse_case(raw).machine.dies),l1_k=256)
    # Physical work is compared to the executed Engine rather than semantic
    # OpSpec estimates, which intentionally exclude padded K arithmetic.
    for role,amount in plan.work.resources.items():
        name=('memory' if role=='hbm' else role)+'@r0:s0'
        assert result['serviced'].get(name,0)==pytest.approx(amount)
    assert result['operator']['flops']==2*sum(rows)*n*((k+63)//64*64)
    assert not result['service_before_input'] and not result['live_tasks']
    assert not any(result['remaining_buffers'].values())


def test_m16_load_payload_does_not_pad_actual_mmad_rows(raw,actual):
    a=build(op(actual,(1,1),4096,256),Geometry.from_devices(parse_case(raw).machine.dies))
    b=build(op(actual,(7,7),4096,256),Geometry.from_devices(parse_case(raw).machine.dies))
    assert a.work.resources['mte1']==b.work.resources['mte1']
    assert b.work.matrix['fp8_e4m3']==7*a.work.matrix['fp8_e4m3']
    assert b.work.resources['fixpipe']==7*a.work.resources['fixpipe']


@pytest.mark.parametrize('rows,n,k', [((1,3),256,64),((3,7),384,192),((3,7),384,193),((257,7),384,320)])
def test_compact_and_detailed_same_owner_pipeline(raw,actual,rows,n,k):
    base=parse_case(raw)
    dies=tuple(replace(d,compute_units=1,kernel_invocations=(),
        resource_units=tuple((key,1) for key,_ in d.resource_units),
        buffer_units=tuple((key,1) for key,_ in d.buffer_units)) for d in base.machine.dies)
    case=replace(base,machine=replace(base.machine,dies=dies))
    value=op(actual,rows,n,k)
    compact=price(case,value,l1_k=256);detailed=price(case,value,compact=False,l1_k=256)
    assert compact['serviced']==pytest.approx(detailed['serviced'])
    assert compact['duration_ms']==pytest.approx(detailed['duration_ms'],rel=1e-10)
    assert compact['peak_buffers_per_unit']==detailed['peak_buffers_per_unit']


@pytest.mark.parametrize('axis',['hbm','mte2_aic','mte1','fixpipe','cube'])
def test_physical_service_slowdown_changes_runtime_without_work_change(raw,actual,axis):
    base=parse_case(raw);value=op(actual,(2,3),512,256)
    normal=price(base,value)
    dies=tuple(replace(d,rates=tuple((k,v*(.001 if k==axis else 1)) for k,v in d.rates),
        compute_peaks=tuple((k,v*(.001 if axis=='cube' else 1)) for k,v in d.compute_peaks))
        for d in base.machine.dies)
    slow=price(replace(base,machine=replace(base.machine,dies=dies)),value)
    assert slow['duration_ms']>normal['duration_ms']
    assert slow['operator']['flops']==normal['operator']['flops']
    assert slow['operator']['hbm_bytes']==normal['operator']['hbm_bytes']


def test_one_invocation_despite_many_banks_owners_and_k_panels(raw,actual):
    from mindie_sim.schema.invocation import InvocationTiming
    base=parse_case(raw)
    dies=tuple(replace(d,kernel_invocations=(('cube',InvocationTiming(minimum_ms=1)),))
               for d in base.machine.dies)
    value=op(actual,(16,16),512,320)
    result=price(replace(base,machine=replace(base.machine,dies=dies)),value,ready=7,l1_k=256)
    assert result['duration_ms']==pytest.approx(1)
    assert not result['service_before_input']


def test_transposed_nz_l0b_tail_has_n32_payload_but_n16_scales():
    load, steps=panel_work(3,33,128,128)
    assert load.resources['mte2_aic']==3*128+64*128+(3+33)*4
    assert steps[0][0].resources['mte1']==(16+64)*128+(16+48)*4
    assert steps[0][1].matrix['fp8_e4m3']==2*3*33*128


def source_owner_oracle(rows,n,tile,cores):
    """Literal per-block GetTileIdx/GetBlockShape arithmetic, not list painting."""
    owners=[[] for _ in range(cores)];end=cores-1
    for bank,m in enumerate(rows):
        if not m:continue
        bm=((m+((m+tile.m-1)//tile.m)-1)//((m+tile.m-1)//tile.m)+15)//16*16
        nm=(m+bm-1)//bm;nn=(n+tile.n-1)//tile.n;total=nm*nn
        mtail=m-(nm-1)*bm;ntail=n-(nn-1)*tile.n
        main=min(4,nm);mainrow=nm//main-1;tw=nm-main*mainrow
        start=(end+1)%cores;end=(start+total-1)%cores
        oldend=end;count=min(end+1,total);ms=ns=1
        if bank==len(rows)-1 and end+1<=cores//2:
            remain=(cores-end-1)//count+1
            ms=min((mtail+15)//16,remain);ns=min((ntail+127)//128,remain)
            while ms*ns>remain:
                if ms>=ns:ms-=1
                else:ns-=1
        split=ms*ns;newend=end+count*(split-1);base=end+1-count if split>1 else 0
        for owner in range(cores):
            rounds=(total+cores-1)//cores
            if (start>end and end<owner<start) or (start<=end and (owner>end or owner<start)):
                rounds-=1
            if oldend<owner<=newend:rounds+=1
            local_split=split if owner<=newend else 1
            for r in range(rounds):
                mapped=owner
                if r==rounds-1 and local_split>1:
                    mapped=(base+((owner-base)//local_split)*local_split)//local_split
                index=mapped+r*cores
                if owner<start:index+=cores-start
                elif local_split>1 and newend+1>=local_split*total:
                    index-=(base+((start-base)//local_split)*local_split)//local_split
                else:index-=start
                window=index//nn//main
                if window<mainrow:mi=window*main+index%main;ni=(index//main)%nn
                else:
                    window=mainrow;j=index-mainrow*main*nn;mi=mainrow*main+j%tw;ni=(j//tw)%nn
                if window%2:ni=nn-1-ni
                mm=bm if mi!=nm-1 else mtail;width=tile.n if ni!=nn-1 else ntail
                mo=no=0
                if local_split>1 and r==rounds-1:
                    sm=(mm+ms-1)//ms;sn=((width+ns-1)//ns+127)//128*128
                    mo=((owner%local_split)%ms)*sm;no=((owner%local_split)//ms)*sn
                    if mo>=mm or no>=width:continue
                    mm=min(mm-mo,sm);width=min(width-no,sn)
                owners[owner].append((bank,mm,width,mi*bm+mo,ni*tile.n+no))
        end=newend
    return tuple(tuple(v) for v in owners)


def test_rolling_aswt_matches_literal_source_block_index_oracle():
    tile=Mapping(256,256,128,256,1)
    for rows in ((1,3),(257,9),(0,31,0),(17,515,23),(1031,17,3),(9,19,29,49)):
        for n in (32,160,384,768,1280):
            for cores in (3,7,16,32):
                assert owner_tiles(rows,n,tile,cores)==source_owner_oracle(rows,n,tile,cores)
