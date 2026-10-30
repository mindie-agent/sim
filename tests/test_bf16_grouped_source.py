"""Standard nonquant grouped source work and finite Engine controls."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.bf16_grouped import Mapping, owner_tiles, recipe
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def raw():
    case = json.loads((ROOT / 'examples/cases/high-concurrency-20261030/kimi-variants/'
                      'a5_kimi_tp8_e16_bf16_c32_p4k.json').read_text())
    case['runtime']['kernels'].setdefault('families', {})['grouped_gemm'] = {
        'kind': 'ascend_bf16_grouped'}
    return case


def op(rows, n, k):
    value = gemm_op(op_id='source', deps=(), stage=0, layer=1, requests=(),
                    m=sum(rows), n=n, k=k, quant=QuantSpec('bf16'), weight_count=sum(v>0 for v in rows),
                    category='routed_expert', phase='decode', family='grouped_gemm')
    return replace(value, metadata=tuple({**dict(value.metadata), 'matrix_group_rows': rows}.items()))


def price(raw, value, compact=True, ready=0):
    return operator_costs(raw, (value,), recipes=RecipeBook(families={
        'grouped_gemm': recipe(compact=compact)}), input_ready_ms=ready)['operators'][0]


def test_ordinary_full_catalog_frontend_selects_grouped_source(raw):
    graphs = target_graphs(raw, phase='decode', tokens=1, context=4096, sequences=32, scope='body')
    group = [o for o in graphs[0] if o.layer==1 and o.family=='grouped_gemm']
    assert [dict(o.metadata)['matrix_group_rows'] for o in group] == [(16,16),(16,16)]
    result = operator_costs(raw, group)['operators']
    assert result[0]['operator']['metadata']['program_execution']['source_mapping']['output_tiles_by_owner'] == (2,)*16+(1,)*16
    assert result[1]['operator']['metadata']['program_execution']['source_mapping']['output_tiles_by_owner'] == (1,)*28+(0,)*4
    assert result[0]['operator']['flops'] == group[0].flops
    assert result[0]['serviced']['fixpipe@r0:s0'] == 32*6144*2
    assert not any('mte3' in k for k in result[0]['serviced'])
    assert any(o.family=='swiglu' or 'situ' in o.recipe for o in graphs[0])


@pytest.mark.parametrize('rows,n,k', [((4,4),6144,3584), ((0,3,0,17),272,320),
                                     ((257,5),528,193), ((0,0),128,64)])
def test_integer_bank_work_and_real_resource_owners(raw, rows, n, k):
    value=op(rows,n,k)
    result=price(raw,value,ready=7)
    nn=(n+255)//256
    expected_reads=sum(2*m*k*nn + 2*n*k*((m+255)//256) for m in rows)
    writes=2*sum(rows)*n
    assert result['serviced']['memory@r0:s0'] == pytest.approx(expected_reads+writes+32*len(rows)*8)
    if sum(rows):
        assert result['serviced']['mte2_aic@r0:s0'] == pytest.approx(expected_reads)
        assert result['serviced']['mte1@r0:s0'] == pytest.approx(expected_reads)
        assert result['serviced']['fixpipe@r0:s0'] == pytest.approx(writes)
        assert result['operator']['flops']==2*sum(rows)*n*k
    assert not result['service_before_input'] and result['live_tasks']==0
    assert not any(result['remaining_buffers'].values())


def test_source_owner_aswt_tail_window_and_cross_bank_offset():
    # Independent source GetTileIdx coordinate oracle, including final window >4.
    rows=(4*256+3,0,2*256+5);n=3*256+11;cores=7;tile=Mapping(256,256,64,256)
    expected=[[] for _ in range(cores)];start=0
    for bank,m in enumerate(rows):
        nm=(m+255)//256;nn=(n+255)//256
        if not nm:continue
        main=min(4,nm);mainrow=nm//main-1;tail=nm-main*mainrow
        for index in range(nm*nn):
            row=index//nn//main
            if row<mainrow:mi=row*main+index%main;ni=(index//main)%nn
            else:
                row=mainrow;j=index-mainrow*main*nn;mi=mainrow*main+j%tail;ni=(j//tail)%nn
            if row%2:ni=nn-1-ni
            expected[(start+index)%cores].append((bank,min(256,m-mi*256),min(256,n-ni*256)))
        start+=nm*nn
    assert owner_tiles(rows,n,tile,cores)==tuple(tuple(v) for v in expected)


@pytest.mark.parametrize('rows,n,k', [((1,3),256,64),((16,16),272,320),((257,7),528,193)])
def test_compact_k_recurrence_matches_detailed_on_one_owner(raw, rows,n,k):
    case=parse_case(raw)
    dies=tuple(replace(d,compute_units=1,kernel_invocations=(),
                       resource_units=tuple((key,1) for key,_ in d.resource_units),
                       buffer_units=tuple((key,1) for key,_ in d.buffer_units)) for d in case.machine.dies)
    case=replace(case,machine=replace(case.machine,dies=dies))
    value=op(rows,n,k)
    compact=price(case,value);detailed=price(case,value,compact=False)
    assert compact['serviced']==pytest.approx(detailed['serviced'])
    assert compact['duration_ms']==pytest.approx(detailed['duration_ms'],rel=1e-10)
    assert compact['peak_buffers_per_unit']==detailed['peak_buffers_per_unit']


@pytest.mark.parametrize('axis', ['hbm','mte2_aic','mte1','fixpipe','cube'])
def test_source_respects_physical_rates_without_changing_work(raw,axis):
    base=parse_case(raw);value=op((16,16),6144,3584);normal=price(base,value)
    dies=tuple(replace(d,rates=tuple((k,v*(.01 if k==axis else 1)) for k,v in d.rates),
                       compute_peaks=tuple((k,v*(.01 if axis=='cube' else 1)) for k,v in d.compute_peaks))
               for d in base.machine.dies)
    slow=price(replace(base,machine=replace(base.machine,dies=dies)),value)
    assert slow['duration_ms']>normal['duration_ms']
    assert slow['operator']['flops']==normal['operator']['flops']
    assert slow['operator']['hbm_bytes']==normal['operator']['hbm_bytes']


def test_one_inclusive_invocation_for_all_banks_and_source_owners(raw):
    from mindie_sim.schema.invocation import InvocationTiming
    case=parse_case(raw)
    dies=tuple(replace(d,kernel_invocations=(('cube',InvocationTiming(minimum_ms=1)),))
               for d in case.machine.dies)
    result=price(replace(case,machine=replace(case.machine,dies=dies)),op((16,16),6144,3584),ready=7)
    assert result['duration_ms']==pytest.approx(1)
    assert result['operator']['flops']==2*32*6144*3584
