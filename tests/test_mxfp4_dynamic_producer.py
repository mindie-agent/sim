"""Source-selected producer overlap conserves work on ordinary physical paths."""
from dataclasses import replace
from copy import deepcopy

import pytest

from mindie_sim import operator_costs
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mxfp4_grouped import recipe
from mindie_sim.prediction.mxfp4_panels import compiled_panels
from mindie_sim.prediction.mxfp4_source_refill import source_panel
from mindie_sim.prediction.mxfp4_pipeline import SourcePipeline, Dependencies, Slots
from mindie_sim.prediction.program import RecipeBook
from test_mxfp4_grouped import source_case, gate, physical_op
from test_mxfp4_panel_ir_cache import source_ir


def cost(raw,op,compact):
    return operator_costs(raw,(op,),recipes=RecipeBook(families={
        'grouped_gemm':recipe(producer_mode='dynamic_buffer4',compact_panels=compact)}),
        input_ready_ms=7)['operators'][0]


@pytest.mark.parametrize('k,m,n',((64,1,32),(512,17,128),(4160,3,160),(5120,1,256)))
def test_actual_source_detail_and_compact_conserve_tails(source_case,gate,k,m,n):
    op=physical_op(gate,(m,0,m),n,k)
    detail,compact=(cost(source_case,op,flag) for flag in (False,True))
    assert detail['operator']['metadata']['program_execution']['mxa8w4']['producer_mode']=='dynamic_buffer4'
    assert compact['operator']['metadata']['program_execution']['mxa8w4']['producer_mode']=='dynamic_buffer4'
    for key,value in detail['serviced'].items():
        assert compact['serviced'][key]==pytest.approx(value,rel=1e-12,abs=1e-12)
    for row in (detail,compact):
        assert not row['service_before_input']
        assert row['live_tasks']==0
        assert not any(row['remaining_buffers'].values())
    if k==64:
        # AIV1 still advances its flag/slot protocol but has no weight payload.
        metadata=detail['operator']['metadata']['program_execution']['mxa8w4']
        assert metadata['valid_rows']==2*m


def test_modes_and_slot_contracts_do_not_share_cached_ir(source_case,gate):
    context,_,clients=source_ir(source_case,physical_op(gate,(1,),160,5120))
    compiled_panels.cache_clear()
    args=dict(k=5120,m=1,n=160,repeat=1,owner=0,cores=32,aivs=64,shared_clients=clients)
    serial=compiled_panels(context,**args)
    dynamic=compiled_panels(context,**args,producer_mode='dynamic_buffer4')
    one_slot=compiled_panels(context,**args,producer_mode='dynamic_buffer4',producer_slots=Slots(1,1,1,1,1))
    assert serial is not dynamic and one_slot is not dynamic
    assert compiled_panels.template_cache_info().misses==3
    assert compiled_panels(context,**args) is serial


@pytest.mark.parametrize('role',('hbm','mte2_aic','mte2_aiv','mte3','mte1','vector'))
def test_real_pipe_rate_controls_remain_monotone(source_case,gate,role):
    durations=[]
    for factor in (.5,1,2):
        raw=deepcopy(source_case)
        raw['machine']['device']['rates']={
            **{r:1e18 for r in ('hbm','mte2_aic','mte2_aiv','mte3','mte1','vector')},
            role:1e8*factor}
        durations.append(cost(raw,physical_op(gate,(1,)*4,256,5120),False)['duration_ms'])
    assert durations[0]>durations[1]>durations[2]


def test_bounded_source_submission_matches_eager_dag_with_periodic_scale(source_case,gate):
    context,_,_=source_ir(source_case,physical_op(gate,(1,),128,32768))
    def panel(i):
        return source_panel(context,k=32768,m=1,n=128,repeat=1,index=i,
                            owner=0,cores=32,aivs=64,directional=False)
    units={name:context.resource_units.get(role,1) for role,name in context.resource_names.items()}
    def run(eager):
        engine=Engine([Resource(name,rate,units=units.get(name,1)) for name,rate in context.rates.items()])
        ready=engine.submit('input',release=7)
        if eager:
            state=Dependencies(ready,Slots())
            for i in range(64):
                def emit(label,stage,deps):
                    return stage.submit(engine,f'source:panel{i}:{label}',deps,lambda x:x,'')
                done=state.advance(panel(i),emit)
        else:
            done=SourcePipeline(64,panel,activation_panels=4,scale_panels=8).start(engine,'source',ready)
        engine.advance(until=(done,))
        assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
        return done.time,dict(engine.serviced)
    actual,expected=run(False),run(True)
    assert actual[0]==pytest.approx(expected[0],rel=1e-12)
    assert actual[1]==pytest.approx(expected[1],rel=1e-12,abs=1e-12)


def test_zero_experts_only_initializes_source_permutation(source_case,gate):
    row=cost(source_case,physical_op(gate,(0,0),32,64),False)
    assert row['serviced']['memory@r0:s0']==pytest.approx(64*256,rel=1e-12)
    assert row['serviced']['mte2_aiv@r0:s0']==pytest.approx(64*256,rel=1e-12)
    assert row['serviced'].get('matrix@r0:s0',0)==0
    assert row['live_tasks']==0
    assert not any(row['remaining_buffers'].values())
