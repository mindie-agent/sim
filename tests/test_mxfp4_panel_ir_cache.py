"""Cached physical panel IR retains the existing Engine task schedule exactly."""
from dataclasses import replace
from copy import deepcopy
from collections import Counter
from math import lcm
import weakref
import gc

import pytest

from mindie_sim import operator_costs
from mindie_sim.prediction import mxfp4_grouped
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mxfp4_panels import compiled_panels, panel_parts
from mindie_sim.prediction.mxfp4_geometry import panel_sizes, tile_runs
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.refill_steady import compact_refill
from mindie_sim.prediction.program import RecipeBook
from test_mxfp4_grouped import source_case, gate, physical_op


def source_ir(raw,op, *, directional=False, owner=0):
    """Capture the ordinary recipe context and compile an uncached source IR.

    This deliberately does not intercept the matrix module's internal helper:
    production can use the cache and delete its duplicate panel builder.
    """
    contexts=[]
    def build(op,geometry):
        plan=mxfp4_grouped.build(op,geometry,mte1_mode='directional' if directional else 'base')
        def capture(context):
            contexts.append(context)
            return plan.program(context)
        return replace(plan,program=capture)
    operator_costs(raw,(op,),recipes=RecipeBook(families={
        'grouped_gemm':PlannedRecipe(build)}))
    context=contexts[0]
    config=mxfp4_grouped.config_from_op(op)
    cores=context.core_count
    schedules=tuple(tuple(tile_runs(config,index,cores,True)) for index in range(cores))
    m,n,repeat=schedules[owner][0]
    clients={context.resource_names['hbm']:Counter(schedules)[schedules[owner]]}
    kb,ka,_=panel_sizes(m,n)
    panels=(config.k+kb-1)//kb
    period=lcm(ka,4096)//kb
    count,remainder=divmod(panels,period)
    if not remainder and config.k%kb:count-=1
    def parts(index):
        return panel_parts(context,k=config.k,m=m,n=n,repeat=repeat,index=index,
                           owner=owner,cores=cores,aivs=context.resource_units['vector'],
                           directional=directional)
    original=compact_refill(context,tuple(parts(i) for i in range(period)) if count else (),
                             count,tuple(parts(i) for i in range(count*period,panels)),
                             shared_clients=clients)
    return context,original,clients


def execute(program,context):
    units={physical:context.resource_units.get(role,1) for role,physical in context.resource_names.items()}
    engine=Engine([Resource(name,rate,units=units.get(name,1)) for name,rate in context.rates.items()],trace_limit=100)
    done=program.start(engine,'source',engine.submit('input',release=7),owner='caller')
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
    return done.time,dict(engine.serviced),engine.completed_count,list(engine.trace)


@pytest.mark.parametrize('k,rows,repeat', [(64,(1,),1),(320,(3,),1),(4096,(1,),1),
                                          (5120,(17,),1),(24576,(1,),1),(320,(1,)*64,2)])
def test_cached_panels_match_existing_source_tasks_traces_and_service(source_case,gate,k,rows,repeat):
    context,original,clients=source_ir(source_case,physical_op(gate,rows,160,k))
    arguments=dict(k=k,m=rows[0],n=160,repeat=repeat,owner=0,cores=32,aivs=64,
                   directional=False,shared_clients=clients)
    cached=compiled_panels(context,**arguments)
    assert cached==original
    assert execute(cached,context)==execute(original,context)
    # Reusing one IR creates fresh task/Event state and cannot consume a plan.
    assert execute(cached,context)==execute(cached,context)
    assert compiled_panels(context,**arguments) is cached
    for stage in cached.stages:
        with pytest.raises(TypeError):stage.work['injected']=1
    child=cached.stages[0].program.stages[0]
    with pytest.raises(TypeError):child.placement['injected']=(0,)


def test_cache_key_covers_owner_rates_units_modes_peaks_and_peer_share(source_case,gate):
    context,_,clients=source_ir(source_case,physical_op(gate,(1,),160,5120))
    base_rate=context.rates[context.resource_names['mte1']]*1000
    context=replace(context,service_modes={**context.service_modes,'mte1_l0a':('mte1',base_rate),
                                           'mte1_l0b':('mte1',base_rate*.5)})
    args=dict(k=5120,m=1,n=160,repeat=1,owner=0,cores=32,aivs=64,directional=True,shared_clients=clients)
    cached=compiled_panels(context,**args)
    hbm=context.resource_names['hbm'];mte=context.resource_names['mte2_aic']
    variants=[(replace(context,rates={**context.rates,hbm:context.rates[hbm]*.5}),args),
              (replace(context,rates={**context.rates,mte:context.rates[mte]*.5}),args),
              (replace(context,resource_units={**context.resource_units,'mte2_aic':16}),args),
              (replace(context,compute_peaks={**context.compute_peaks,'fp8_e4m3':context.compute_peaks['fp8_e4m3']*.5}),args),
              (replace(context,service_modes={**context.service_modes,'mte1_l0a':('mte1',1e10)}),args),
              (context,{**args,'owner':1}),
              (context,{**args,'repeat':2}),
              (context,{**args,'shared_clients':{hbm:32}})]
    for changed,options in variants:
        new=compiled_panels(changed,**options)
        assert new is not cached
        # A non-bottleneck rate can retain the same IR; the key must still
        # invalidate, since another geometry may put it on the critical path.
        assert execute(new,changed)==execute(new,changed)
    # Invocation/trace state is deliberately excluded: it belongs to the caller.
    assert compiled_panels(replace(context,op=None,setup_ms=10,minimum_ms=100),**args) is cached


def test_cache_does_not_retain_original_context_or_operator(source_case,gate):
    compiled_panels.cache_clear()
    context,_,clients=source_ir(source_case,physical_op(gate,(1,),160,64))
    context_ref=weakref.ref(context);op_ref=weakref.ref(context.op)
    cached=compiled_panels(context,k=64,m=1,n=160,repeat=1,owner=0,cores=32,aivs=64,shared_clients=clients)
    del context
    gc.collect()
    assert context_ref() is None and op_ref() is None
    # A complete source IR has no live callbacks, events or allocation handles.
    assert all(not s.acquire and not s.free for s in cached.stages)


def test_cache_entry_bound_evicts_without_changing_compilation(source_case,gate):
    context,_,clients=source_ir(source_case,physical_op(gate,(1,),160,64))
    compiled_panels.cache_clear()
    args=dict(k=64,m=1,n=160,repeat=1,owner=0,cores=32,aivs=64,shared_clients=clients)
    first=compiled_panels(context,**args)
    for repeat in range(2,1026):compiled_panels(context,**{**args,'repeat':repeat})
    assert compiled_panels.cache_info().currsize==1024
    assert compiled_panels.template_cache_info().currsize==128
    again=compiled_panels(context,**args)
    assert again is not first and again==first


def test_directional_modes_and_nonzero_owner_match_existing_source(source_case,gate):
    raw=deepcopy(source_case)
    raw['machine']['device']['service_modes']={
        'mte1_l0a':{'resource':'mte1','rate':1e10},
        'mte1_l0b':{'resource':'mte1','rate':2e10}}
    context,original,clients=source_ir(raw,physical_op(gate,(3,3),160,4160),
                                       directional=True,owner=1)
    cached=compiled_panels(context,k=4160,m=3,n=160,repeat=1,owner=1,cores=32,aivs=64,
                           directional=True,shared_clients=clients)
    assert cached==original
    assert execute(cached,context)==execute(original,context)


def test_shared_ir_has_independent_events_under_same_engine_contention(source_case,gate):
    context,original,clients=source_ir(source_case,physical_op(gate,(1,),160,5120))
    cached=compiled_panels(context,k=5120,m=1,n=160,repeat=1,owner=0,cores=32,aivs=64,
                           shared_clients=clients)
    def concurrent(program):
        units={physical:context.resource_units.get(role,1) for role,physical in context.resource_names.items()}
        engine=Engine([Resource(name,rate,units=units.get(name,1))
                       for name,rate in context.rates.items()],trace_limit=200)
        ready=engine.submit('input',release=7)
        done=tuple(program.start(engine,f'copy{i}',ready,owner=f'caller{i}') for i in range(2))
        engine.advance(horizon=6)
        assert not any(engine.serviced.values())
        engine.advance(until=done)
        assert engine.summary()['live_tasks']==engine.summary()['live_programs']==0
        return tuple(e.time for e in done),dict(engine.serviced),engine.completed_count,list(engine.trace)
    assert concurrent(cached)==concurrent(original)
