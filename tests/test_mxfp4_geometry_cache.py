"""Exact geometry reuse without retaining mutable operator or Engine state."""
from dataclasses import replace
from itertools import product

import pytest

from mindie_sim import operator_costs
from mindie_sim.prediction.instructions import InstructionInventory
from mindie_sim.prediction.mechanism import PlannedRecipe, Work
from mindie_sim.prediction.mxfp4_geometry import Config, geometry_ledger, owner_tiles, tile_work, tile_inventory
from mindie_sim.prediction.mxfp4_grouped import build
from mindie_sim.prediction.program import RecipeBook
from test_mxfp4_grouped import source_case, gate, physical_op


@pytest.mark.parametrize('config', [Config(320,320,(1,0,2)), Config(448,544,(0,257,0,3)),
                                    Config(5120,4608,(17,)), Config(64,32,(0,0))])
def test_cached_work_matches_individual_integer_tiles_and_owner_counts(config):
    for cores, compact, directional in product((1,3,32),(False,True),(False,True)):
        expected=Work(resources={'hbm':512*cores,'mte2_aiv':512*cores})
        inventory=InstructionInventory({});counts=[];runs=[]
        for owner in range(cores):
            tiles=list(owner_tiles(config,owner,cores));counts.append(len(tiles))
            shape=None;run_count=0
            for *_,m,n in tiles:
                expected+=tile_work(config,m,n,directional_mte1=directional)
                inventory+=tile_inventory(config,m,n)
                run_count+=not compact or shape!=(m,n);shape=(m,n)
            runs.append(run_count)
        actual=geometry_ledger(config,cores,compact,directional)
        assert actual.work==expected
        assert actual.inventory==inventory
        assert actual.counts==tuple(counts) and actual.runs==tuple(runs)
        assert geometry_ledger(config,cores,compact,directional) is actual


def test_apply_and_inventory_exports_cannot_mutate_later_cached_predictions(gate):
    cfg=Config(320,320,(1,0,2));saved=geometry_ledger(cfg,32)
    values=dict(saved.work.resources);counts=dict(saved.inventory.counts)
    op=saved.work.apply(physical_op(gate,cfg.rows,cfg.n,cfg.k))
    dict(op.metadata)['resource_work']['mte2_aic']=123
    scaled=saved.work.scaled(2);scaled.resources['mte2_aiv']=456
    exported=saved.inventory.to_dict()
    first=next(iter(exported['instructions']))
    exported['instructions'][first]['calls']=789
    with pytest.raises(TypeError):saved.work.resources['hbm']=0
    with pytest.raises(TypeError):saved.work.matrix['fp8_e4m3']=0
    with pytest.raises(TypeError):saved.inventory.counts[first]=None
    again=geometry_ledger(cfg,32)
    assert dict(again.work.resources)==values and dict(again.inventory.counts)==counts


def test_actual_source_execution_keeps_cached_work_immutable(source_case,gate):
    cfg=Config(320,160,(1,0,2));op=physical_op(gate,cfg.rows,cfg.n,cfg.k)
    ledger=geometry_ledger(cfg,32)
    before=(dict(ledger.work.matrix),dict(ledger.work.resources),ledger.inventory.to_dict())
    # The exact existing source program executes with the cached ledger as its
    # work contract. No substitute timing recipe or mocked Engine is used.
    def cached(op,geometry):
        plan=build(op,geometry)
        return replace(plan,work=geometry_ledger(cfg,geometry.compute_units).work)
    uncached=operator_costs(source_case,(op,),recipes=RecipeBook(families={
        'grouped_gemm':PlannedRecipe(build)}),input_ready_ms=7)['operators'][0]
    for _ in range(2):
        actual=operator_costs(source_case,(op,),recipes=RecipeBook(families={
            'grouped_gemm':PlannedRecipe(cached)}),input_ready_ms=7)['operators'][0]
        assert actual['duration_ms']==uncached['duration_ms']
        assert actual['serviced']==uncached['serviced']
        assert not actual['service_before_input'] and actual['live_tasks']==0
        assert not any(actual['remaining_buffers'].values())
    assert before==(dict(ledger.work.matrix),dict(ledger.work.resources),ledger.inventory.to_dict())


def test_cache_retention_is_bounded_and_eviction_does_not_change_work():
    geometry_ledger.cache_clear()
    first=geometry_ledger(Config(64,32,(0,0)),2)
    for i in range(1,130):geometry_ledger(Config((i+1)*64,32,(0,0)),2)
    info=geometry_ledger.cache_info()
    assert info.currsize==128
    recreated=geometry_ledger(Config(64,32,(0,0)),2)
    assert recreated==first and recreated is not first
    assert geometry_ledger.cache_info().misses==info.misses+1
