"""Actual owner/queue controls for full-load source DSQ, without timing fitting."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_operators
from mindie_sim.prediction.grouped_dequant_arch35 import Config, assignments, inventories, build
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.family_grouped_quant import tensor_contract
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]


def ordinary(batch=64, capacity=2048):
    raw = json.loads((ROOT / f'examples/cases/high-concurrency-20261030/qwen35-hccl-conditional/qwen35-35b-a5-tp4-c{batch}-hccl-source.json').read_text())
    raw['model']['layers'] = 1
    kernels = raw['runtime']['kernels']
    kernels['native_moe_protocol'] = {'kind':'arch35_fullmesh', 'max_inflight_per_rank':4,
                                      'dispatch_quant':'per_token_int8'}
    kernels['moe_token_partition'] = {'kind':'vllm_ascend_mc2_tp_slice',
                                     'gather_output_materialization':'view'}
    kernels['lowerings']['grouped_int8_caller'] = {
        'implementation':'ascend_grouped_int32_dequant_swiglu',
        'options': {'dequant_program':'arch35_full_load', 'input_capacity_rows':capacity}}
    original = deepcopy(raw)
    ops = target_operators(raw, scope='body', phase='decode', tokens=4,
                           context=4096, sequences=batch)
    assert raw == original
    return parse_case(raw), next(o for o in ops if o.recipe == 'grouped_int32_dequant_swiglu'), ops


def selected(rows, width=512, capacity=2048):
    case, op, _ = ordinary()
    config = {**dict(op.metadata)['grouped_dequant'], 'expert_rows':tuple(rows),
              'width':width, 'input_capacity_rows':capacity}
    op = replace(op, local_shape=(sum(rows),width), deps=(), metadata=tuple({
        **dict(op.metadata), 'grouped_dequant':config}.items()))
    return case, op


@pytest.mark.parametrize('batch,allocated,active,each', ((4,128,32,1),(64,2048,8,64),(128,4096,16,64)))
def test_real_frontend_uses_allocated_dispatch_but_only_valid_payload(batch, allocated, active, each):
    case, op, ops = ordinary(batch, allocated)
    plan = build(op, Geometry.from_devices(case.machine.dies))
    diag = plan.metadata['grouped_dequant_schedule']
    valid = batch*8
    assert op.local_shape == (valid,512)
    assert diag['input_capacity_rows'] == allocated and diag['valid_rows'] == valid
    assert diag['launched_aiv'] == 64
    abi = tensor_contract(op.to_dict())
    assert abi[2][0][0] == (allocated,1024) and abi[2][2][0] == (allocated,)
    assert abi[3][0][0] == (allocated,512)
    assert dict(op.metadata)['activation_scale_shape'] == (valid,)
    assert diag['rows_per_owner'] == (each,)*active + (0,)*(64-active)
    assert diag['weight_scale_read_bytes'] == (32 if batch == 4 else active*64)*4096
    assert 's0.l0.expert_up.quantize' not in {o.op_id for o in ops}
    down = next(o for o in ops if o.op_id == 's0.l0.expert_down')
    assert dict(down.metadata)['activation_scale_producer'] == op.op_id
    result = graph_costs(case, [replace(op,deps=())], input_ready_ms=7, isolated=False)
    service = result['execution']['serviced']
    assert service['memory@r0:s0'] == pytest.approx(plan.work.resources['hbm'])
    assert service['mte2_aiv@r0:s0'] == pytest.approx(valid*(4096+4)+diag['weight_scale_read_bytes'])
    assert service['mte3@r0:s0'] == pytest.approx(valid*516)
    assert all(row['ready_ms'] >= 7 for row in result['timeline'])
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert not any(result['execution']['live_allocated'].values())


def test_same_valid_groups_different_capacity_changes_owner_branch_not_payload():
    rows = (8,)*64
    tight = Config(rows,512,512,64,262144)
    capacity = Config(rows,512,2048,64,262144)
    assert tight.special and not capacity.special
    a = [assignments(tight, i) for i in range(tight.blocks)]
    b = [assignments(capacity, i) for i in range(capacity.blocks)]
    assert [sum(n for _,n in x) for x in a] == [8]*64
    assert [sum(n for _,n in x) for x in b] == [64]*8+[0]*56
    assert sum(n for x in a for _,n in x) == sum(n for x in b for _,n in x) == 512


@pytest.mark.parametrize('batch', (4,64,128))
def test_explicit_valid_row_allocation_follows_actual_received_expert_rows(batch):
    case, op, _ = ordinary(batch, 'valid_rows')
    config = dict(op.metadata)['grouped_dequant']
    rows = sum(config['expert_rows'])
    assert config['input_capacity_rows'] == rows == batch * 8
    abi = tensor_contract(op.to_dict())
    assert abi[2][0][0] == (rows, 1024)
    assert abi[3][0][0] == (rows, 512)
    declared_case, declared_op, _ = ordinary(batch, rows)
    geometry = Geometry.from_devices(case.machine.dies)
    variable = build(op, geometry)
    fixed = build(declared_op, geometry)
    assert variable.work == fixed.work
    assert variable.metadata['grouped_dequant_schedule'] == fixed.metadata['grouped_dequant_schedule']
    report = graph_costs(case, [replace(op, deps=())], isolated=False)
    assert report['duration_ms'] > 0
    assert report['execution']['live_tasks'] == 0


@pytest.mark.parametrize('rows,capacity,cores', (((0,5,69,0,7),1024,8),((1,)*33,33,16),((30,0,17),128,4)))
def test_owner_partition_and_ub_tile_tail_independently_conserve_every_bank(rows,capacity,cores):
    cfg = Config(rows,33,capacity,cores,8192)
    assigned = [assignments(cfg,i) for i in range(cfg.blocks)]
    by_bank = [sum(n for owner in assigned for group,n in owner if group==g) for g in range(len(rows))]
    assert by_bank == list(rows)
    assert cfg.storage <= cfg.ub-1024
    # All non-special rows belong to contiguous ceil-sized chunks, never free
    # cores reassigned to other groups in the same group iteration.
    if not cfg.special:
        for g,n in enumerate(rows):
            chunk = (n+cfg.blocks-1)//cfg.blocks
            assert [dict(owner).get(g,0) for owner in assigned] == [max(0,min(chunk,n-i*chunk)) for i in range(cfg.blocks)]


def test_typed_scopes_preserve_issued_calls_on_column_tail_and_no_vector_double_charge():
    dequant, reduction, quant = inventories(3,65)
    assert dequant.counts['Load'].calls == 16  # 4 weight loads + 12 data loads
    assert dequant.counts['Load'].read_bytes == 4096
    assert dequant.counts['Cast32'].active_lanes == 390
    assert reduction.counts['ReduceMax'].active_lanes == 3*64
    assert reduction.counts['Div'].active_lanes == 3*64  # source unmasked scale division
    assert quant.counts['Store'].write_bytes == 3*65
    case, op = selected((3,),65,16)
    work = build(op,Geometry.from_devices(case.machine.dies)).work
    assert work.resources.get('vector',0) == 0
    assert work.resources['reg_special_lanes'] > 0


@pytest.mark.parametrize('role', ('hbm','mte2_aiv','mte3','reg_issue_slots','reg_special_lanes'))
def test_finite_owner_schedule_respects_rate_axis_without_changing_work(role):
    case, op = selected((3,1,2,0),65,16)
    values=[]; services=[]
    for factor in (.5,1,2):
        dies=tuple(replace(d,rates=tuple((k, 1e7*factor if k==role else v)
                                        for k,v in d.rates),
                           service_modes=tuple((k,(r,1e7*factor if k==role else v))
                                               for k,(r,v) in d.service_modes)) for d in case.machine.dies)
        changed=replace(case,machine=replace(case.machine,dies=dies))
        report=graph_costs(changed,[op],isolated=False,input_ready_ms=7)
        values.append(report['duration_ms'])
        measured = dict(report['execution']['serviced'])
        if role == 'reg_issue_slots':
            measured['vector@r0:s0'] *= factor  # normalize the explicit issue service mode
        services.append(measured)
        assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
        assert not any(report['execution']['live_allocated'].values())
    assert values[0] > values[1] > values[2]
    assert services[0] == pytest.approx(services[1])
    assert services[1] == pytest.approx(services[2])


def direct(rows, *, ub=1440):
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.program import RecipeContext
    _, op = selected(rows,1,max(1,sum(rows)))
    # Isolate queue dependencies: typed arithmetic is fast, GM read/write each
    # cost one ms/byte. Width1 still has the source's padded32B UB structures.
    roles = ('hbm','mte2_aiv','mte3','vector','reg_issue_slots','reg_data_writes',
             'reg_ub_accesses','reg_ub_stores','reg_alu_lanes','reg_reduce_lanes','reg_special_lanes')
    rates = {r:(1 if r in {'mte2_aiv','mte3'} else 1e12) for r in roles}
    geometry=Geometry(1,{r:1 for r in roles},{'ub':ub},{'ub':1})
    plan=build(op,geometry)
    engine=Engine([Resource(r,v,units=1) for r,v in rates.items()],
                  pools={'ub':ub},pool_units={'ub':1},trace_limit=10000)
    context=RecipeContext(op,{},dict(zip(roles,roles)),rates,1,
                          resource_units={r:1 for r in roles},pool_names={'ub':'ub'},pool_units={'ub':1})
    ready=engine.submit('ready',release=7)
    done=plan.lower(context).start(engine,'dsq',ready)
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not engine._tasks and not engine._promises and not engine.allocations
    assert not any(engine.allocated.values())
    return plan,engine,done.time-7


def test_one_scale_slot_and_cross_group_weight_reuse_have_real_engine_dependencies():
    plan,engine,span=direct((3,))
    assert plan.metadata['grouped_dequant_schedule']['ub_tile_rows'] == 1
    # One8B weight prologue, three12B input reads; tail4Bscale+1Boutput.
    # Read2 overlaps output1, so 8+3*12+5=49, not3*(8+12+5).
    assert span == pytest.approx(49,abs=1e-7)
    records={x['task']:x for x in engine.trace}
    end=lambda i,name: next(x['end_ms'] for key,x in records.items() if key.endswith(f':tile{i}:{name}'))
    start=lambda i,name: next(x['start_ms'] for key,x in records.items() if key.endswith(f':tile{i}:{name}'))
    for i in (1,2):
        assert start(i,'dequant_swiglu') >= end(i-1,'scale_write')
        assert start(i,'dequant_swiglu') >= end(i-1,'quantize')
        assert start(i,'scale_write') >= end(i-1,'output_write')
    assert start(1,'read') < end(0,'output_write')
    _,separate,span2=direct((1,1,1))
    # Each group owns different weight scales. The next scale prologue can
    # overlap previous MTE3 output, but cannot overtake its vector consumer.
    assert span2 == pytest.approx(3*(8+12)+5,abs=1e-7)
    assert separate.serviced['mte2_aiv']-engine.serviced['mte2_aiv'] == pytest.approx(16)


def test_large_row_loop_retains_only_a_finite_window_of_engine_tasks():
    _,engine,_=direct((65,))
    assert engine.completed_count > 500
    assert engine.peak_tasks < 40
    assert engine.peak_allocated['ub'] == 416  # source queues, excluding1KiBhost reserve
