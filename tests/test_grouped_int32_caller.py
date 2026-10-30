"""Caller traffic and scale dependencies independently from observed durations."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import target_operators, target_graphs, graph_costs
from mindie_sim.prediction.family_grouped_quant import RECIPE, lower, tensor_contract
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


CASE = Path(__file__).resolve().parents[1] / 'examples/cases/high-concurrency-20261030/qwen-source-gdn-native/qwen35-35b-a5-tp4-c128-graph-source-gdn-native.json'


def graph(batch=3, selected=True):
    raw = json.loads(CASE.read_text())
    raw['model']['layers'] = 1
    raw['runtime']['kernels'] = {'lowerings': {}}
    if selected:
        raw['runtime']['kernels']['lowerings']['grouped_int8_caller'] = 'ascend_grouped_int32_dequant_swiglu'
    ops = target_operators(raw, scope='body', phase='decode', tokens=4, sequences=batch, context=256)
    return raw, ops


@pytest.mark.parametrize('batch', (1, 3, 128))
def test_real_frontend_removes_only_epilogue_and_activation_intermediate(batch):
    raw, before = graph(batch, False)
    _, after = graph(batch)
    gate = next(o for o in after if o.recipe == 'balanced_grouped_gate_up')
    down = next(o for o in after if o.recipe == 'balanced_grouped_down')
    consumer = next(o for o in after if o.recipe == RECIPE)
    gm = dict(gate.metadata)
    rows, n, k = gate.local_shape
    groups = gm['matrix_group_rows']
    assert sum(groups) == rows
    assert gate.mte2_bytes == rows*k + sum(x>0 for x in groups)*n*k
    assert gate.mte3_bytes == rows*n*4
    assert gate.vector_ops == 0 and 'matrix_epilogue' not in gm
    assert gm['output_format'] == 'int32'
    scale = next(o for o in after if o.op_id == consumer.deps[1])
    assert 'activation_scales' in scale.returns
    assert consumer.deps[0] == gate.op_id and down.deps == (consumer.op_id,)
    removed = {o.op_id for o in before} - {o.op_id for o in after}
    assert removed == {down.op_id + '.quantize'}
    old = {o.op_id:o for o in before}
    for op in after:
        if op.op_id not in {gate.op_id, consumer.op_id, down.op_id}:
            assert op == old[op.op_id]
    contract = tensor_contract(consumer.to_dict())
    assert contract[2][0] == ((rows,n), 'int32','ND')
    assert contract[2][1] == ((len(groups),n), 'fp32','ND')
    assert contract[3][0] == ((rows,n//2),'int8','ND')
    group = dict(consumer.metadata)['structure_subgraphs'][0]
    assert group['tensor_view']['intermediate_dtype'] == 'int32'
    assert group['implementation']['kernel_sequence'] == ('GroupedMatmul', 'DequantSwigluQuant')
    assert group['implementation']['native_kinds'] == []


@pytest.mark.parametrize('sequences', ([1, 8], [0, 8]))
def test_asymmetric_dp_uses_received_scale_boundary_not_local_sender_shape(sequences):
    from test_moe_quant_transport import transport_case
    raw = transport_case()
    raw['runtime']['routing'] = {'policy': 'balanced'}
    raw['runtime']['kernels']['lowerings'] = {
        'grouped_int8_caller': 'ascend_grouped_int32_dequant_swiglu'}
    raw['machine'] = graph()[0]['machine']
    graphs = target_graphs(raw, scope='body', phase='decode', tokens=1,
                           context=128, sequences=sequences)
    for ops in graphs.values():
        gate = next(o for o in ops if o.recipe == 'balanced_grouped_gate_up')
        scale = next(o for o in ops if o.recipe == 'expert_scale_dispatch')
        consumer = next(o for o in ops if o.recipe == RECIPE)
        assert consumer.deps == (gate.op_id, scale.op_id)
        assert dict(consumer.metadata)['input_row_scale_producer'] == scale.op_id
        assert consumer.local_shape[0] == sum(sequences)
    report = graph_costs(raw, graphs, isolated=False, input_ready_ms=7,
                         external_inputs={'analysis_input': 7})
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert report['duration_ms'] > 0


def test_exclusive_chain_keeps_extra_consumer_visible_instead_of_erasing_it():
    raw, ops = graph(3, False)
    gate = next(o for o in ops if o.recipe == 'balanced_grouped_gate_up')
    other = replace(gate, op_id='external_intended_bf16_reader', deps=(gate.op_id,),
                    family='tensor_copy',recipe='copy')
    with pytest.raises(ModelingError, match='exclusive_activation_chain'):
        lower((*ops,other), {'grouped_int8_caller':'ascend_grouped_int32_dequant_swiglu'})


def test_consumer_owner_work_is_conserved_and_obeys_input_and_bandwidth():
    from test_gdn_decode_family import execute_one
    raw, ops = graph(3)
    consumer = replace(next(o for o in ops if o.recipe == RECIPE),deps=())
    case = parse_case(raw)
    execute_one(case, (), consumer)
    n, width = consumer.local_shape
    rows = dict(consumer.metadata)['grouped_dequant']['expert_rows']
    units = case.machine.dies[0].units('vector')
    # Each positive group has <=units rows in this control: one row per owner,
    # and every participating owner reloads that expert's complete channel scale.
    assert max(rows) <= units
    expected_reads = len(rows)*8*units + n*(8*width+4) + n*8*width
    expected_writes = n*(width+4)
    measurements=[]
    for factor in (.5,1,2):
        dies=tuple(replace(d,rates=tuple((key, value*factor if key in {'mte2_aiv','mte3','vector','hbm'} else value)
                                        for key,value in d.rates)) for d in case.machine.dies)
        selected=replace(case,machine=replace(case.machine,dies=dies))
        report=graph_costs(selected,[consumer],isolated=False,input_ready_ms=7)
        execution=report['execution']
        serviced=execution['serviced']
        assert serviced['mte2_aiv@r0:s0'] == pytest.approx(expected_reads)
        assert serviced['mte3@r0:s0'] == pytest.approx(expected_writes)
        assert serviced['memory@r0:s0'] == pytest.approx(expected_reads+expected_writes)
        assert not execution['live_tasks'] and not execution['live_programs']
        assert report['duration_ms'] > 0
        measurements.append(report['duration_ms'])
    assert measurements[0] >= measurements[1] >= measurements[2]
    assert measurements[0] > measurements[2]


def test_semantics_do_not_depend_on_operator_identifiers():
    raw, ops=graph(3,False)
    renames={op.op_id:f'opaque_{i}' for i,op in enumerate(ops)}
    renamed=tuple(replace(o,op_id=renames[o.op_id],deps=tuple(renames.get(d,d) for d in o.deps)) for o in ops)
    external={d for o in ops for d in o.deps if d not in renames}
    lowered=lower(renamed,{'grouped_int8_caller':'ascend_grouped_int32_dequant_swiglu'})
    assert sum(o.recipe==RECIPE for o in lowered)==1
    assert all(dep in renames.values() or dep in external for o in lowered for dep in o.deps)
    assert lower(lowered,{'grouped_int8_caller':'ascend_grouped_int32_dequant_swiglu'}) == lowered
