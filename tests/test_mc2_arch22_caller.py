"""A3 native MC2 ABI, masked transport and complete-call timing boundary."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from test_mc2_per_token_int8 import case as int8_case, graphs
from test_mc2_mx_dispatch import protocol_only
from test_joint_collective_execution import wire


def case():
    raw = int8_case()
    kernels = deepcopy(raw.runtime.kernels)
    kernels['native_moe_protocol'] = {
        'kind': 'arch22_v2_caller', 'dispatch_quant': 'per_token_int8',
        'max_inflight_per_rank': 2, 'global_bs': 6}
    return replace(raw, runtime=replace(raw.runtime, kernels=kernels))


def test_native_v2_produces_payload_scales_and_return_indices_once():
    raw = case(); before = deepcopy(raw)
    declared = graphs(raw); ops = {op.op_id: op for op in declared[0]}
    send, received, up, combine, output = (ops['s0.l0.' + name] for name in
        ('pack', 'dispatch', 'expert_up', 'combine', 'unpack'))
    assert raw == before
    assert 's0.l0.expert_up.quantize' not in ops
    assert 's0.l0.expert_down.quantize' in ops
    meta = dict(up.metadata)
    assert up.deps == (meta['activation_payload_producer'],) == (meta['activation_scale_producer'],)
    assert up.deps == (received.op_id,)
    assert {'activation_scales', 'assist_info', 'expert_token_nums', 'ep_send_counts'} <= set(received.returns)
    received_meta = dict(received.metadata)
    assert received_meta['activation_payload_shape'] == (2,2048)
    assert received_meta['activation_scale_shape'] == (2,)
    assert received_meta['activation_scale_bytes'] == 8
    assert received_meta['assist_info_shape'] == (2,3)
    assert received_meta['native_output_capacity_rows'] == 12
    assert received_meta['native_abi_outputs'][0]['shape'] == (12,2048)
    assert received_meta['native_abi_outputs'][1]['quantity'] == 'dynamic_scales'
    assert received_meta['native_abi_outputs'][6]['quantity'] == 'expand_scales'
    assert dict(combine.metadata)['expand_scales_buffer_producer'] == received.op_id
    assert 'expand_scales' in received.returns
    assert received.local_shape == combine.local_shape == (2,2048)
    assert received.op_id in combine.deps
    assert output.local_shape == (1,2048)
    native = [op for op in ops.values() if 'device_kernel_group' in dict(op.metadata)]
    assert sum(dict(o.metadata)['device_kernel_group']['entry'] for o in native) == 2
    assert sum(dict(o.metadata)['device_kernel_group']['exit'] for o in native) == 2
    assert {dict(o.metadata)['native_kind'] for o in native} == {
        'MoeDistributeDispatchV2','MoeDistributeCombineV2'}
    assert send.local_shape == (1,2048)  # actual X input, not token-expert pair capacity
    report = graph_costs(raw, declared, input_ready_ms=7, external_inputs={'analysis_input':7})
    timeline = {row['id']:row for row in report['timeline']}
    assert timeline['r0:s0.l0.expert_up']['ready_ms'] >= timeline['r0:s0.l0.dispatch']['complete_ms']
    assert min(row['ready_ms'] for row in report['timeline']) >= 7
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


@pytest.mark.parametrize('sequences', [(1,1),(0,3),(3,0),(1,3)])
def test_logical_wire_and_typed_receiver_rows_survive_idle_unequal_sources(sequences):
    raw = case(); declared = graphs(raw, sequences)
    report = graph_costs(raw, protocol_only(declared), isolated=False)
    # Two ranks/topk2: half of each sender's assignments are remote. Count
    # channel sends 4 experts *4B /2 once per physical sender, including idle.
    assert sum(wire(report).values()) == sum(sequences) * (2048+4+12+4096) + 16
    for rank, ops in declared.items():
        by_id={op.op_id:op for op in ops}
        received=dict(by_id['s0.l0.dispatch'].metadata)
        assert received['activation_scale_shape'] == (sum(sequences),)
        assert by_id['s0.l0.pack'].local_shape[0] == sequences[rank]
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


def test_complete_call_response_is_one_boundary_not_one_per_local_or_network_phase():
    raw=case(); kernels=deepcopy(raw.runtime.kernels)
    kernels['service_regimes']=[{
        'id':'dispatch-v2-control','match':{'kernel':'MoeDistributeDispatchV2'},
        'response':{'fixed_us':100},'provenance':{'kind':'synthetic_control'}}, {
        'id':'combine-v2-control','match':{'kernel':'MoeDistributeCombineV2'},
        'response':{'fixed_us':100},'provenance':{'kind':'synthetic_control'}}]
    # Synthetic fast resources expose precisely two 100us inclusive groups.
    raw=replace(raw,runtime=replace(raw.runtime,kernels=kernels),
        machine=replace(raw.machine,dies=tuple(replace(d,
            rates=tuple((r,1e15) for r,v in d.rates), launch_us=0) for d in raw.machine.dies),
            links=tuple(replace(l,latency_us=0,capacity_bytes_per_ms=1e12) for l in raw.machine.links)))
    report=graph_costs(raw,protocol_only(graphs(raw)),isolated=False)
    assert report['duration_ms'] == pytest.approx(.2)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


def test_caller_right_padding_preserves_mask_and_native_quantization():
    raw=case(); kernels=deepcopy(raw.runtime.kernels)
    kernels['moe_token_partition']={'kind':'vllm_ascend_mc2_tp_slice','padding':'right_to_tp'}
    raw=replace(raw,runtime=replace(raw.runtime,kernels=kernels,
        parallel=replace(raw.runtime.parallel,tp=2,dp=1,ep=2)))
    declared=target_graphs(raw,phase='decode',tokens=1,sequences=3,context=128,scope='body')
    ops={o.op_id:o for o in declared[0]}
    assert ops['s0.l0.mc2_input_pad'].local_shape == (4,2048)
    assert ops['s0.l0.pack'].local_shape == (2,2048)
    assert ops['s0.l0.mc2_tp_gather'].local_shape == (4,2048)
    assert ops['s0.l0.mc2_output_trim'].local_shape == (3,2048)
    md=dict(ops['s0.l0.dispatch.payload_network'].metadata)
    matrix=md['peer_traffic']['payload_bytes']
    assert matrix[0][1] == 2*matrix[1][0] # second rank has one valid source row
    reverse=dict(ops['s0.l0.combine.payload_network'].metadata)['peer_traffic']['payload_bytes']
    assert reverse[0][1]/4096 == matrix[1][0]/2064
    assert 's0.l0.expert_up.quantize' not in ops


@pytest.mark.parametrize('concurrency,bucket', [(2,12),(8,48)])
@pytest.mark.parametrize('fused',[False,True])
def test_glm78_capture_rows_feed_w4a8_from_native_payload_and_scales(concurrency,bucket,fused):
    name='a3_glm53_w4a8_tp16_' + ('tpot30_c2' if concurrency==2 else 'tpot88_c8')
    path=Path(__file__).resolve().parents[1]/'examples/cases/high-concurrency-20261030/glm78-capture-rows'/f'{name}.json'
    raw=json.loads(path.read_text())
    if fused:
        raw['runtime']['kernels'].setdefault('lowerings',{})['native_fusion']={
            'implementation':'declared_native_fusion','options':{
                'roles':['grouped_gate_up_activation_quant'],
                'program_profiles':{'grouped_gate_up_activation_quant':'ascend_w4a8_msd'}}}
    graphs=target_graphs(raw,phase='decode',tokens=6,sequences=concurrency,context=4096,scope='body')
    ops={o.op_id:o for o in graphs[0]}
    assert sum(dict(o.metadata).get('device_kernel_group',{}).get('entry',False)
               for o in ops.values())==150
    for layer in range(3,78):
        p=f's0.l{layer}.'
        assert ops[p+'pack'].local_shape==(bucket//16+(bool(bucket%16)),6144)
        assert p+'expert_up.quantize' not in ops
        consumer='grouped_gate_up_activation_quant.native' if fused else 'expert_up'
        assert ops[p+consumer].deps==(p+'dispatch',)
        assert 'activation_scales' in ops[p+'dispatch'].returns
        md=dict(ops[p+'dispatch'].metadata)
        capacity=(bucket//16+(bool(bucket%16)))*16*8
        assert md['native_abi_outputs'][0]['shape']==(capacity,6144)
        assert md['native_abi_outputs'][1]['shape']==(capacity,)
        assert md['native_abi_outputs'][2]['shape']==(capacity*128,)
        assert md['native_abi_outputs'][6]['shape']==(capacity,)
        assert md['activation_expected_rows']==bucket*8/16
        gm=dict(ops[p+consumer].metadata)
        assert gm['activation_payload_producer']==gm['activation_scale_producer']==p+'dispatch'
        assert gm['native_input_capacity_shape']==(capacity,6144)
        assert gm['native_input_scale_capacity_shape']==(capacity,)
        assert gm['native_output_capacity_shape'][0]==capacity
        assert p+'dispatch' in ops[p+'combine'].deps
        assert ops[p+'router'].local_shape[0]==bucket
        assert ops[p+'mc2_tp_gather'].deps==(p+'unpack',)
    assert sum(o.recipe=='tensor_concat' for o in ops.values())==(150 if concurrency==2 else 0)
