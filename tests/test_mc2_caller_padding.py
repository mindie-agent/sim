"""MC2 padding stays after full-row router/shared work and is masked on wire."""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs, target_graphs
from test_moe_token_partition import case, graph


def padded_case(*, source=False, native=False):
    raw = case(tp=4, experts=16)
    kernels = raw['runtime']['kernels']
    kernels['moe_token_partition']['padding'] = 'right_to_tp'
    if not native:
        kernels.pop('native_moe_protocol')
    if source:
        kernels['force_eplb_source'] = {'kind': 'vllm_ascend_mc2_tp', 'global_bs': 0}
    return raw


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('source', [False, True])
def test_valid_input_projection_padding_and_gather_trim(native, source):
    raw = padded_case(native=native, source=source)
    ops = graph(raw, 3); by_id = {op.op_id: op for op in ops}
    assert by_id['s0.l0.qkv'].local_shape[0] == 3
    assert by_id['s0.l0.router'].local_shape[0] == 3
    assert by_id['s0.l0.shared.gate_up'].local_shape[0] == 3
    for name, width, dtype_bytes in (('input', 32, 2), ('logits', 16, 2)):
        pad = by_id[f's0.l0.mc2_{name}_pad']
        assert pad.local_shape == (4, width)
        assert dict(pad.metadata)['input_shapes'] == ((3, width), (1, width))
        assert pad.mte2_bytes == pad.mte3_bytes == 4 * width * dtype_bytes
        assert by_id[f's0.l0.mc2_{name}_view'].deps == (pad.op_id,)
    assert by_id['s0.l0.topk'].local_shape[0] == 1
    gather = by_id['s0.l0.mc2_tp_gather']
    assert gather.local_shape == (4,32)
    assert gather.link_bytes == 3 * 32 * 2
    trim = by_id['s0.l0.mc2_output_trim']
    assert trim.local_shape == (3,32) and trim.deps == (gather.op_id,)
    assert trim.hbm_bytes == 0
    assert trim.op_id in by_id['s0.l0.join'].deps
    ledger = dict(by_id['s0.l0.expert_up'].metadata)['routing_realization']
    assert sum(ledger['global_expert_rows']) == 3 * 2
    assert sum(ledger['rank_assignment_rows']) == 3 * 2
    networks = [op for op in ops if dict(op.metadata).get('peer_traffic') and
                dict(op.metadata).get('channel') != 'count_status']
    assert len(networks) == 2
    forward, reverse = [dict(op.metadata)['peer_traffic']['payload_bytes'] for op in networks]
    assert not any(forward[3])  # allocated row on last rank is an invalid MC2 mask bit
    assert all(reverse[i][j] == forward[j][i] for i in range(4) for j in range(4))
    report = graph_costs(raw, {0:ops}, external_inputs={'analysis_input':7}, input_ready_ms=7)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert all(row['ready_ms'] >= 7 for row in report['timeline'])


def test_explicit_capture_padding_remains_physical_before_mc2_padding():
    raw = padded_case()
    raw['runtime'].update(graph_mode='auto', graph_capture_mode='full', graph_capture_keys=[4])
    ops = graph(raw, 3); by_id = {op.op_id: op for op in ops}
    assert by_id['s0.l0.qkv'].local_shape[0] == 4
    assert by_id['s0.l0.router'].local_shape[0] == 4
    assert not any(op.recipe == 'tensor_concat' for op in ops)
    assert by_id['s0.l0.mc2_tp_gather'].local_shape == (4,32)
    ledger = dict(by_id['s0.l0.expert_up'].metadata)['routing_realization']
    assert sum(ledger['global_expert_rows']) == 4 * 2


def test_rank_copy_materialization_precedes_trim_and_preserves_allocation_bytes():
    raw = padded_case()
    raw['runtime']['kernels']['moe_token_partition']['gather_output_materialization'] = 'per_rank_copy'
    ops = graph(raw,3); by_id={o.op_id:o for o in ops}
    copies = [o for o in ops if '.mc2_gather_copy' in o.op_id]
    assert len(copies)==4
    assert sum(o.mte2_bytes for o in copies)==sum(o.mte3_bytes for o in copies)==4*32*2
    assert by_id['s0.l0.mc2_output_trim'].deps==('s0.l0.mc2_gather_complete',)


def test_tp16_m12_valid_counts_do_not_turn_into_16_routed_tokens():
    raw = padded_case(source=True)
    raw['runtime'].update(tp=16,ep=16)
    raw['machine']['num_devices']=16
    raw['machine']['links'][0]['members']=list(range(16))
    raw['model'].update(num_heads=16,num_kv_heads=16,num_experts=256,top_k=8)
    ops=graph(raw,12); by_id={o.op_id:o for o in ops}
    ledger=dict(by_id['s0.l0.expert_up'].metadata)['routing_realization']
    assert ledger['allocated_global_tokens']==16
    assert ledger['valid_global_tokens']==12
    assert ledger['valid_tokens_per_rank']==(1,)*12+(0,)*4
    assert sum(ledger['global_expert_rows'])==96
    assert by_id['s0.l0.router'].local_shape==(12,256,32)
    assert by_id['s0.l0.mc2_tp_gather'].local_shape==(16,32)
    assert by_id['s0.l0.mc2_output_trim'].local_shape==(12,32)
