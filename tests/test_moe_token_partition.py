"""MC2 caller token ownership, restored TP output and grouped work oracles."""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.moe_token_partition import KIND
from test_prediction_system import case as base_case


def case(tp=4, experts=16, shared=True):
    raw = base_case(tp=tp, ep=tp)
    raw['model'].update(layers=1, ffn='moe', num_experts=experts, top_k=2,
                        moe_intermediate_size=32, num_shared_experts=int(shared))
    raw['machine']['device'].update(
        rates={'memory': 1e9, 'vector': 1e9, 'mte2_aic': 1e9, 'mte2_aiv': 1e9, 'mte3': 1e9},
        resource_units={'vector': 4, 'mte2_aiv': 4, 'mte3': 4})
    raw['runtime']['routing'] = {'policy': 'balanced'}
    raw['runtime']['kernels'] = {
        'moe_token_partition': {'kind': KIND},
        'native_moe_protocol': {'kind': 'arch35_fullmesh', 'max_inflight_per_rank': 4},
    }
    return raw


def graph(raw, rows=8):
    return target_graphs(raw, phase='decode', tokens=1, sequences=rows,
                         context=128, scope='body')[0]


@pytest.mark.parametrize('tp,rows,experts', [(2, 2, 16), (4, 8, 16), (4, 64, 16)])
def test_full_router_local_selection_global_expert_ledger_and_final_gather(tp, rows, experts):
    raw = case(tp, experts)
    ops = graph(raw, rows)
    by_id = {o.op_id: o for o in ops}
    local = rows // tp
    assert by_id['s0.l0.router'].local_shape[0] == rows
    assert by_id['s0.l0.topk'].local_shape == (local, experts)
    assert by_id['s0.l0.topk'].deps == ('s0.l0.mc2_logits_view',)
    for suffix in ('mc2_input_view', 'mc2_logits_view'):
        view = by_id['s0.l0.' + suffix]
        assert view.family == 'barrier'
        assert view.flops == view.vector_ops == view.hbm_bytes == 0
    for suffix in ('pack', 'unpack'):
        phase = dict(by_id['s0.l0.' + suffix].metadata)['mc2_phase']
        assert phase['source_tokens'] == local
        assert phase['source_assignments' if suffix == 'pack' else 'received_assignments'] == local * 2
    gate = by_id['s0.l0.expert_up']
    ledger = dict(gate.metadata)['routing_realization']
    assert sum(ledger['global_expert_rows']) == rows * 2
    assert ledger['source_partitions'] == 1
    assert gate.local_shape[0] == max(ledger['rank_assignment_rows'])
    assert gate.flops * tp == 2 * rows * 2 * 64 * 32
    assert by_id['s0.l0.shared.gate_up'].local_shape[0] == rows
    assert by_id['s0.l0.shared.tp'].family == 'allreduce'
    gather = by_id['s0.l0.mc2_tp_gather']
    assert gather.deps == ('s0.l0.unpack',)
    assert gather.local_shape == (rows, 32)
    assert gather.link_bytes == (tp - 1) * local * 32 * 2
    assert set(by_id['s0.l0.join'].deps) == {'s0.l0.mc2_tp_gather', 's0.l0.shared.tp'}
    assert not any(o.op_id.endswith('.expert_reduce') for o in ops)
    # All expert-bank work is invariant to where the caller slices source rows.
    control = deepcopy(raw)
    control['runtime']['kernels'] = {}
    old = {o.op_id: o for o in graph(control, rows)}
    for suffix in ('expert_up', 'expert_down', 'expert_act', 'router',
                   'shared.gate_up', 'shared.swiglu', 'shared.down', 'shared.tp'):
        actual, previous = by_id['s0.l0.' + suffix], old['s0.l0.' + suffix]
        assert (actual.flops, actual.vector_ops, actual.hbm_bytes,
                actual.mte2_bytes, actual.mte3_bytes, actual.link_bytes) == (
                previous.flops, previous.vector_ops, previous.hbm_bytes,
                previous.mte2_bytes, previous.mte3_bytes, previous.link_bytes)


def test_reconstructed_output_waits_for_native_combine_and_has_one_allgather():
    raw = case()
    graphs = target_graphs(raw, phase='decode', tokens=1, sequences=8,
                           context=128, scope='body')
    report = graph_costs(raw, graphs, external_inputs={'analysis_input': 7},
                         input_ready_ms=7, trace_limit=300)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    timeline = {row['id'].split(':', 1)[-1]: row for row in report['timeline']}
    gather, combine, join = (timeline['s0.l0.' + name]
                             for name in ('mc2_tp_gather', 'unpack', 'join'))
    assert gather['ready_ms'] >= combine['complete_ms']
    assert join['ready_ms'] >= gather['complete_ms']
    assert min(row['ready_ms'] for row in report['timeline']) >= 7
    assert sum(o.family == 'allgather' for o in graphs[0]) == 1


def test_hccl_source_expands_the_real_local_input_allgather():
    raw = case()
    raw['runtime']['kernels']['hccl'] = {
        'allgather': {'expansion': 'AICPU_TS', 'template': 'mesh', 'axes': ['tp']}}
    ops = graph(raw, 8)
    grouped = [o for o in ops if dict(o.metadata).get('hccl_source_group') ==
               's0.l0.mc2_tp_gather.hccl']
    assert grouped
    for op in grouped:
        selection = dict(op.metadata)['hccl_selection']
        assert selection['payload_bytes'] == 2 * 32 * 2
        assert selection['ranks'] == 4
    report = graph_costs(raw, {0: ops}, external_inputs={'analysis_input': 0})
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


def test_shared_absent_returns_full_rows_and_padding_is_not_invented():
    raw = case(shared=False)
    ops = graph(raw, 4)
    gather = next(o for o in ops if o.op_id == 's0.l0.mc2_tp_gather')
    assert gather.local_shape == (4, 32)
    assert any(gather.op_id in o.deps for o in ops if o.op_id != gather.op_id)
    with pytest.raises(ValueError, match='tp_divisible'):
        graph(raw, 3)


def test_caller_partition_preserves_generic_transport_and_integer_peer_work():
    raw = case()
    raw['runtime']['kernels'].pop('native_moe_protocol')
    declared = graph(raw, 8)
    by_id = {op.op_id: op for op in declared}
    # The caller also wraps V2 transports. Selecting its tensor views must not
    # install the arch35 V3 protocol, nor change global expert-bank work.
    assert by_id['s0.l0.topk'].local_shape == (2, 16)
    assert by_id['s0.l0.dispatch'].local_shape == (4, 32)
    assert dict(by_id['s0.l0.dispatch'].metadata)['source_assignment_partitions'] == 1
    assert by_id['s0.l0.dispatch'].link_bytes == 4 * 32 * 2 * 3 / 4
    assert by_id['s0.l0.combine'].link_bytes == by_id['s0.l0.dispatch'].link_bytes
    assert not any(dict(op.metadata).get('mc2_phase') for op in declared)
    assert by_id['s0.l0.mc2_tp_gather'].deps == ('s0.l0.unpack',)
    report = graph_costs(raw, {0: declared}, external_inputs={'analysis_input': 9},
                         input_ready_ms=9)
    timeline = {row['op_id']: row for row in report['timeline']}
    assert timeline['s0.l0.mc2_tp_gather']['ready_ms'] >= timeline['s0.l0.unpack']['complete_ms']
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert all(row['ready_ms'] >= 9 for row in report['timeline'])


def test_explicit_gather_materialization_copies_each_real_rank_slice_once():
    from mindie_sim.structure import export_structure

    raw = case()
    raw['runtime']['kernels']['moe_token_partition']['gather_output_materialization'] = 'per_rank_copy'
    ops = graph(raw, 8)
    copies = [o for o in ops if '.mc2_gather_copy' in o.op_id]
    assert len(copies) == 4
    assert all(o.local_shape == (2, 32) and o.dtype == 'bf16' for o in copies)
    assert sum(o.mte2_bytes for o in copies) == sum(o.mte3_bytes for o in copies) == 8 * 32 * 2
    assert all(o.deps == ('s0.l0.mc2_tp_gather',) for o in copies)
    by_id = {o.op_id: o for o in ops}
    complete = by_id['s0.l0.mc2_gather_complete']
    assert set(complete.deps) == {o.op_id for o in copies}
    assert 's0.l0.mc2_gather_complete' in by_id['s0.l0.join'].deps
    manifest = export_structure(raw, phase='decode', tokens=1, sequences=8,
                                context=128, scope='body')
    declared_copies = [node for node in manifest['nodes']
                       if '.mc2_gather_copy' in node['node_id']]
    assert len(declared_copies) == len(copies)
    tensor = [[2, 32], 'bf16', 'ND']
    assert all(node['contract'] == ['native_tensor_abi', 'TensorMove', [tensor], [tensor]]
               for node in declared_copies)
    assert {node['operator']['metadata']['gathered_slice'] for node in declared_copies} == set(range(4))
    assert len({node['physical_signature'] for node in declared_copies}) == 1
    assert sum(node['physical_work']['hbm_bytes'] for node in declared_copies) == 2 * 8 * 32 * 2
    control = case()
    base = graph_costs(control, {0: graph(control, 8)}, external_inputs={'analysis_input': 0})
    copied = graph_costs(raw, {0: ops}, external_inputs={'analysis_input': 0})
    assert copied['duration_ms'] > base['duration_ms']
    assert copied['execution']['work']['hbm_bytes'] - base['execution']['work']['hbm_bytes'] == 2 * 8 * 32 * 2
