"""Source cursor, valid-mask and physical transport versus independent integer oracles."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.mc2_graph import NETWORK
from mindie_sim.prediction.mc2_phase import contract
from mindie_sim.prediction.source_force_eplb import ledger
from test_joint_collective_execution import wire

ROOT = Path(__file__).resolve().parents[1]


def brute(experts, ep, top_k, allocated, valid):
    banks = [0] * experts
    peers = [[0] * ep for _ in range(ep)]
    for source in range(ep):
        for token in range(allocated):
            if source * allocated + token >= valid:
                continue
            for slot in range(top_k):
                cursor = (allocated * top_k + 1) * source + token * top_k + slot
                dest = cursor % ep
                bank = (cursor // ep) % (experts // ep)
                banks[dest * (experts // ep) + bank] += 1
                peers[source][dest] += 1
    return tuple(banks), tuple(map(tuple, peers))


def test_closed_cursor_counts_match_token_oracle_including_wrap_and_idle_sources():
    for ep in (2, 4, 8):
        for experts in (ep, 2 * ep, 3 * ep):
            for top_k in (1, ep, experts - 1):
                for allocated in (1, 2, 5):
                    for valid in sorted({1, allocated, allocated * ep // 2, allocated * ep - 1, allocated * ep}):
                        actual = ledger(experts, ep, top_k, allocated, valid)
                        banks, peers = brute(experts, ep, top_k, allocated, valid)
                        assert actual.expert_rows == banks
                        assert actual.peer_assignments == peers
                        assert sum(banks) == sum(map(sum, peers)) == valid * top_k
                        for dst in range(ep):
                            assert sum(actual.rank_expert_rows[dst]) == sum(row[dst] for row in peers)
                        assert tuple(sum(row) for row in peers) == tuple(v * top_k for v in actual.valid_tokens_per_rank)


def case(source=True):
    path = ROOT / 'examples/cases/high-concurrency-20261030/kimi-hccl-source-conditional/a5_kimi_tp8_e16_bf16_c32_p4k_hccl_observed.json'
    raw = json.loads(path.read_text())
    raw['model']['layers'] = 2
    raw['runtime']['kernels'].pop('service_regimes', None)
    if source:
        raw['runtime']['kernels']['force_eplb_source'] = {
            'kind': 'vllm_ascend_sp_mc2', 'global_bs': 0}
    return raw


def graphs(raw, batch):
    return target_graphs(raw, scope='body', phase='decode', tokens=1,
                         context=4104, sequences=batch, apply_body_defaults=True)


@pytest.mark.parametrize('batch,expected', [(1, (1, 0)), (2, (2, 0)), (4, (3, 1)),
                                          (8, (5, 3)), (16, (8, 8)), (32, (16, 16))])
def test_ordinary_sp_caller_keeps_allocated_router_but_filters_actual_expert_work(batch, expected):
    raw = case(); original = deepcopy(raw)
    ops = {op.op_id: op for op in graphs(raw, batch)[0]}
    plain = {op.op_id: op for op in graphs(case(False), batch)[0]}
    assert raw == original
    up = ops['s0.l1.expert_up']; down = ops['s0.l1.expert_down']
    for op in (up, down):
        md = dict(op.metadata)
        assert md['matrix_group_rows'] == expected
        assert md['matrix_groups'] == sum(v > 0 for v in expected)
        assert op.local_shape[0] == sum(expected)
        assert op.flops == 2 * sum(expected) * op.local_shape[1] * op.local_shape[2]
        routing = md['routing_realization']
        assert routing['expert_placement'] == 'contiguous_ep'
        assert routing['representative_expert_ids'] == (0, 1)
        assert routing['valid_global_tokens'] == batch
        assert routing['allocated_global_tokens'] == 8 * max(1, (batch + 7) // 8)
    # Padding belongs to allocated router/top-k tensors; masks do not compact
    # their shapes or silently change the original balanced scenario.
    for name in ('router', 'topk'):
        before, after = plain[f's0.l1.{name}'], ops[f's0.l1.{name}']
        assert (before.local_shape, before.flops, before.vector_ops) == (after.local_shape, after.flops, after.vector_ops)
    if batch <= 8:
        assert dict(plain['s0.l1.expert_up'].metadata)['matrix_group_rows'] == (4, 4)
    dispatch = ops['s0.l1.dispatch']
    md = dict(dispatch.metadata)
    assert md['mc2_phase']['source_tokens'] == max(1, (batch + 7) // 8)
    assert md['mc2_phase']['source_assignments'] == max(1, (batch + 7) // 8) * 8
    assert md['mc2_phase']['route_matrix'] == ledger(
        16, 8, 8, max(1, (batch + 7) // 8), batch).peer_assignments


def protocol(ops):
    selected = [op for op in ops if 'mc2_phase' in dict(op.metadata)]
    names = {op.op_id for op in selected}
    return tuple(replace(op, deps=tuple(d for d in op.deps if d in names)) for op in selected)


@pytest.mark.parametrize('batch', [1, 4, 32])
def test_masked_peer_matrix_transposes_real_rows_and_engine_wires_once(batch):
    raw = case(); ops = graphs(raw, batch)[0]
    indexed = {op.op_id: op for op in ops}
    source = ledger(16, 8, 8, max(1, (batch + 7) // 8), batch)
    rank = source.representative_rank
    matrix = source.peer_assignments
    ep = len(matrix)
    expected = {}
    payload_total = 0
    full_frames = {}
    for kind, op_id in (('dispatch', 's0.l1.dispatch'),
                        ('combine', 's0.l1.unpack')):
        op = indexed[op_id]
        phase_raw = dict(op.metadata)['mc2_phase']
        assert phase_raw['route_matrix'] == matrix
        assert phase_raw['route_reverse'] is (kind == 'combine')
        packets = contract(phase_raw, 64).layout.packet_bytes
        cells = (matrix[rank]
                 if kind == 'dispatch'
                 else tuple(matrix[source_rank][rank]
                            for source_rank in range(ep)))
        frames = tuple((peer, cells[peer]) for peer in range(ep)
                       if peer != rank and cells[peer] > 0)
        expected[kind] = (op_id, frames, packets)
        off_diagonal = sum(
            matrix[src][dst] for src in range(ep)
            for dst in range(ep) if src != dst)
        full_frames[kind] = off_diagonal
        payload_total += off_diagonal * packets
    # Count/status is an integral known channel, separate from payload frames.
    count_bytes = 16 * 32 * (ep - 1)
    total = payload_total + count_bytes
    result = graph_costs(
        raw, {0: protocol(ops)}, isolated=False, compact=True,
        input_ready_ms=7)
    bridge = result['profile']['native_bridge']
    assert bridge['status'] == 'actual'
    assert bridge['known_vs_expected'] == 'exact_directed_source_frames'
    assert bridge['transport_tx_bytes'] == pytest.approx(total)
    assert bridge['transport_rx_bytes'] == pytest.approx(total)
    assert bridge['transport_frames'] == (
        full_frames['dispatch'] + full_frames['combine'])
    payload_edges = 2 * sum(
        1 for src in range(ep) for dst in range(ep)
        if src != dst and matrix[src][dst] > 0)
    assert bridge['payload_edges'] == bridge['transport_edges'] == payload_edges
    assert bridge['count_edges'] == ep * (ep - 1)
    groups = {group['op_id']: group for group in bridge['operators']}
    for kind, (op_id, frames, packets) in expected.items():
        group = groups[op_id]
        assert group['kind'] == kind
        name = next(iter(group['payload_frames']))
        assert group['payload_frames'][name] == dict(frames)
        assert group['directed_frames'] == full_frames[kind]
        assert group['transport_frames'] == full_frames[kind]
        links = group['link_by_phase_by_rank'][name]['payload_exchange']
        assert sum(links.values()) == sum(
            count for _peer, count in frames) * packets
        if kind == 'dispatch':
            status = group['transport_by_rank'][name]['count_status_exchange']
            assert status['edges'] == ep * (ep - 1)
            assert status['tx_bytes'] == status['rx_bytes'] == count_bytes
        else:
            assert 'count_status_exchange' not in group['transport_by_rank'][name]
        work = dict(group['work_by_rank'][name])
        assert sum(value for pipe, value in work.items()
                   if 'link' in pipe) == pytest.approx(
            group['transport_tx_bytes'] + group['transport_rx_bytes'])
    serviced = result['execution']['serviced']
    tx = sum(value for key, value in serviced.items() if ':tx:' in key)
    rx = sum(value for key, value in serviced.items() if ':rx:' in key)
    # A directed frame is one coupled transfer: source TX and destination RX
    # are the same reservation, served exactly once at every real hop.
    assert tx == pytest.approx(total)
    assert rx == pytest.approx(total)
    assert sum(wire(result).values()) == pytest.approx(2 * total)
    # Two senders into one destination reserve the same real RX port; that
    # port's busy time is the serialized sum, not the maximum of one transfer.
    incoming = [(destination_index, tuple(
        src for src in range(ep)
        if src != destination_index and matrix[src][destination_index] > 0))
        for destination_index in range(ep)]
    destination_index, senders = max(incoming, key=lambda item: len(item[1]))
    if len(senders) >= 2:
        rx_port = next(
            name for name in serviced
            if f':rx:port{destination_index}' in name)
        column_frames = sum(matrix[src][destination_index] for src in senders)
        row_frames = sum(matrix[destination_index][src]
                         for src in range(ep) if src != destination_index)
        expected_rx = (column_frames * expected['dispatch'][2]
                       + row_frames * expected['combine'][2]
                       + (count_bytes // ep))
        assert serviced[rx_port] == pytest.approx(expected_rx)
        assert result['execution']['resource_busy_ms'][rx_port] > 0
        detail = graph_costs(
            raw, {0: protocol(ops)}, isolated=False, input_ready_ms=7,
            trace_limit=4096)
        events = {row['task']: row for row in detail['events']}
        tx_ports = set()
        for source in senders:
            task = (f"r0:i0:s0.l1.dispatch:payload_exchange:"
                    f"{source}->{destination_index}:hop0:service")
            assert task in events
            demand = events[task]['demand']
            assert rx_port in demand
            tx_ports.update(name for name in demand if ':tx:' in name)
        # Two independent senders reserve the same destination RX while each
        # owns a distinct source TX; the coupled demand is one transfer.
        assert len(tx_ports) == len(senders)
    assert result['profile']['steps_by_replica'][0]['start_ms'] == 7
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert not any(result['execution']['live_allocated'].values())


def test_ordinary_uniform_expectation_keeps_payload_work_and_labels_expected():
    raw = case(False); ops = graphs(raw, 4)[0]
    native = [op for op in ops if op.recipe == 'arch35_fullmesh_resource_phase']
    for op in native:
        phase_raw = dict(op.metadata)['mc2_phase']
        assert 'route_matrix' not in phase_raw
        assert phase_raw['uniform_payload_bytes'] == 8 * 7680
    result = graph_costs(
        raw, {0: protocol(ops)}, isolated=False, compact=True,
        input_ready_ms=7)
    bridge = result['profile']['native_bridge']
    assert bridge['status'] == 'actual'
    assert bridge['known_vs_expected'] == 'declared_destination_expectation'
    assert bridge['transport_frames'] == bridge['directed_frames'] == 0
    assert bridge['transport_tx_bytes'] == pytest.approx(863744)
    assert bridge['transport_rx_bytes'] == pytest.approx(863744)
    groups = {group['op_id']: group for group in bridge['operators']}
    for op_id in ('s0.l1.dispatch', 's0.l1.unpack'):
        group = groups[op_id]
        name = next(iter(group['payload_frames']))
        assert group['payload_frames'][name] == {}
        payload = group['transport_by_rank'][name]['payload_exchange']
        assert payload['edges'] == 56
        assert payload['frames'] == 0
        assert payload['tx_bytes'] == payload['rx_bytes'] == 430080
        assert group['link_by_phase_by_rank'][name][
            'payload_exchange']['link_hccs'] == 53760
    # TX/RX and timing remain physical even without witnessed integer frames.
    serviced = result['execution']['serviced']
    assert sum(value for key, value in serviced.items()
               if ':tx:' in key) == pytest.approx(863744)
    assert sum(value for key, value in serviced.items()
               if ':rx:' in key) == pytest.approx(863744)
    assert sum(wire(result).values()) == pytest.approx(2 * 863744)
    assert result['profile']['steps_by_replica'][0]['start_ms'] == 7
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0


def test_masked_transport_keeps_work_under_physical_link_scaling():
    from mindie_sim.prediction.structure import parse_case
    raw = parse_case(case()); ops = {0: protocol(graphs(raw, 4)[0])}
    results = []
    for scale in (.5, 1, 2):
        links = tuple(replace(link, capacity_bytes_per_ms=100 * scale) for link in raw.machine.links)
        modified = replace(raw, machine=replace(raw.machine, links=links))
        results.append(graph_costs(modified, ops, isolated=False))
    assert results[0]['duration_ms'] > results[1]['duration_ms'] > results[2]['duration_ms']
    assert results[0]['execution']['work'] == results[1]['execution']['work'] == results[2]['execution']['work']


def test_logical_valid_rows_drive_source_mass_not_physical_padding():
    from test_prediction_system import case as base_case
    raw = base_case(dp=16, tp=1, ep=16)
    raw['model'].update(ffn='moe', num_experts=64, top_k=6,
                        moe_intermediate_size=32, layers=1)
    raw['runtime']['routing'] = {'policy': 'balanced'}
    raw['runtime']['kernels'] = {
        'native_moe_protocol': {'kind': 'arch35_fullmesh',
                                'max_inflight_per_rank': 8}}
    raw['machine']['device'].update(
        rates={'memory': 1e9, 'vector': 1e9, 'mte2_aic': 1e9,
               'mte2_aiv': 1e9, 'mte3': 1e9},
        resource_units={'vector': 4, 'mte2_aiv': 4, 'mte3': 4})
    raw['machine']['links'][0].update(
        semantics='per_pair', capacity_bytes_per_ms=1000)
    sequences = [1, 1] + [0] * 14
    graphs = target_graphs(raw, phase='decode', tokens=1,
                           sequences=sequences, scope='body')
    source = received = 0
    destination_rows = tuple(dict(dict(
        next(op for op in graphs[0]
             if op.op_id == 's0.l0.expert_down').metadata)
        ['routing_realization'])['rank_assignment_rows'])
    for replica, ops in graphs.items():
        indexed = {op.op_id: op for op in ops}
        dispatch = dict(indexed['s0.l0.dispatch'].metadata)['mc2_phase']
        combine = dict(indexed['s0.l0.unpack'].metadata)['mc2_phase']
        source += dispatch['source_assignments']
        received += combine['source_assignments']
        assert dispatch['received_assignments'] == destination_rows[replica]
        assert combine['source_assignments'] == destination_rows[replica]
        if not sequences[replica]:
            # Physical padding remains a capacity/ID row, not an outgoing row.
            assert dispatch['source_tokens'] == 0
            assert dispatch['source_assignments'] == 0
    assert source == received == 12
    indexed = {op.op_id: op for op in graphs[0]}
    routing = dict(dict(indexed['s0.l0.expert_down'].metadata)
                   ['routing_realization'])
    assert routing['global_tokens'] == 2
    assert sum(routing['rank_assignment_rows']) == 12
    assert routing['expected_rows_per_expert'] == 0.75
    dispatch_raw = dict(indexed['s0.l0.dispatch'].metadata)['mc2_phase']
    combine_raw = dict(indexed['s0.l0.unpack'].metadata)['mc2_phase']
    dispatch_matrix = dispatch_raw['route_matrix_bytes']
    combine_matrix = combine_raw['route_matrix_bytes']
    dispatch_packet = contract(dispatch_raw, 4).layout.packet_bytes
    combine_packet = contract(combine_raw, 4).layout.packet_bytes
    assert all(combine_matrix[j][i] / combine_packet ==
               pytest.approx(dispatch_matrix[i][j] / dispatch_packet)
               for i in range(16) for j in range(16) if i != j)
    expected_offdiag = 11.0
    assert sum(map(sum, dispatch_matrix)) / dispatch_packet == pytest.approx(12.0)
    assert sum(map(sum, combine_matrix)) / combine_packet == pytest.approx(12.0)
    assert expected_offdiag == pytest.approx(
        sum(dispatch_matrix[i][j] / dispatch_packet
            for i in range(16) for j in range(16) if i != j))
    assert sum(map(sum, combine_matrix)) / combine_packet == pytest.approx(12.0)
    payload_edges = 2 * 11 * 2  # 2 occupied origins x 11 peers, both directions
    result = graph_costs(
        raw, graphs, isolated=False, compact=True, input_ready_ms=7,
        external_inputs={'analysis_input': 7})
    bridge = result['profile']['native_bridge']
    assert bridge['known_vs_expected'] == 'declared_destination_expectation'
    assert bridge['transport_tx_bytes'] == bridge['transport_rx_bytes'] > 0
    assert bridge['payload_edges'] == payload_edges
    assert bridge['count_edges'] == 16 * 15
    assert result['execution']['live_tasks'] == 0
    assert result['execution']['live_programs'] == 0
    assert all(step['device_end_ms'] == step['end_ms']
               for step in result['profile']['steps_by_replica'].values())
