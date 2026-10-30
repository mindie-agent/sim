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
    md = dict(ops['s0.l1.pack'].metadata)
    assert md['mc2_phase']['source_tokens'] == max(1, (batch + 7) // 8)
    assert md['mc2_phase']['source_assignments'] == max(1, (batch + 7) // 8) * 8


def protocol(ops):
    selected = [op for op in ops if 'mc2_phase' in dict(op.metadata) or op.recipe == NETWORK]
    names = {op.op_id for op in selected}
    return tuple(replace(op, deps=tuple(d for d in op.deps if d in names)) for op in selected)


@pytest.mark.parametrize('batch', [1, 4, 32])
def test_masked_peer_matrix_transposes_real_rows_and_engine_wires_once(batch):
    raw = case(); ops = graphs(raw, batch)[0]
    indexed = {op.op_id: op for op in ops}
    source = ledger(16, 8, 8, max(1, (batch + 7) // 8), batch)
    total = 0
    for kind in ('dispatch', 'combine'):
        net = indexed[f's0.l1.{kind}.payload_network']; meta = dict(net.metadata)
        phase_op = next(op for op in ops if dict(op.metadata).get('mc2_phase', {}).get('kind') == kind)
        width = contract(dict(phase_op.metadata)['mc2_phase'], 64).layout.packet_bytes
        expected = source.payload_matrix(width, reverse=kind == 'combine')
        assert meta['peer_traffic']['payload_bytes'] == expected
        assert meta['peer_volume_basis'] == 'group_mean'
        assert net.link_bytes == sum(map(sum, expected)) / 8
        assert 'reverse_of_collective' not in meta
        total += sum(map(sum, expected))
    total += 8 * 16 * 32 * 7 / 8  # Count/status remains collective even for idle sources.
    result = graph_costs(raw, {0: protocol(ops)}, isolated=False, input_ready_ms=7)
    # Tx and Rx are simultaneous capacity reservations, not duplicate packets.
    demands = wire(result)
    assert sum(v for k, v in demands.items() if ':tx:' in k) == pytest.approx(total)
    assert sum(v for k, v in demands.items() if ':rx:' in k) == pytest.approx(total)
    assert min(row['ready_ms'] for row in result['timeline']) >= 7
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert not any(result['execution']['live_allocated'].values())


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
