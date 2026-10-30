"""Full-peer AIV programs routed over two physical levels, with one algorithm."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction import hccl_aiv, hccl_aiv_gather
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_graph import lower
from mindie_sim.prediction.structure import parse_case


def declared(ranks, collective, template):
    return {
        'model': {'name': 'geometry', 'layers': 1, 'hidden_size': 1024,
                  'num_heads': 32, 'num_kv_heads': 32, 'head_dim': 32,
                  'intermediate_size': 2048, 'vocab_size': 128},
        'machine': {'backend': 'ascend', 'sku': 'Ascend910_9362',
                    'num_devices': ranks, 'ranks_per_node': ranks//2,
                    'dies_per_package': 2,
                    # Exercise the retained source's 192.5KiB active UB contract
                    # on these physical links; the 192KiB A3 default cannot fit
                    # this kernel. This is a declared capacity control, not an
                    # assertion that the captured A3 binary uses this program.
                    'device': {'buffers': {'ub': 256*1024}}},
        'runtime': {'tp': ranks, 'kernels': {'hccl': {collective: {
            'expansion': 'AIV', 'template': template, 'topology_levels': 2,
            'block_limit': 40}}}},
        'workload': {'requests': [{'id': 'r', 'prompt_tokens': 4, 'output_tokens': 1}]},
    }


def caller(raw, collective):
    # Start with the ordinary body's real TP tensor. AllGather's declaration
    # consumes a local shard rather than the AllReduce traffic-factor field.
    unbound = deepcopy(raw)
    unbound['runtime']['kernels'] = {}
    ops = target_graphs(unbound, scope='body', phase='decode', tokens=32, context=128)[0]
    original = next(op for op in ops if op.family == 'allreduce')
    ranks = raw['runtime']['tp']
    payload = round(original.link_bytes / (2*(ranks-1)/ranks))
    factor = ranks-1 if collective == 'allgather' else 2*(ranks-1)/ranks
    original = replace(original, family=collective, recipe=collective,
                       deps=(), link_bytes=payload*factor,
                       metadata=tuple({**dict(original.metadata), 'collective': collective}.items()))
    parsed = parse_case(raw)
    return lower((original,), parsed.runtime, parsed.model.activation_dtype), payload


@pytest.mark.parametrize('ranks', (8, 16))
@pytest.mark.parametrize('collective,template', (
    ('allreduce', 'mesh_oneshot'), ('allreduce', 'mesh_twoshot'), ('allgather', 'mesh')))
def test_two_physical_levels_retain_one_full_peer_algorithm_and_routed_bytes(ranks, collective, template):
    raw = declared(ranks, collective, template)
    graph, payload = caller(raw, collective)
    selection = select(raw['runtime']['kernels']['hccl'][collective],
                       collective=collective, ranks=ranks, payload_bytes=payload)
    module = hccl_aiv_gather if collective == 'allgather' else hccl_aiv
    plan = module.contract(selection, 40)
    # The source TopoMatchOneLevel picks a full-group accessible level. It
    # never inserts an invented SIO reduction followed by a second algorithm.
    assert plan.phases == module.contract(replace(selection, topology_levels=1), 40).phases
    expected_wire = sum(sum(map(sum, phase.exchange)) for phase in plan.phases)
    report = graph_costs(raw, graph, input_ready_ms=7, trace_limit=10000)
    serviced = report['execution']['serviced']
    tx = {k: v for k, v in serviced.items() if k.startswith('link:') and ':tx:' in k}
    rx = {k: v for k, v in serviced.items() if k.startswith('link:') and ':rx:' in k}
    assert sum(tx.values()) == pytest.approx(expected_wire)
    assert sum(rx.values()) == pytest.approx(expected_wire)
    for link in ('d2d_intra_package', 'd2d_intra_node', 'd2d_inter_node'):
        assert any(link in k and v for k, v in tx.items())
    assert all(row['start_ms'] >= 7 for row in report['events'] if row['demand'])
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert not any(report['execution']['live_allocated'].values())
    assert report['bounds']['combined_ms'] <= report['duration_ms'] + 1e-9


@pytest.mark.parametrize('channel', ('d2d_intra_package', 'd2d_intra_node', 'd2d_inter_node'))
def test_each_physical_level_changes_arrival_without_changing_source_work(channel):
    raw = declared(8, 'allreduce', 'mesh_twoshot')
    graph, _ = caller(raw, 'allreduce')
    reports = []
    for factor in (.5, 1, 2):
        varied = deepcopy(raw)
        # Make this independently selected physical link the limiting supply.
        varied['machine']['transfer_channels'] = {
            channel: {'effective_uni_bw_gbps': factor}}
        reports.append(graph_costs(varied, graph, isolated=False))
    assert reports[0]['duration_ms'] > reports[1]['duration_ms'] > reports[2]['duration_ms']
    assert reports[0]['execution']['serviced'] == pytest.approx(reports[1]['execution']['serviced'])
    assert reports[1]['execution']['serviced'] == pytest.approx(reports[2]['execution']['serviced'])
