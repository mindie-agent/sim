"""A3 TP8/TP16 physical links beneath an explicit Mesh/NHR source schedule.

[Role] Verify ordinary caller phases on registered SIO and HCCS_SW links.
[Invariants] This is an explicit algorithm declaration, not captured selection.
  Endpoint TWO_DIE_REGULAR is distinct from two NPU ranks in one package.
[Timeouts & Locks] Small sequential controls; no subprocesses or locks.
[Side-Effect Policy] Prints a compact report; no measurements or remote work.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.hccl_hierarchy import contract
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[2]


def declared(ranks):
    return {
        'machine': {'backend': 'ascend', 'sku': 'Ascend910_9362',
                    'num_devices': ranks, 'ranks_per_node': 16, 'dies_per_package': 2},
        'model': {'name': 'geometry', 'layers': 1, 'hidden_size': 1024,
                  'num_heads': 32, 'num_kv_heads': 32, 'head_dim': 32,
                  'intermediate_size': 2048, 'vocab_size': 128},
        'workload': {'requests': [{'id': 'r', 'prompt_tokens': 4, 'output_tokens': 2}]},
        'runtime': {'tp': ranks, 'max_num_batched_tokens': 128, 'kernels': {
            'hccl': {'allreduce': {
                'expansion': 'AICPU_TS', 'template': 'parallel_mesh_nhr_route0',
                'topology_levels': 2, 'level0_size': 2, 'max_loop_bytes': 8388608,
                'signal_bytes': 4}}}},
        'metadata': {'selection': 'source-compatible declaration, not observed A3 executor',
                     'signal_bytes': 'explicit control geometry, not measured wire framing'},
    }


def main():
    started = time.monotonic()
    reports = []
    for ranks in (8, 16):
        raw = declared(ranks)
        graphs = target_graphs(raw, phase='decode', tokens=32, context=128, scope='body')
        chosen = tuple(o for o in graphs[0] if 'hccl_source_group' in dict(o.metadata))
        ids = {o.op_id for o in chosen}
        isolated = {0: tuple(replace(o, deps=tuple(d for d in o.deps if d in ids))
                             for o in chosen)}
        first = next(o for o in chosen if 'hccl_phase' in dict(o.metadata))
        s = dict(first.metadata)['hccl_selection']
        plan = contract(select(raw['runtime']['kernels']['hccl']['allreduce'],
                               collective='allreduce', ranks=ranks,
                               payload_bytes=s['payload_bytes']))
        calls = sum(bool(dict(o.metadata).get('device_kernel_group', {}).get('entry'))
                    for o in chosen)
        payload = sum(sum(map(sum, p.exchange)) for p in plan.phases if p.exchange)
        signals = sum(len(p.signals) for p in plan.phases) * 4
        scans = []
        for channel in ('d2d_intra_package', 'd2d_intra_node'):
            for factor in (.5, 1, 2):
                varied = deepcopy(raw)
                nominal = 448e9 if channel == 'd2d_intra_package' else 196e9
                varied['machine']['transfer_channels'] = {
                    channel: {'effective_uni_bw_gbps': nominal * factor / 1e9}}
                result = graph_costs(varied, isolated, isolated=False, input_ready_ms=7)
                serviced = result['execution']['serviced']
                tx = sum(v for k, v in serviced.items() if k.startswith('link:') and ':tx:' in k)
                rx = sum(v for k, v in serviced.items() if k.startswith('link:') and ':rx:' in k)
                assert tx == rx == calls * (payload + signals)
                assert not result['execution']['live_tasks']
                assert not result['execution']['live_programs']
                assert all(row['ready_ms'] >= 7 for row in result['timeline'])
                scans.append({'channel': channel, 'factor': factor,
                              'duration_ms': result['duration_ms'],
                              'wire_bytes': tx, 'tx_capacity_bytes': tx, 'rx_capacity_bytes': rx})
        for channel in ('d2d_intra_package', 'd2d_intra_node'):
            values = [row['duration_ms'] for row in scans if row['channel'] == channel]
            assert values[0] > values[1] > values[2]
        normalized = parse_case(raw)
        reports.append({'ranks': ranks, 'case': raw, 'calls': calls,
                        'links': [link.to_dict() for link in normalized.machine.links],
                        'selection': plan.selection.to_dict(),
                        'payload_wire_bytes_per_call': payload,
                        'declared_signal_wire_bytes_per_call': signals, 'scans': scans})
    paths = ['mindie_sim/prediction/hccl_hierarchy.py', 'mindie_sim/prediction/topology.py',
             'mindie_sim/backends/ascend.py', 'mindie_sim/backends/profiles/ascend.json']
    print(json.dumps({'schema': 'mindie-sim.a3-physical-hierarchy/1',
        'boundary': 'two isolated ordinary-body AllReduce calls per TP configuration',
        'coverage': 'explicit source-compatible AICPU schedule on declared A3 topology',
        'unmodeled': ['captured native selector', 'independent controller issue-rate calibration',
                      'CCU endpoint-die classification', 'interlevel chunk overlap'],
        'source_sha256': {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        'results': reports, 'wall_seconds': time.monotonic() - started}, indent=2))


if __name__ == '__main__':
    main()
