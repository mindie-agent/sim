"""TP8/16 public model-caller controls for explicit CCU endpoint ownership."""
import hashlib
import json
import math
import resource
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_ccu_two_die import contract, channels

ROOT = Path(__file__).resolve().parents[2]
start = time.monotonic()
path = ROOT / 'examples/communication/ccu-two-die-allgather-tp8.json'
base = json.loads(path.read_text())
rows = []
domains = []
for ranks in (8, 16):
    case = deepcopy(base)
    case['machine']['num_devices'] = ranks
    case['machine']['links'][0]['members'] = list(range(ranks))
    case['runtime']['tp'] = ranks
    case['model']['vocab_size'] = 384 * ranks // 2
    declaration = case['runtime']['kernels']['hccl']['allgather']
    declaration['channel_die_map'] = [[None if src == dst else int((dst-src)%ranks >= ranks//2)
                                      for dst in range(ranks)] for src in range(ranks)]
    graphs = target_graphs(case, phase='decode', tokens=1, context=16, scope='model')
    calls = {}
    for replica, ops in graphs.items():
        selected = tuple(op for op in ops if 'hccl_source_group' in dict(op.metadata))
        ids = {op.op_id for op in selected}
        calls[replica] = tuple(replace(op, deps=tuple(d for d in op.deps if d in ids)) for op in selected)
    selection = select(declaration, collective='allgather', ranks=ranks, payload_bytes=384)
    plan = contract(selection)
    totals = {}
    for phase in plan.phases:
        for role, value in phase.work:
            totals[role] = totals.get(role, 0) + value
    domains.append({'ranks': ranks, 'selection': selection.to_dict(),
                    'rank0_peer_groups': channels(selection)[0][0],
                    'rank0_self_copy_owner': channels(selection)[0][1],
                    'per_rank_local_work': totals,
                    'full_group_payload_bytes': ranks * (ranks-1) * 384,
                    'full_group_notify_count': 3 * ranks * (ranks-1),
                    'unmodeled': plan.unmodeled})
    for scan in ('ccu_copy', 'ccu_buffer_bytes', 'ccu_control', 'memory', 'link', 'switch'):
        for factor in (.5, 1, 2):
            varied = deepcopy(case)
            if scan == 'link':
                varied['machine']['links'][0]['capacity_bytes_per_ms'] *= factor
            elif scan == 'switch':
                varied['machine']['links'][0]['switch_capacity_bytes_per_ms'] *= factor
            else:
                varied['machine']['device']['rates'][scan] *= factor
            result = graph_costs(varied, calls, isolated=False, input_ready_ms=7, trace_limit=10000)
            service = result['execution']['serviced']
            tx = sum(value for key, value in service.items() if ':tx:' in key)
            rx = sum(value for key, value in service.items() if ':rx:' in key)
            switch = sum(value for key, value in service.items() if key.endswith(':switch_plane'))
            # N384 -> two 288-byte transactions; every 8-byte notification has
            # its own 288-byte frame. Tx/Rx/plane are simultaneous capacities.
            expected_wire = ranks * (ranks-1) * (2 * 288 + 3 * 288)
            assert math.isclose(tx, expected_wire) and math.isclose(rx, tx) and math.isclose(switch, tx)
            assert math.isclose(service['memory@r0:s0'], (ranks+2)*384)
            assert math.isclose(service['ccu_copy@r0:s0'], 3*384)
            assert all(event['start_ms'] >= 7 for event in result['events'] if event['demand'])
            copies = [event for event in result['events'] if event['demand'].get('ccu_copy@r0:s0')]
            owner_work = {str(owner): sum(event['demand']['ccu_copy@r0:s0'] for event in copies
                          if tuple(event['placement']['ccu_copy@r0:s0']) == (owner,)) for owner in (0, 1)}
            assert owner_work == {'0': 768, '1': 384}
            assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
            rows.append({'ranks': ranks, 'scan': scan, 'factor': factor,
                         'duration_ms': result['duration_ms'], 'physical_wire_bytes': tx,
                         'tx_capacity_service_bytes': tx, 'rx_capacity_service_bytes': rx,
                         'switch_capacity_service_bytes': switch, 'ccu_copy_bytes_by_owner': owner_work,
                         'local_work': {role: service[role+'@r0:s0'] for role in
                                        ('memory', 'ccu_copy', 'ccu_buffer_bytes', 'ccu_control', 'hccl_control')},
                         'live_tasks': result['execution']['live_tasks'],
                         'live_programs': result['execution']['live_programs']})
    for scan in ('ccu_copy', 'ccu_buffer_bytes', 'ccu_control', 'memory', 'link', 'switch'):
        values = [row['duration_ms'] for row in rows if row['ranks'] == ranks and row['scan'] == scan]
        assert values[0] + 1e-10 >= values[1] and values[1] + 1e-10 >= values[2]
files = ['mindie_sim/prediction/hccl_ccu_two_die.py', 'mindie_sim/prediction/hccl_contracts.py',
         'mindie_sim/prediction/hccl_graph.py', 'mindie_sim/prediction/hccl_signals.py']
print(json.dumps({
    'boundary': 'one ordinary model logits AllGather per TP8/TP16 graph, isolated from unrelated compute',
    'timing_basis': 'explicit synthetic rates; no device measurements or selector cost coefficients',
    'upstream_source_receipt': 'hccl-ccu-two-die-20261030.sources.json',
    'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
    'domains': domains, 'scans': rows, 'wall_seconds': time.monotonic()-start,
    'maxrss_platform_native_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}, indent=2))
