"""Batch query ownership without merging independent causal sequences.

[Role] Exercise the public indexed-MLA graph and its common Engine program.
[Invariants] Batching changes scheduling, never each sequence's selected rows
  or physical work. Explicit source/recipe boundaries remain authoritative.
[Timeouts & Locks] Finite component graphs; no subprocesses or locks.
[Side-Effect Policy] In-memory cases only; no devices or measurements.
"""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.indexed_attention import (
    IMPLEMENTATION, build, sequence_query_positions,
)
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.operator_analysis import _target_ops
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure_contracts import predicted_contract


def case(*, serial=0, compact=True):
    pipeline = {'kind': 'indexed_attention_pipeline', 'scratch_policy': 'offchip',
                'operand_modes': True, 'compact': compact}
    return {
        'model': {'catalog_key': 'glm-5.3', 'quantization': 'w4a8', 'layers': 1},
        'machine': {'backend': 'ascend', 'sku': 'Ascend910_9362', 'num_devices': 16},
        'runtime': {'tp': 16, 'ep': 16, 'kernels': {
            'programs': {'indexed_attention': pipeline},
            'recipes': {f'latent_attention{i}': deepcopy(pipeline) for i in range(serial)}}},
        'workload': {'requests': [{'id': 'r', 'prompt_tokens': 1, 'output_tokens': 1}]},
    }


def attention_ops(raw, *, batch=1, query=1, context=4096):
    return [op for op in target_graphs(raw, phase='decode', tokens=query,
                                      sequences=batch, context=context, scope='body')[0]
            if dict(op.metadata).get('program_kind') == 'indexed_attention']


def run(raw, ops, *, trace=0):
    return graph_costs(raw, ops, input_ready_ms=5,
                       external_inputs={dep: 7 for op in ops for dep in op.deps},
                       isolated=False, trace_limit=trace)


def elapsed(report):
    return report['complete_ms'] - 7


def test_eight_one_query_sequences_share_owners_and_one_call_not_eight_serial_calls():
    raw = case()
    one = run(raw, attention_ops(raw))
    batched_ops = attention_ops(raw, batch=8)
    assert len(batched_ops) == 1
    contract = predicted_contract(batched_ops[0].to_dict())
    assert contract[0:2] == ('sparse_attention_active', 'SparseFlashAttention')
    assert contract[2][0][0] == (8, 4, 512)
    batch = run(raw, batched_ops, trace=500)
    old = case(serial=8)
    separate_ops = attention_ops(old, batch=8)
    assert len(separate_ops) == 8
    serial = run(old, separate_ops)
    assert elapsed(serial) == pytest.approx(8 * elapsed(one))
    assert elapsed(batch) == pytest.approx(elapsed(one))
    assert batch['execution']['work'] == serial['execution']['work']
    assert all(row['start_ms'] >= 7 for row in batch['events'] if row['demand'])
    owners = {unit for row in batch['events'] for name, units in row['placement'].items()
              if name.startswith('matrix@') for unit in units}
    assert owners == set(range(8))
    assert batch['execution']['live_tasks'] == batch['execution']['live_programs'] == 0


def test_batch_q6_remains_hbm_limited_and_conserves_work_through_batch_eight():
    previous = 0
    for batch in range(1, 9):
        raw, old = case(), case(serial=batch)
        result = run(raw, attention_ops(raw, batch=batch, query=6))
        serial = run(old, attention_ops(old, batch=batch, query=6))
        assert result['execution']['work'] == serial['execution']['work']
        assert elapsed(result) >= previous - 1e-12
        assert elapsed(result) <= elapsed(serial) + 1e-12
        if batch > 1:
            assert elapsed(result) < elapsed(serial)
        previous = elapsed(result)
    # Shared HBM is real: 48 query owners do not give eight times throughput.
    bandwidth = dict(parse_case(raw).machine.dies[0].rates)['hbm']
    assert elapsed(result) >= result['execution']['work']['hbm_bytes'] / bandwidth * 1000 - 1e-12


def test_ragged_batches_keep_sequence_local_positions_and_causal_contexts():
    sequences = ((1, 0), (0, 90), (2, 510), (1, 1024))
    assigned = [item for owner in range(2)
                for item in sequence_query_positions(owner, sequences, 2)]
    assert assigned == [(0, 0), (2, 0), (2, 1), (3, 0)]
    raw = case()
    parsed = parse_case(raw)
    views = tuple(RequestView(str(i), query, context, False, 'decode')
                  for i, (query, context) in enumerate(sequences) if query)
    ops = [op for op in _target_ops(parsed, views, 'body')
           if dict(op.metadata).get('program_kind') == 'indexed_attention']
    assert len(ops) == 1
    shape = dict(ops[0].metadata)['indexed_attention_geometry']
    assert shape['sequences'] == ((1, 0), (2, 510), (1, 1024))
    assert dict(ops[0].metadata)['logical_pairs'] == 1 + 511 + 512 + 1025
    geometry = Geometry.from_devices(parsed.machine.dies)
    combined = build(ops[0], geometry, scratch_policy='offchip').work
    independent = Work()
    for query, context in shape['sequences']:
        single_shape = {key: value for key, value in shape.items() if key != 'sequences'}
        single_shape.update(query=query, context=context)
        single = replace(ops[0], metadata=(('indexed_attention_geometry', single_shape),))
        independent += build(single, geometry, scratch_policy='offchip').work
    assert combined == independent
    compact = run(raw, ops)
    expanded_raw = case(compact=False)
    expanded = run(expanded_raw, ops)
    assert compact['execution']['serviced'] == pytest.approx(expanded['execution']['serviced'])
    assert compact['execution']['live_tasks'] == expanded['execution']['live_tasks'] == 0


def test_explicit_family_and_source_boundaries_are_not_silently_replaced():
    family = case()
    family['runtime']['kernels']['families'] = {'attention': {'kind': 'envelope'}}
    assert len(attention_ops(family, batch=2)) == 2
    source = case()
    source['runtime']['kernels']['lowerings'] = {
        'indexed_attention': {'implementation': IMPLEMENTATION,
                              'options': {'scratch_policy': 'offchip'}}}
    with pytest.raises(ModelingError, match='single_contiguous_sequence'):
        attention_ops(source, batch=2)
    # INT8 KV is still outside the BF16 physical pipeline domain.
    parsed = parse_case(case())
    op = attention_ops(case(), batch=2)[0]
    shape = {**dict(op.metadata)['indexed_attention_geometry'], 'cache_dtype': 'int8'}
    invalid = replace(op, metadata=(('indexed_attention_geometry', shape),))
    with pytest.raises(ModelingError, match='source_geometry_outside_reviewed_domain'):
        build(invalid, Geometry.from_devices(parsed.machine.dies), scratch_policy='offchip')
