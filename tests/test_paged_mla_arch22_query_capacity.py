"""Zero-KV host oracle and native capacity preserve active attention work."""
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.paged_mla_arch22 import IMPLEMENTATION, build
from mindie_sim.prediction.paged_mla_arch22_dispatch import source_dispatch
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure_contracts import predicted_contract

ROOT = Path(__file__).resolve().parents[1]
ORACLE = json.loads((ROOT/'tests/fixtures/paged_mla_arch22_query_capacity.json').read_text())


def test_unmodified_host_skips_zero_kv_rows_without_changing_active_owners():
    for case in ORACLE['cases']:
        raw = case['host_result']
        count = case['active_sequence_count']
        active = source_dispatch(tuple(raw['contexts'][:count]), raw['heads'], raw['cores'])
        expected = {**active, 'contexts': raw['contexts'],
                    'ends': [list(end) for end in active['ends']]}
        # The terminal endpoint crosses the zero-work storage rows. Every
        # active owner boundary and every FD merge remains the same.
        expected['ends'][-1][0] = case['native_query_capacity']
        assert raw == expected


@pytest.mark.parametrize('tp,active,capacity', [(8, 6, 8), (16, 12, 16)])
def test_query_capacity_changes_native_abi_without_inventing_kv_work(tp, active, capacity):
    raw = json.loads((ROOT/'examples/cases/high-concurrency-20261030/kimi-source/'
                      'a3_kimi4_tp8_w4a8_c1_source.json').read_text())
    raw['machine']['num_devices'] = tp
    raw['runtime'].update(tp=tp, ep=tp)
    binding = {'implementation': IMPLEMENTATION, 'options': {}}
    raw['runtime']['kernels']['lowerings']['paged_mla_attention'] = binding

    def attention():
        return next(op for op in target_graphs(raw, scope='body', phase='decode',
                    tokens=1, sequences=active, context=4096)[0] if op.recipe==IMPLEMENTATION)

    baseline = attention()
    binding['options']['native_query_capacity'] = capacity
    padded = attention()
    geometry = Geometry.from_devices(parse_case(raw).machine.dies)
    before, after = build(baseline, geometry), build(padded, geometry)
    assert before.work == after.work
    assert before.metadata == after.metadata
    assert baseline.semantic_shape == padded.semantic_shape
    assert padded.local_shape[0] == active
    assert padded.padded_shape[0] == padded.launch_shape[0] == capacity
    assert predicted_contract(baseline.to_dict())[-1] == active
    contract = predicted_contract(padded.to_dict())
    assert contract[-1] == capacity
    assert contract[2][0][0] == padded.padded_shape
    outputs = operator_costs(raw, [baseline, padded], input_ready_ms=7)['operators']
    assert outputs[0]['duration_ms'] == outputs[1]['duration_ms']
    for result in outputs:
        assert not result['service_before_input']
        assert result['live_tasks'] == 0
        assert not any(result['remaining_buffers'].values())


@pytest.mark.parametrize('tp,active,capacity', [(8, 6, 8), (16, 12, 16)])
def test_graph_query_capacity_reaches_caller_and_explicit_override_wins(tp, active, capacity):
    raw = json.loads((ROOT/'examples/cases/high-concurrency-20261030/kimi-source/'
                      'a3_kimi4_tp8_w4a8_c1_source.json').read_text())
    raw['machine']['num_devices'] = tp
    raw['runtime'].update(tp=tp, ep=tp)
    binding = {'implementation': IMPLEMENTATION, 'options': {}}
    raw['runtime']['kernels']['lowerings']['paged_mla_attention'] = binding

    def graph():
        ops = target_graphs(raw, scope='body', phase='decode', tokens=1,
                            sequences=active, context=126000)[0]
        return next(op for op in ops if op.recipe == IMPLEMENTATION), ops

    eager, _ = graph()
    raw['runtime'].update(graph_mode='auto', graph_capture_mode='full',
                          graph_capture_keys=[8, 16])
    captured, ops = graph()
    # The selected graph bucket reaches the existing projection caller. TP
    # padding is separate: an eager B6 on TP8 still has native capacity 6.
    query = next(op for op in ops if op.layer == captured.layer and op.recipe == 'q_a')
    assert query.local_shape[0] == capacity
    assert dict(captured.metadata)['graph_captured']
    binding['options']['native_query_capacity'] = active
    overridden, _ = graph()

    geometry = Geometry.from_devices(parse_case(raw).machine.dies)
    plans = [build(op, geometry) for op in (eager, captured, overridden)]
    assert plans[0].work == plans[1].work == plans[2].work
    assert plans[0].metadata == plans[1].metadata == plans[2].metadata
    assert eager.semantic_shape == captured.semantic_shape == overridden.semantic_shape
    for op, expected in ((eager, active), (captured, capacity), (overridden, active)):
        native = dict(op.metadata)['paged_mla_geometry']
        assert native['contexts'] == (126000,) * active
        assert native['active_sequence_count'] == op.local_shape[0] == active
        assert native['inactive_sequence_kv_length'] == 0
        assert native['native_query_capacity'] == expected
        assert op.padded_shape[0] == op.launch_shape[0] == expected
        assert predicted_contract(op.to_dict())[-1] == expected
