"""Source-macro boundaries and logical SIMT accesses on the real Engine."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim.prediction.moe_init_instructions import source_operations
from test_moe_init_mxfp8 import case, graph, RECIPE
from test_source_scalar_services import execute


def declared(raw, intervals):
    raw = deepcopy(raw)
    raw['runtime']['kernels']['invocations'] = [{'match': {}, 'startup_ms': 0, 'minimum_ms': 0}]
    raw['machine'].setdefault('device', {})['compute_arch'] = {'instruction_services': {
        role: {'unit': 'input_blocks32' if '32_blocks' in role else 'int32_thread_accesses',
               'owner_axis': 'vector', 'issue_ns': ns,
               'provenance': {'kind': 'synthetic_supported_shape_only_not_hardware_measurement'}}
        for role, ns in intervals.items()}}
    return raw


@pytest.mark.parametrize('factor', (.5, 1, 2))
def test_sort_macros_execute_once_in_source_order_on_owner_zero(factor):
    raw = declared(case(16), {})
    op = next(op for op in graph(raw, 16) if op.recipe == RECIPE)
    before, baseline, _ = execute(raw, op)
    intervals = dict(source_concat32_blocks=2000/factor,
                     source_sort32_blocks=3000/factor, source_extract32_blocks=5000/factor)
    after, engine, prepared = execute(declared(raw, intervals), op)
    assert after-before == pytest.approx(16 * .010 / factor)
    assert {k: engine.serviced[k] for k in baseline.serviced} == pytest.approx(baseline.serviced)
    rows = [row for row in engine.trace if row['task'].endswith('_macro')]
    assert len(rows) == 3
    for row, (role, ns) in zip(rows, intervals.items()):
        assert row['serviced'][role+'@r0:s0'] == 16
        assert row['placement'][role+'@r0:s0'] == (0,)
        assert row['end_ms']-row['start_ms'] == pytest.approx(16*ns/1e6)
    assert rows[0]['end_ms'] <= rows[1]['start_ms'] < rows[1]['end_ms'] <= rows[2]['start_ms']
    post = next(row for row in engine.trace if row['task'].endswith('sort_post_lane_work'))
    assert rows[-1]['end_ms'] <= post['start_ms']
    diag = dict(prepared.metadata)['program_execution']['source_operation_services']
    assert diag['sort']['sort']['priced_service_work'] == {'source_sort32_blocks': 16}


@pytest.mark.parametrize('assignments,units', ((15, 64), (65, 8), (1024, 40)))
def test_logical_simt_accesses_match_independent_thread_iteration_counts(assignments, units):
    per = (assignments + units - 1)//units
    active = (assignments + per - 1)//per
    _, count, inverse = source_operations(assignments, per, active, 288)
    for owner in range(active):
        rows = min(per, assignments-owner*per)
        first = last = 0
        for thread in range(2048):
            for i in range(thread, rows, 2048):
                first += 1 + (i > 0)
                last += 1 + (i < rows-1)
        assert count[owner].counts == {'source_simt_gm_load32': first+last}
        assert inverse[owner].counts == {'source_simt_gm_load32': rows, 'source_simt_gm_store32': rows}


def test_simt_declared_access_capacity_uses_real_owners_and_preserves_payload():
    raw = declared(case(16), {})
    op = next(op for op in graph(raw,16) if op.recipe == RECIPE)
    meta = dict(op.metadata)
    meta['moe_init_config'] = dict(meta['moe_init_config'], tokens=3, topk=5)
    op = replace(op, metadata=tuple(meta.items()))
    _, baseline, _ = execute(raw, op)
    _, engine, prepared = execute(declared(raw, {'source_simt_gm_load32': 2000,
                                                'source_simt_gm_store32': 3000}), op)
    assert engine.serviced['source_simt_gm_load32@r0:s0'] == pytest.approx(45)  # 15 owners: count2 + inverse1
    assert engine.serviced['source_simt_gm_store32@r0:s0'] == pytest.approx(15)
    assert {k: engine.serviced[k] for k in baseline.serviced} == pytest.approx(baseline.serviced)
    accesses = [row for row in engine.trace if any('source_simt_' in k for k in row['demand'])]
    owners = {row['placement']['source_simt_gm_load32@r0:s0'] for row in accesses}
    assert owners == {(i,) for i in range(15)}
    diag = dict(prepared.metadata)['program_execution']['source_operation_services']
    assert diag['atomic_count']['payload_bytes_per_owner'] == 288*4
    assert not any('atomic_rmw' in k or 'simt_warp' in k for k in engine.serviced)


def test_unavailable_macro_services_do_not_remove_known_vector_or_direct_memory_work():
    raw=case(16);op=next(op for op in graph(raw,16) if op.recipe==RECIPE)
    _, engine, prepared = execute(raw,op)
    assert any(row['task'].endswith('sort_known_lane_work') for row in engine.trace)
    assert not any(row['task'].endswith('_macro') for row in engine.trace)
    diag=dict(prepared.metadata)['program_execution']['source_operation_services']
    assert diag['sort']['sort']['unpriced_service_work']=={'source_sort32_blocks':16}
    assert sum(row['counts']['source_simt_gm_load32'] for row in diag['count_by_owner'])==1920
