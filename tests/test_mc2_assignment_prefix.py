"""Independent sender-index scan arithmetic and physical owner service checks."""
from collections import defaultdict

import pytest

from mindie_sim.prediction.operators.mc2_fullmesh import (
    assignment_prefix_work, dispatch_layout, dispatch_plan)
from test_mc2_owner_and_prefix import run_phase


@pytest.mark.parametrize('tokens,topk,owners', [
    (0, 8, 56), (1, 8, 56), (2, 3, 5), (4, 8, 56), (17, 5, 7), (64, 8, 56), (128, 8, 56)])
def test_closed_form_equals_actual_source_indices_not_uniform_row_counts(tokens, topk, owners):
    naive = defaultdict(int)
    for index in range(tokens * topk):
        if index >= topk:
            naive[index % owners] += 5 * index - 1
    actual = dict(assignment_prefix_work(tokens, topk, tokens * topk, tuple(range(owners))))
    assert actual == {owner: naive[owner] for owner in range(owners)}
    assert sum(actual.values()) == sum(5 * i - 1 for i in range(topk, tokens * topk))


def test_missing_retained_indices_do_not_invent_a_first_n_routing_mask():
    row = dispatch_layout(240)
    for tokens, retained in ((2, 3), (2, .5), (.5, 4)):
        plan = dispatch_plan(row, source_tokens=tokens, top_k=8, source_assignments=retained,
                             received_assignments=8, experts=16, ep=8, aiv_count=64)
        phase = next(p for p in plan.phases if p.name == 'send_read_quantize')
        prefix = dict(phase.details)['assignment_prefix_scan']
        assert prefix['scope'] == 'unresolved_retained_indices'
        assert prefix['owner_vector_work'] is None
        assert dict(phase.work)['hbm'] == retained * 480
        assert any('retained valid assignment indices' in x for x in plan.unmodeled_features)


@pytest.mark.parametrize('tokens,topk,units', [(2, 3, 8), (4, 8, 64), (17, 5, 16)])
def test_engine_keeps_owner_specific_prefix_then_payload_and_exact_total_work(tokens, topk, units):
    elapsed, engine, plan, contract = run_phase('send_read_quantize', tokens=tokens,
                                             top_k=topk, units=units)
    prefix = dict(next(p for p in contract.phases if p.name == 'send_read_quantize').details)['assignment_prefix_scan']
    expected = dict(prefix['owner_vector_work'])
    assert engine.serviced['vector'] == pytest.approx(sum(expected.values()))
    for owner in plan.metadata['active_owner_domain']:
        trace = [x for x in engine.trace if f'owner{owner}:' in x['task']]
        scan = next(x for x in trace if x['task'].endswith(':assignment_prefix_scan'))
        read = next(x for x in trace if x['task'].endswith(':read'))
        # Resource rates are per-device; each owner gets exactly one lane.
        assert scan['end_ms'] - scan['start_ms'] == pytest.approx(expected[owner] / (64000 / units))
        assert read['start_ms'] >= scan['end_ms']
    assert elapsed > 0
    counts = next(p for p in contract.phases if p.name == 'count_compute')
    c = len(counts.owner_domain)
    assert dict(counts.work)['vector'] == 16 * (5 * tokens * topk - 1) + c * 9 * 16
    assert engine.serviced['hbm'] == pytest.approx(tokens * topk * 480)


def test_vector_rate_scaling_preserves_prefix_work_and_compact_task_count():
    results = [run_phase('send_read_quantize', tokens=4096, top_k=8,
                         units=64, rate=64000 * factor) for factor in (.5, 1, 2)]
    assert results[0][0] > results[1][0] > results[2][0]
    for _, engine, plan, _ in results:
        # 32768 assignments must not become an expanded quadratic scan graph.
        assert engine.summary()['completed_tasks'] < 4 * 64
        assert plan.metadata['source_service_details']['assignment_prefix_scan']['vector_work'] == sum(
            5 * i - 1 for i in range(8, 32768))
    assert results[0][1].serviced == pytest.approx(results[2][1].serviced)
