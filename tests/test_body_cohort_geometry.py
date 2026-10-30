"""Per-DP query counts retain physical expert ownership and idle rendezvous."""
from copy import deepcopy

import pytest

from mindie_sim import graph_costs, operator_costs, predict, predict_pd, target_graphs
from mindie_sim.prediction.peer_traffic import PeerTraffic
from test_balanced_experts import balanced_case


def cohort_case(*, tp=1, dp=8, ep=8):
    raw = balanced_case(tp=tp, dp=dp, ep=ep, experts=48, tokens=1)
    raw['model']['top_k'] = 6
    raw['runtime'].update(graph_mode='auto', graph_capture_mode='full',
                          graph_capture_keys=[1, 4, 8])
    return raw


def routing(ops):
    return next(dict(op.metadata)['routing_realization'] for op in ops
                if 'routing_realization' in dict(op.metadata))


def wire(report):
    return sum(value for key, value in report['execution']['serviced'].items()
               if key.startswith('link:'))


def test_single_active_dp_has_six_expert_rows_and_idle_collective_participants():
    raw = cohort_case()
    graphs = target_graphs(raw, phase='decode', tokens=1, context=32,
                           sequences=[1, 0, 0, 0, 0, 0, 0, 0], scope='body')
    ledger = [routing(ops) for ops in graphs.values()]
    assert [sum(item['local_expert_rows']) for item in ledger] == [1] * 6 + [0, 0]
    assert sum(sum(item['local_expert_rows']) for item in ledger) == 6
    assert [item['representative_rank'] for item in ledger] == list(range(8))
    for rank, ops in graphs.items():
        exchanges = [op for op in ops if op.family == 'alltoall']
        assert len(exchanges) == 2
        if rank:
            assert not any(op.family in ('attention', 'norm', 'router') for op in ops)
            assert ledger[rank]['padded_local_tokens'] == 0
        if rank >= 6:
            assert not any(op.family == 'grouped_gemm' for op in ops)
    report = graph_costs(raw, graphs, input_ready_ms=7,
                         external_inputs={'analysis_input': 7}, trace_limit=100000)
    assert report['replay_scope_by_replica'] == {0: 'full', **{r: 'none' for r in range(1, 8)}}
    assert report['replay_scope'] == 'mixed'
    assert all(row['ready_ms'] >= 7 for row in report['timeline'])
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    # Uniform-peer transport remains a separate declared expectation. No token
    # IDs or per-destination integer router realization have been invented.
    assert wire(report) == pytest.approx(2 * 6 * 32 * 2 * 7 / 8)
    result = operator_costs(raw, graphs)
    assert result['replicas'] == list(range(8))
    assert all(not row['service_before_input'] for row in result['operators'])


def test_symmetric_counts_preserve_scalar_work_and_capture_padding_is_per_dp():
    raw = cohort_case()
    scalar = predict(raw, scope='body', context=32, sequences=1)
    listed = predict(raw, scope='body', context=32, sequences=[1] * 8)
    assert scalar['body_forward_ms'] == listed['body_forward_ms']
    assert scalar['execution']['serviced'] == listed['execution']['serviced']
    assert listed['geometry']['valid_query_tokens'] == 8
    counts = [2, 0, 1, 0, 0, 0, 0, 0]
    padded = predict(raw, scope='body', context=32, sequences=counts)
    assert list(padded['geometry']['padded_query_tokens_by_replica'].values()) == [4, 0, 1, 0, 0, 0, 0, 0]
    assert padded['geometry']['valid_query_tokens'] == 3
    graphs = target_graphs(raw, phase='decode', tokens=1, context=32,
                           sequences=counts, scope='body')
    assert sum(sum(routing(ops)['local_expert_rows']) for ops in graphs.values()) == 5 * 6


def test_tp_representative_is_local_max_not_global_rank_zero():
    raw = cohort_case(tp=4, dp=4, ep=16)
    graphs = target_graphs(raw, phase='decode', tokens=1, sequences=[1, 0, 0, 0], scope='body')
    values = [routing(ops) for ops in graphs.values()]
    assert [v['representative_rank'] for v in values] == [0, 4, 8, 12]
    assert [sum(v['local_expert_rows']) for v in values] == [1, 1, 0, 0]
    assert [v['represented_ep_ranks'] for v in values] == [tuple(range(i, i + 4)) for i in (0, 4, 8, 12)]
    assert all(v['rank_timing_assumption'] == 'heaviest_local_tp_shard_upper_bound' for v in values)
    assert sum(values[0]['rank_assignment_rows']) == 6
    # Multiplying representative work by TP is an upper bound, not an exact
    # sum of actual shard work. The global histogram retains the exact six.
    assert sum(sum(v['local_expert_rows']) * 4 for v in values) == 8


def test_unequal_source_rows_preserve_reverse_peer_matrix():
    raw = cohort_case(dp=4, ep=4)
    raw['runtime']['graph_mode'] = 'eager'
    graphs = target_graphs(raw, phase='decode', tokens=1, context=16,
                           sequences=[1, 0, 3, 2], scope='body')
    forward = next(op for op in graphs[0] if op.recipe == 'expert_dispatch')
    backward = next(op for op in graphs[0] if op.recipe == 'expert_combine')
    f = PeerTraffic.from_mapping(dict(forward.metadata)['peer_traffic'], 4)
    b = PeerTraffic.from_mapping(dict(backward.metadata)['peer_traffic'], 4)
    assert all(b.payload(i, j) == f.payload(j, i) for i in range(4) for j in range(4))
    for rank, ops in graphs.items():
        ret = next(op for op in ops if op.recipe == 'expert_combine')
        assert ret.link_bytes == b.row_total(rank)
    report = graph_costs(raw, graphs, external_inputs={'analysis_input': 2}, input_ready_ms=2)
    assert wire(report) == pytest.approx(2 * 6 * 6 * 32 * 2 * 3 / 4)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


def test_pd_body_retains_full_cohort_counts_in_both_pools():
    raw = cohort_case()
    machine, runtime = raw.pop('machine'), raw.pop('runtime')
    raw['pools'] = {name: {'machine': deepcopy(machine), 'runtime': deepcopy(runtime)}
                    for name in ('prefill', 'decode')}
    raw['connector'] = {'bytes_per_s': 1e9}
    result = predict_pd(raw, scope='body', sequences=[1] + [0] * 7, context=16)
    for pool in result['pools'].values():
        assert pool['geometry']['valid_query_tokens'] == 1
        assert pool['geometry']['padded_query_tokens'] == 1
        assert pool['execution']['live_tasks'] == pool['execution']['live_programs'] == 0


def test_serving_single_request_uses_actual_dp_expert_shards():
    raw = cohort_case()
    raw['runtime']['graph_mode'] = 'eager'
    raw['workload']['requests'][0].update(prompt_tokens=1, output_tokens=1)
    result = predict(raw)
    graph = graph_costs(raw, target_graphs(raw, phase='prefill', tokens=1,
                         sequences=[1] + [0] * 7, scope='model'))
    # Serving additionally samples/commits; expert matrix and wire service are
    # nevertheless the same physical cohort, including zero-query replicas.
    grouped_matrix = lambda out: sum(v for k, v in out['execution']['serviced'].items() if k.startswith('matrix@'))
    assert grouped_matrix(result) == pytest.approx(grouped_matrix(graph))
    assert wire(result) == pytest.approx(wire(graph))
    assert result['execution']['live_tasks'] == 0


def test_model_scope_moe_mtp_preserves_replica_rows_and_target_completion():
    from dataclasses import replace
    from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
    from mindie_sim.prediction.structure import parse_case
    parsed = parse_case(cohort_case())
    draft = DraftSpec(kind='mtp', layers=1, hidden_size=32, intermediate_size=64,
                      num_heads=4, num_kv_heads=2, head_dim=8, q_width=32, kv_width=16,
                      vocab_size=128, num_experts=48, top_k=6, expert_intermediate_size=64,
                      target_shared_embedding=True, target_shared_lm_head=True)
    parsed = replace(parsed, model=replace(parsed.model, auxiliary=AuxiliarySpec(mtp=draft)),
                     runtime=replace(parsed.runtime, metadata=(('draft_mode', 'mtp'),)))
    graphs = target_graphs(parsed, phase='prefill', tokens=1, sequences=[1] + [0] * 7,
                           scope='model', draft_prepare=True)
    totals = {'target': 0, 'draft': 0}
    for rank, ops in graphs.items():
        for op in ops:
            if op.recipe == 'balanced_grouped_gate_up':
                md = dict(op.metadata)['routing_realization']
                assert md['representative_rank'] == rank
                totals['draft' if op.op_id.startswith('draft') else 'target'] += sum(md['local_expert_rows'])
                assert op.phase == ('draft_prefill' if op.op_id.startswith('draft') else 'prefill')
        assert not any(op.family in ('sampling', 'token_commit') for op in ops)
        assert any(op.family == 'logits' for op in ops) == (rank == 0)
    assert totals == {'target': 6, 'draft': 6}
    report = graph_costs(parsed, graphs, input_ready_ms=3, isolated=False)
    target_head = next(row for row in report['timeline'] if row['replica'] == 0 and row['family'] == 'logits')
    draft_first = next(row for row in report['timeline'] if row['replica'] == 0 and row['recipe'] == 'draft_features')
    assert draft_first['ready_ms'] >= target_head['complete_ms']
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    # The other call site generates each autoregressive draft proposal before
    # verification. Each proposal owns six assignments; verification owns 18.
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.schema.workload import IterationPlan, RequestView
    view = RequestView('verify', 1, 32, True, 'decode', draft_proposal_tokens=2)
    assignments = {}
    for rank in range(8):
        work = Frontend()(IterationPlan((view,) if rank == 0 else ()),
                          parsed.model, parsed.runtime, parsed.machine,
                          global_views=(view,), global_tokens=3, replica=rank)
        for op in work.ops:
            if op.recipe == 'balanced_grouped_gate_up':
                key = op.op_id.split('.')[0]
                assignments[key] = assignments.get(key, 0) + sum(
                    dict(op.metadata)['routing_realization']['local_expert_rows'])
    assert assignments == {'draft0': 6, 'draft1': 6, 's0': 18}


def test_idle_expert_program_uses_actual_mixed_cohort_phase():
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.workload import IterationPlan, RequestView
    parsed = parse_case(cohort_case(dp=2, ep=2))
    views = (RequestView('prefill', 1, 0, True, 'prefill'),
             RequestView('decode', 1, 32, True, 'decode'))
    idle = Frontend()(IterationPlan(()), parsed.model, parsed.runtime, parsed.machine,
                      global_views=views, global_tokens=2, replica=1)
    grouped = [op for op in idle.ops if op.family == 'grouped_gemm']
    assert grouped and all(op.phase == 'mixed' for op in grouped)
    assert all(dict(op.metadata)['routing_realization']['representative_rank'] == 1 for op in grouped)
    assert not any(op.family == 'logits' for op in idle.ops)
