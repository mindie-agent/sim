"""Real Kimi catalog controls for source AllToAll's physical padded rows."""
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.balanced_routing import cohort_padded_tokens
from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.sequence_parallel import DECLARATION, select
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.schema.workload import RequestView

ROOT = Path(__file__).resolve().parents[1]


def case(*, graph_bucket=None):
    raw = json.loads((ROOT / 'examples/cases/kimi-k3-a5-current-checkpoint-conditional.json').read_text())
    runtime = raw['runtime']
    runtime['routing'] = {'policy': 'balanced'}
    runtime.setdefault('kernels', {})['sequence_parallel'] = dict(DECLARATION, padding='right_to_tp')
    runtime['kernels']['programs'] = {'matmul': {'kind': 'steady_matmul'}}
    runtime['graph_mode'] = 'eager' if graph_bucket is None else 'auto'
    if graph_bucket is not None:
        runtime['graph_buckets'] = [graph_bucket]
        runtime['graph_capture_keys'] = [graph_bucket]
    return parse_case(raw)


def views(rows, request='r'):
    return (RequestView(request, rows, 4096, False, 'decode'),) if rows else ()


@pytest.mark.parametrize('rows,physical,valid_per_tp', [
    (1, 8, (1, 0, 0, 0, 0, 0, 0, 0)),
    (4, 8, (1, 1, 1, 1, 0, 0, 0, 0)),
    (9, 16, (2, 2, 2, 2, 1, 0, 0, 0)),
    (16, 16, (2, 2, 2, 2, 2, 2, 2, 2)),
])
def test_real_catalog_padding_geometry_does_not_change_valid_queries(rows, physical, valid_per_tp):
    parsed = case()
    assert parsed.model.layers == 93
    assert {layer.num_experts for layer in parsed.model.layer_plans if layer.ffn == 'moe'} == {896}
    batch = views(rows)
    plan = select(parsed.runtime, parsed.model, batch, tokens=rows, global_tokens=physical * 4)
    assert plan.valid_global == plan.input_rows == rows
    assert plan.physical_global == plan.tokens_global == physical
    assert plan.tokens_local == physical // 8
    assert plan.valid_per_tp == valid_per_tp
    assert sum(plan.valid_per_tp) == rows
    assert batch[0].q_len == rows
    assert plan.cohort_tokens == physical * 4


@pytest.fixture(scope='module')
def full_body_graphs():
    parsed = case()
    return parsed, {q: target_graphs(parsed, phase='decode', tokens=q, context=4096,
                                    scope='body', resource_fusion=False) for q in (1, 4)}


def test_full_93_layer_graph_keeps_sp_padding_outside_recurrent_query_loop(full_body_graphs):
    parsed, by_query = full_body_graphs
    for q, graphs in by_query.items():
        assert set(graphs) == {0, 1, 2, 3}
        for ops in graphs.values():
            by_id = {op.op_id: op for op in ops}
            pad = by_id['sp_shard']
            metadata = dict(pad.metadata)
            assert pad.mte2_bytes == q * parsed.model.hidden_size * 2
            assert pad.mte3_bytes == 8 * parsed.model.hidden_size * 2
            assert metadata['valid_global'] == q
            assert metadata['physical_global'] == 8
            assert metadata['valid_per_tp'] == (1,) * q + (0,) * (8 - q)
            assert by_id['s0.l0.linear_q'].local_shape[0] == q
            assert by_id['s0.l0.recurrent_scan0'].local_shape[0] == q
            assert by_id['s0.l0.norm1'].local_shape[0] == 1
            gather = by_id['s0.l0.attention.sp_gather']
            assert gather.link_bytes == 7 * parsed.model.hidden_size * 2
            trim = by_id['sp_output_trim']
            assert trim.local_shape == (q, parsed.model.hidden_size)
            assert trim.deps == ('sp_output_gather',)
            assert trim.hbm_bytes == trim.link_bytes == 0
            assert {op.layer for op in ops if op.layer >= 0} == set(range(93))


def test_padding_executes_alltoall_router_rows_but_is_not_valid_token_count(full_body_graphs):
    _, by_query = full_body_graphs
    contracts = {}
    for q, graphs in by_query.items():
        ops = graphs[0]
        grouped = [op for op in ops if op.family == 'grouped_gemm']
        assert grouped
        for op in grouped:
            metadata = dict(op.metadata)
            routing = metadata['routing_realization']
            assert routing['global_tokens'] == 4 * 8
            assert sum(routing['global_expert_rows']) == 4 * 8 * 16
            assert routing['valid_local_tokens'] == q
            assert routing['padded_local_tokens'] == 8
            assert routing['source_partitions'] == 1
            assert sum(routing['rank_assignment_rows']) == 4 * 8 * 16
        dispatch = next(op for op in ops if op.recipe == 'expert_dispatch')
        # Each source physical rank has one actual padded router row.
        assert dispatch.local_shape == (16, 3584)
        assert dispatch.link_bytes == 16 * 3584 * 2 * 31 / 32
        contracts[q] = [(op.recipe, op.flops, op.mte2_bytes, op.mte3_bytes) for op in grouped]
    assert contracts[1] == contracts[4]


def test_unequal_and_idle_dp_keep_per_replica_padding_and_inverse_traffic():
    parsed = case()
    # Retain the real first dense/KDA and routed/KDA layers for a fast Engine
    # contract. The full-depth frontend geometry is checked above.
    populations = (1, 4, 9, 0)
    groups = tuple(views(n, f'r{i}') for i, n in enumerate(populations))
    cohort = cohort_padded_tokens(groups, parsed.runtime)
    assert cohort == 8 + 8 + 16
    frontend = Frontend()
    graphs = {}
    for replica, group in enumerate(groups):
        ops, _ = frontend.body(parsed.model, parsed.runtime, group, ready='input',
                               tokens=populations[replica], mode='decode', global_tokens=cohort)
        ops = tuple(op for op in ops if op.layer in (0, 1) or op.op_id == 'sp_shard')
        graphs[replica] = lower_implementations(ops, parsed.runtime, parsed.model.activation_dtype,
                                               resource_fusion=False)
    graphs = bind_cohort_traffic(parsed, graphs)
    row_bytes = 16 * 3584 * 2 * 31 / 32
    for replica, ops in graphs.items():
        dispatch = next(op for op in ops if op.recipe == 'expert_dispatch')
        assert dispatch.link_bytes == (1, 1, 2, 0)[replica] * row_bytes
        grouped = next(op for op in ops if op.family == 'grouped_gemm')
        routing = dict(grouped.metadata)['routing_realization']
        assert routing['global_tokens'] == 32
        assert routing['valid_local_tokens'] == populations[replica]
        assert routing['padded_local_tokens'] == (8, 8, 16, 0)[replica]
        assert sum(routing['global_expert_rows']) == 512
        combine = next(op for op in ops if dict(op.metadata).get('reverse_of_collective'))
        traffic = dict(combine.metadata)['peer_traffic']
        assert traffic['direction'] == 'transpose'
        assert traffic['source_bytes'] == [row_bytes] * 16 + [2 * row_bytes] * 8 + [0] * 8
    assert not [op for op in graphs[3] if dict(op.metadata).get('sequence_parallel_role')]
    report = graph_costs(parsed, graphs, external_inputs={'input': 7})
    assert report['duration_ms'] > 7
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert min(row['ready_ms'] for row in report['timeline']) >= 7


def test_graph_capture_rows_and_sp_rows_are_separate_from_valid_rows():
    parsed = case(graph_bucket=4)
    plan = select(parsed.runtime, parsed.model, views(1), tokens=4, global_tokens=32)
    assert (plan.valid_global, plan.input_rows, plan.physical_global, plan.tokens_local) == (1, 4, 8, 1)
    assert cohort_padded_tokens((views(1), views(9), ()), parsed.runtime) == 8 + 16
    from mindie_sim.prediction.sequence_parallel import input_shard
    pad = input_shard(plan=plan, op_id='pad', deps=('input',), stage=0, layer=-1,
                      requests=('r',), phase='decode', width=7168, dtype='bf16')
    assert pad.mte2_bytes == 4 * 7168 * 2
    assert pad.mte3_bytes == 8 * 7168 * 2
    assert pad.vector_ops == 4 * 7168


def test_public_cohort_padding_is_independent_of_expert_histogram_policy():
    # Padding describes the physical source tensor even if the caller chooses
    # probability moments rather than an integer balanced scenario.
    from dataclasses import replace
    parsed = case()
    parsed = replace(parsed, runtime=replace(parsed.runtime, routing={}))
    graphs = target_graphs(parsed, phase='decode', tokens=1, context=32,
                           scope='body', resource_fusion=False)
    assert set(graphs) == {0, 1, 2, 3}
    for ops in graphs.values():
        pad = next(op for op in ops if op.op_id == 'sp_shard')
        assert dict(pad.metadata)['cohort_tokens'] == 32
        assert dict(pad.metadata)['valid_global'] == 1
        dispatched = next(op for op in ops if op.recipe == 'expert_dispatch')
        assert dispatched.local_shape[0] == 16
