"""The source's attention slice and SP repad preserve distinct row domains."""
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.sequence_parallel import select
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import RequestView

ROOT = Path(__file__).resolve().parents[1]


def case(batch=1):
    path = ROOT / ('examples/cases/high-concurrency-20261030/kimi-source-paged-host/'
                   f'a5_kimi_tp8_e16_bf16_c{batch}_p4k_source.json')
    return json.loads(path.read_text())


@pytest.mark.parametrize('batch', [1, 2, 4, 8, 32])
def test_complete_source_graph_slices_attention_and_preserves_padded_ffn(batch):
    raw = case(batch)
    graph = target_graphs(raw, scope='body', phase='decode', tokens=1,
                          sequences=batch, context=4096, apply_body_defaults=True)[0]
    by_id = {op.op_id: op for op in graph}
    physical = max(8, batch)
    assert {op.layer for op in graph if op.layer >= 0} == set(range(93))
    for layer in range(93):
        prefix = f's0.l{layer}'
        ops = [op for op in graph if op.layer == layer]
        is_mla = any(op.recipe == 'q_a' for op in ops)
        first = by_id[prefix + ('.q_a' if is_mla else '.linear_q')]
        assert first.local_shape[0] == batch
        if not is_mla:
            conv = by_id[prefix + '.short_convolution']
            assert conv.local_shape == (batch * 4608,)
            core = by_id[prefix + '.recurrent_scan0']
            assert dict(core.metadata)['kda_recurrent_params']['lengths'] == (1,) * batch
        gather = by_id[prefix + '.attention.sp_gather']
        assert gather.local_shape == (physical, 7168)
        assert gather.link_bytes == physical * 7168 * 2 * 7 / 8
        pad = [op for op in ops if op.recipe == 'sequence_parallel_reduce_scatter_pad']
        sliced = by_id.get(prefix + '.attention.sp_input_view')
        if batch < physical:
            assert sliced.local_shape == (batch, 7168)
            assert sliced.deps == (gather.op_id,)
            assert first.deps == (sliced.op_id,)
            assert sliced.hbm_bytes == sliced.link_bytes == 0
            assert len(pad) == 1
            assert pad[0].mte2_bytes == batch * 7168 * 2
            assert pad[0].mte3_bytes == physical * 7168 * 2
            assert pad[0].vector_ops == (physical - batch) * 7168
            scatter = next(op for op in ops if op.deps == (pad[0].op_id,))
            assert scatter.family == 'reduce_scatter'
            assert scatter.link_bytes == gather.link_bytes
        else:
            assert sliced is None and pad == []
            assert first.deps == (gather.op_id,)
    # FFN and residual tensors are still SP-local, while shared/dense TP
    # matrices execute their physical allgather rows.
    assert by_id['s0.l0.gate_up'].local_shape[0] == physical
    assert by_id['s0.l1.router'].local_shape[0] == physical // 8
    for op in graph:
        if op.family == 'grouped_gemm':
            route = dict(op.metadata)['routing_realization']
            assert route['global_tokens'] == physical
            assert route['valid_local_tokens'] == batch
            assert sum(route['global_expert_rows']) == physical * 8
        if op.recipe == 'attn_res_fwd':
            assert op.local_shape[0] == physical // 8


def test_graph_bucket_rows_are_preserved_but_sp_rows_do_not_reach_attention():
    parsed = parse_case(case())
    views = (RequestView('r', 1, 4096, False, 'decode'),)
    plan = select(parsed.runtime, parsed.model, views, tokens=4, global_tokens=8)
    assert (plan.valid_global, plan.attention_tokens, plan.physical_global) == (1, 4, 8)
    graph, _ = Frontend().body(parsed.model, parsed.runtime, views, ready='input',
                               tokens=4, mode='decode', global_tokens=8)
    by_id = {op.op_id: op for op in graph}
    assert by_id['s0.l0.linear_q'].local_shape[0] == 4
    assert by_id['s0.l3.q_a'].local_shape[0] == 4
    assert by_id['s0.l0.attention.sp_input_view'].local_shape == (4, 7168)
    pad = by_id['s0.l0.linear_reduce.reduce_scatter.pad']
    assert (pad.mte2_bytes, pad.mte3_bytes) == (4 * 7168 * 2, 8 * 7168 * 2)
    assert dict(by_id['s0.l0.recurrent_scan0'].metadata)['sequence_tokens'] == (1,)


@pytest.mark.parametrize('batch', [1, 8])
def test_real_source_slice_and_repad_wait_for_input_and_release_engine_state(batch):
    raw = case(batch)
    graph = target_graphs(raw, scope='body', phase='decode', tokens=1,
                          sequences=batch, context=4096, apply_body_defaults=True)[0]
    # Actual first dense/KDA and routed/KDA blocks; complete depth is checked
    # above and recorded separately by the bounded full-body producer.
    graph = tuple(op for op in graph if op.layer in (0, 1) or op.op_id == 'sp_shard')
    result = graph_costs(raw, {0: graph}, external_inputs={'analysis_input': 7})
    assert min(row['ready_ms'] for row in result['timeline']) >= 7
    assert result['duration_ms'] > 7
    execution = result['execution']
    assert execution['live_tasks'] == execution['live_programs'] == 0
    assert all(value == 0 for value in execution['live_allocated'].values())
