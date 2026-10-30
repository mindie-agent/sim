"""Retained Prefill ALLTOALL caller contracts and causal output reconstruction.

[Role] Check physical source rows, expert work and the five communication calls.
[Invariants] TP slicing changes caller ownership once, never global expert work.
  INT32 count transport, INT8 rows and BF16 returns keep independent byte widths.
[Timeouts & Locks] In-memory graph construction and one small Engine execution.
[Side-Effect Policy] No device work, capture reads or generated artifacts.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.moe_token_partition import ALLTOALL_KIND
from mindie_sim.prediction.structure import parse_case
from test_moe_quant_transport import SELECTOR, transport_case

ROOT = Path(__file__).resolve().parents[1]


def caller(raw, *, copies=False):
    raw['runtime'].setdefault('kernels', {}).update({
        'moe_token_partition': {
            'kind': ALLTOALL_KIND, 'layer_scopes': ['body'],
            'gather_output_materialization': 'per_rank_copy' if copies else 'view',
        },
        'moe_transport': {**SELECTOR, 'histogram_dtype': 'int32', 'layer_scopes': ['body']},
    })
    return raw


@pytest.mark.parametrize('name,rows,tp,hidden,experts', [
    ('v4f-a3-w8a8-tp8', 4096, 8, 4096, 256),
    ('v4pro-a3-w4a8-tp16', 1024, 16, 7168, 384),
])
def test_retained_prefill_local_rows_typed_counts_and_five_collectives(
        name, rows, tp, hidden, experts):
    # pre4096.kernel_details.csv: F router/gating/histogram/permute lines
    # 150/157/159/174; Pro 152/159/161/176. These are tensor ABI controls.
    raw = json.loads((ROOT / 'examples/cases' / (name + '.json')).read_text())
    raw['model']['layers'] = 1
    caller(raw, copies=True)
    graph = target_graphs(raw, phase='prefill', tokens=rows, scope='body')[0]
    ops = {op.op_id: op for op in graph}
    local, assignments = rows // tp, rows // tp * 6
    router, topk = ops['s0.l0.router'], ops['s0.l0.topk']
    assert router.local_shape[0] == rows
    assert topk.local_shape == (local, experts)
    assert topk.deps == ('s0.l0.alltoall_logits_view',)
    assert ops['s0.l0.pack'].local_shape == (local, 6)
    assert ops['s0.l0.pack'].mte3_bytes == assignments * (hidden * 2 + 4)
    assert ops['s0.l0.unpack'].local_shape == (local, 6, hidden)
    assert ops['s0.l0.shared.gate_up'].local_shape[0] == rows

    hist = ops['s0.l0.moe_histogram']
    counts = ops['s0.l0.moe_histogram_gather']
    assert hist.dtype == counts.dtype == 'int32'
    assert hist.local_shape == (experts,)
    assert (hist.mte2_bytes, hist.mte3_bytes, hist.vector_ops) == (
        assignments * 4, experts * 4, assignments)
    assert counts.link_bytes == (tp - 1) * experts * 4
    assert dict(hist.metadata)['counts'] is None  # no invented capture histogram
    quant = ops['s0.l0.moe_transport_quantize']
    assert quant.local_shape == (assignments, hidden)
    assert quant.mte2_bytes == assignments * hidden * 2
    assert quant.mte3_bytes == assignments * (hidden + 4)
    assert dict(quant.metadata)['source_assignment_partitions'] == 1

    scale, payload, back = (ops['s0.l0.' + suffix] for suffix in (
        'moe_scale_dispatch', 'moe_payload_dispatch', 'expert_combine'))
    remote = (tp - 1) / tp
    assert scale.link_bytes == assignments * 4 * remote
    assert payload.link_bytes == assignments * hidden * remote
    assert back.link_bytes == assignments * hidden * 2 * remote
    assert [op.family for op in (scale, payload, back)] == ['alltoall'] * 3
    gathers = [op for op in graph if op.family == 'allgather']
    assert len(gathers) == 2
    gather = ops['s0.l0.alltoall_tp_gather']
    assert gather.link_bytes == (tp - 1) * local * hidden * 2
    assert gather.deps == ('s0.l0.unpack',)
    copies = [op for op in graph if '.alltoall_gather_copy' in op.op_id]
    assert len(copies) == tp
    assert all(op.local_shape == (local, hidden) for op in copies)
    assert sum(op.mte2_bytes for op in copies) == rows * hidden * 2
    assert sum(op.mte3_bytes for op in copies) == rows * hidden * 2

    control = deepcopy(raw)
    del control['runtime']['kernels']['moe_token_partition']
    previous = {op.op_id: op for op in target_graphs(
        control, phase='prefill', tokens=rows, scope='body')[0]}
    # Communication is already per-source. The new explicit slice must not
    # divide it again or change destination experts, router or shared matrices.
    for suffix in ('router', 'expert_up', 'expert_down', 'expert_act',
                   'shared.gate_up', 'shared.down', 'shared.tp',
                   'moe_transport_quantize', 'moe_scale_dispatch',
                   'moe_payload_dispatch', 'expert_combine'):
        op, old = ops['s0.l0.' + suffix], previous['s0.l0.' + suffix]
        assert (op.flops, op.vector_ops, op.hbm_bytes, op.link_bytes) == (
            old.flops, old.vector_ops, old.hbm_bytes, old.link_bytes)
    assert target_graphs(parse_case(raw).to_dict(), phase='prefill', tokens=rows,
                         scope='body')[0] == graph


def test_quantized_exchange_and_reconstructed_output_keep_causal_readiness():
    raw = caller(transport_case(dp=1, tp=2, ep=2, tokens=4))
    graphs = target_graphs(raw, phase='prefill', tokens=4, scope='body')
    result = graph_costs(raw, graphs, external_inputs={'analysis_input': 7},
                         input_ready_ms=7, trace_limit=200)
    timeline = {row['op_id']: row for row in result['timeline']}
    for after, before in (
        ('moe_histogram_gather', 'moe_histogram'),
        ('pack', 'moe_histogram_gather'),
        ('moe_transport_quantize', 'pack'),
        ('moe_scale_dispatch', 'moe_transport_quantize'),
        ('moe_payload_dispatch', 'moe_scale_dispatch'),
        ('expert_up', 'moe_payload_dispatch'),
        ('expert_combine', 'expert_down'),
        ('unpack', 'expert_combine'),
        ('alltoall_tp_gather', 'unpack'),
    ):
        assert timeline['s0.l0.' + after]['ready_ms'] >= timeline['s0.l0.' + before]['complete_ms']
    assert min(row['ready_ms'] for row in result['timeline']) >= 7
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0


def test_body_caller_selection_preserves_the_separate_bf16_draft_boundary():
    raw = json.loads((ROOT / 'tests/fixtures/verification/'
                     'v4flash-communication-case.json').read_text())
    caller(raw)
    graph = target_graphs(raw, phase='prefill', tokens=8, scope='model',
                          draft_prepare=True)[0]
    target = [op for op in graph if op.recipe == 'expert_quantized_dispatch']
    draft = [op for op in graph if op.phase == 'draft_prefill'
             and op.recipe == 'expert_dispatch']
    assert len(target) == 43 and len(draft) == 1
    assert all(op.phase == 'prefill' for op in target)
    target_views = [op for op in graph if op.recipe == 'alltoall_tp_token_view']
    assert len(target_views) == 2 * 43
    assert all(op.local_shape[0] == 1 and op.phase == 'prefill' for op in target_views)
    assert draft[0].link_bytes == 8 * 6 * 4096 * 2 / 8 * 7 / 8
    assert not any('moe_token_partition' in dict(op.metadata) for op in graph
                   if op.phase == 'draft_prefill')
