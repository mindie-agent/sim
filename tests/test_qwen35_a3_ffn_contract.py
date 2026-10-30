"""Qwen35 A3 retained FFN operands and TP sum have the same physical boundary.

[Role] Verify the source-bound case at low/high native row counts.
[Invariants] BF16 shared weights do not change routed INT8 work. The final TP
  sum owns token rows after weighted routed/shared combination.
[Timeouts & Locks] One in-memory model layer; no subprocess or device context.
[Side-Effect Policy] Reads retained ABI evidence and case; writes nothing.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import target_graphs
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'examples/cases/longctx-decode-20261030/qwen35-35b-a3-tp4.json'
EVIDENCE = json.loads((ROOT / 'tests/fixtures/verification/qwen35-a3-ffn-contract.json').read_text())


def local_graph(raw, query, batch):
    raw = deepcopy(raw)
    raw['model']['layers'] = 1
    case = parse_case(raw)
    ops = target_graphs(case, phase='decode', tokens=query, sequences=batch,
                        context=126000, scope='body')[0]
    return case, {op.op_id: op for op in ops}


@pytest.mark.parametrize('capture', EVIDENCE['captures'], ids=lambda row: row['id'])
def test_shared_operand_bytes_and_late_sum_match_both_native_row_counts(capture):
    raw = json.loads(CASE.read_text())
    case, ops = local_graph(raw, capture['query_tokens'], capture['batch_slots'])
    m = capture['query_tokens'] * capture['batch_slots']
    rows = capture['rows']
    native_up = next(row for row in rows if row['Type'] == 'MatMulV2'
                     and row['csv_line'] in (33934, 182))
    native_down = next(row for row in rows if row['Type'] == 'MatMulV2'
                       and row['csv_line'] in (33936, 184))
    terms = {term.name: term for term in layer_weight_terms(case.model.layer_plans[0], case.model)}
    for op_id, n, k, names, native in (
        ('s0.l0.shared.gate_up', 256, 2048, ('shared_gate', 'shared_up'), native_up),
        ('s0.l0.shared.down', 2048, 128, ('shared_down',), native_down),
    ):
        op = ops[op_id]
        assert native['Input Shapes'].strip('"') == f'{m},{k};{n},{k}'
        assert native['Input Data Types'] == 'DT_BF16;DT_BF16'
        assert op.local_shape == (m, n, k) and op.dtype == 'bf16'
        assert op.flops == 2 * m * n * k
        assert op.hbm_bytes == 2 * (m*k + n*k + m*n)
        assert op_id + '.quantize' not in ops
        assert dict(op.metadata)['packed_weight_bytes'] == sum(
            term_bytes(terms[name], case.model.quant, case.runtime.parallel) for name in names)
    reduced, = [op for op in ops.values() if op.recipe == 'moe_output_reduce']
    assert reduced.local_shape == (m, 2048)
    assert reduced.link_bytes == 2 * 3 / 4 * m * 2048 * 2
    joined = ops[reduced.deps[0]]
    assert joined.recipe == 'expert_add'
    assert joined.local_shape == (m, 2048)
    assert joined.hbm_bytes == 3 * m * 2048 * 2
    assert joined.mte2_bytes == 2 * m * 2048 * 2
    assert joined.mte3_bytes == m * 2048 * 2
    assert {ops[dep].recipe for dep in joined.deps} == {'route_unpack', 'shared_gate_apply'}
    native_reduce, = [row for row in rows if row['Type'] == 'hcom_allReduce_']
    native_join, = [row for row in rows if row['Type'] == 'ForeachAddList']
    assert native_reduce['csv_line'] > native_join['csv_line'] > native_down['csv_line']
    assert not any(op.recipe == 'expert_tensor_reduce' for op in ops.values())
    assert 's0.l0.shared.tp' not in ops


def test_case_correction_preserves_all_routed_matrix_work_and_nonshared_precision():
    raw = json.loads(CASE.read_text())
    old = deepcopy(raw)
    old['model']['quant']['category_overrides'].pop('shared_expert')
    old['runtime']['moe_recipe'] = 'alltoall'
    _, before = local_graph(old, 2, 1)
    _, after = local_graph(raw, 2, 1)
    for op_id in ('s0.l0.expert_up', 's0.l0.expert_down', 's0.l0.router',
                  's0.l0.linear_qkvz', 's0.l0.linear_beta_alpha', 's0.l0.linear_output'):
        for field in ('dtype', 'local_shape', 'flops', 'hbm_bytes', 'mte2_bytes', 'mte3_bytes'):
            assert getattr(before[op_id], field) == getattr(after[op_id], field)
    assert after['s0.l0.expert_up'].dtype == after['s0.l0.expert_down'].dtype == 'int8'
    assert set(before) - set(after) == {'s0.l0.shared.gate_up.quantize', 's0.l0.shared.down.quantize',
                                      's0.l0.shared.tp', 's0.l0.expert_reduce'}
    assert set(after) - set(before) == {'s0.l0.moe_output_reduce'}
