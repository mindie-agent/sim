"""Packed column-scale ABI controls native service selection independently of rows."""
from copy import deepcopy
import json
from pathlib import Path

from mindie_sim import operator_costs, target_graphs

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'examples/cases/longctx-decode-20261030/qwen35-a5-tp4-ep1-grouped-joint.json'


def execute(raw):
    ops = target_graphs(raw, scope='body', phase='decode', tokens=4,
                        sequences=1, context=126976)[0]
    op = next(op for op in ops if op.recipe == 'grouped_gate_up')
    return op, operator_costs(raw, [op], input_ready_ms=7)['operators'][0]


def test_changed_column_scale_format_restores_the_analytical_grouped_program():
    raw = json.loads(CASE.read_text())
    for rule in raw['runtime']['kernels']['service_regimes']:
        rule['match'].update({'metadata.weight_scale_format': 'bf16',
                              'metadata.weight_scale_bytes': 2})
    op, native = execute(raw)
    assert dict(op.metadata)['activation_scale_format'] == 'fp32'
    assert dict(op.metadata)['weight_scale_format'] == 'bf16'
    assert native['program'] == 'JointNativeService'

    changed = deepcopy(raw)
    changed['model']['quant']['category_overrides']['routed_expert'].update(
        scale_format='fp32', scale_bytes=4)
    op, result = execute(changed)
    assert dict(op.metadata)['weight_scale_format'] == 'fp32'
    assert dict(op.metadata)['activation_scale_format'] == 'fp32'
    plain = deepcopy(changed)
    plain['runtime']['kernels']['service_regimes'] = []
    _, analytical = execute(plain)
    assert result['program'] == analytical['program'] == 'ExpectedGroupedMatmul'
    assert result['duration_ms'] == analytical['duration_ms']
    assert result['serviced'] == analytical['serviced']
    assert not result['service_before_input'] and result['live_tasks'] == 0
