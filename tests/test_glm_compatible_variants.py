"""Execute declared GLM alternatives without overriding unrelated mechanisms.

[Role] Keep retained conditional cases connected to ordinary body prediction.
[Invariants] Only supported attention/down programs change, never hardware,
  ordinary matrix mapping, accepted-token state ABI or fused gate/up schedule.
[Timeouts & Locks] Three four-layer shared-Engine witnesses; no processes/locks.
[Side-Effect Policy] Read small repository cases; all execution is in memory.
"""
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'tests/fixtures/verification/glm-supported-variants.json'


@pytest.mark.parametrize('case_id,name', [
    ('glm53flash-a5-tp8-c3', 'attention-only'),
    ('glm53flash-a5-tp8-c3', 'attention-plus-down'),
    ('glm53flash-a3-tp16-c2', 'attention-only'),
])
def test_retained_variant_uses_supported_programs_and_full_layer_dependencies(case_id, name):
    receipt = json.loads(RECEIPT.read_text())
    row = next(row for row in receipt['cases'] if row['case_id'] == case_id)
    variant = next(v for v in row['variants'] if v['name'] == name)
    original = json.loads((ROOT / row['baseline']['path']).read_text())
    raw = json.loads((ROOT / variant['path']).read_text())
    assert raw['machine'] == original['machine']
    assert raw['runtime'].get('routing') == original['runtime'].get('routing')
    assert raw['model']['quant'] == original['model']['quant']
    raw['model']['layers'] = 4
    original['model']['layers'] = 4
    geometry = row['geometry']
    kwargs = dict(scope='body', phase='decode', tokens=geometry['tokens_per_sequence'],
                  sequences=geometry['sequences_per_replica'], context=geometry['context_tokens'])
    baseline = target_graphs(original, **kwargs)
    graphs = target_graphs(raw, **kwargs)
    assert graphs.effective_kernels['programs']['matmul'] == baseline.effective_kernels['programs']['matmul']
    # Source selection changes neither router output nor the surviving generic
    # accepted-state boundary; actual graph execution checks the producers.
    for prior, after in zip(
        (o for o in baseline[0] if o.family == 'recurrent_scan'),
        (o for o in graphs[0] if o.family == 'recurrent_scan'), strict=True,
    ):
        assert prior == after
    result = graph_costs(raw, graphs, input_ready_ms=7,
                         external_inputs={'analysis_input': 7}, isolated=False)
    assert result['model_coverage'] == 'partial'
    assert result['prediction_scope'] == 'known_components_with_declared_degradations'
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert all(abs(value) < 1e-7 for value in result['execution']['live_allocated'].values())
    attention = [r for r in result['timeline'] if r['family'] == 'attention']
    assert len(attention) == 1 and attention[0]['program'] == 'OperatorPlan'
    if name == 'attention-plus-down':
        i = next(i for i, op in enumerate(graphs[0]) if op.op_id.endswith('.expert_down'))
        down = operator_costs(raw, graphs[0][i:i + 1])['operators'][0]
        plan = down['operator']['metadata']['program_execution']
        assert plan['source_program']['branch'].startswith('MX NZ FP8')
        assert sum(plan['source_mapping']['group_rows']) == geometry['query_rows'] * 8
        # Fixpipe owns the one BF16 output; no repeated HBM epilogue/vector pass.
        assert down['serviced']['fixpipe@r0:s0'] == pytest.approx(2 * geometry['query_rows'] * 8 * 4096)
        assert not any('vector' in key or 'mte3' in key for key in down['serviced'])
        up = next(op for op in graphs[0] if op.recipe == 'balanced_grouped_gate_up')
        before_up = next(op for op in baseline[0] if op.op_id == up.op_id)
        assert up == before_up
