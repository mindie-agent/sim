"""Real Kimi inputs select distinct source domains without changing checkpoints."""
from collections import Counter
import json
from pathlib import Path

import pytest

from mindie_sim import target_graphs
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'examples/cases/high-concurrency-20261030'


@pytest.mark.parametrize('name,layers,kda,implementation', [
    ('a5_kimi_tp8_e16_bf16_c1_p4k', 93, 69, 'vllm_ascend_recurrent_kda_arch35'),
    ('a5_kimi_tp8_e16_bf16_c32_p4k', 93, 69, 'vllm_ascend_recurrent_kda_arch35'),
    ('a3_kimi4_tp8_w4a8_c1', 4, 3, 'vllm_ascend_recurrent_kda_arch22'),
    ('a3_kimi4_tp8_w4a8_c32', 4, 3, 'vllm_ascend_recurrent_kda_arch22'),
])
def test_complete_source_inputs_preserve_geometry_and_replace_native_boundaries(
        name, layers, kda, implementation):
    base = json.loads((CASES / 'kimi-variants' / (name + '.json')).read_text())
    source = json.loads((CASES / 'kimi-source' / (name + '_source.json')).read_text())
    assert source['model'] == base['model']
    assert source['machine'] == base['machine']
    assert source['runtime']['routing'] == base['runtime']['routing']
    assert len(parse_case(source).model.layer_plans) == layers
    batch = source['runtime']['metadata']['client']['concurrency']
    graph = target_graphs(source, phase='decode', tokens=1, context=4096,
                          sequences=batch, scope='body')[0]
    selected = [op for op in graph if op.recipe == implementation]
    assert len(selected) == kda
    for op in selected:
        params = dict(op.metadata)['kda_recurrent_params']
        assert params['lengths'] == (1,) * batch
        assert (params['gate_dtype'], params['beta_dtype'], params['state_dtype']) == (
            'bf16', 'fp32', 'fp32')
        assert not params['has_accepted_tokens']
    if name.startswith('a5_'):
        groups = Counter(dict(op.metadata).get('device_kernel_group', {}).get('id')
                         for op in graph if 'device_kernel_group' in dict(op.metadata))
        assert len(groups) == 2 * 92  # First transformer block has a dense FFN.
        assert not any(op.recipe in {'expert_dispatch', 'expert_combine'} for op in graph)
        grouped = [op for op in graph if op.family == 'grouped_gemm']
        assert len(grouped) == 2 * 92
        assert all(len(dict(op.metadata)['matrix_group_rows']) == 2 for op in grouped)
        assert sum(op.recipe == 'mla_query_pad_nope' for op in graph) == 24
    else:
        assert not any('device_kernel_group' in dict(op.metadata) for op in graph)
        assert any(op.recipe == 'expert_dispatch' for op in graph)
