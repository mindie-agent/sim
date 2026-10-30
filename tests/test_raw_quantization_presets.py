"""Generic structure presets must produce the same physical operands as QuantSpec."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]


def model_case(preset='w8a8_mxfp8'):
    raw = json.loads((ROOT/'examples/quickstart.json').read_text())
    raw['model'].update(layers=1, ffn='moe', num_experts=8, top_k=2,
                        moe_intermediate_size=256, num_shared_experts=1, quantization=preset)
    raw['machine'] = {'backend': 'ascend', 'sku': 'Ascend950DT_9582', 'num_devices': 4}
    raw['runtime'].update(tp=2, ep=4, dp=2, graph_mode='none', routing={'policy': 'balanced'})
    raw['runtime']['kernels'] = {'programs': {'matmul': {'kind': 'steady_matmul'}},
        'recipes': {'balanced_grouped_down': {'kind': 'ascend_mxfp8_grouped_nz'}}}
    return raw


@pytest.mark.parametrize('preset', ['w8a8_mxfp8', 'w8a8-mxfp8', 'mxfp8', 'fp8_mxfp8', 'fp8'])
def test_registered_mx_presets_equal_explicit_operands_but_keep_residuals_bf16(preset):
    raw = model_case(preset)
    explicit = deepcopy(raw)
    explicit['model'].pop('quantization')
    explicit['model']['quant'] = quant_preset(preset).to_dict()
    case, reference = parse_case(raw), parse_case(explicit)
    assert case.model.quant == reference.model.quant
    assert case.model.activation_dtype == case.model.compute_dtype == 'bf16'
    assert case.model.weight_dtype == 'fp8_e4m3'
    kwargs = dict(phase='decode', tokens=1, sequences=2, context=128, scope='body')
    graphs = target_graphs(case, **kwargs)
    assert graphs == target_graphs(reference, **kwargs)
    ops = graphs[0]
    norms = [op for op in ops if op.family == 'rmsnorm']
    assert norms and all(op.dtype == 'bf16' for op in norms)
    matrices = [op for op in ops if dict(op.metadata).get('weight_format') == 'fp8_e4m3']
    assert matrices and all(op.dtype == 'fp8_e4m3' for op in matrices)
    assert all(dict(op.metadata)['activation_scale_format'] == 'fp8_e8m0' for op in matrices)
    assert any(op.family == 'quantize' for op in ops)
    if preset == 'w8a8_mxfp8':
        result = graph_costs(case, graphs, input_ready_ms=7, external_inputs={"analysis_input": 7})
        assert result['duration_ms'] > 0
        assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
        assert not any(result['execution']['live_allocated'].values())
        selected = [row for row in result['isolated']['operators']
                    if row['operator']['recipe'] == 'balanced_grouped_down']
        assert selected and all(row['program'] == 'OperatorPlan' for row in selected)


def test_explicit_storage_and_residual_dtypes_keep_precedence_over_preset_defaults():
    raw = model_case()
    raw['model'].update(weight_dtype='fp16', activation_dtype='fp32',
                        compute_dtype='fp32', kv_dtype='fp16')
    model = parse_case(raw).model
    assert model.weight_dtype == model.quant.weight_format == model.quant.expert_weight_format == 'fp16'
    assert model.kv_dtype == model.quant.kv_dtype == 'fp16'
    assert model.activation_dtype == model.compute_dtype == 'fp32'
    explicit = quant_preset('bf16')
    raw['model']['quant'] = explicit.to_dict()
    assert parse_case(raw).model.quant == explicit


@pytest.mark.parametrize('name', ['w8a8_fp8', 'fp8_e4m3', 'custom_fp8_unblocked'])
def test_unregistered_fp8_format_still_requires_a_full_representation(name):
    raw = model_case(name)
    with pytest.raises(ModelingError, match='explicit_quant_block_and_scale_geometry'):
        parse_case(raw)
    raw['model']['quant'] = quant_preset('w8a8_mxfp8').to_dict()
    assert parse_case(raw).model.quant.weight_block == (1, 32)
