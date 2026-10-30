"""Explicit HF wrappers must drive actual precision, work and coverage."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs, target_operators, study
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.model import ModelSpec
from test_prediction_system import case

ROOT = Path(__file__).resolve().parents[1]


def test_actual_kimi_bf16_wrapper_changes_expert_banks_and_real_graph():
    path = ROOT / 'examples/cases/high-concurrency-20261030/kimi-variants/a5_kimi_tp8_e16_bf16_c1_p4k.json'
    raw = json.loads(path.read_text())
    raw['model']['layers'] = 2
    original = deepcopy(raw)
    parsed = parse_case(raw)
    assert parsed.model.quant.expert_weight_format == 'bf16'
    assert parsed.model.quantization == 'bf16'
    assert len(parsed.model.layer_plans) == 2
    assert dict(parsed.model.metadata)['model_coverage'] == 'partial'
    model_roundtrip = ModelSpec.from_mapping(parsed.model.to_dict())
    assert json.dumps(model_roundtrip.to_dict(), sort_keys=True) == json.dumps(parsed.model.to_dict(), sort_keys=True)
    graphs = target_graphs(raw, phase='decode', tokens=1, context=128, scope='body')
    gate = next(o for o in graphs[0] if o.recipe == 'balanced_grouped_gate_up')
    assert dict(gate.metadata)['weight_format'] == 'bf16'
    assert dict(gate.metadata)['activation_format'] == 'bf16'
    # The source-verified new checkpoint has 2 local BF16 expert banks,
    # K=3584 and gate/up N=6144. Neither capacity nor the old FP4 preset wins.
    assert dict(gate.metadata)['packed_weight_bytes'] == 2 * 3584 * 6144 * 2
    result = graph_costs(raw, graphs, external_inputs={'analysis_input': 0}, isolated=False)
    assert result['duration_ms'] > 0 and result['execution']['live_tasks'] == 0
    assert result['model_coverage'] == 'partial'
    assert raw == original


def wrapped():
    raw = case()
    raw['machine']['device']['compute_peaks']['int8'] = 2e9
    raw['model'] = {'hf_config': {
        'model_type': 'llama', 'num_hidden_layers': 2, 'hidden_size': 32,
        'num_attention_heads': 4, 'num_key_value_heads': 2,
        'intermediate_size': 64, 'vocab_size': 128},
        'quantization': 'w8a8', 'layers': 1,
        'metadata': {'model_coverage': 'partial', 'unmodeled_features': ['custom_capture_layout']}}
    return raw


def test_hf_wrapper_preset_and_coverage_reach_physical_graph():
    raw = wrapped()
    model = parse_case(raw).model
    assert model.quant.weight_format == model.quant.activation_format == 'int8'
    assert model.activation_dtype == model.compute_dtype == 'bf16'
    assert model.layers == 1
    operators = target_operators(raw, phase='prefill', tokens=4)
    assert any(dict(o.metadata).get('weight_format') == 'int8' for o in operators)
    assert any(o.family == 'quantize' for o in operators)
    report = graph_costs(raw, target_graphs(raw, phase='prefill', tokens=4))
    assert report['unmodeled_features'] == ['custom_capture_layout']
    assert report['execution']['live_tasks'] == 0


def test_study_hf_representation_overrides_stay_outside_native_config():
    raw = wrapped()
    report = study([
        {'id': 'int8', 'case': raw},
        {'id': 'bf16', 'case': raw, 'overrides': {'model': {'quantization': 'bf16'}}},
    ], [{'id': 'p4', 'phase': 'prefill', 'tokens': 4}], level='graph')
    assert report['status'] == 'completed'
    one, two = report['evaluations']
    assert one['operators']['work']['hbm_bytes'] < two['operators']['work']['hbm_bytes']
    assert all(row['model_coverage'] == 'partial' for row in report['evaluations'])


def test_existing_outer_hf_dimensions_and_quant_do_not_shadow_study_updates():
    from mindie_sim.study import _override_case
    path = ROOT / 'examples/cases/high-concurrency-20261030/kimi-variants/a5_kimi_tp8_e16_bf16_c1_p4k.json'
    raw = json.loads(path.read_text())
    raw['model']['layers'] = 2
    changed = _override_case(raw, {'model': {'num_hidden_layers': 3, 'quantization': 'w8a8'}})
    parsed = parse_case(changed)
    assert parsed.model.layers == 3
    assert parsed.model.quant.expert_weight_format == 'int8'
    assert raw['model']['layers'] == 2 and raw['model']['quant']['expert_weight_format'] == 'bf16'
    ops = target_operators(changed, phase='decode', tokens=1, context=128, scope='body')
    gates = [o for o in ops if o.recipe == 'balanced_grouped_gate_up']
    assert len(gates) == 2
    assert all(dict(o.metadata)['weight_format'] == 'int8' for o in gates)
