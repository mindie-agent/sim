"""Source-GDN case overlays preserve deployment geometry and real tensor work."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.gdn_decode_programs import CONV, GATING
from mindie_sim.prediction.gdn_recurrent import IMPLEMENTATION
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import target_graphs

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'tests/fixtures/verification/qwen-gdn-native-contracts.json'
CASES = json.loads(EVIDENCE.read_text())['cases']


@pytest.mark.parametrize('evidence', CASES, ids=lambda x: x['case_id'])
def test_source_overlay_preserves_model_precision_topology_and_only_adds_primitive_service(evidence):
    original = json.loads((ROOT / evidence['base_case_identity']['path']).read_text())
    selected = json.loads((ROOT / evidence['selected_case_identity']['path']).read_text())
    assert selected['model'] == original['model']
    assert selected['workload'] == original['workload']
    a, b = deepcopy(original['runtime']), deepcopy(selected['runtime'])
    condition = b['metadata'].pop('gdn_source_native_condition')
    binding = b['kernels']['lowerings'].pop('gdn_decode')
    if not b['kernels']['lowerings']:
        b['kernels'].pop('lowerings')
    if not b['kernels']:
        b.pop('kernels')
    assert a == b
    assert binding['options']['cache_index_row_stride'] == 4
    params = json.loads((ROOT / condition['primitive_parameters_identity']['path']).read_text())
    assert original['machine']['sku'] == selected['machine']['sku'] == params['hardware']
    base, source = parse_case(original), parse_case(selected)
    assert base.model == source.model and base.runtime.parallel == source.runtime.parallel
    assert len(base.machine.dies) == len(source.machine.dies)
    for before, after in zip(base.machine.dies, source.machine.dies):
        rates = dict(after.rates); units = dict(after.resource_units)
        assert rates.pop('scalar_store_issue') == params['per_unit_issue_rate_per_s'] * 64
        assert units.pop('scalar_store_issue') == 64
        base_rates, base_units = dict(before.rates), dict(before.resource_units)
        base_rates.pop('scalar_store_issue', None); base_units.pop('scalar_store_issue', None)
        assert rates == base_rates and units == base_units


@pytest.mark.parametrize('evidence', CASES, ids=lambda x: x['case_id'])
def test_each_source_gdn_layer_uses_its_tp_shard_abi_and_active_rows(evidence):
    raw = json.loads((ROOT / evidence['selected_case_identity']['path']).read_text())
    case = parse_case(raw); geo = evidence['geometry']
    graphs = target_graphs(case, phase='decode', tokens=geo['tokens_per_sequence'],
                          sequences=geo['sequences_per_replica'],
                          context=raw['workload']['requests'][0]['prompt_tokens'], scope='body')
    samples = {r['kind']: r for r in evidence['samples']}
    recurrent = samples['RecurrentGatedDeltaRule']['native_abi']; inputs, outputs = recurrent[2:]
    conv_native = samples['CausalConv1d']['native_abi']
    # The gate itself is not a complete ABI observation. Its output precision
    # follows the independently retained recurrence inputs and source caller.
    assert samples['fused_gdn_gating_kernel']['native_abi'] is None
    expected_layers = {p.index for p in case.model.layer_plans if p.attention == 'gdn'}
    for ops in graphs.values():
        cores = [op for op in ops if op.recipe == IMPLEMENTATION]
        assert {op.layer for op in cores} == expected_layers
        for core in cores:
            meta = dict(core.metadata); M, H, DV = outputs[0][0]
            assert core.local_shape == (M, H, inputs[0][0][-1], DV)
            assert core.dtype == inputs[0][1] == inputs[1][1] == inputs[2][1] == outputs[0][1] == 'bf16'
            assert meta['state_dtype'] == inputs[4][1] == outputs[1][1] == 'fp32'
            assert meta['key_heads'] == inputs[0][0][1] == inputs[1][0][1]
            assert meta['heads'] == H and meta['sequence_tokens'] == (geo['tokens_per_sequence'],) * geo['sequences_per_replica']
            assert [list(layout['shape'][1:]) for layout in meta['source_layouts']] == [i[0] for i in inputs[:3]]
            local = [op for op in ops if op.layer == core.layer]
            conv = next(op for op in local if op.recipe == CONV)
            gate = next(op for op in local if op.recipe == GATING)
            assert list(conv.local_shape) == conv_native[2][0][0] == conv_native[3][0][0]
            assert dict(conv.metadata)['state_history_slots'] == conv_native[2][3][0][1]
            assert dict(conv.metadata)['kernel_width'] == conv_native[2][1][0][0]
            assert gate.local_shape == (M, H)
            assert inputs[3][1] == 'bf16' and inputs[7][1] == 'fp32'
            assert dict(gate.metadata)['beta_store']['resource'] == 'scalar_store_issue'
            assert core.local_shape[0] == geo['query_rows'] != inputs[4][0][0]
            assert sum(op.family == 'recurrent_vector' for op in local) == 1
