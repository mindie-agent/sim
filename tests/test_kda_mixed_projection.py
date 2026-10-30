"""Retained TP8/TP16 projection groups, physical storage and caller joins.

[Role] Verify one source-declared QKV/BFG pair with real KDA dimensions.
[Invariants] Native durations are not inputs. Offline composition changes
  online FLOPs and resident matrices; layout views do not create free copies.
[Timeouts & Locks] Small in-memory graphs and finite one-layer Engine controls.
[Side-Effect Policy] Read source ABI fixtures only; no hardware or subprocesses.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/verification/kimi-mixed-projection-contract.json').read_text())


def case(tp):
    return json.loads((ROOT / f'examples/cases/kimi-mixed-projection-20261030/a3-kimi4-tp{tp}.json').read_text())


def first_layer(raw, batch):
    parsed = parse_case(raw)
    full = target_graphs(parsed, phase='decode', tokens=1, sequences=batch,
                         context=126000, scope='body', resource_fusion=False)[0]
    stop = next(index for index, op in enumerate(full) if op.layer > 0)
    return parsed, full[:stop]


@pytest.mark.parametrize('capture', FIXTURE['captures'], ids=lambda row: f"tp{row['tp']}")
def test_retained_projection_groups_have_one_quantized_input_and_exact_output_views(capture):
    raw = case(capture['tp'])
    parsed, ops = first_layer(raw, capture['batch_slots'])
    by = {op.recipe: op for op in ops}
    bfg, qkv = by['linear_bfg'], by['linear_qkv']
    m, heads, hidden = capture['batch_slots'], 96 // capture['tp'], 7168
    assert bfg.local_shape == (m, heads * 257, hidden)
    assert qkv.local_shape == (m, heads * 384, hidden)
    assert bfg.dtype == 'bf16' and qkv.dtype == 'int8'
    native_bfg = next(row for row in capture['rows'] if row['Type'] == 'MatMulV2')
    native_qkv = next(row for row in capture['rows'] if row['Type'] == 'QuantBatchMatmulV3')
    assert native_bfg['Input Shapes'].strip('"') == f'{m},7168;{heads*257},7168'
    assert native_bfg['Input Data Types'] == 'DT_BF16;DT_BF16'
    assert native_qkv['Input Shapes'].strip('"') == f'{m},7168;{heads*12},448,16,32;{heads*384};{m}'
    assert native_qkv['Input Data Types'] == 'INT8;INT8;DT_BF16;FLOAT'
    conversions = [op for op in ops if op.op_id == qkv.op_id + '.quantize']
    quant, = conversions
    assert quant.local_shape == (m, hidden)
    assert quant.mte2_bytes == m * hidden * 2
    assert quant.mte3_bytes == m * hidden + m * 4
    assert qkv.deps == (quant.op_id, bfg.op_id)
    assert not any(op.recipe in {'linear_q','linear_k','linear_v','linear_decay_a','linear_decay_b'} for op in ops)
    assert activation_quantization(ops, 'bf16') == list(ops)
    slices = dict(bfg.metadata)['packed_projection_slices']
    assert [item['shape'][1] for item in slices] == [heads, heads*128, heads*128]
    assert [item['offset_elements'] for item in slices] == [0, heads, heads*129]
    assert all(item['strides'] == (heads*257,1) for item in slices)
    assert ('kda_beta_contiguous' in by) == (m > 1)
    if m > 1:
        assert by['kda_beta_contiguous'].mte2_bytes == by['kda_beta_contiguous'].mte3_bytes == m*heads*2
    cast = by['linear_beta_cast']
    assert cast.mte2_bytes == m*heads*2 and cast.mte3_bytes == m*heads*4
    assert by['linear_beta_sigmoid'].dtype == 'fp32'
    assert by['linear_beta_sigmoid'].vector_ops == 4*m*heads
    assert bfg.flops == 2*m*hidden*heads*257
    assert qkv.flops == 2*m*hidden*heads*384
    assert dict(bfg.metadata)['packed_weight_bytes'] == hidden*heads*257*2
    # Grouping preserves the case's existing scale+offset residency policy;
    # the BF16 native scale argument alone does not identify retained buffers.
    assert dict(qkv.metadata)['packed_weight_bytes'] == hidden*heads*384 + heads*384*8


@pytest.mark.parametrize('tp', (8,16))
def test_composed_decay_residency_retains_checkpoint_factors_and_does_not_rewrite_model_weights(tp):
    raw = case(tp)
    selected = parse_case(raw)
    off = deepcopy(raw)
    off['runtime']['kernels']['lowerings'].pop('kda_input_projection')
    plain = parse_case(off)
    before = {term.name: term for term in layer_weight_terms(plain.model.layer_plans[0], plain.model)}
    after = {term.name: term for term in layer_weight_terms(selected.model.layer_plans[0], selected.model)}
    assert set(after) - set(before) == {'kda_composed_f'}
    for name in before:
        assert after[name] == before[name]
    derived = after['kda_composed_f']
    assert not derived.source_owned
    assert term_bytes(derived, selected.model.quant, selected.runtime.parallel) == 96//tp*128*7168*2
    assert term_bytes(derived, selected.model.quant, selected.runtime.parallel, source=True) == 0
    _, ops = first_layer(raw, 2)
    bfg = next(op for op in ops if op.recipe == 'linear_bfg')
    active = sum(term_bytes(after[name], selected.model.quant, selected.runtime.parallel)
                 for name in ('kda_b','kda_composed_f','kda_g'))
    assert active == dict(bfg.metadata)['packed_weight_bytes']
    # Parsed cases can change software declarations without keeping a stale
    # prepared matrix resident after its selector has been removed.
    roundtrip = selected.to_dict()
    roundtrip['runtime']['kernels']['lowerings'].pop('kda_input_projection')
    restored = parse_case(roundtrip)
    assert 'kda_input_projection' not in dict(restored.model.metadata)
    assert layer_weight_terms(restored.model.layer_plans[0], restored.model) == list(before.values())


def test_two_stage_caller_join_runs_quant_with_bfg_then_beta_with_qkv():
    raw = case(8)
    _, ops = first_layer(raw, 2)
    report = graph_costs(raw, ops, input_ready_ms=7, external_inputs={'analysis_input':7},
                         isolated=False, trace_limit=0)
    rows = {row['recipe']: row for row in report['timeline']}
    qkv, bfg, beta = (rows[name] for name in ('linear_qkv','linear_bfg','linear_beta_sigmoid'))
    quant = next(row for row in report['timeline'] if row['op_id'] == qkv['op_id']+'.quantize')
    cast = rows['linear_beta_cast']
    assert quant['ready_ms'] < bfg['complete_ms']
    assert qkv['ready_ms'] >= max(quant['complete_ms'], bfg['complete_ms'])
    assert cast['ready_ms'] >= quant['complete_ms']
    assert beta['ready_ms'] < qkv['complete_ms']
    assert rows['short_convolution']['ready_ms'] >= max(qkv['complete_ms'],beta['complete_ms'])
    assert min(row['ready_ms'] for row in report['timeline']) >= 7
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0


def test_explicit_consumer_layout_copies_only_the_two_gate_views():
    raw = case(8)
    _, normal = first_layer(raw, 3)
    assert not any(op.recipe in {'kda_decay_contiguous','kda_output_gate_contiguous'} for op in normal)
    raw['runtime']['tensor_layouts'] = {'kda_recurrent':{'gate':'contiguous'},
                                      'linear_output_norm_gate':{'gate':'contiguous'}}
    _, copied = first_layer(raw, 3)
    by = {op.recipe: op for op in copied}
    for name, offset in (('kda_decay_contiguous',12),('kda_output_gate_contiguous',1548)):
        op = by[name]
        assert op.mte2_bytes == op.mte3_bytes == 3*12*128*2
        layout = dict(op.metadata)['source_layout']
        assert layout['offset'] == offset
        assert 3084 in layout['strides']
    gate_layout = dict(by['linear_output_norm_gate'].metadata)['gate_input_layout']
    assert gate_layout['strides'] == (1536,128,1)
    core = next(op for op in copied if op.family == 'recurrent_scan')
    assert 's0.l0.kda_decay_contiguous' in core.deps


def test_incompatible_qkv_precision_cannot_borrow_the_fused_native_contract():
    raw = case(8)
    raw['model']['quant']['category_overrides']['kda.q'] = {
        'weight_format':'bf16', 'activation_format':'bf16'}
    with pytest.raises(ValueError,match='qkv_group_requires_one_operand'):
        first_layer(raw, 2)
