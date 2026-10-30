"""Independent projection precision from the retained BF16 output-A ABI.

[Role] Keep ordinary matrix work and resident weights on one quant category.
[Invariants] Child precision only changes that projection; parent MLA overrides
  still apply. Native timing and capture binary identity are not test inputs.
[Timeouts & Locks] Small in-memory graphs and bounded Engine execution.
[Side-Effect Policy] No hardware or output writes.
"""
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs, target_operators
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]
BF16 = {"weight_format": "bf16", "activation_format": "bf16"}
INT8 = {"weight_format": "int8", "activation_format": "int8"}


def raw_case(overrides=None, tp=8):
    raw = json.loads((ROOT / 'examples/cases/high-concurrency-20261030/new17/'
                     'v4flash-a3-tp8-c4.json').read_text())
    raw['model']['layers'] = 1
    raw['runtime']['tp'] = raw['runtime']['ep'] = tp
    raw['model']['quant']['category_overrides'] = overrides or {}
    return raw


def graph(raw, batch=4):
    return target_graphs(raw, scope='body', phase='decode', tokens=2,
                         sequences=batch, context=4096)


@pytest.mark.parametrize('tp,batch', [(8,4), (4,8)])
def test_grouped_output_a_bf16_matches_work_and_resident_tensor_contract(tp, batch):
    raw = raw_case({'mla.o_a': BF16}, tp)
    c = parse_case(raw)
    ops = graph(raw, batch)[0]
    by_recipe = {o.recipe: o for o in ops}
    a, b, qb = (by_recipe[r] for r in ('grouped_output_a','grouped_output_b','q_b'))
    groups = 8 // tp
    assert a.local_shape == (2*batch*groups,1024,4096)
    md = dict(a.metadata)
    assert md['matrix_groups'] == groups
    assert a.dtype == md['activation_format'] == md['weight_format'] == 'bf16'
    assert md['packed_weight_bytes'] == groups*4096*1024*2
    assert md.get('activation_producer') != 'explicit_quantization'
    assert not any(o.op_id == a.op_id+'.quantize' for o in ops)
    assert b.dtype == qb.dtype == 'int8'
    assert dict(b.metadata)['activation_producer'] == 'explicit_quantization'
    terms = {t.name:t for t in layer_weight_terms(c.model.layer_plans[0],c.model)}
    assert term_bytes(terms['wo_a'],c.model.quant,c.runtime.parallel) == md['packed_weight_bytes']
    parsed = parse_case(c.to_dict())
    assert graph(parsed,batch) == graph(c,batch)
    assert layer_weight_terms(parsed.model.layer_plans[0],parsed.model) == list(terms.values())


def test_default_and_parent_category_inheritance_preserve_physical_work():
    default, explicit = graph(raw_case())[0], graph(raw_case({'mla.o_a':INT8}))[0]
    assert default == explicit
    parent = graph(raw_case({'mla':BF16}))[0]
    explicit_child = graph(raw_case({'mla':BF16,'mla.o_a':BF16}))[0]
    assert parent == explicit_child
    child_int8 = graph(raw_case({'mla':BF16,'mla.o_a':INT8}))[0]
    for op in child_int8:
        if op.recipe in ('q_b','grouped_output_b'):
            assert op.dtype == 'bf16'
        elif op.recipe == 'grouped_output_a':
            assert op.dtype == 'int8'


def test_output_b_child_changes_only_its_operands_and_resident_weights():
    raw = raw_case({'mla.o_b': BF16})
    c = parse_case(raw)
    ops = graph(raw)[0]
    by_recipe = {op.recipe: op for op in ops}
    a, b, qb = (by_recipe[r] for r in ('grouped_output_a', 'grouped_output_b', 'q_b'))
    assert b.dtype == 'bf16'
    assert a.dtype == qb.dtype == 'int8'
    assert not any(op.op_id == b.op_id + '.quantize' for op in ops)
    assert dict(a.metadata)['activation_producer'] == 'explicit_quantization'
    terms = {term.name: term for term in layer_weight_terms(c.model.layer_plans[0], c.model)}
    assert term_bytes(terms['wo_b'], c.model.quant, c.runtime.parallel) == 4096 * 1024 * 2
    assert dict(b.metadata)['packed_weight_bytes'] == 4096 * 1024 * 2
    assert graph(parse_case(c.to_dict())) == graph(c)


@pytest.mark.parametrize('name,tokens,hidden,expert_format', [
    ('v4f-a3-w8a8-tp8', 4096, 4096, 'int8'),
    ('v4pro-a3-w4a8-tp16', 1024, 7168, 'int4'),
])
def test_retained_prefill_output_abi_preserves_other_quantized_roles(name, tokens, hidden, expert_format):
    # Real pre4096.kernel_details.csv tensor contracts: V4F lines 143-144,
    # V4Pro lines 144-145. TransposeBatchMatMul and MatMulV3 both read BF16.
    raw = json.loads((ROOT / 'examples/cases' / (name + '.json')).read_text())
    raw['model']['layers'] = 1
    c = parse_case(raw)
    ops = target_operators(c, scope='body', phase='prefill', tokens=tokens, context=0)
    by_recipe = {op.recipe: op for op in ops}
    terms = {term.name: term for term in layer_weight_terms(c.model.layer_plans[0], c.model)}
    for role, weight, n, k in (('grouped_output_a', 'wo_a', 1024, 4096),
                               ('grouped_output_b', 'wo_b', hidden, 1024)):
        op = by_recipe[role]
        assert op.local_shape == (tokens, n, k)
        assert op.dtype == 'bf16'
        assert op.flops == 2 * tokens * n * k
        assert op.hbm_bytes == 2 * (tokens * k + n * k + tokens * n)
        assert dict(op.metadata)['packed_weight_bytes'] == n * k * 2
        assert term_bytes(terms[weight], c.model.quant, c.runtime.parallel) == n * k * 2
        assert not any(other.op_id == op.op_id + '.quantize' for other in ops)
    for role in ('q_a', 'kv_a', 'q_b', 'dense_gate_up', 'dense_down'):
        op = by_recipe[role]
        assert op.dtype == 'int8'
        assert dict(op.metadata)['activation_producer'] == 'explicit_quantization'
    assert dict(by_recipe['grouped_gate_up'].metadata)['weight_format'] == expert_format
    assert c.model.quant.index_dtype == 'int8'
    base_raw = {**raw, 'model': {key: value for key, value in raw['model'].items() if key != 'quant'}}
    base_quant = parse_case(base_raw).model.quant.to_dict()
    actual_quant = c.model.quant.to_dict()
    for metadata in ('source', 'notes', 'category_overrides'):
        base_quant.pop(metadata)
        actual_quant.pop(metadata)
    assert actual_quant == base_quant


def test_real_ordinary_graph_preserves_dependency_and_releases_resources():
    raw = raw_case({'mla.o_a': BF16})
    graphs = graph(raw)
    result = graph_costs(raw,graphs,input_ready_ms=7,isolated=False,
                         external_inputs={'analysis_input':7})
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert all(row['ready_ms'] >= 7 for row in result['timeline'])
    a = next(o for o in graphs[0] if o.recipe=='grouped_output_a')
    assert a.deps and not a.deps[0].endswith('quantize')


@pytest.mark.parametrize('role,category', [
    ('grouped_output_a', 'mla.o_a'), ('grouped_output_b', 'mla.o_b')])
def test_prepared_mx_validation_uses_the_same_child_as_runtime_and_capacity(role, category):
    raw = json.loads((ROOT/'examples/cases/high-concurrency-20261030/'
        'v41-dynamic-rotary-conditional/a5_v41_fp8_tp1_dp8_ep8_c128_p2k-dynamic-rotary-source.json').read_text())
    raw['model']['layers']=1
    raw['model']['quant']['category_overrides']={category:BF16}
    raw['runtime']['kernels']['lowerings']['mx_matrix']={
        'implementation':'ascend_prepared_mx32_matrix', 'options':{
            'projection_recipes':[role],
            'scale_storage':'replace_checkpoint_scales'}}
    # Preparation cannot accept parent MX when the actual child matrix is BF16.
    with pytest.raises(ModelingError,match='prepared_mx32_requires_e4m3_operands'):
        parse_case(raw)
    raw['runtime']['kernels']['lowerings']['mx_matrix']['options']['projection_recipes']=['q_b']
    c=parse_case(raw)
    output=next(o for o in graph(c)[0] if o.recipe==role)
    assert output.dtype=='bf16'
