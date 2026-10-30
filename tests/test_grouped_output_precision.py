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

from mindie_sim import graph_costs, target_graphs
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


def test_real_ordinary_graph_preserves_dependency_and_releases_resources():
    raw = raw_case({'mla.o_a': BF16})
    graphs = graph(raw)
    result = graph_costs(raw,graphs,input_ready_ms=7,isolated=False,
                         external_inputs={'analysis_input':7})
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert all(row['ready_ms'] >= 7 for row in result['timeline'])
    a = next(o for o in graphs[0] if o.recipe=='grouped_output_a')
    assert a.deps and not a.deps[0].endswith('quantize')


def test_prepared_mx_validation_uses_the_same_child_as_runtime_and_capacity():
    raw = json.loads((ROOT/'examples/cases/high-concurrency-20261030/'
        'v41-dynamic-rotary-conditional/a5_v41_fp8_tp1_dp8_ep8_c128_p2k-dynamic-rotary-source.json').read_text())
    raw['model']['layers']=1
    raw['model']['quant']['category_overrides']={'mla.o_a':BF16}
    raw['runtime']['kernels']['lowerings']['mx_matrix']={
        'implementation':'ascend_prepared_mx32_matrix', 'options':{
            'projection_recipes':['grouped_output_a'],
            'scale_storage':'replace_checkpoint_scales'}}
    # Preparation cannot accept parent MX when the actual child matrix is BF16.
    with pytest.raises(ModelingError,match='prepared_mx32_requires_e4m3_operands'):
        parse_case(raw)
    raw['runtime']['kernels']['lowerings']['mx_matrix']['options']['projection_recipes']=['q_b','grouped_output_b']
    c=parse_case(raw)
    a=next(o for o in graph(c)[0] if o.recipe=='grouped_output_a')
    assert a.dtype=='bf16'
