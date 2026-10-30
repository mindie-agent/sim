"""Prepared scale residency and AIC publication in real ordinary graphs."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import target_graphs, operator_costs
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.prediction.family_mx_matrix import PROFILE
from mindie_sim.prediction.structure import parse_case

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('q_b', 'grouped_output_a', 'grouped_output_b')


def selected(raw):
    case = deepcopy(raw)
    case['runtime']['kernels']['lowerings']['mx_matrix'] = {
        'implementation': 'ascend_prepared_mx32_matrix', 'options': {
            'projection_recipes': ROLES, 'scale_storage': 'replace_checkpoint_scales'}}
    return case


@pytest.fixture(scope='module')
def raw():
    return json.loads((ROOT / 'examples/cases/high-concurrency-20261030/v41-dynamic-rotary-conditional/'
                      'a5_v41_fp8_tp1_dp8_ep8_c128_p2k-dynamic-rotary-source.json').read_text())


@pytest.fixture(scope='module')
def controls(raw):
    result = {}
    for batch in (1, 16):
        args = dict(scope='body',phase='decode',tokens=1,context=2048,sequences=[batch]+[0]*7)
        baseline = target_graphs(raw, **args)[0]
        variant = target_graphs(selected(raw), **args)[0]
        result[batch] = baseline, variant
    return result


def test_only_selected_projections_change_in_full_40_layer_graph(raw, controls):
    for base, variant in controls.values():
        changed = []
        assert len(base) == len(variant)
        for old, new in zip(base, variant):
            if old.recipe not in ROLES:
                assert old == new
                continue
            changed.append(new)
            md = dict(new.metadata)
            m,n,k = old.local_shape
            g = dict(old.metadata)['matrix_groups']
            assert md['semantic_recipe'] == old.recipe
            assert new.deps == old.deps and new.op_id == old.op_id
            assert new.flops == old.flops and new.mte3_bytes == 0
            assert md['packed_weight_bytes'] == g*n*k + g*n*k//32
            assert md['fixpipe_bytes'] == 2*m*n
            assert new.to_dict()['layout'] == md['weight_representation'] == 'fp8_block1x32_fp8_e8m0'
        assert len(changed) == 120


@pytest.mark.parametrize('batch', [1,16])
@pytest.mark.parametrize('role', ROLES)
def test_real_matrix_work_output_and_input_readiness(raw, controls, batch, role):
    op = next(o for o in controls[batch][1] if dict(o.metadata).get('semantic_recipe') == role)
    result = operator_costs(selected(raw), (op,), input_ready_ms=7)['operators'][0]
    m,n,k = op.local_shape; meta = dict(op.metadata)
    assert result['serviced']['fixpipe@r0:s0'] == pytest.approx(2*m*n)
    assert result['serviced']['memory@r0:s0'] == pytest.approx(op.hbm_bytes)
    assert not any(result['serviced'].get(name+'@r0:s0',0) for name in ('mte3','vector'))
    assert not result['service_before_input'] and not result['live_tasks']
    assert not any(result['remaining_buffers'].values())
    if role == 'grouped_output_a':
        view = meta['native_tensor_view']
        assert view['input_stride'] == (8*k,k,1)
        assert view['output_stride'] == (8*n,n,1)
        assert meta['matrix_groups'] == 8 and m//8 == batch


def test_checkpoint_bytes_runtime_replacement_roundtrip_and_unbind(raw):
    base, prepared = parse_case(raw), parse_case(selected(raw))
    terms = lambda model: {t.name:t for t in layer_weight_terms(model.layer_plans[0], model)}
    old,new = terms(base.model),terms(prepared.model)
    assert old.keys() == new.keys()
    for name in old:
        a,b = old[name],new[name]
        assert term_bytes(a,base.model.quant,base.runtime.parallel,source=True) == term_bytes(
            b,prepared.model.quant,prepared.runtime.parallel,source=True)
        before = term_bytes(a,base.model.quant,base.runtime.parallel)
        after = term_bytes(b,prepared.model.quant,prepared.runtime.parallel)
        if name in ('q_b','wo_a','wo_b'):
            payload = b.rows*b.cols
            assert after-before == payload/32-payload/1024
            assert b.runtime_quant.weight_block == (1,32)
        else:
            assert after == before
    roundtrip = parse_case(prepared.to_dict())
    assert terms(roundtrip.model) == new
    kernels = deepcopy(prepared.runtime.kernels);del kernels['lowerings']['mx_matrix']
    unbound = parse_case(replace(prepared,runtime=replace(prepared.runtime,kernels=kernels)))
    assert terms(unbound.model) == old
    assert 'prepared_mx32_matrix' not in dict(unbound.model.metadata)


@pytest.mark.parametrize('axis', ['hbm','mte1','mte2_aic','fixpipe','cube'])
def test_resource_sensitivity_preserves_work(raw, controls, axis):
    op = next(o for o in controls[16][1] if dict(o.metadata).get('semantic_recipe') == 'grouped_output_a')
    case = parse_case(selected(raw));results=[]
    for factor in (.5,1,2):
        dies = tuple(replace(d,rates=tuple((k,v*(factor if k==axis else 1)) for k,v in d.rates),
            compute_peaks=tuple((k,v*(factor if axis=='cube' else 1)) for k,v in d.compute_peaks))
            for d in case.machine.dies)
        report=operator_costs(replace(case,machine=replace(case.machine,dies=dies)),(op,))['operators'][0]
        results.append(report)
    assert results[0]['duration_ms'] >= results[1]['duration_ms'] >= results[2]['duration_ms']
    for factor,result in zip((.5,1,2),results):
        # Matrix service is peak-normalized ms; recover invariant FLOP work.
        serviced = dict(result['serviced'])
        if axis == 'cube':
            serviced['matrix@r0:s0'] *= factor
        assert serviced == pytest.approx(results[1]['serviced'])
        assert result['operator']['flops'] == op.flops
    # A dedicated bottleneck control is strict even when normal scans plateau.
    dies=tuple(replace(d,rates=tuple((k,v*(.001 if k==axis else 1)) for k,v in d.rates),
        compute_peaks=tuple((k,v*(.001 if axis=='cube' else 1)) for k,v in d.compute_peaks))
        for d in case.machine.dies)
    slow=operator_costs(replace(case,machine=replace(case.machine,dies=dies)),(op,))['operators'][0]
    assert slow['duration_ms'] > results[1]['duration_ms']
