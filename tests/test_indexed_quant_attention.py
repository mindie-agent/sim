"""Packed-cache work, ownership and actual Engine controls; no fitted timings."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.indexed_quant_attention import build, phases
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.structure import parse_case


ROOT = Path(__file__).resolve().parents[1]


def case(compact=True):
    path = ROOT / 'examples/cases/high-concurrency-20261030/matrix-extension/a3_glm53flash_tp16_aclgraph_c32.json'
    raw = json.loads(path.read_text())
    raw['model']['layers'] = 4  # First ordinary indexed-attention layer is 3.
    raw['runtime']['kernels']['programs']['indexed_attention'] = {
        'kind': 'indexed_attention_packed_int8', 'scratch_policy': 'offchip',
        'compact': compact}
    return raw


def ops(raw, batch=1, query=1, context=4096):
    return [op for op in target_graphs(
        raw, scope='body', phase='decode', tokens=query, context=context,
        sequences=batch)[0] if dict(op.metadata).get('program_kind') == 'indexed_attention']


def run(raw, operators, *, trace=0):
    return graph_costs(raw, operators, input_ready_ms=5, isolated=False,
                       external_inputs={d: 7 for op in operators for d in op.deps},
                       trace_limit=trace)


def test_packed_rows_read_aliased_cache_once_and_convert_before_publish():
    stages = dict(phases(17, 4, 512, 0, first=True, last=True,
                         scratch_policy='resident'))
    assert stages['gather'].resources == {'hbm': 17 * (528 + 8),
                                          'mte2_aiv': 17 * (528 + 8)}
    # Four casts/scale element passes plus explicit ND->NZ lane copy.
    assert stages['dequantize'].resources == {'vector': 17 * (5 * 512 + 32)}
    assert stages['publish'].resources['l2_write'] == 17 * 512 * 2 + 1024
    assert sum(p.matrix.get('bf16', 0) for p in stages.values()) == 2 * 16 * 32 * 1024
    assert list(stages)[:3] == ['gather', 'dequantize', 'publish']


def rope64_case(compact=True):
    path = ROOT/'examples/cases/high-concurrency-20261030/glm78/a3_glm53_w4a8_tp16_tpot88_c8.json'
    raw = json.loads(path.read_text())
    raw['model']['layers'] = 1
    raw['runtime']['kernels']['programs']['indexed_attention'] = {
        'kind': 'indexed_attention_packed_int8', 'scratch_policy': 'offchip',
        'compact': compact, 'operand_modes': True}
    return raw


def test_rope64_packed_cache_keeps_bf16_rotary_bytes_and_vector_copy():
    zero = dict(phases(17, 4, 512, 0, first=True, last=True, scratch_policy='resident'))
    rotary = dict(phases(17, 4, 512, 64, first=True, last=True, scratch_policy='resident'))
    assert rotary['gather'].resources == {'hbm': 17*(656+8), 'mte2_aiv': 17*(656+8)}
    assert rotary['dequantize'].resources['vector']-zero['dequantize'].resources['vector'] == 17*64
    assert rotary['publish'].resources['l2_write'] == 17*576*2+1024
    assert rotary['pv'].matrix == zero['pv'].matrix
    assert rotary['qk'].matrix['bf16']-zero['qk'].matrix['bf16'] == 2*16*32*64


def test_rope64_batched_call_uses_one_response_and_independent_contexts():
    from mindie_sim import operator_costs, predict
    raw = rope64_case()
    attention = ops(raw, batch=8, query=6)
    assert len(attention) == 1
    shape = dict(attention[0].metadata)['indexed_attention_geometry']
    assert shape['sequences'] == ((6,4096),)*8
    raw['runtime']['kernels']['service_regimes'] = [{
        'id':'whole-packed-attention-control',
        'match':{'family':'attention','metadata.indexed_attention_geometry.rope':64},
        'response':{'fixed_us':10000}, 'provenance':{'basis':'synthetic_boundary_control'}}]
    result = operator_costs(raw, attention, input_ready_ms=7)
    assert result['operator_sum_ms'] == pytest.approx(10)
    graph = run(raw, attention)
    assert graph['parameters']['service_regimes'][0]['queries'] == {'observable_response':1}
    assert graph['complete_ms'] == pytest.approx(17)
    plan = result['operators'][0]['operator']['metadata']['program_execution']
    assert plan['packed_cache']['row_bytes'] == 656
    assert plan['packed_cache']['local_padded_row_bytes'] == 672
    assert result['operators'][0]['service_before_input'] == {}
    # The normal public Body path installs the same batched pipeline without
    # an explicit recipe or empirical response.
    raw['runtime']['kernels']['programs'].pop('indexed_attention')
    raw['runtime']['kernels'].pop('service_regimes')
    body = predict(raw,scope='body',phase='decode',tokens=6,context=4096,sequences=8)
    rows = [r for r in body['timeline'] if r['family']=='attention']
    assert len(rows) == 1
    assert rows[0]['recipe'] == 'latent_attention_batch'
    assert body['execution']['live_tasks'] == body['execution']['live_programs'] == 0


def test_rope64_compact_and_detailed_share_work_and_paired_publish_ownership():
    reports = []
    for compact in (False,True):
        raw = rope64_case(compact)
        reports.append(run(raw,ops(raw,batch=2,query=1,context=16),trace=2000))
    assert reports[0]['execution']['serviced'] == pytest.approx(reports[1]['execution']['serviced'])
    for report in reports:
        assert not any(report['execution']['live_allocated'].values())
        assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
        assert all(e['start_ms']>=7 for e in report['events'] if e['demand'])


@pytest.mark.parametrize('sequences', [((1, 0),), ((2, 510), (1, 1024)), ((4, 2047),)])
def test_detailed_and_compact_actual_engine_conserve_tail_work_and_readiness(sequences):
    raw = case(); geometry = Geometry.from_devices(parse_case(raw).machine.dies)
    op = ops(raw)[0]; shape = {**dict(op.metadata)['indexed_attention_geometry'],
                              'sequences': sequences}
    op = replace(op, metadata=(('indexed_attention_geometry', shape),))
    reports = [run(case(compact), [op], trace=20000) for compact in (False, True)]
    expanded, compact = reports
    assert expanded['execution']['serviced'] == pytest.approx(compact['execution']['serviced'])
    for report in reports:
        assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
        assert all(e['start_ms'] >= 7 for e in report['events'] if e['demand'])
    plan = build(op, geometry, scratch_policy='offchip')
    separate = Work()
    for query, context in sequences:
        one = {k: v for k, v in shape.items() if k != 'sequences'}
        one.update(query=query, context=context)
        separate += build(replace(op, metadata=(('indexed_attention_geometry', one),)),
                          geometry, scratch_policy='offchip').work
    assert plan.work == separate


def test_real_batched_c32_c64_ordinary_branch_and_bandwidth_response():
    raw = case(); small, large = ops(raw, 32, 6), ops(raw, 64, 6)
    assert len(small) == len(large) == 1
    for batch, operators in [(32, small), (64, large)]:
        meta = dict(operators[0].metadata)
        assert meta['attention_tensor_view']['query_shape'] == (6 * batch, 4, 512)
        assert meta['indexed_attention_geometry']['sequences'] == ((6, 4096),) * batch
    lo, hi = run(raw, small), run(raw, large)
    assert hi['execution']['work']['hbm_bytes'] == pytest.approx(2 * lo['execution']['work']['hbm_bytes'])
    assert hi['complete_ms'] > lo['complete_ms']
    values = []
    for factor in (0.5, 1, 2):
        trial = deepcopy(raw); trial['machine']['hbm_bw_bytes_per_s'] = 1.2e12 * factor
        values.append(run(trial, small)['complete_ms'])
    assert values[0] > values[1] > values[2]


def test_single_row_dequant_and_gather_do_not_borrow_idle_paired_aiv():
    raw = case(False); operators = ops(raw, context=0)
    report = run(raw, operators, trace=500)
    dequant = [e for e in report['events'] if ':dequantize' in e['task'] and e['demand']]
    assert dequant
    gather = [e for e in report['events'] if ':gather' in e['task'] and e['demand']]
    assert {tuple(v) for e in gather for k, v in e['placement'].items()
            if k.startswith('mte2_aiv@')} == {(0,)}
    assert {tuple(v) for e in dequant for k, v in e['placement'].items()
            if k.startswith('vector@')} == {(0,)}
    softmax = [e for e in report['events'] if ':softmax' in e['task'] and e['demand']]
    assert {tuple(v) for e in softmax for k, v in e['placement'].items()
            if k.startswith('vector@')} == {(0,)}
    assert report['execution']['work']['hbm_bytes'] > 0
    assert report['execution']['peak_allocated_per_unit'] == {
        'ub@r0:s0:unit0': 189952, 'ub@r0:s0:unit1': 189952}
    assert not any(report['execution']['live_allocated'].values())


def test_outer_invocation_is_paid_once_for_many_queries_and_tiles():
    from mindie_sim import operator_costs
    raw = case(); operators = ops(raw, 4, 6)
    raw['runtime']['kernels']['invocations'] = [{
        'match': {}, 'startup_ms': 1, 'minimum_ms': 10,
        'provenance': {'kind': 'causal_control'}}]
    report = operator_costs(raw, operators, input_ready_ms=7)
    assert report['operator_sum_ms'] == pytest.approx(10)
    assert report['operators'][0]['service_before_input'] == {}
    execution = report['operators'][0]['operator']['metadata']['program_execution']
    assert execution['packed_cache']['row_bytes'] == 528
    assert execution['coverage'] == 'analytic_adaptation'


@pytest.mark.parametrize('keys', [1, 2, 3, 17, 512])
def test_pair_partition_respects_heaviest_aiv_tail_without_inventing_work(keys):
    raw = case(False)
    raw['machine']['device'] = {'rates': {'vector': 1e6}}
    raw['runtime']['kernels']['invocations'] = [{
        'match': {}, 'startup_ms': 0, 'minimum_ms': 0,
        'provenance': {'kind': 'resource_control'}}]
    result = run(raw, ops(raw, context=keys-1))
    work = dict(phases(keys, 4, 512, 0, first=True, last=True,
                       scratch_policy='offchip'))
    first = min(keys, ((keys + 3) // 4) * 2)
    vector_demand = sum(p.resources.get('vector', 0) for p in work.values())
    dequant = work['dequantize'].resources['vector']
    expected_ms = (vector_demand - dequant + dequant * first / keys) / (1e6 / 40) * 1000
    elapsed = result['complete_ms'] - 7
    # Other physical pipes remain present but are >10^6 faster than the
    # deliberately constrained vector. This isolates the source pair split.
    assert elapsed == pytest.approx(expected_ms, rel=1e-5)


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('keys', [1, 2])
def test_short_tail_does_not_contend_with_idle_sibling_vector(compact, keys):
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.cost import CostModel
    raw = case(compact)
    raw['machine']['device'] = {'rates': {'vector': 1e6}}
    parsed = parse_case(raw)
    op = ops(raw, context=keys - 1)[0]
    plan = build(op, Geometry.from_devices(parsed.machine.dies),
                 scratch_policy='offchip', compact=compact)

    def execute(contender):
        execution = Execution(parsed, CostModel(parsed), include_submission=False)
        ctx = execution._context(plan.work.apply(op), 0, 'none', recipe=plan)
        engine = execution.engine
        ready = engine.submit('input', release=7)
        vector = ctx.resource_names['vector']
        if contender:
            engine.submit('idle_sibling', {vector: plan.work.resources['vector'] * 10},
                          release=7, placement={vector: (1,)})
        done = plan.lower(ctx).start(engine, 'source', ready)
        engine.advance(horizon=6)
        assert not any(engine.serviced.values())
        engine.advance(until=(done,))
        elapsed = done.time - 7
        engine.advance()
        assert not engine.summary()['live_tasks'] and not engine.summary()['live_programs']
        assert not any(engine.allocated.values())
        return elapsed

    assert execute(True) == pytest.approx(execute(False), rel=1e-12)


@pytest.mark.parametrize('keys', [1, 2, 3, 17])
def test_pair_publish_includes_each_aiv_descriptor_and_its_own_payload(keys):
    raw = case(False)
    raw['machine']['device'] = {'rates': {'mte3': 1e6}}
    raw['runtime']['kernels']['invocations'] = [{
        'match': {}, 'startup_ms': 0, 'minimum_ms': 0,
        'provenance': {'kind': 'resource_control'}}]
    report = run(raw, ops(raw, context=keys - 1), trace=500)
    publish = next(e for e in report['events'] if ':publish' in e['task'] and e['demand'])
    first = min(keys, ((keys + 3) // 4) * 2)
    expected_ms = (first * 512 * 2 + 512) / (1e6 / 40) * 1000
    assert publish['end_ms'] - publish['start_ms'] == pytest.approx(expected_ms, rel=1e-10)
    assert next(v for k, v in publish['placement'].items() if k.startswith('mte3@')) == (0, 1)
