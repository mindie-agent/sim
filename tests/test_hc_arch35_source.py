"""Real frontend/Engine controls for the explicit HC arch35 source branch."""
import copy
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.family_hc_arch35 import PRE_PROFILE, POST_PROFILE, selected_plan
from mindie_sim.prediction.hc_arch35_program import pre_geometry, pre_owner, total_work
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operator_analysis import target_graphs, operator_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]


def raw_case():
    raw = json.loads((ROOT / 'examples/source-families/v41-hc-source-control.json').read_text())
    opts = raw['runtime']['kernels']['lowerings']['native_fusion']['options']
    opts['traffic_profile'] = {'hc_pre': PRE_PROFILE, 'hc_post': POST_PROFILE}
    return raw


@lru_cache(None)
def sample(batch=1):
    case = parse_case(raw_case())
    ops = target_graphs(case, scope='body', phase='decode', tokens=1, context=2048, sequences=batch)[0]
    pre = next(o for o in ops if o.layer == 0 and dict(o.metadata).get('fusion_role') == 'hc_pre')
    post = next(o for o in ops if o.layer == 0 and dict(o.metadata).get('fusion_role') == 'hc_post')
    return case, pre, post


def changed_view(op, **fields):
    metadata = dict(op.metadata)
    metadata['fusion_tensor_view'] = {**metadata['fusion_tensor_view'], **fields}
    return replace(op, metadata=tuple(metadata.items()))


def geometry(case):
    return Geometry.from_devices(case.machine.dies)


def run(case, op):
    row = operator_costs(case, (op,), input_ready_ms=7)['operators'][0]
    assert not row['service_before_input']
    assert row['live_tasks'] == 0 and not any(row['remaining_buffers'].values())
    return row


@pytest.mark.parametrize('batch', (1, 8, 65))
def test_pre_independent_payload_and_physical_matrix_oracle(batch):
    case, pre, _ = sample(batch)
    row = run(case, pre)
    view = dict(pre.metadata)['fusion_tensor_view']
    h, k, cores = view['hidden'], 4 * view['hidden'], case.machine.dies[0].compute_units
    span = ((k + cores - 1) // cores + 255) // 256 * 256
    partitions = (k + span - 1) // span
    rows_per_aiv = (batch + 2 * cores - 1) // (2 * cores)
    active_second = (batch + rows_per_aiv - 1) // rows_per_aiv
    batches = sum((min(rows_per_aiv, batch-i*rows_per_aiv) + 3) // 4
                  for i in range(active_second)) if rows_per_aiv > 4 else active_second
    # Source stage2 rowFactor fits rowFormer for these cases; scalar scales
    # fetched once per owner batch, not once per token.
    expected_hbm = (batch*k*2 + 24*k*4 + batch*partitions*25*4
        + active_second*96 + batch*partitions*25*4 + batches*12
        + batch*k*2 + batch*h*2 + batch*80)
    work = row['operator']
    assert work['hbm_bytes'] == expected_hbm
    assert work['matrix_flops'] == {'hf32': 2 * ((batch+15)//16*16) * 32 * k}
    assert row['serviced']['mte1@r0:s0'] == pytest.approx(((batch+15)//16*16 + 32) * k * 4)
    # UB→L1 is private MTE3 work, never another HBM write.
    assert row['serviced']['mte3@r0:s0'] == pytest.approx(batch*k*4 + batch*partitions*4 + batch*h*2 + batch*80)
    assert row['serviced']['fixpipe@r0:s0'] == pytest.approx(batch*partitions*24*4)
    assert all(v <= 262144 for k,v in row['peak_buffers_per_unit'].items() if k.startswith('ub'))


@pytest.mark.parametrize('batch,hidden', ((1, 5120), (8, 5120), (3, 4128)))
def test_post_source_tail_cast_and_fixed_buffer_are_executed(batch, hidden):
    case, _, post = sample(batch)
    post = changed_view(post, hidden=hidden)
    rows = dict(post.metadata)['fusion_tensor_view']['tokens']
    row = run(case, post)
    resources = row['serviced']
    assert resources['mte2_aiv@r0:s0'] == pytest.approx(rows * (10*hidden + 80))
    assert resources['mte3@r0:s0'] == pytest.approx(rows * 8*hidden)
    assert resources['vector@r0:s0'] == pytest.approx(rows * (56*hidden + 16384*((hidden+4095)//4096)))
    assert len(row['peak_buffers_per_unit']) == rows
    assert set(row['peak_buffers_per_unit'].values()) == {213152}
    assert not any('mte1' in key or 'matrix' in key for key in resources)


def test_source_topology_preserves_l1_release_and_odd_row_aiv_ownership():
    case, pre, _ = sample()
    g = pre_geometry({**dict(pre.metadata)['fusion_tensor_view'], 'hidden': 32768}, geometry(case))
    phases = pre_owner(g, 0)
    nodes = {p.name:p for p in phases}
    # width131072 /32 =4096 => four L1 panels. Slot0 reuse waits last
    # MTE1 of panel0, not its Cube completion or whole-kernel drain.
    assert 'p0.k3.operands' in nodes['p2.weight'].after
    assert 'p0.k3.cube' not in nodes['p2.weight'].after
    assert 'p0.k3.operands' in nodes['p2.v1.c0.publish_l1'].after
    read_owners = {p.owner for p in phases if p.work.resources.get('mte2_aiv', 0)}
    assert read_owners == {1}  # source gives floor(M/2) to even AIV, ceil to odd.
    assert total_work((phases,)).resources['mte2_aiv'] == 4096*2
    assert nodes['p1.v1.c0.read'].after  # finite two-slot input ring


def test_real_buffers_limit_source_domain_without_changing_legacy_profile():
    case, pre, post = sample()
    geom = geometry(case)
    for op, pool, limit in ((pre,'l1',524287),(pre,'l0a',65535),(post,'ub',213151)):
        small = replace(geom, buffers={**geom.buffers,pool:limit})
        with pytest.raises(ModelingError, match='capacity'):
            selected_plan(op, small)
    # No silent source substitution for old explicit selection.
    meta=dict(pre.metadata);meta['fusion_boundary']={**meta['fusion_boundary'],
        'traffic_profile':'hc_pre_m_k_split_175ef4d4'}
    assert selected_plan(replace(pre,metadata=tuple(meta.items())),geom) is None


@pytest.mark.parametrize('axis', ('mte1','aiv','hbm'))
def test_three_axis_rates_act_on_the_same_source_work(axis):
    case, pre, post = sample(8)
    rates = dict(case.machine.dies[0].rates)
    roles = {'mte1':('mte1',),'aiv':('vector','mte2_aiv','mte3'),'hbm':('hbm',)}[axis]
    times=[];works=[]
    for factor in (.5,1,2):
        raw=raw_case();raw['machine'].setdefault('device',{})['rates']={k:rates[k]*factor for k in roles}
        row=run(parse_case(raw),pre)
        times.append(row['duration_ms']);works.append(row['operator']['hbm_bytes'])
    assert times[0] > times[1] > times[2]
    assert works[0] == works[1] == works[2]


def test_explicit_frontend_scope_preserves_all_external_hc_consumers():
    raw=raw_case();before=copy.deepcopy(raw)
    case=parse_case(raw)
    ops=target_graphs(case,scope='body',phase='decode',tokens=1,context=2048,sequences=1)[0]
    ids={o.op_id for o in ops}
    assert raw == before
    fused=[o for o in ops if dict(o.metadata).get('fusion_role') in {'hc_pre','hc_post'}]
    assert len(fused) == 160  # two pre + two post per 40-layer body.
    assert all(o.source_boundary == 'kernel' for o in fused)
    assert all(dep in ids for op in ops for dep in op.deps if dep.startswith('s0.'))
    costs=operator_costs(case,(fused[0],),input_ready_ms=7)['operators'][0]
    raw['runtime']['kernels']['invocations']=[{'match':{},'startup_ms':10,'minimum_ms':0}]
    launched=operator_costs(raw,(fused[0],),input_ready_ms=7)['operators'][0]
    assert launched['duration_ms'] == pytest.approx(costs['duration_ms'] + 10)
    assert launched['serviced'] == pytest.approx(costs['serviced'])
    assert costs['operator']['metadata']['program_execution']['source']['revision'].startswith('175ef4d4')


def test_repeated_layers_share_only_source_ir_not_bound_execution(monkeypatch):
    from mindie_sim.prediction import family_hc_arch35 as hc
    hc._pre_template.cache_clear()
    hc._post_template.cache_clear()
    case, pre, post = sample(3)
    geom = geometry(case)
    calls = []
    original = hc.pre_owner

    def counted(g, owner):
        calls.append(owner)
        return original(g, owner)

    monkeypatch.setattr(hc, 'pre_owner', counted)
    first = selected_plan(pre, geom)
    second = selected_plan(replace(pre, op_id='other_layer'), geom)
    assert first.work is second.work
    assert len(calls) == pre_geometry(dict(pre.metadata)['fusion_tensor_view'], geom).k_parts
    # Shared dictionaries cannot let a consumer corrupt later layers or devices.
    with pytest.raises(TypeError):
        first.work.resources['hbm'] = 0
    baseline = run(case, pre)
    repeated = run(case, pre)
    assert baseline['duration_ms'] == repeated['duration_ms']
    assert baseline['serviced'] == repeated['serviced']
    # A cache hit still performs target-local buffer validation.
    with pytest.raises(ModelingError, match='capacity'):
        selected_plan(pre, replace(geom, buffers={**geom.buffers, 'l1': 524287}))
    post_first = selected_plan(post, geom)
    assert post_first.work is selected_plan(replace(post, op_id='other_post'), geom).work
    # Owner splitting is a physical cache key, not just tensor dimensions.
    fewer_aivs = replace(geom, resource_units={**geom.resource_units, 'vector': 2})
    post_other = selected_plan(post, fewer_aivs)
    assert post_other.work is not post_first.work
    assert post_other.work == post_first.work
    assert post_other.metadata['active_owners'] == 2


@pytest.mark.parametrize('tokens,sequences', ((1024, 1), (2048, 1), (1025, 1), (4097, 1)))
def test_pre_and_post_large_m_follow_source_mk_tiling_and_tails(tokens, sequences):
    case = parse_case(raw_case())
    ops = target_graphs(case, scope='body', phase='prefill', tokens=tokens,
                        context=2048, sequences=sequences)[0]
    pre = next(o for o in ops if o.layer == 0 and
               dict(o.metadata).get('fusion_role') == 'hc_pre')
    post = next(o for o in ops if o.layer == 0 and
                dict(o.metadata).get('fusion_role') == 'hc_post')
    view = dict(pre.metadata)['fusion_tensor_view']
    rows = view['tokens']
    g = pre_geometry(view, geometry(case))
    cores = case.machine.dies[0].compute_units
    assert g.rows == rows
    assert g.m_dim == min(cores, (rows + 255) // 256)
    assert g.single_core_m == ((rows + g.m_dim - 1) // g.m_dim + 15) // 16 * 16
    assert g.m_l1 == min(256, g.single_core_m)
    assert g.m_tiles == (g.single_core_m + g.m_l1 - 1) // g.m_l1
    assert g.k_dim == cores // g.m_dim
    assert g.k_parts == (4 * view['hidden'] + g.k_span - 1) // g.k_span
    assert g.k_parts <= g.k_dim
    assert g.tiling_key == (1001 if g.k_dim == 1 else 1000)
    # Owner count is the source M x K core split, not a repeated 256 launch.
    assert g.m_dim * g.k_parts <= cores
    phases = pre_owner(g, g.m_dim * g.k_parts - 1)
    assert phases and len({p.name for p in phases}) == len(phases)
    row = run(case, pre)
    assert row['duration_ms'] > 0 and row['live_tasks'] == 0
    assert not any(row['remaining_buffers'].values())
    assert all(v <= 262144 for k, v in row['peak_buffers_per_unit'].items()
               if k.startswith('ub'))
    if g.tiling_key == 1001:
        # Key1001 is one local M-split kernel: no global K partials.
        assert g.k_parts == 1
    post_row = run(case, post)
    assert post_row['duration_ms'] > 0 and post_row['live_tasks'] == 0
    assert not any(post_row['remaining_buffers'].values())


def test_key1001_boundary_and_physical_tail_flops():
    case = parse_case(raw_case())
    for tokens in (257, 4096, 4097):
        ops = target_graphs(case, scope='body', phase='prefill', tokens=tokens,
                            context=2048, sequences=1)[0]
        pre = next(o for o in ops if o.layer == 0 and
                   dict(o.metadata).get('fusion_role') == 'hc_pre')
        view = dict(pre.metadata)['fusion_tensor_view']
        g = pre_geometry(view, geometry(case))
        assert g.tiling_key == (1001 if g.k_dim == 1 else 1000)
        row = run(case, pre)
        meta = row['operator']['metadata']['program_execution']
        assert meta['m_k_tiling']['tiling_key'] == g.tiling_key
        # Diagnostic physical flops come from the actual owned/tail phase Work,
        # not mDim*mTiles*mL1 for every tile.
        first_work = total_work(tuple(pre_owner(g, owner) for owner in
                                      range(g.m_dim * g.k_parts)))
        assert meta['physical_projection_flops'] == first_work.matrix.get('hf32', 0)
        if tokens == 257:
            # mDim=2, m_l1=144; second M tile is 113 rows -> 128 aligned, not 144.
            assert g.m_dim == 2 and g.m_l1 == 144
            naive = 2 * g.m_dim * g.m_tiles * g.m_l1 * g.width * 32
            assert meta['physical_projection_flops'] < naive
        if tokens == 4096:
            assert g.tiling_key == 1000 and g.m_dim == 16 and g.k_dim == 2
        if tokens == 4097:
            assert g.tiling_key == 1001 and g.m_dim == 17 and g.k_dim == 1
            assert g.k_parts == 1
