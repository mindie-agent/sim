"""Source slice traffic and independent native dependency boundaries."""
from pathlib import Path
import json

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.family_v41_rotary import declare
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import RequestView

ROOT=Path(__file__).resolve().parents[1]


def builder(batch=1):
    case=parse_case(json.loads((ROOT/'examples/cases/high-concurrency-20261030/v41-source-conditional/a5_v41_fp8_tp1_dp8_ep8_tpot5ms_c1-source.json').read_text()))
    views=tuple(RequestView(str(i),1,2048,False,'decode') for i in range(batch))
    return Builder('s0.l0',case.model.layer_plans[0],0,views,case.model,'decode')


def test_q_kv_and_inverse_have_three_independent_native_boundaries():
    b=builder()
    q=declare(b,name='q_rotary',deps=('q_projection',),tokens=1,heads=64,head_dim=512,rotary_width=64)
    kv=declare(b,name='kv_rotary',deps=('kv_norm',),tokens=1,heads=1,head_dim=512,rotary_width=64)
    inverse=declare(b,name='output_inverse_rotary',deps=('attention_output',),tokens=1,heads=64,head_dim=512,rotary_width=64,inverse=True)
    assert len({q,kv,inverse})==len(b.ops)==3
    assert [op.deps for op in b.ops]==[('q_projection',),('kv_norm',),('attention_output',)]
    # Real captured views are BF16[B,1,64,512] / BF16[B,1,1,512],
    # each with FP32[B,1,1,64] cosine and sine. No full-width copy is created.
    assert [op.mte2_bytes for op in b.ops]==[8192+512,128+512,8192+512]
    assert [op.mte3_bytes for op in b.ops]==[8192,128,8192]
    assert b.ops[0].vector_ops==b.ops[2].vector_ops
    assert dict(b.ops[2].metadata)['native_tensor_view']['sine_sign']=='caller_negated'


def test_nope_width_does_not_create_inplace_slice_traffic_or_vector_work():
    b=builder(16)
    for width in (128,512):
        declare(b,name=f'rotary_{width}',deps=('projection',),tokens=16,heads=32,head_dim=width,rotary_width=64)
    short,long=b.ops
    assert short.hbm_bytes==long.hbm_bytes==16*(32*64*2*2+2*64*4)
    assert short.vector_ops==long.vector_ops
    assert short.local_shape==(16,32,128) and long.local_shape==(16,32,512)
    assert dict(long.metadata)['native_tensor_view']['partial_slice']==(448,512)


def test_batch_scaling_and_coefficient_reuse_are_explicit_not_free():
    b=builder()
    for batch in (1,16):
        declare(b,name=f'query_{batch}',deps=('q',),tokens=batch,heads=64,head_dim=512,rotary_width=64)
    first,last=b.ops
    assert last.hbm_bytes==16*first.hbm_bytes
    assert last.vector_ops==16*first.vector_ops
    work=dict(last.metadata)['partial_rotary_work']
    assert work['coefficient_bytes']==16*2*64*4
    assert work['vector_lane_operations']['coefficient_broadcast']==16*2*63*64
    assert work['native_owner_split_and_reload_not_selected']


def test_public_caller_replaces_combined_rotation_and_orders_inverse_before_output_projection():
    raw = json.loads((ROOT / 'examples/source-families/v41-mqsmla-source-control.json').read_text())
    raw['model']['layers'] = 3
    probe = dict(scope='body', phase='decode', tokens=1, sequences=2, context=129)
    original = target_graphs(raw, **probe)
    raw['runtime']['kernels']['lowerings']['v41_rotary'] = 'v41_partial_rotary'
    selected = target_graphs(raw, **probe)
    for rank, ops in selected.items():
        before = original[rank]
        assert sum(op.flops for op in ops) == sum(op.flops for op in before)
        assert not any(op.recipe == 'rotary' for op in ops)
        for layer in range(3):
            by_recipe = {op.recipe: op for op in ops if op.layer == layer}
            q, kv, inverse = (by_recipe[name] for name in
                              ('q_rotary', 'kv_rotary', 'output_inverse_rotary'))
            assert q.deps == (by_recipe['q_b'].op_id,)
            assert kv.deps == (by_recipe['kv_norm'].op_id,)
            assert by_recipe['kv_store'].deps == (kv.op_id,)
            wo_a = by_recipe['grouped_output_a']
            quantize = next(op for op in ops if op.op_id == wo_a.deps[0])
            assert quantize.family == 'quantize' and quantize.deps == (inverse.op_id,)
            assert all(op.source_boundary == 'kernel' for op in (q, kv, inverse))
            attention = [op for op in ops if op.layer == layer and op.family == 'attention']
            assert attention and set(inverse.deps) == {op.op_id for op in attention}
            assert all(q.op_id in op.deps for op in attention)
    result = graph_costs(raw, selected, input_ready_ms=7,
                         external_inputs={'analysis_input': 7}, isolated=False)
    assert result['duration_ms'] > 0
    assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
    assert all(abs(value) < 1e-7 for value in result['execution']['live_allocated'].values())
    rotations = [row for row in result['timeline'] if row['recipe'] in
                 ('q_rotary', 'kv_rotary', 'output_inverse_rotary')]
    assert len(rotations) == 3 * 3 * len(selected)
    assert all(row['ready_ms'] >= 7 for row in rotations)
