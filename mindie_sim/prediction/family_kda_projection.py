"""Explicit mixed-precision KDA input projection groups.

[Role] Pack QKV separately from beta/composed decay/output gate and preserve
  the retained caller's two-stream joins through ordinary matrix/vector ops.
[Invariants] One quantized payload feeds one QKV GEMM. Offline F_b @ F_a adds
  a resident full-rank F weight while both original factors remain resident.
  Split outputs are views; only real consumer layout/cast boundaries copy.
[Timeouts & Locks] O(KDA layer count + projection roles), no I/O or locks.
[Side-Effect Policy] Returns prepared model metadata and appends local DAG ops.
"""
from dataclasses import replace

from mindie_sim.catalog.quant import projection_quant
from mindie_sim.schema.common import ModelingError
from .projections import input_projections, _representation
from .transforms import activation_quantization

IMPLEMENTATION = 'vllm_ascend_kda_mixed_projection'
SOURCE = {
    'repository': 'vllm-ascend',
    'revision': '8c30e4f491f73cba737a1dacc522894e437cdf3d',
    'path': 'vllm_ascend/ops/kimi_kda.py',
    'sha256': 'b36208b3a9a6fffb05c9437fdecaf62bb2af0e0a5be9edf0bab4f1bd5ab10617',
    'symbols': ['_KDAFusedBFGLinear._maybe_fuse_f_proj',
                'AscendKimiK3DeltaAttention._run_overlapped_qkv_bfg',
                'AscendKimiK3DeltaAttention._postprocess_bfg'],
    'scope': 'source-compatible grouping and caller order; captured binary identity unverified',
}


def recipes():
    return {}


def _validate(plan, model, tp):
    heads = plan.linear_value_heads or plan.linear_heads
    keys = plan.linear_key_heads or plan.linear_heads
    dk = plan.linear_key_head_dim or plan.linear_head_dim
    if (heads != keys or heads % tp or plan.gate_rank
            or dk != plan.linear_head_dim or model.activation_dtype != 'bf16'):
        raise ModelingError('kda_mixed_projection_requires_integral_equal_heads_and_full_rank_bf16_gate')
    quant = projection_quant(model.quant, plan)
    if len({_representation(quant, 'kda.' + role) for role in ('q', 'k', 'v')}) != 1:
        raise ModelingError('kda_qkv_group_requires_one_operand_representation')
    if any(quant.for_category('kda.' + role).format_for('kda.' + role) != 'bf16'
           for role in ('b', 'f_a', 'f_b', 'f', 'g')):
        raise ModelingError('kda_bfg_group_and_retained_decay_factors_require_bf16_weights')
    if len({_representation(quant, 'kda.' + role) for role in ('b', 'f', 'g')}) != 1:
        raise ModelingError('kda_bfg_group_requires_one_operand_representation')
    return {'dtype': 'bf16', 'decay_rows': heads * dk,
            'resident_policy': 'retain_original_f_a_f_b_and_add_composed_f',
            'offline_composition': 'F_b.float() @ F_a.float(), stored as BF16',
            'groups': (('q', 'k', 'v'), ('b', 'f', 'g'))}


def prepare_model(model, runtime):
    layers = {str(plan.index): _validate(plan, model, runtime.parallel.tp)
              for plan in model.layer_plans if plan.attention == 'kda'}
    metadata = {**dict(model.metadata), 'kda_input_projection': {
        'implementation': IMPLEMENTATION, 'layers': layers, 'source': SOURCE}}
    return replace(model, metadata=tuple(sorted(metadata.items())))


def _placement(b, start, *, stream, after=()):
    """Place actual caller operations, retaining the first-stage quant join."""
    for index in range(start, len(b.ops)):
        op = b.ops[index]
        b.ops[index] = replace(op, stream=stream, deps=tuple(dict.fromkeys((*op.deps, *after))))


def _view(tensors, b, producer, role):
    op = next(op for op in b.ops if op.op_id == producer)
    part, = [part for part in dict(op.metadata)['packed_projection_slices'] if part['role'] == role]
    return tensors.result(producer, op.local_shape[:2]).narrow(
        1, part['offset_elements'], part['shape'][1])


def lower(b, tensors, *, plan, tokens, hidden, key_width, value_width, heads, dk, deps):
    """Return QKV producers, output gate, FP32 beta, raw decay, and join event."""
    # The three hidden-input groups use the same K dimension. Composition is
    # performed on weights at load time, so online F is one hidden->decay GEMM.
    bfg = input_projections(
        b, name='linear_bfg', deps=deps, tokens=tokens, hidden=hidden,
        roles=(('b', 'linear_beta', heads, 'kda.b'),
               ('f', 'linear_decay', heads * dk, 'kda.f'),
               ('g', 'linear_gate', value_width, 'kda.g')), packed=True)
    bfg_id = bfg['b']
    b.ops[-1] = replace(b.ops[-1], stream='kda_bfg', metadata=b.ops[-1].metadata + (
        ('projection_group', ('b', 'f', 'g')), ('source', SOURCE),
        ('offline_composed_weight', {'category': 'kda.f', 'factors': ('kda.f_b', 'kda.f_a')})))
    qkv = input_projections(
        b, name='linear_qkv', deps=deps, tokens=tokens, hidden=hidden,
        roles=(('q', 'linear_q', key_width, 'kda.q'),
               ('k', 'linear_k', key_width, 'kda.k'),
               ('v', 'linear_v', value_width, 'kda.v')), packed=True)
    matrix = b.ops.pop()
    # Quantize before attaching the BFG join: the input conversion overlaps
    # BFG, while the two matrix kernels are explicitly serialized in source.
    lowered = activation_quantization((matrix,), b.model.activation_dtype)
    conversion = next((op.op_id for op in lowered if op.family == 'quantize'), None)
    if conversion is None:
        raise ModelingError('kda_mixed_projection_requires_quantized_qkv_group')
    b.ops.extend(lowered[:-1])
    matrix = lowered[-1]
    b.ops.append(replace(matrix, deps=(*matrix.deps, bfg_id), metadata=matrix.metadata + (
        ('projection_group', ('q', 'k', 'v')), ('source', SOURCE),
        ('projection_schedule', 'quant_parallel_bfg_then_qkv_parallel_beta'))))

    beta_view = _view(tensors, b, bfg_id, 'b')
    start = len(b.ops)
    # A multi-row beta slice has the BFG row stride. Native Cast consumes a
    # compact slice; a one-row view is already compact and has no Slice launch.
    beta_view = tensors.contiguous(beta_view, 'kda_beta_contiguous',
                                   reason='packed_BFG_beta_Cast.AutoContiguous')
    cast = tensors.materialize(beta_view, 'linear_beta_cast',
                               reason='packed_BFG_beta_float', dtype='fp32')
    beta = b.vector('linear_beta_sigmoid', (cast.producer,), tokens * heads, 4,
                    dtype='fp32', family='sigmoid', shape=(tokens, heads))
    _placement(b, start, stream='kda_bfg', after=(conversion,))

    raw_gate = _view(tensors, b, bfg_id, 'f')
    output_gate = _view(tensors, b, bfg_id, 'g')
    # Default consumers accept the source strided views. Explicit layout
    # contracts materialize exactly their selected payload, never the BFG bank.
    raw_gate = tensors.reshape(raw_gate, (1, tokens, heads, dk), name='kda_decay_reshape')
    raw_gate = tensors.require(raw_gate, 'kda_recurrent', 'gate', name='kda_decay_contiguous')
    output_gate = tensors.reshape(output_gate, (tokens, heads, value_width // heads),
                                  name='kda_output_gate_reshape')
    output_gate = tensors.require(output_gate, 'linear_output_norm_gate', 'gate',
                                  name='kda_output_gate_contiguous')
    return {'sources': [qkv[role] for role in ('q', 'k', 'v')],
            'output_gate': output_gate.producer, 'decay': raw_gate.producer,
            'beta': beta, 'raw_beta': beta_view.producer, 'join': beta,
            'views': {'decay': raw_gate.layout.to_dict(),
                      'output_gate': output_gate.layout.to_dict()}}
