"""Packed INT8 latent, BF16 RoPE and FP32 scales in a native MLA prolog.

[Role] Describe the per-tile cache ABI and its explicit analytic producer.
[Invariants] D=656 is 512 INT8 + 64 BF16 + 4 FP32 scales, not 656 INT8 values.
  Allocated cache capacity is not per-call traffic. UK and query outputs stay
  BF16. This profile does not reuse the arch22 BF16-cache timing program.
[Timeouts & Locks] Constant geometry arithmetic; no source/capture reads.
[Side-Effect Policy] Builds immutable graph records only.
"""
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.operators import (
    FieldDisposition, ProgramFragment, SemanticAttr, SemanticRef, StateEffect,
    WorkItem, WorkSet,
)
from mindie_sim.schema.subgraphs import (
    EventPortSpec, PortSet, StatePortSpec, TensorPortSpec,
)


TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCE_REF = SourceRef(
    repository="vllm-project/vllm-ascend",
    revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
    relative_path="vllm_ascend/attention/sfa_v1.py",
    source_sha256="058aeb723a4ef3cdb0e07a5fc0180ec53f8227f62d98710144ecffb9cc66678f",
)
PACKED_KV_GROUP_SIZE = 128
PACKED_E8M0_GROUP_SIZE = 32
PACKED_KV_LATENT_DIM = 512
PACKED_KV_QUANT_MODE = 3
PACKED_KV_ROPE_DIM = 64
PACKED_KV_SCALE_BYTES = 4

LAYOUT = 'packed_int8_bf16_rope_fp32_scale'
PROFILE = 'mla_prolog_packed_int8_tile128_analytic'
#: Declared query/token projection quant modes.  C8 keeps one INT8 payload
#: row; only the normalized query and projection operands change precision.
INT8_QUERY_FORMAT = 'int8'
MXFP8_QUERY_FORMAT = 'fp8_e4m3'
QUERY_GROUP_SIZES = {INT8_QUERY_FORMAT: 128, MXFP8_QUERY_FORMAT: 32}
SUPPORTED_QUERY_FORMATS = frozenset(QUERY_GROUP_SIZES)
NATIVE_KIND_V3 = 'aclnnMlaPrologV3WeightNz'
NATIVE_KIND_V4 = 'aclnnMlaPrologV4WeightNz'
#: The analytic writer charges normalization, group amax/clip and conversion
#: separately from one 64-bit slot index and one scalar row prefix.
QUANT_NORM_OPS_PER_ELEMENT = 9
QUANT_GROUP_RENORM_OPS = 4
TOKEN_PROJECTION_COUNT = 3
BF16_BYTES = int(dtype_bytes('bf16'))
INT64_BYTES = int(dtype_bytes('int64'))
INT32_BYTES = int(dtype_bytes('int32'))
SOURCE = {
    'caller': {'repository': 'vllm-project/vllm-ascend',
               'revision': '8c30e4f491f73cba737a1dacc522894e437cdf3d',
               'path': 'vllm_ascend/attention/sfa_v1.py', 'lines': '1181-1289'},
    'layout': {'path': 'vllm_ascend/attention/utils.py', 'lines': '69-80'},
    'cache_writer': {'repository': 'ops-transformer',
                     'revision': 'a3925fd49cd256ad478d2faf983bf2961b8ae2c8',
                     'path': 'attention/mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h',
                     'lines': '1454-1556,1628-1645'},
    'capture_binary_identity': 'not_established; explicit_representation_and_analytic_program',
}
#: CANN 9.2.1 V4 caller: MXFP8 query/projection with the same C8 cache row.
SOURCE_V4 = {
    'repository': 'vllm-ascend',
    'base_revision': 'nightly-ci-main-a5-openeuler-cann9.2.1-20261111',
    'source_state': 'committed',
    'path': 'vllm_ascend/attention/sfa_v1.py',
    'symbol': 'PreprocessType.PROLOG_V3/aclnnMlaPrologV4WeightNz',
    'lines': '1533-1596',
    'cache_layout': {'path': 'vllm_ascend/attention/utils.py', 'lines': '69-80'},
    'capture_binary_identity': 'not_established; declared_cann_921_contract',
}


def view(
    latent,
    rope,
    group_size=PACKED_KV_GROUP_SIZE,
    *,
    query_format=INT8_QUERY_FORMAT,
    cache_dtype=None,
):
    if latent != PACKED_KV_LATENT_DIM or rope != PACKED_KV_ROPE_DIM or group_size != PACKED_KV_GROUP_SIZE:
        raise ModelingError('mla_prolog_packed_int8_requires_latent512_rope64_tile128')
    if query_format not in SUPPORTED_QUERY_FORMATS:
        raise ModelingError('mla_prolog_packed_query_format_unsupported')
    allowed_cache = (
        {INT8_QUERY_FORMAT, MXFP8_QUERY_FORMAT}
        if query_format == MXFP8_QUERY_FORMAT
        else {INT8_QUERY_FORMAT}
    )
    if cache_dtype is None:
        latent_dtype = INT8_QUERY_FORMAT
    else:
        latent_dtype = str(cache_dtype).strip().lower()
        if latent_dtype not in allowed_cache:
            raise ModelingError('mla_prolog_v3_cache_layout_dtype_mismatch')
    return {'layout': LAYOUT, 'latent_dtype': latent_dtype, 'rope_dtype': 'bf16',
            'scale_dtype': 'fp32', 'group_size': group_size,
            'payload_bytes': latent, 'rope_bytes': rope * 2,
            'scale_bytes': latent // group_size * PACKED_KV_SCALE_BYTES,
            'row_bytes': latent + rope * 2 + latent // group_size * PACKED_KV_SCALE_BYTES,
            'component_offsets': {'latent': 0, 'rope': latent, 'scales': latent + rope * 2},
            'cache_mode': 'PA_BSND', 'kv_cache_quant_mode': PACKED_KV_QUANT_MODE,
            'ckvkr_repo_mode': 1, 'quant_scale_repo_mode': 1}


def packed_operand_scales(
    *, tokens: int, hidden: int, local_heads: int, q_lora_rank: int,
    kv_lora_rank: int, qk_nope_head_dim: int, qk_rope_head_dim: int,
    query_format: str,
) -> tuple[str, tuple[int, int], str, tuple[tuple[int, int], ...]]:
    """Return CANN-exact (token_scale_dtype, shape, proj_scale_dtype, shapes) by query format."""
    if query_format not in SUPPORTED_QUERY_FORMATS:
        raise ModelingError('mla_prolog_packed_query_format_unsupported')
    uq_qr_cols = local_heads * (qk_nope_head_dim + qk_rope_head_dim)
    dkv_kr_cols = kv_lora_rank + qk_rope_head_dim
    if query_format == MXFP8_QUERY_FORMAT:
        hidden_groups = hidden // PACKED_E8M0_GROUP_SIZE
        q_lora_groups = q_lora_rank // PACKED_E8M0_GROUP_SIZE
        return (
            'fp8_e8m0',
            (tokens, hidden_groups),
            'fp8_e8m0',
            ((q_lora_rank, hidden_groups), (uq_qr_cols, q_lora_groups), (dkv_kr_cols, hidden_groups)),
        )
    return (
        'fp32',
        (tokens, 1),
        'fp32',
        ((1, q_lora_rank), (1, uq_qr_cols), (1, dkv_kr_cols)),
    )


def _packed_plan(*, plan, rows, query_format=INT8_QUERY_FORMAT, cache_dtype=None):
    cache = view(
        plan.kv_lora_rank, plan.qk_rope_head_dim,
        query_format=query_format, cache_dtype=cache_dtype,
    )
    if type(rows) is not int or isinstance(rows, bool) or rows < 0:
        raise ModelingError('mla_prolog_packed_requires_nonnegative_rows')
    if query_format not in SUPPORTED_QUERY_FORMATS:
        raise ModelingError('mla_prolog_packed_query_format_unsupported')
    width = plan.kv_lora_rank
    query_groups = -(-width // QUERY_GROUP_SIZES[query_format])
    groups = width // cache['group_size']
    # The semantic norm producer is BF16. The native source uses FP32 norm
    # scratch; this analytic producer includes widening, group abs/max, clip,
    # normalize and three-step low-precision conversion, without instruction
    # tiling.  MXFP8 groups 32 columns per E8M0 scale; C8 cache scales stay
    # one per 128 payload columns.
    vector = rows * (QUANT_NORM_OPS_PER_ELEMENT * width
                     + QUANT_GROUP_RENORM_OPS * query_groups)
    read = (rows * ((width + plan.qk_rope_head_dim) * BF16_BYTES + INT64_BYTES)
            + INT32_BYTES)
    return {
        'cache': cache, 'rows': rows, 'vector': vector,
        'read': read, 'write': rows * cache['row_bytes'],
        'hbm': read + rows * cache['row_bytes'],
        'groups': groups, 'query_groups': query_groups,
        'query_format': query_format,
        'query_group_size': QUERY_GROUP_SIZES[query_format],
    }


def packed_contract(*, plan, rows, occurrence=None, query_format=INT8_QUERY_FORMAT, cache_dtype=None):
    """Typed packed-cache writer contract with component offsets and scales."""
    if occurrence is not None and not isinstance(occurrence, Occurrence):
        raise ModelingError('mla_prolog_packed_occurrence_invalid')
    spec = _packed_plan(plan=plan, rows=rows, query_format=query_format, cache_dtype=cache_dtype)
    cache = spec['cache']
    return ProgramFragment(
        'mla_prolog_packed_cache',
        occurrence=occurrence,
        semantic_refs=(SemanticRef('subgraph', 'mla_attention', role='mla_prolog_cache_store'),),
        attrs=(
            SemanticAttr('implementation', 'mla_prolog_packed_int8_tile128_analytic',
                         source='selected_program'),
            SemanticAttr('layout', cache['layout']),
            SemanticAttr('latent_dtype', cache['latent_dtype']),
            SemanticAttr('rope_dtype', cache['rope_dtype']),
            SemanticAttr('scale_dtype', cache['scale_dtype']),
            SemanticAttr('group_size', cache['group_size'], unit='row'),
            SemanticAttr('payload_bytes', cache['payload_bytes'], unit='byte'),
            SemanticAttr('rope_bytes', cache['rope_bytes'], unit='byte'),
            SemanticAttr('scale_bytes', cache['scale_bytes'], unit='byte'),
            SemanticAttr('row_bytes', cache['row_bytes'], unit='byte'),
            SemanticAttr('component_offsets', cache['component_offsets'], unit='byte'),
            SemanticAttr('cache_mode', cache['cache_mode']),
            SemanticAttr('kv_cache_quant_mode', cache['kv_cache_quant_mode']),
            SemanticAttr('ckvkr_repo_mode', cache['ckvkr_repo_mode']),
            SemanticAttr('quant_scale_repo_mode', cache['quant_scale_repo_mode']),
            SemanticAttr('query_format', spec['query_format']),
            SemanticAttr('query_group_size', spec['query_group_size'], unit='column'),
            SemanticAttr('rows', spec['rows'], unit='row'),
            SemanticAttr('source', SOURCE, source='source_binding'),
            SemanticAttr('unresolved_fields',
                         ('capture_binary_identity', 'native_instruction_tiling'),
                         source='specific_request'),
        ),
        ports=PortSet(
            tensor_in=(
                TensorPortSpec('kv_latent_norm', 'in', 'activation',
                               dtype='bf16', layout_kind='ND', scope='layer'),
                TensorPortSpec('rope_input', 'in', 'activation',
                               dtype='bf16', layout_kind='ND', scope='layer'),
            ),
            tensor_out=(
                TensorPortSpec('packed_cache', 'out', 'packed_latent_cache',
                               dtype=('mixed_fp8_bf16' if cache['latent_dtype'] == MXFP8_QUERY_FORMAT else 'mixed_int8_bf16'),
                               layout_kind=cache['cache_mode'], scope='cohort'),
                TensorPortSpec('packed_scale', 'out', 'quant_scale',
                               dtype=cache['scale_dtype'], layout_kind='ND',
                               scope='cohort'),
            ),
            state_out=(
                StatePortSpec('kv_cache_state', 'out', 'paged_latent_kv',
                              scope='cohort', readiness='producer_complete'),
            ),
            event_in=(EventPortSpec('kv_latent_ready', 'in', 'ready'),),
            event_out=(EventPortSpec('done', 'out', 'complete'),),
        ),
        work=WorkSet((
            WorkItem('hbm', spec['hbm'], 'bytes', 'issued', 'selected_packed_writer'),
            WorkItem('mte2', spec['read'], 'bytes', 'issued', 'selected_packed_writer'),
            WorkItem('mte3', spec['write'], 'bytes', 'issued', 'selected_packed_writer'),
            WorkItem('vector', spec['vector'], 'operations', 'issued',
                     'analytic_quantization_and_scatter'),
        )),
        state_effects=(
            StateEffect('kv_cache', 'write', condition='active_rows',
                        arbiter='packed_cache_writer'),
        ),
        field_effects=(
            FieldDisposition('kv_latent.dtype', 'rewrite',
                             source=f"bf16 normalized latent -> {cache['latent_dtype']} payload"),
            FieldDisposition('rope.dtype', 'preserve'),
            FieldDisposition('scale.dtype', 'rewrite',
                             source='fp32 group scale written in packed row'),
            FieldDisposition('layout', 'rewrite', source='PA_BSND packed row'),
            FieldDisposition('cache.alias', 'preserve',
                             source='cache_output_alias'),
        ),
        notes=('D=656 is 1-byte latent payload + bf16 rope + fp32 scales, not one dtype',
               'work is the actual packed writer traffic and quantization ops'),
    )


def store(builder, name, deps, rows, plan, query_format=INT8_QUERY_FORMAT, *, cache_dtype=None):
    resolved_cache_dtype = (
        cache_dtype if cache_dtype is not None
        else getattr(getattr(builder, 'quant', None), 'kv_dtype', None)
    )
    spec = _packed_plan(
        plan=plan, rows=rows, query_format=query_format, cache_dtype=resolved_cache_dtype,
    )
    cache = spec['cache']
    return builder.work(
        name, deps, (rows, cache['row_bytes']), family='cache_store', dtype=cache['latent_dtype'],
        read=spec['read'], write=spec['write'], vector=spec['vector'],
        metadata=(('cache_kind', name), ('cache_visible', True), ('shared_owner', False),
                  ('packed_cache', cache), ('source', SOURCE),
                  ('cache_producer', 'analytic_tile128_quantization_and_packed_scatter'),
                  ('slot_index_dtype', 'int64'),
                  ('query_format', spec['query_format']),
                  ('cache_output_alias', True)))


def validate_view(declaration):
    """One C8 row ABI; INT8 and MXFP8 query/projection modes share it."""
    token = declaration.get('token_dtype')
    if token not in SUPPORTED_QUERY_FORMATS:
        raise ModelingError(
            'mla_prolog_packed_requires_declared_token_projections_bf16_uk_and_tile128_cache')
    allowed_cache = {'int8', 'fp8_e4m3'} if token == MXFP8_QUERY_FORMAT else {'int8'}
    cache_dtype = declaration.get('cache_dtype')
    if cache_dtype not in allowed_cache:
        raise ModelingError(
            'mla_prolog_packed_requires_declared_token_projections_bf16_uk_and_tile128_cache')
    expected = view(
        declaration['latent_rank'], declaration['rope_width'],
        query_format=token, cache_dtype=cache_dtype,
    )
    if (declaration.get('packed_cache') != expected
            or tuple(declaration.get('projection_weight_dtypes', ())
                     ) != (token,) * TOKEN_PROJECTION_COUNT
            or declaration.get('absorbed_weight_dtype') != 'bf16'
            or declaration.get('norm_dtype') != 'bf16'
            or declaration.get('query_norm_dtype') != token):
        raise ModelingError(
            'mla_prolog_packed_requires_declared_token_projections_bf16_uk_and_tile128_cache')
