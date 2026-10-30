"""Packed INT8 latent, BF16 RoPE and FP32 scales in a native MLA prolog.

[Role] Describe the per-tile cache ABI and its explicit analytic producer.
[Invariants] D=656 is 512 INT8 + 64 BF16 + 4 FP32 scales, not 656 INT8 values.
  Allocated cache capacity is not per-call traffic. UK and query outputs stay
  BF16. This profile does not reuse the arch22 BF16-cache timing program.
[Timeouts & Locks] Constant geometry arithmetic; no source/capture reads.
[Side-Effect Policy] Builds immutable graph records only.
"""
from mindie_sim.schema.common import ModelingError

LAYOUT = 'packed_int8_bf16_rope_fp32_scale'
PROFILE = 'mla_prolog_packed_int8_tile128_analytic'
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


def view(latent, rope, group_size=128):
    if latent != 512 or rope != 64 or group_size != 128:
        raise ModelingError('mla_prolog_packed_int8_requires_latent512_rope64_tile128')
    return {'layout': LAYOUT, 'latent_dtype': 'int8', 'rope_dtype': 'bf16',
            'scale_dtype': 'fp32', 'group_size': group_size,
            'payload_bytes': latent, 'rope_bytes': rope * 2,
            'scale_bytes': latent // group_size * 4,
            'row_bytes': latent + rope * 2 + latent // group_size * 4,
            'component_offsets': {'latent': 0, 'rope': latent, 'scales': latent + rope * 2},
            'cache_mode': 'PA_BSND', 'kv_cache_quant_mode': 3,
            'ckvkr_repo_mode': 1, 'quant_scale_repo_mode': 1}


def store(builder, name, deps, rows, plan):
    cache = view(plan.kv_lora_rank, plan.qk_rope_head_dim)
    # The semantic norm producer is BF16. The native source uses FP32 norm
    # scratch; this analytic producer explicitly includes the widening cast,
    # group abs/max, clip, normalize and three-step FP32->INT8 conversion.
    # Source-specific instruction issue/tiling is intentionally not inferred.
    width, groups = plan.kv_lora_rank, plan.kv_lora_rank // 128
    vector = rows * (9 * width + 4 * groups)
    return builder.work(
        name, deps, (rows, cache['row_bytes']), family='cache_store', dtype='int8',
        read=rows * ((width + plan.qk_rope_head_dim) * 2 + 8) + 4,
        write=rows * cache['row_bytes'], vector=vector,
        metadata=(('cache_kind', name), ('cache_visible', True), ('shared_owner', False),
                  ('packed_cache', cache), ('source', SOURCE),
                  ('cache_producer', 'analytic_tile128_quantization_and_packed_scatter'),
                  ('slot_index_dtype', 'int64'),
                  ('cache_output_alias', True)))


def validate_view(declaration):
    expected = view(declaration['latent_rank'], declaration['rope_width'])
    if (declaration.get('packed_cache') != expected
            or declaration.get('token_dtype') != 'int8'
            or tuple(declaration.get('projection_weight_dtypes', ())) != ('int8',) * 3
            or declaration.get('absorbed_weight_dtype') != 'bf16'
            or declaration.get('norm_dtype') != 'bf16'
            or declaration.get('cache_dtype') != 'int8'
            or declaration.get('query_norm_dtype') != 'int8'):
        raise ModelingError('mla_prolog_packed_requires_int8_token_projections_bf16_uk_and_tile128_cache')
