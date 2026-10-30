"""Dense paged MLA core binding separate from its prolog and padding caller.

[Role] Bind the selected BF16 kernel once across all independent requests.
[Invariants] Logical heads, prepared prolog weights and shared KV stay distinct.
  An existing head caller owns its copies; this core never adds them twice.
[Timeouts & Locks] O(batch) metadata only, no locks or I/O.
[Side-Effect Policy] Appends immutable ordinary graph operators.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from .mechanism import PlannedRecipe
from . import paged_mla

IMPLEMENTATION = paged_mla.IMPLEMENTATION


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(paged_mla.build)}


def validate(*, views, plan, parallel, model, quant, padded_tokens, kv_splits=1, dispatch=None):
    if (not views or any(v.phase != 'decode' or v.q_len != 1 for v in views)
            or parallel.cp != 1 or parallel.pcp != 1 or padded_tokens < len(views)
            or plan.mla_variant != 'latent_mla' or plan.cache_kind != 'dense'
            or plan.index_topk or plan.window or plan.noncausal_attention
            or (plan.kv_lora_rank, plan.qk_rope_head_dim) != (512, 64)
            or model.activation_dtype != 'bf16' or quant.kv_dtype != 'bf16'):
        raise ModelingError('paged_mla_requires_q1_bf16_dense_latent512_rope64')
    if type(kv_splits) is not int or kv_splits < 1:
        raise ModelingError('paged_mla_requires_positive_declared_split_count')


def lower_attention(b, *, views, plan, heads, runtime, query, rotary, store,
                    head_contract=None, padded_tokens=None, kv_splits=1, dispatch=None):
    batch = len(views)
    logical = plan.num_heads // runtime.parallel.tp
    physical = 1 << (logical-1).bit_length()
    if head_contract is None and physical != heads:
        # Generic prolog has contiguous logical outputs. Only F.pad is needed;
        # the native padded-prolog's strided q_nope copy belongs to its caller.
        packed = []
        for part, producer, width in (('nope', query, 512), ('rope', rotary, 64)):
            packed.append(b.work('paged_mla_query_pad_' + part, (producer,),
                (batch, physical, width), read=batch*heads*width*2,
                write=batch*physical*width*2, family='copy', dtype='bf16',
                metadata=(('materialization', 'F.pad logical heads to physical'),
                          ('zero_fill_bytes', batch*(physical-heads)*width*2))))
        query, rotary = packed
    name = b.work('paged_mla_decode', (query, rotary, store),
                  (batch, physical, 1, 512), family='attention', dtype='bf16')
    meta = {
        'paged_mla_geometry': {'contexts': tuple(v.context for v in views),
                              'heads': physical, 'kv_splits': kv_splits,
                              'page_size': runtime.kv_block_size, 'dispatch': dispatch},
        'implementation_binding': {'selector': 'paged_mla_attention',
            'implementation': IMPLEMENTATION, 'capture_dispatch': 'not_established',
            'call_boundary': 'one_native_kernel_for_independent_single_query_sequences'},
        'attention_tensor_view': {'kind': 'paged_latent', 'query_shape': (batch,physical,1,512),
            'query_rope_shape': (batch,physical,1,64), 'output_shape': (physical,batch,1,512),
            'query_dtype':'bf16','cache_dtype':'bf16','output_dtype':'bf16',
            'cache_head_shape':(1,512),'rope_cache_head_shape':(1,64)},
        'mla_head_contract': {'logical_heads': logical, 'physical_heads': physical},
        'output_view': 'NBSD[:logical_heads]; leading-head contiguous slice; no copy',
    }
    b.ops[-1] = replace(b.ops[-1], recipe=IMPLEMENTATION, source_boundary='kernel',
                        metadata=tuple(meta.items()))
    return name
