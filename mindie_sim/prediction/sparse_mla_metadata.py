"""Source caller's AI_CPU sparse MLA metadata invocation.

[Role] Connect one metadata build and buffer publication to batch attention.
[Invariants] Metadata is a model-body dependency, not Vector work or host
  submission. One declared metadata group invokes it once per batch. Controller service requires declared
  capacity or an inclusive observed response; no compiled instruction count is
  inferred from the caller. The persistent buffer copy is a distinct operation.
[Timeouts & Locks] O(requests) metadata; no I/O, waits or locks.
[Side-Effect Policy] Constructs operators and ordinary Engine programs only.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, Stage

IMPLEMENTATION = 'vllm_ascend_sparse_mla_metadata'
NATIVE = 'SparseFlashMlaMetadata'
SERVICE = 'aicpu_metadata_call'
SOURCE = {
    'revision': '8c30e4f491f73cba737a1dacc522894e437cdf3d',
    'path': 'vllm_ascend/attention/sfa_v1.py',
    'sha256': 'f5fbfc63cf488f9839f85a3814a65a080bf484ac67c25aced0ee4a42162577b6',
    'lines': '72-104,175-228',
    'boundary': 'NoPE BF16 paged sparse caller; capture binary not identified',
}


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(build)}


def lower(b, *, views, plan, parallel, quant, deps, padded_tokens, stage_of,
          layer_groups, schedule='pre_forward'):
    if schedule not in {'pre_forward', 'layer_ordered'}:
        raise ModelingError('sparse_mla_metadata_schedule_requires_explicit_caller_boundary')
    # The source builder groups layers by attention/cache configuration and
    # keeps one state per draft index. Captured per-layer calls therefore need
    # explicit singleton groups; neither a layer count nor a name proves them.
    members = [int(layer) for group in layer_groups for layer in group]
    if len(members) != len(set(members)) or any(not group for group in layer_groups):
        raise ModelingError('sparse_mla_metadata_groups_must_be_disjoint_nonempty')
    group = next((tuple(sorted(g)) for g in layer_groups if plan.index in g), None)
    if group is None:
        return None
    fields = ('attention', 'mla_variant', 'cache_kind', 'qk_rope_head_dim',
              'kv_lora_rank', 'num_heads', 'index_topk')
    by_index = {p.index: p for p in b.model.layer_plans}
    if any(stage_of(i) != b.common['stage'] or i not in by_index
           or any(getattr(by_index[i], field) != getattr(plan, field) for field in fields)
           for i in group):
        raise ModelingError('sparse_mla_metadata_group_requires_same_stage_and_geometry')
    key = b.prefix.rsplit('.l', 1)[0] + f'.sparse_metadata_group{group[0]}'
    if plan.index != group[0]:
        return key + '.publish'
    if (plan.mla_variant != 'latent_mla' or plan.cache_kind != 'dense'
            or plan.qk_rope_head_dim or not plan.index_topk
            or quant.kv_dtype != 'bf16' or parallel.cp != 1 or parallel.pcp != 1
            or not views or any(v.phase != 'decode' for v in views)):
        raise ModelingError('sparse_mla_metadata_requires_nope_bf16_paged_decode')
    batch, rows = len(views), padded_tokens
    # cu_seqlens_q[B+1], seqused_ori_kv[B], ori_topk_length[M] are INT32.
    # The caller reserves and copies 1024 INT32 metadata words. No reload
    # amplification or undocumented internal controller operations are inferred.
    reads, writes = 4 * (2 * batch + 1 + rows), 1024 * 4
    spec = {'batch': batch, 'rows': rows, 'valid_rows': sum(v.q_len for v in views),
            'layer_group': group, 'group_boundary': 'explicit_caller_metadata_group',
            'schedule': schedule,
            'heads': plan.num_heads // parallel.tp,
            'head_dim': plan.kv_lora_rank, 'topk': plan.index_topk,
            'max_query': max(v.q_len for v in views),
            'max_context': max(v.kv_len for v in views),
            'input_read_bytes': reads, 'output_write_bytes': writes,
            'output_shape': (1024,), 'output_dtype': 'int32',
            'query_layout': 'TND', 'cache_layout': 'PA_BBND'}
    result = b.work('sparse_mla_metadata', deps, (batch, rows),
                    family='attention_metadata', dtype='int32', metadata=(
        ('kernel', NATIVE), ('native_kind', NATIVE), ('implementation', IMPLEMENTATION),
        ('invocation_class', 'aicpu'), ('execution_domain', 'AI_CPU'),
        ('metadata_schedule', schedule),
        ('source', SOURCE), ('metadata_call', spec)))
    result = key + '.build'
    b.ops[-1] = replace(b.ops[-1], op_id=result, recipe=IMPLEMENTATION, source_boundary='kernel')
    b.work('sparse_mla_metadata_publish', (result,), (1024,),
        family='copy', dtype='int32', read=writes, write=writes,
        metadata=(('caller', SOURCE), ('metadata_buffer_publication', True),
                  ('metadata_schedule', schedule)))
    b.ops[-1] = replace(b.ops[-1], op_id=key + '.publish')
    return key + '.publish'


def build(op, geometry):
    spec = dict(op.metadata)['metadata_call']
    read = Work(resources={'hbm': spec['input_read_bytes']})
    write = Work(resources={'hbm': spec['output_write_bytes']})
    priced = SERVICE in geometry.resource_units
    controller = Work(resources={SERVICE: 1}) if priced else Work()

    def program(context):
        return KernelProgram((
            Stage('inputs', context.bind(read)),
            Stage('controller', context.bind(controller), after=('inputs',),
                  placement={context.resource_names[SERVICE]: (0,)} if priced else {}),
            Stage('metadata_output', context.bind(write), after=('controller',)),
        ))
    return OperatorPlan(read + controller + write, program, metadata={
        'execution_class': 'aicpu', 'count_boundary': 'one_source_native_call_not_ISA',
        'priced_service_work': {SERVICE: 1} if priced else {},
        'unpriced_service_work': {} if priced else {SERVICE: 1},
        'internal_controller_work_and_cache_reload': 'not_reconstructed',
    })


def before_forward(ops, ready):
    """The runner builds metadata groups before entering layer execution."""
    calls = [op for op in ops if dict(op.metadata).get('metadata_schedule') == 'pre_forward']
    if not calls:
        return ops
    ids = {op.op_id for op in calls}
    publications = tuple(op.op_id for op in calls
                         if dict(op.metadata).get('metadata_buffer_publication'))
    prepared = [replace(op, deps=(ready,)) if op.recipe == IMPLEMENTATION else op
                for op in calls]
    rest = [replace(op, deps=tuple(dict.fromkeys((*op.deps, *publications))))
            if ready in op.deps else op for op in ops if op.op_id not in ids]
    return [*prepared, *rest]
