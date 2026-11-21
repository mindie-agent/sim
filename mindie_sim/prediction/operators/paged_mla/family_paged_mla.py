"""Dense paged MLA core binding separate from its prolog and padding caller.

[Role] Bind the selected BF16 kernel once across all independent requests and
  expose the active/bucket geometry the native call actually consumes.
[Invariants] Logical heads, prepared prolog weights and shared KV stay distinct.
  An existing head caller owns its copies; this core never adds them twice.
  Active sequence rows and native query capacity are separate values: only
  active rows own KV work, while capacity is a storage/ABI bucket dimension.
  Positions, KV lengths, page entries, dispatch and retained KV come from the
  actual request views, runtime and selected implementation, never from a
  model name or measured duration.  ``native_contract`` and graph lowering
  project one normalized declaration; the selected ``paged_mla*.build`` recipe
  remains the only physical-plan authority.
[Timeouts & Locks] O(batch) metadata only, no locks or I/O.
[Side-Effect Policy] Appends immutable ordinary graph operators.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.invocation import Occurrence as Occurrence
from mindie_sim.schema.operators import (
    FieldDisposition as FieldDisposition, OpSpec, ProgramFragment as ProgramFragment,
    SemanticAttr as SemanticAttr, SemanticRef as SemanticRef,
    StateEffect as StateEffect, WorkSet as WorkSet,
)
from mindie_sim.schema.subgraphs import (
    EventPortSpec as EventPortSpec, PortSet as PortSet,
    StatePortSpec as StatePortSpec, TensorPortSpec as TensorPortSpec,
)

from mindie_sim.prediction.operators.paged_mla import paged_mla
from .family_paged_mla_contract import _fragment
from mindie_sim.prediction.mechanism import PlannedRecipe

IMPLEMENTATION = paged_mla.IMPLEMENTATION
PAGE_TILE = paged_mla.KEY_TILE
_ARCH35_BRANCH = 'arch35_bf16_noquant_mla'
_ARCH35_RECIPE = 'mindie_sim.prediction.paged_mla.build'
_LATENT_WIDTH = 512
_ROPE_WIDTH = 64


def _augment_plan(plan, op):
    """Attach the selected state-access mechanism without changing work/program.

    The actual owner/version belongs to the occurrence metadata and fragment.
    A layer-specific owner must not enter the reusable physical signature; the
    plan carries the mechanism and readiness, while the op binds the occurrence.
    """
    state = dict(op.metadata).get('native_state_access')
    if not isinstance(state, dict) or 'native_state_access' in dict(plan.metadata):
        return plan
    return replace(plan, metadata={
        **dict(plan.metadata), 'native_state_access': dict(state)})


class _PagedMlaPlannedRecipe(PlannedRecipe):
    def prepare(self, op, geometry):
        prepared, plan = super().prepare(op, geometry)
        return prepared, _augment_plan(plan, op)


def planned_recipe(build):
    return _PagedMlaPlannedRecipe(build)


def recipes():
    return {IMPLEMENTATION: planned_recipe(paged_mla.build)}


def _active_views(views):
    views = tuple(views)
    if not views:
        raise ModelingError('paged_mla_requires_active_decode_sequence')
    for view in views:
        if view.phase != 'decode' or view.q_len != 1:
            raise ModelingError('paged_mla_requires_single_token_decode_views')
        if view.padding:
            raise ModelingError('paged_mla_rejects_padding_view_as_active_request')
        if view.draft_proposal_tokens:
            raise ModelingError(
                'paged_mla_does_not_declare_speculative_draft_token_views')
        if type(view.context) is not int or view.context < 0:
            raise ModelingError('paged_mla_requires_nonnegative_integer_context')
    return views


def _capacity(*, active, padded_tokens, native_query_capacity):
    if padded_tokens is not None and (
            type(padded_tokens) is not int or padded_tokens < active):
        raise ModelingError('paged_mla_padded_tokens_below_active_sequences')
    capacity = active if padded_tokens is None else padded_tokens
    if native_query_capacity is not None:
        if (type(native_query_capacity) is not int
                or native_query_capacity < active):
            raise ModelingError(
                'paged_mla_native_query_capacity_below_active_sequences')
        capacity = native_query_capacity
    return capacity


def _retained_rows(retained_kv_rows):
    if (type(retained_kv_rows) is not int or isinstance(retained_kv_rows, bool)
            or retained_kv_rows < 0):
        raise ModelingError('paged_mla_retained_kv_rows_requires_nonnegative_int')
    return retained_kv_rows


def _validate_branch(plan, parallel):
    """The selected dense paged-MLA branch's semantic applicability."""
    if (parallel.cp != 1 or plan.mla_variant != 'latent_mla'
            or plan.cache_kind != 'dense' or plan.index_topk
            or plan.window or plan.noncausal_attention):
        raise ModelingError(
            'paged_mla_requires_q1_bf16_dense_latent512_rope64')


def native_geometry(*, views, plan, parallel, runtime, padded_tokens=None,
                    kv_splits=1, native_query_capacity=None, dispatch=None,
                    source_host_dispatch=False, retained_kv_rows=0):
    """Return active rows, bucket capacity and physical page geometry.

    The values are declarations consumed by the native binding and by tests.
    They do not add work: the selected recipe derives work from ``contexts``.
    ``retained_kv_rows`` is preserved as an implementation option; its bytes,
    capacity and lifetime are computed only by the selected arch22 recipe.
    """
    _validate_branch(plan, parallel)
    views = _active_views(views)
    active = len(views)
    if type(kv_splits) is not int or kv_splits < 1:
        raise ModelingError('paged_mla_requires_positive_declared_split_count')
    if plan.num_heads <= 0 or plan.num_heads % parallel.tp:
        raise ModelingError('paged_mla_requires_integral_local_heads')
    logical = plan.num_heads // parallel.tp
    physical = 1 << (logical - 1).bit_length()
    capacity = _capacity(active=active, padded_tokens=padded_tokens,
                         native_query_capacity=native_query_capacity)
    page = getattr(runtime, 'kv_block_size', None)
    if type(page) is not int or page < 1:
        raise ModelingError('paged_mla_requires_positive_kv_block_size')
    retained = _retained_rows(retained_kv_rows)
    contexts = tuple(view.context for view in views)
    kv_lengths = tuple(view.kv_len for view in views)
    page_counts = tuple((length + page - 1) // page for length in kv_lengths)
    # A selected physical program has one split authority. Declared ``kv_splits``
    # is authoritative only on the dispatch-free, non-source-host path; source
    # SplitCore assigns its own segments and explicit dispatch carries an oracle.
    if dispatch is None and not source_host_dispatch:
        split_source = 'declared_kv_splits'
        effective_splits = tuple(
            min(kv_splits, (length + PAGE_TILE - 1) // PAGE_TILE)
            for length in kv_lengths)
    else:
        split_source = ('explicit_dispatch' if isinstance(dispatch, dict)
                        else 'source_host_dispatch')
        effective_splits = None
    return {
        'active_sequence_count': active,
        'native_query_capacity': capacity,
        'logical_heads': logical,
        'physical_heads': physical,
        'contexts': contexts,
        'kv_lengths': kv_lengths,
        'position_ranges': tuple((view.context, view.context + view.q_len)
                                 for view in views),
        'page_size': page,
        'page_counts': page_counts,
        'page_entries': sum(page_counts),
        'kv_splits': kv_splits,
        'kv_split_source': split_source,
        'effective_kv_splits': effective_splits,
        'dispatch': dispatch,
        'retained_kv_rows': retained,
        'retained_kv_layout': ('source_s2_tile_adjacent_generations'
                               if retained else None),
        'retained_kv_capacity_owner': ('per_active_aic_owner'
                                       if retained else None),
        'inactive_sequence_kv_length': 0,
    }


def _declared_operands(*, plan, query_dtype='bf16', cache_dtype='bf16',
                       output_dtype=None, query_layout='BNSD',
                       cache_layout='PA_BSND', output_layout='NBSD',
                       quant_scales=None, attention_scale=None,
                       attention_scale_source='unresolved'):
    latent = int(plan.kv_lora_rank)
    rope = int(plan.qk_rope_head_dim)
    if (latent, rope) != (_LATENT_WIDTH, _ROPE_WIDTH):
        raise ModelingError(
            f'paged_mla_selected_branch_requires_latent512_rope64:{latent}:{rope}')
    output_dtype = output_dtype or query_dtype
    if (query_dtype, cache_dtype, output_dtype) != ('bf16', 'bf16', 'bf16'):
        raise ModelingError(
            'paged_mla_selected_bf16_branch_requires_bf16:'
            f'{query_dtype}:{cache_dtype}:{output_dtype}')
    if (query_layout, cache_layout, output_layout) != (
            'BNSD', 'PA_BSND', 'NBSD'):
        raise ModelingError(
            'paged_mla_selected_branch_layout_mismatch:'
            f'{query_layout}:{cache_layout}:{output_layout}')
    if quant_scales is not None:
        raise ModelingError('paged_mla_selected_bf16_branch_rejects_quant_scales')
    if attention_scale is not None and (
            isinstance(attention_scale, bool)
            or not isinstance(attention_scale, (int, float))):
        raise ModelingError('paged_mla_attention_scale_requires_number_or_none')
    return {
        'latent_width': latent, 'rope_width': rope,
        'query_dtype': query_dtype, 'cache_dtype': cache_dtype,
        'output_dtype': output_dtype, 'query_layout': query_layout,
        'cache_layout': cache_layout, 'output_layout': output_layout,
        'quant_scales': None, 'attention_scale': attention_scale,
        'attention_scale_source': attention_scale_source,
    }


def _build_declaration_metadata(
    *, geometry, operands, abi, kv_splits, dispatch, is_arch22, host_dispatch,
    implementation, program_recipe, logical, physical, active, capacity,
    kv_owner, owner_source, kv_version, kv_condition, kv_producer, kv_output,
    kv_storage, spans, request_ids, request_rank,
):
    geometry_meta = {
        'contexts': geometry['contexts'], 'kv_lengths': geometry['kv_lengths'],
        'position_ranges': geometry['position_ranges'], 'heads': physical,
        'logical_heads': logical, 'kv_splits': kv_splits,
        'kv_split_source': geometry['kv_split_source'],
        'effective_kv_splits': geometry['effective_kv_splits'],
        'page_size': geometry['page_size'], 'page_counts': geometry['page_counts'],
        'dispatch': dispatch, 'retained_kv_rows': geometry['retained_kv_rows'],
        'retained_kv_layout': geometry['retained_kv_layout'],
        'retained_kv_capacity_owner': geometry['retained_kv_capacity_owner'],
        'active_sequence_count': active, 'native_query_capacity': capacity,
        'inactive_sequence_kv_length': 0,
    }
    if is_arch22:
        geometry_meta.pop('kv_splits', None)
    tensor_view = {
        'kind': 'paged_latent', 'active_query_shape': abi['active_query_shape'],
        'query_shape': abi['query_storage_shape'], 'query_rope_shape': abi['query_rope_shape'],
        'output_shape': abi['output_shape'], 'query_dtype': operands['query_dtype'],
        'cache_dtype': operands['cache_dtype'], 'output_dtype': operands['output_dtype'],
        'cache_head_shape': abi['cache_head_shape'], 'rope_cache_head_shape': abi['rope_cache_head_shape'],
        'contexts': geometry['contexts'], 'kv_lengths': geometry['kv_lengths'],
        'position_ranges': geometry['position_ranges'], 'active_sequence_count': active,
        'native_query_capacity': capacity, 'inactive_sequence_kv_length': 0,
        'page_size': geometry['page_size'], 'page_counts': geometry['page_counts'],
        'page_entries': geometry['page_entries'], 'page_table_dtype': 'int32',
        'kv_split_source': geometry['kv_split_source'],
        'effective_kv_splits': geometry['effective_kv_splits'],
        'retained_kv_rows': geometry['retained_kv_rows'],
        'retained_kv_layout': geometry['retained_kv_layout'],
        'retained_kv_capacity_owner': geometry['retained_kv_capacity_owner'],
        'quant_scales': operands['quant_scales'], 'attention_scale': operands['attention_scale'],
        'attention_scale_source': operands['attention_scale_source'],
        'kv_owner': kv_owner, 'kv_owner_source': owner_source, 'kv_version': kv_version,
        'kv_condition': kv_condition, 'kv_producer': kv_producer, 'kv_output': kv_output,
        'kv_storage': kv_storage, 'kv_span': spans, 'kv_request_ids': request_ids,
        'kv_request_rank': request_rank,
        'output_alias': 'NBSD[:logical_heads]; leading-head slice; no copy',
    }
    binding = {
        'selector': 'paged_mla_attention', 'implementation': implementation,
        'program_recipe': program_recipe, 'capture_dispatch': 'not_established',
        'call_boundary': 'one_native_kernel_for_independent_single_query_sequences',
    }
    if is_arch22:
        binding['host_dispatch'] = host_dispatch
    return {
        'paged_mla_geometry': geometry_meta, 'native_attention_abi': abi,
        'implementation_binding': binding, 'attention_tensor_view': tensor_view,
        'mla_head_contract': {'logical_heads': logical, 'physical_heads': physical},
        'native_state_access': {
            'target': 'kv_cache', 'action': 'read', 'owner': kv_owner,
            'owner_source': owner_source, 'version': kv_version, 'condition': kv_condition,
            'producer': kv_producer, 'output': kv_output, 'storage': kv_storage,
            'span': spans, 'request_ids': request_ids, 'request_rank': request_rank,
            'representation': 'paged_latent_kv', 'readiness': 'producer_complete',
        },
        'output_view': 'NBSD[:logical_heads]; leading-head contiguous slice; no copy',
    }


def _declaration(*, views, plan, parallel, runtime, padded_tokens=None,
                 kv_splits=1, native_query_capacity=None, dispatch=None,
                 source_host_dispatch=False, retained_kv_rows=0,
                 implementation=IMPLEMENTATION, branch=_ARCH35_BRANCH,
                 program_recipe=_ARCH35_RECIPE,
                 query_dtype='bf16', cache_dtype='bf16', output_dtype=None,
                 query_layout='BNSD', cache_layout='PA_BSND',
                 output_layout='NBSD', quant_scales=None,
                 attention_scale=None, attention_scale_source='unresolved',
                 kv_owner='', kv_version=None,
                 kv_condition='active_contexts_only',
                 kv_producer='', kv_output='kv_cache',
                 kv_storage='paged_latent_kv', kv_span=None,
                 request_rank=None):
    """Normalize the one declaration consumed by graph and bare-native entries."""
    if isinstance(retained_kv_rows, bool) or type(retained_kv_rows) is not int:
        raise ModelingError('paged_mla_retained_kv_rows_requires_nonnegative_int')
    if retained_kv_rows and implementation == IMPLEMENTATION:
        raise ModelingError('paged_mla_arch35_retained_kv_not_supported')
    geometry = native_geometry(
        views=views, plan=plan, parallel=parallel, runtime=runtime,
        padded_tokens=padded_tokens, kv_splits=kv_splits, dispatch=dispatch,
        native_query_capacity=native_query_capacity,
        source_host_dispatch=source_host_dispatch,
        retained_kv_rows=retained_kv_rows)
    operands = _declared_operands(
        plan=plan, query_dtype=query_dtype, cache_dtype=cache_dtype,
        output_dtype=output_dtype, query_layout=query_layout,
        cache_layout=cache_layout, output_layout=output_layout,
        quant_scales=quant_scales, attention_scale=attention_scale,
        attention_scale_source=attention_scale_source)
    active = geometry['active_sequence_count']
    capacity = geometry['native_query_capacity']
    logical = geometry['logical_heads']
    physical = geometry['physical_heads']
    is_arch22 = implementation != IMPLEMENTATION
    host_dispatch = (
        'source_common_SplitCore_q1_geometry' if dispatch is None
        else 'declared_common_SplitCore_endpoints')
    # The model adapter supplies the actual selected store owner.  A bare
    # wrapper without that explicit value keeps the neutral self binding
    # instead of guessing ownership from the consumer layer.
    owner_source = 'explicit' if kv_owner else 'default_self'
    if kv_producer and not kv_owner:
        raise ModelingError('paged_mla_state_producer_requires_owner')
    kv_owner = kv_owner or 'self'
    request_ids = tuple(view.request_id for view in views)
    spans = (tuple(tuple(item) for item in kv_span)
             if kv_span is not None
             else tuple((0, length) for length in geometry['kv_lengths']))
    unresolved = ['native_binary_identity', 'actual_split_segments_before_plan',
                  'kv_state_version_before_invocation']
    if not kv_producer:
        unresolved.append('kv_producer_output_unresolved')
    if request_rank is None:
        unresolved.append('kv_request_rank_unresolved')
    if operands['attention_scale'] is None:
        unresolved.append('attention_scale_source_unresolved')
    abi = {
        'call': 'FusedInferAttentionScore',
        'branch': branch,
        'implementation': implementation,
        'program_recipe': program_recipe,
        'query_layout': operands['query_layout'],
        'output_layout': operands['output_layout'],
        'cache_layout': operands['cache_layout'],
        'query_dtype': operands['query_dtype'],
        'cache_dtype': operands['cache_dtype'],
        'output_dtype': operands['output_dtype'],
        'dtype': operands['query_dtype'],
        'active_query_shape': (active, physical, 1, operands['latent_width']),
        'query_storage_shape': (capacity, physical, 1, operands['latent_width']),
        'query_rope_shape': (capacity, physical, 1, operands['rope_width']),
        'output_shape': (physical, capacity, 1, operands['latent_width']),
        'cache_head_shape': (1, operands['latent_width']),
        'rope_cache_head_shape': (1, operands['rope_width']),
        'latent_width': operands['latent_width'],
        'rope_width': operands['rope_width'],
        'page_size': geometry['page_size'],
        'page_table_dtype': 'int32',
        'page_entries': geometry['page_entries'],
        'position_ranges': geometry['position_ranges'],
        'kv_lengths': geometry['kv_lengths'],
        'kv_split_source': geometry['kv_split_source'],
        'effective_kv_splits': geometry['effective_kv_splits'],
        'retained_kv_rows': geometry['retained_kv_rows'],
        'retained_kv_layout': geometry['retained_kv_layout'],
        'retained_kv_capacity_owner': geometry['retained_kv_capacity_owner'],
        'inactive_sequence_kv_length': 0,
        'quant_scales': operands['quant_scales'],
        'attention_scale': operands['attention_scale'],
        'attention_scale_source': operands['attention_scale_source'],
        'active_sequence_count': active,
        'native_query_capacity': capacity,
        'logical_heads': logical,
        'physical_heads': physical,
        'kv_owner': kv_owner,
        'kv_owner_source': owner_source,
        'kv_version': kv_version,
        'kv_condition': kv_condition,
        'kv_producer': kv_producer,
        'kv_output': kv_output,
        'kv_storage': kv_storage,
        'kv_span': spans,
        'kv_request_ids': request_ids,
        'kv_request_rank': request_rank,
    }
    metadata = _build_declaration_metadata(
        geometry=geometry, operands=operands, abi=abi, kv_splits=kv_splits,
        dispatch=dispatch, is_arch22=is_arch22, host_dispatch=host_dispatch,
        implementation=implementation, program_recipe=program_recipe,
        logical=logical, physical=physical, active=active, capacity=capacity,
        kv_owner=kv_owner, owner_source=owner_source, kv_version=kv_version,
        kv_condition=kv_condition, kv_producer=kv_producer, kv_output=kv_output,
        kv_storage=kv_storage, spans=spans, request_ids=request_ids,
        request_rank=request_rank)
    return {
        'geometry': geometry,
        'operands': operands,
        'implementation': implementation,
        'branch': branch,
        'program_recipe': program_recipe,
        'metadata': metadata,
        'dispatch': dispatch,
        'retained_kv_rows': geometry['retained_kv_rows'],
        'kv_owner': kv_owner,
        'kv_owner_source': owner_source,
        'kv_version': kv_version,
        'kv_condition': kv_condition,
        'kv_producer': kv_producer,
        'kv_output': kv_output,
        'kv_storage': kv_storage,
        'kv_span': spans,
        'request_ids': request_ids,
        'request_rank': request_rank,
        'unresolved_fields': tuple(unresolved),
    }


def _declaration_op(declaration, *, views, plan, deps=(), op_id='paged_mla_decode'):
    geometry = declaration['geometry']
    operands = declaration['operands']
    active = geometry['active_sequence_count']
    capacity = geometry['native_query_capacity']
    physical = geometry['physical_heads']
    shape = (active, physical, 1, operands['latent_width'])
    padded = (capacity, physical, 1, operands['latent_width'])
    return OpSpec(
        op_id, 'attention', 'decode', 0, plan.index,
        tuple(view.request_id for view in views), shape, shape, padded, padded,
        deps=tuple(deps), recipe=declaration['implementation'],
        metadata=tuple(declaration['metadata'].items()),
        dtype=operands['query_dtype'], source_boundary='kernel')


def native_contract(*, views, plan, parallel, runtime, padded_tokens=None,
                    kv_splits=1, native_query_capacity=None, dispatch=None,
                    source_host_dispatch=False, retained_kv_rows=0,
                    implementation=IMPLEMENTATION, branch=_ARCH35_BRANCH,
                    program_recipe=_ARCH35_RECIPE,
                    query_dtype='bf16', cache_dtype='bf16', output_dtype=None,
                    query_layout='BNSD', cache_layout='PA_BSND',
                    output_layout='NBSD', quant_scales=None,
                    attention_scale=None, attention_scale_source='unresolved',
                    kv_owner='', kv_version=None,
                    kv_condition='active_contexts_only',
                    kv_producer='', kv_output='kv_cache',
                    kv_storage='paged_latent_kv', kv_span=None,
                    request_rank=None, occurrence=None):
    """Independent typed contract for the selected paged-MLA native call.

    It consumes explicit values and the selected implementation, projects the
    same normalized declaration as graph lowering, and never invents work.
    """
    declaration = _declaration(
        views=views, plan=plan, parallel=parallel, runtime=runtime,
        padded_tokens=padded_tokens, kv_splits=kv_splits, dispatch=dispatch,
        native_query_capacity=native_query_capacity,
        source_host_dispatch=source_host_dispatch,
        retained_kv_rows=retained_kv_rows, implementation=implementation,
        branch=branch, program_recipe=program_recipe,
        query_dtype=query_dtype, cache_dtype=cache_dtype,
        output_dtype=output_dtype, query_layout=query_layout,
        cache_layout=cache_layout, output_layout=output_layout,
        quant_scales=quant_scales, attention_scale=attention_scale,
        attention_scale_source=attention_scale_source, kv_owner=kv_owner,
        kv_version=kv_version, kv_condition=kv_condition,
        kv_producer=kv_producer, kv_output=kv_output,
        kv_storage=kv_storage, kv_span=kv_span,
        request_rank=request_rank)
    return _fragment(declaration, views=views, plan=plan, occurrence=occurrence)


def native_plan(*, views, plan, parallel, runtime, geometry,
                padded_tokens=None, kv_splits=1, native_query_capacity=None,
                dispatch=None, source_host_dispatch=False,
                retained_kv_rows=0, implementation=IMPLEMENTATION,
                branch=_ARCH35_BRANCH, program_recipe=_ARCH35_RECIPE,
                build=None, deps=(), op_id='paged_mla_decode',
                query_dtype='bf16', cache_dtype='bf16', output_dtype=None,
                query_layout='BNSD', cache_layout='PA_BSND',
                output_layout='NBSD', quant_scales=None,
                attention_scale=None, attention_scale_source='unresolved',
                kv_owner='', kv_version=None,
                kv_condition='active_contexts_only',
                kv_producer='', kv_output='kv_cache',
                kv_storage='paged_latent_kv', kv_span=None,
                request_rank=None):
    """Return the unique recipe OperatorPlan for this same declaration.

    The caller supplies physical ``geometry`` and, for arch22, its existing
    build function.  No SplitCore or cache arithmetic is duplicated here.
    """
    declaration = _declaration(
        views=views, plan=plan, parallel=parallel, runtime=runtime,
        padded_tokens=padded_tokens, kv_splits=kv_splits, dispatch=dispatch,
        native_query_capacity=native_query_capacity,
        source_host_dispatch=source_host_dispatch,
        retained_kv_rows=retained_kv_rows, implementation=implementation,
        branch=branch, program_recipe=program_recipe,
        query_dtype=query_dtype, cache_dtype=cache_dtype,
        output_dtype=output_dtype, query_layout=query_layout,
        cache_layout=cache_layout, output_layout=output_layout,
        quant_scales=quant_scales, attention_scale=attention_scale,
        attention_scale_source=attention_scale_source, kv_owner=kv_owner,
        kv_version=kv_version, kv_condition=kv_condition,
        kv_producer=kv_producer, kv_output=kv_output,
        kv_storage=kv_storage, kv_span=kv_span,
        request_rank=request_rank)
    op = _declaration_op(declaration, views=views, plan=plan, deps=deps,
                         op_id=op_id)
    plan = (build or paged_mla.build)(op, geometry)
    return _augment_plan(plan, op)


def _validate_dispatch(dispatch, *, contexts, heads):
    if isinstance(dispatch, str):
        if dispatch != 'source_split_core_v2':
            raise ModelingError('paged_mla_dispatch_policy_unknown')
        return
    if dispatch is None:
        return
    if not isinstance(dispatch, dict):
        raise ModelingError(
            'paged_mla_dispatch_requires_policy_or_explicit_oracle')
    declared = tuple(dispatch.get('contexts') or ())
    if declared and declared != tuple(contexts):
        raise ModelingError(
            'paged_mla_dispatch_contexts_must_match_active_views')
    declared_heads = dispatch.get('heads')
    if declared_heads is not None:
        try:
            mismatch = int(declared_heads) != int(heads)
        except (TypeError, ValueError) as exc:
            raise ModelingError(
                'paged_mla_dispatch_heads_must_be_integer') from exc
        if mismatch:
            raise ModelingError(
                'paged_mla_dispatch_heads_must_match_physical_heads')


def validate(*, views, plan, parallel, model, quant, padded_tokens, kv_splits=1,
             dispatch=None, native_query_capacity=None):
    _validate_branch(plan, parallel)
    views = _active_views(views)
    active = len(views)
    _declared_operands(
        plan=plan, query_dtype=model.activation_dtype,
        cache_dtype=quant.kv_dtype, output_dtype=model.activation_dtype)
    if padded_tokens is not None and padded_tokens < active:
        raise ModelingError('paged_mla_requires_q1_bf16_dense_latent512_rope64')
    if type(kv_splits) is not int or kv_splits < 1:
        raise ModelingError('paged_mla_requires_positive_declared_split_count')
    if plan.num_heads <= 0 or plan.num_heads % parallel.tp:
        raise ModelingError('paged_mla_requires_integral_local_heads')
    logical = plan.num_heads // parallel.tp
    physical = 1 << (logical - 1).bit_length()
    _capacity(active=active, padded_tokens=padded_tokens,
              native_query_capacity=native_query_capacity)
    _validate_dispatch(dispatch, contexts=tuple(v.context for v in views),
                       heads=physical)


def lower_attention(b, *, views, plan, heads, runtime, query, rotary, store,
                    head_contract=None, padded_tokens=None, kv_splits=1,
                    dispatch=None, native_query_capacity=None,
                    source_host_dispatch=False, retained_kv_rows=0,
                    implementation=IMPLEMENTATION, branch=_ARCH35_BRANCH,
                    program_recipe=_ARCH35_RECIPE,
                    attention_scale=None, attention_scale_source='unresolved',
                    kv_owner='', kv_version=None,
                    kv_condition='active_contexts_only',
                    kv_producer='', kv_output='kv_cache',
                    kv_storage='paged_latent_kv', kv_span=None,
                    request_rank=None):
    declaration = _declaration(
        views=views, plan=plan, parallel=runtime.parallel, runtime=runtime,
        padded_tokens=padded_tokens, kv_splits=kv_splits, dispatch=dispatch,
        native_query_capacity=native_query_capacity,
        source_host_dispatch=source_host_dispatch,
        retained_kv_rows=retained_kv_rows, implementation=implementation,
        branch=branch, program_recipe=program_recipe,
        query_dtype=b.model.activation_dtype, cache_dtype=b.quant.kv_dtype,
        output_dtype=b.model.activation_dtype, attention_scale=attention_scale,
        attention_scale_source=attention_scale_source, kv_owner=kv_owner,
        kv_version=kv_version, kv_condition=kv_condition,
        kv_producer=kv_producer, kv_output=kv_output,
        kv_storage=kv_storage, kv_span=kv_span,
        request_rank=request_rank)
    geometry = declaration['geometry']
    operands = declaration['operands']
    active = geometry['active_sequence_count']
    capacity = geometry['native_query_capacity']
    logical = geometry['logical_heads']
    physical = geometry['physical_heads']
    declared_heads = int(heads)
    if declared_heads not in (logical, physical):
        raise ModelingError('paged_mla_caller_heads_must_be_logical_or_physical')
    if head_contract is not None and (
            int(head_contract.get('logical_heads', -1)) != logical
            or int(head_contract.get('physical_heads', -1)) != physical):
        raise ModelingError('paged_mla_head_contract_mismatch')
    _validate_dispatch(dispatch, contexts=geometry['contexts'], heads=physical)
    if head_contract is None and physical != declared_heads:
        # Generic prolog has contiguous logical outputs. Only F.pad is needed;
        # the native padded-prolog's strided q_nope copy belongs to its caller.
        byte = dtype_bytes(operands['query_dtype'])
        packed = []
        for part, producer, width in (
                ('nope', query, operands['latent_width']),
                ('rope', rotary, operands['rope_width'])):
            packed.append(b.work('paged_mla_query_pad_' + part, (producer,),
                (capacity, physical, width),
                read=capacity*declared_heads*width*byte,
                write=capacity*physical*width*byte, family='copy',
                dtype=operands['query_dtype'],
                metadata=(('materialization', 'F.pad logical heads to physical'),
                          ('zero_fill_bytes',
                           capacity*(physical-declared_heads)*width*byte))))
        query, rotary = packed
    name = b.work('paged_mla_decode', (query, rotary, store),
                  (active, physical, 1, operands['latent_width']),
                  family='attention', dtype=operands['query_dtype'])
    b.ops[-1] = replace(
        b.ops[-1], recipe=implementation, source_boundary='kernel',
        metadata=tuple(declaration['metadata'].items()),
        padded_shape=(capacity, physical, 1, operands['latent_width']),
        launch_shape=(capacity, physical, 1, operands['latent_width']))
    return name
