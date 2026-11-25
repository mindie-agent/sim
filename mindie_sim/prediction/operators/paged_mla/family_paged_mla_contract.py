"""Typed contract projection for a declared dense paged MLA call.

[Role] Publish tensor ports, request state, dependencies and native ABI fields.
[Invariants] Metadata retains the declared query capacity and active KV spans.
  The fragment shares the binding's declaration and adds no physical work.
[Timeouts & Locks] O(batch) descriptor projection; no I/O or locks.
[Side-Effect Policy] Returns immutable semantic fragments.
"""
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.operators import (
    FieldDisposition, ProgramFragment, SemanticAttr, SemanticRef, StateEffect, WorkSet,
)
from mindie_sim.schema.subgraphs import (
    EventPortSpec, PortSet, StatePortSpec, TensorPortSpec,
)
from .common import FLASH_ABI

TARGET_BACKEND: Literal["ascend"] = "ascend"


def _is_flash_contract(declaration, operands) -> bool:
    return (
        declaration.get('aclnn_symbol') == FLASH_ABI
        or operands.get('query_layout') == 'TND'
    )


def _fragment_ports(declaration, operands):
    is_flash = _is_flash_contract(declaration, operands)
    if is_flash:
        tensor_in = (
            TensorPortSpec('q', 'in', 'activation', dtype=operands['query_dtype'],
                           layout_kind=operands['query_layout'], scope='layer'),
            TensorPortSpec('kv_cache', 'in', 'latent_kv_cache', dtype=operands['cache_dtype'],
                           layout_kind=operands['cache_layout'], scope='cohort'),
            TensorPortSpec('page_table', 'in', 'index', dtype='int32',
                           layout_kind='flat', scope='cohort'),
            TensorPortSpec('cu_seqlens_q', 'in', 'index', dtype='int32',
                           layout_kind='flat', scope='cohort'),
            TensorPortSpec('cache_seqlens', 'in', 'index', dtype='int32',
                           layout_kind='flat', scope='cohort'),
            TensorPortSpec('tile_scheduler_metadata', 'in', 'index', dtype='int32',
                           layout_kind='flat', scope='cohort'),
        )
        event_in = (
            EventPortSpec('q_ready', 'in', 'ready'),
            EventPortSpec('kv_cache_ready', 'in', 'ready'),
            EventPortSpec('page_table_ready', 'in', 'ready'),
            EventPortSpec('metadata_ready', 'in', 'ready'),
        )
    else:
        tensor_in = (
            TensorPortSpec('q', 'in', 'activation', dtype=operands['query_dtype'],
                           layout_kind=operands['query_layout'], scope='layer'),
            TensorPortSpec('q_rope', 'in', 'activation', dtype=operands['query_dtype'],
                           layout_kind=operands['query_layout'], scope='layer'),
            TensorPortSpec('kv_cache', 'in', 'latent_kv_cache', dtype=operands['cache_dtype'],
                           layout_kind=operands['cache_layout'], scope='cohort'),
            TensorPortSpec('page_table', 'in', 'index', dtype='int32',
                           layout_kind='flat', scope='cohort'),
        )
        event_in = (
            EventPortSpec('q_ready', 'in', 'ready'),
            EventPortSpec('kv_cache_ready', 'in', 'ready'),
            EventPortSpec('page_table_ready', 'in', 'ready'),
        )
    return PortSet(
        tensor_in=tensor_in,
        tensor_out=(
            TensorPortSpec('attention_output', 'out', 'activation', dtype=operands['output_dtype'],
                           layout_kind=operands['output_layout'], scope='layer'),
        ),
        state_in=(
            StatePortSpec('kv_cache_state', 'in', 'paged_latent_kv', scope='cohort',
                          readiness='producer_complete', owner=declaration['kv_owner'] or 'self'),
        ),
        event_in=event_in,
        event_out=(EventPortSpec('done', 'out', 'complete'),),
    )


def _fragment(declaration, *, views, plan, occurrence=None):
    if occurrence is not None and not isinstance(occurrence, Occurrence):
        raise ModelingError('paged_mla_native_occurrence_invalid')
    geometry = declaration['geometry']
    operands = declaration['operands']
    is_flash = _is_flash_contract(declaration, operands)
    output_alias = (
        declaration['metadata']['attention_tensor_view']['output_alias']
        if isinstance(declaration.get('metadata'), dict)
        and isinstance(declaration['metadata'].get('attention_tensor_view'), dict)
        and 'output_alias' in declaration['metadata']['attention_tensor_view']
        else 'NBSD[:logical_heads]; leading-head contiguous slice; no copy'
    )
    active = geometry['active_sequence_count']
    capacity = geometry['native_query_capacity']
    logical = geometry['logical_heads']
    physical = geometry['physical_heads']
    field_effects = [
        FieldDisposition('q.dtype', 'preserve'),
        FieldDisposition('q.layout', 'consume',
                         source='native query layout view'),
    ]
    if is_flash:
        field_effects.extend((
            FieldDisposition(
                'cu_seqlens_q.dtype', 'consume',
                source='request view query lengths',
            ),
            FieldDisposition(
                'cache_seqlens.dtype', 'consume',
                source='request view kv lengths',
            ),
            FieldDisposition(
                'tile_scheduler_metadata.dtype', 'consume',
                source='flash_mla_metadata_kernel',
            ),
        ))
    else:
        field_effects.append(FieldDisposition('q_rope.dtype', 'preserve'))
    field_effects.extend((
        FieldDisposition('kv_cache.dtype', 'preserve'),
        FieldDisposition('kv_cache.state_version', 'consume',
                         source='runtime cache request state'),
        FieldDisposition('kv_cache.page_size', 'consume',
                         source='runtime.kv_block_size'),
        FieldDisposition('position', 'consume',
                         source='request view context and query range'),
        FieldDisposition('scale', 'preserve',
                         source=operands['attention_scale_source']),
        FieldDisposition('quant_scales', 'preserve',
                         source='bf16 cache: no quant scales'),
        FieldDisposition('output.alias', 'consume',
                         source='leading logical heads view; no copy'),
    ))
    return ProgramFragment(
        'paged_mla_decode',
        occurrence=occurrence,
        semantic_refs=(SemanticRef('subgraph', 'mla_attention', role='mla_decode'),),
        attrs=(
            SemanticAttr('native_kind', declaration.get('native_kind') or 'FusedInferAttentionScore',
                         source='source_binding'),
            SemanticAttr('implementation', declaration['implementation'],
                         source='selected_program'),
            SemanticAttr('program_recipe', declaration['program_recipe'],
                         source='selected_program'),
            SemanticAttr('branch', declaration['branch'],
                         source='selected_program'),
            SemanticAttr('dispatch', declaration['dispatch'],
                         source='selected_program'),
            SemanticAttr('retained_kv_rows', declaration['retained_kv_rows'],
                         unit='row', source='selected_program'),
            SemanticAttr('retained_kv_layout',
                         geometry['retained_kv_layout'],
                         source='selected_program'),
            SemanticAttr('retained_kv_capacity_owner',
                         geometry['retained_kv_capacity_owner'],
                         source='selected_program'),
            SemanticAttr('dtype', operands['query_dtype'], source='typed_input'),
            SemanticAttr('query_dtype', operands['query_dtype'],
                         source='typed_input'),
            SemanticAttr('cache_dtype', operands['cache_dtype'],
                         source='typed_input'),
            SemanticAttr('output_dtype', operands['output_dtype'],
                         source='typed_input'),
            SemanticAttr('query_layout', operands['query_layout'],
                         source='typed_input'),
            SemanticAttr('output_layout', operands['output_layout'],
                         source='typed_input'),
            SemanticAttr('cache_layout', operands['cache_layout'],
                         source='typed_input'),
            SemanticAttr('latent_width', operands['latent_width'],
                         unit='element'),
            SemanticAttr('rope_width', operands['rope_width'], unit='element'),
            SemanticAttr('logical_heads', logical, unit='head'),
            SemanticAttr('physical_heads', physical, unit='head'),
            SemanticAttr('contexts', geometry['contexts'], unit='token'),
            SemanticAttr('kv_lengths', geometry['kv_lengths'], unit='token'),
            SemanticAttr('position_ranges', geometry['position_ranges'],
                         unit='token'),
            SemanticAttr('active_sequence_count', active, unit='sequence'),
            SemanticAttr('native_query_capacity', capacity, unit='sequence'),
            SemanticAttr('inactive_sequence_kv_length', 0, unit='token'),
            SemanticAttr('page_size', geometry['page_size'], unit='token'),
            SemanticAttr('page_counts', geometry['page_counts'], unit='page'),
            SemanticAttr('page_entries', geometry['page_entries'], unit='entry'),
            SemanticAttr('page_table_dtype', 'int32', source='native_abi'),
            SemanticAttr('kv_split_source', geometry['kv_split_source'],
                         source='selected_program'),
            SemanticAttr('effective_kv_splits', geometry['effective_kv_splits'],
                         unit='split', source='selected_program'),
            SemanticAttr('quant_scales', operands['quant_scales'],
                         source='absent:bf16_cache_has_no_quant_scales'),
            SemanticAttr('attention_scale', operands['attention_scale'],
                         source=operands['attention_scale_source']),
            SemanticAttr('output_alias', output_alias, source='caller_view'),
            SemanticAttr('kv_owner', declaration['kv_owner'] or 'self',
                         source='selected_state_owner'),
            SemanticAttr('kv_owner_source', declaration['kv_owner_source']),
            SemanticAttr('kv_version', declaration['kv_version'],
                         unit='version', source='runtime_state'),
            SemanticAttr('kv_condition', declaration['kv_condition'],
                         source='selected_program'),
            SemanticAttr('kv_producer', declaration['kv_producer'],
                         source='selected_state_producer'),
            SemanticAttr('kv_output', declaration['kv_output'],
                         source='selected_state_output'),
            SemanticAttr('kv_storage', declaration['kv_storage'],
                         source='selected_state_storage'),
            SemanticAttr('kv_span', declaration['kv_span'],
                         unit='range', source='request_span'),
            SemanticAttr('kv_request_ids', declaration['request_ids'],
                         source='request_scope'),
            SemanticAttr('kv_request_rank', declaration['request_rank'],
                         source='request_scope'),
            SemanticAttr('unresolved_fields', declaration['unresolved_fields'],
                         source='specific_request'),
        ),
        ports=_fragment_ports(declaration, operands),
        work=WorkSet(),
        state_effects=(
            StateEffect('kv_cache', 'read', version=declaration['kv_version'],
                        condition=declaration['kv_condition']),
        ),
        field_effects=tuple(field_effects),
        notes=('work supplied by selected recipe plan; no duplicate timing',
               'ports and attrs are executable-contract declarations',
               'dispatch and retained KV are implementation options'),
    )
