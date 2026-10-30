"""Source-compatible arch22 dense MLA and its geometry-derived host partition.

[Role] Bind q1 BF16 latent MLA and the existing caller head materializations.
[Invariants] The caller ABI is shared with arch35; physical execution is not.
  SplitCore is geometry-derived; an explicit host declaration is an override.
[Timeouts & Locks] O(batch + host owners), no locks or I/O.
[Side-Effect Policy] Appends immutable ordinary graph operators only.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from . import family_paged_mla, paged_mla_arch22
from .mechanism import PlannedRecipe

IMPLEMENTATION = paged_mla_arch22.IMPLEMENTATION


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(paged_mla_arch22.build)}


def validate(*, dispatch=None, retained_kv_rows=0, native_query_capacity=None, **kwargs):
    family_paged_mla.validate(**kwargs)
    if native_query_capacity is not None and (
            type(native_query_capacity) is not int
            or native_query_capacity < len(kwargs['views'])):
        raise ModelingError('paged_mla_query_capacity_below_active_sequences')


def lower_attention(b, *, padded_tokens=None, dispatch=None, retained_kv_rows=0,
                    native_query_capacity=None, **kwargs):
    name = family_paged_mla.lower_attention(b, dispatch=dispatch, **kwargs)
    op = b.ops[-1]
    metadata = dict(op.metadata)
    metadata['paged_mla_geometry'].pop('kv_splits')
    metadata['paged_mla_geometry']['retained_kv_rows'] = retained_kv_rows
    metadata['implementation_binding']['implementation'] = IMPLEMENTATION
    metadata['implementation_binding']['host_dispatch'] = (
        'source_common_SplitCore_q1_geometry' if dispatch is None else 'declared_common_SplitCore_output')
    active = len(kwargs['views'])
    # GraphPolicy already selected the incoming query storage rows. SP-only
    # padding has been sliced away by the caller before this boundary.
    capacity = active if padded_tokens is None else padded_tokens
    if native_query_capacity is not None:
        capacity = native_query_capacity
    heads = metadata['paged_mla_geometry']['heads']
    metadata['paged_mla_geometry'].update(
        active_sequence_count=active, native_query_capacity=capacity,
        inactive_sequence_kv_length=0)
    # The source caller pads seq_lens with zero. Common SplitCore assigns no
    # QK/PV work to those rows; query storage capacity is not valid KV length.
    metadata['attention_tensor_view'].update(
        active_query_shape=(active, heads, 1, 512),
        query_shape=(capacity, heads, 1, 512),
        query_rope_shape=(capacity, heads, 1, 64),
        output_shape=(heads, capacity, 1, 512))
    b.ops[-1] = replace(op, recipe=IMPLEMENTATION, metadata=tuple(metadata.items()),
                       padded_shape=(capacity, heads, 1, 512),
                       launch_shape=(capacity, heads, 1, 512))
    return name
