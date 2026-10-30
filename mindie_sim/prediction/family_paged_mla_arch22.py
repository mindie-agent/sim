"""Source-compatible arch22 dense MLA and its geometry-derived host partition.

[Role] Bind q1 BF16 latent MLA and the existing caller head materializations.
[Invariants] The caller ABI is shared with arch35; physical execution is not.
  SplitCore is geometry-derived; an explicit host declaration is an override.
[Timeouts & Locks] O(batch + host owners), no locks or I/O.
[Side-Effect Policy] Appends immutable ordinary graph operators only.
"""
from dataclasses import replace

from . import family_paged_mla, paged_mla_arch22
from .mechanism import PlannedRecipe

IMPLEMENTATION = paged_mla_arch22.IMPLEMENTATION


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(paged_mla_arch22.build)}


def validate(*, dispatch=None, retained_kv_rows=0, **kwargs):
    family_paged_mla.validate(**kwargs)


def lower_attention(b, *, dispatch=None, retained_kv_rows=0, **kwargs):
    name = family_paged_mla.lower_attention(b, dispatch=dispatch, **kwargs)
    op = b.ops[-1]
    metadata = dict(op.metadata)
    metadata['paged_mla_geometry'].pop('kv_splits')
    metadata['paged_mla_geometry']['retained_kv_rows'] = retained_kv_rows
    metadata['implementation_binding']['implementation'] = IMPLEMENTATION
    metadata['implementation_binding']['host_dispatch'] = (
        'source_common_SplitCore_q1_geometry' if dispatch is None else 'declared_common_SplitCore_output')
    b.ops[-1] = replace(op, recipe=IMPLEMENTATION, metadata=tuple(metadata.items()))
    return name
