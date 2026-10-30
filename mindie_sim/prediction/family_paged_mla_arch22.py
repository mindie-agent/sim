"""Source-compatible arch22 dense MLA with an explicitly supplied host partition.

[Role] Bind q1 BF16 latent MLA and the existing caller head materializations.
[Invariants] The caller ABI is shared with arch35; physical execution is not.
  The exact retained SplitCore declaration is mandatory and context-specific.
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


def validate(*, dispatch=None, **kwargs):
    family_paged_mla.validate(**kwargs)
    if dispatch is None:
        raise ModelingError('paged_mla_arch22_requires_source_host_dispatch')


def lower_attention(b, *, dispatch=None, **kwargs):
    name = family_paged_mla.lower_attention(b, dispatch=dispatch, **kwargs)
    op = b.ops[-1]
    metadata = dict(op.metadata)
    metadata['paged_mla_geometry'].pop('kv_splits')
    metadata['implementation_binding']['implementation'] = IMPLEMENTATION
    metadata['implementation_binding']['host_dispatch'] = 'declared_common_SplitCore_output'
    b.ops[-1] = replace(op, recipe=IMPLEMENTATION, metadata=tuple(metadata.items()))
    return name
