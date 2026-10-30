"""Explicit arch22 KDA decode binding on the shared caller contract.

[Role] Register the retained non-arch35 RecurrentKda program.
[Invariants] Selection changes source work/ownership, never a model timer.
  Accepted-token input requires exact addresses or explicit valid-range assumptions.
[Timeouts & Locks] Pure metadata operations; no I/O or locks.
[Side-Effect Policy] Immutable operators and recipe mappings only.
"""
from dataclasses import replace

from . import family_kda_recurrent
from .mechanism import PlannedRecipe
from .operators import kda_recurrent_arch22

IMPLEMENTATION = kda_recurrent_arch22.IMPLEMENTATION
SOURCE = kda_recurrent_arch22.SOURCE
CAPABILITIES = family_kda_recurrent.CAPABILITIES
validate = family_kda_recurrent.validate


def recipes():
    return {kda_recurrent_arch22.RECIPE: PlannedRecipe(kda_recurrent_arch22.build)}


def bind_decode_scans(*args, **kwargs):
    op = family_kda_recurrent.bind_decode_scans(*args, **kwargs)
    metadata = {**dict(op.metadata), "implementation": IMPLEMENTATION, "source": SOURCE,
                "source_binding": "explicit_arch22_kernel_selection_not_binary_equivalence"}
    return replace(op, recipe=kda_recurrent_arch22.RECIPE, metadata=tuple(sorted(metadata.items())))
