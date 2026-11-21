"""Native KDA decode binding for the retained memory-base program.

[Role] Register the retained RecurrentKda program on the shared caller ABI.
[Invariants] Selection determines source work and ownership.
  Accepted-token input requires exact addresses or explicit valid-range assumptions.
[Timeouts & Locks] Pure metadata operations; no I/O or locks.
[Side-Effect Policy] Immutable operators and recipe mappings only.
"""
from mindie_sim.prediction.operators.kda import family_kda_recurrent
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators.kda import kda_recurrent_membase

IMPLEMENTATION = kda_recurrent_membase.IMPLEMENTATION
SOURCE = kda_recurrent_membase.SOURCE
CAPABILITIES = family_kda_recurrent.CAPABILITIES
validate = family_kda_recurrent.validate


def recipes():
    return {kda_recurrent_membase.RECIPE: PlannedRecipe(kda_recurrent_membase.build)}


def bind_decode_scans(*args, **kwargs):
    op = family_kda_recurrent.bind_decode_scans(*args, **kwargs)
    return kda_recurrent_membase.bind_source(op)
