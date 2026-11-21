"""Compatibility entry point for the sparse index scorer recipes.

[Role] Expose the legacy scorer recipe IDs, constructors and planners.
[Invariants] The governed sparse indexer family owns every lowering.
[Timeouts & Locks] Imports only; no I/O or locks.
[Side-Effect Policy] No registration or device setup at import time.
"""
from mindie_sim.prediction.operators.sparse_indexer import (
    LIGHTNING_INDEXER_QSA_RECIPE as LIGHTNING_INDEXER_QSA_RECIPE,
    MSA_INDEX_SCORE_RECIPE as MSA_INDEX_SCORE_RECIPE,
    MSA_INDEX_SCORE_V2_RECIPE as MSA_INDEX_SCORE_V2_RECIPE,
    STEP5_REGION_SCORE_RECIPE as STEP5_REGION_SCORE_RECIPE,
    build as build,
    make_lightning_indexer_qsa_op as make_lightning_indexer_qsa_op,
    make_msa_index_score_op as make_msa_index_score_op,
    make_step5_region_score_op as make_step5_region_score_op,
    recipes as recipes,
)
