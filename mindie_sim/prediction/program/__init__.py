"""Composable kernel programs, recipes, pipelines, and cohort lowering.

[Role] Re-export the public ``mindie_sim.prediction.program`` surface across
  its focused submodules (``stages``, ``pipeline``, ``binding``, ``lowering``)
  together with 3-ledger memory access and vector sub-pipe binding contracts.
[Invariants] Every historical ``mindie_sim.prediction.program`` public symbol
  remains importable directly from this package with identical semantics.
[Timeouts & Locks] Pure module re-export; zero I/O, threads, or locks.
[Side-Effect Policy] Import-only side-effect-free package facade.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
from mindie_sim.schema.operators import ProgramFragment
from ..engine import Allocation, Engine, Event
from ..ownership import unit_group
from ..runtime.memory_physics import (
    AddressRun,
    AddressRunPlan,
    AdmissionMemoryView,
    BoundAccess,
    BurstTurnRule,
    ByteLedger,
    ConstraintTerm,
    CounterTerm,
    DirtyWriteback,
    HbmBusSpec,
    MemoryAccessPattern,
    ReuseKey,
    VisibilityContract,
)
from ..runtime.prepared_terms import BoundVectorOp, VectorInstructionGroup
from .binding import (
    Recipe,
    RecipeContext,
    ResourceEnvelope,
    TransferRecipe,
    proportional_coupled_placement,
)
from .coalesce import (
    _FloorParallelProgram,
    coalesce_execution_program,
    is_pure_static_program,
)
from .lowering import (
    coalesce_programs,
    merge_programs,
    program_signature,
    split_cohort_kernel,
)
from .recipes import (
    GENERIC_MATRIX_PROGRAM,
    HostStaging,
    RecipeBook,
    RecurrentScan,
)
from .pipeline import (
    PIPELINE_COHORT_WINDOW,
    PIPELINE_FOLD_WINDOW,
    PipelinePhase,
    PipelineProgram,
    PipelineTemplate,
    PipelineTemplateProgram,
    fold_interior_pipeline,
    indexed_pipeline_consumer_available,
    interior_fold_plan,
)
from .stages import (
    PIPELINE_UNROLL_LIMIT,
    KernelProgram,
    NativeProgram,
    ParallelProgram,
    Program,
    Stage,
    scope_allocations,
    stage_from_fragment,
)

__all__ = [
    "AddressRun",
    "AddressRunPlan",
    "AdmissionMemoryView",
    "Allocation",
    "BoundAccess",
    "BoundVectorOp",
    "BurstTurnRule",
    "ByteLedger",
    "ConstraintTerm",
    "CounterTerm",
    "DirtyWriteback",
    "Engine",
    "Event",
    "GENERIC_MATRIX_PROGRAM",
    "HbmBusSpec",
    "HostStaging",
    "KernelProgram",
    "MemoryAccessPattern",
    "ModelingError",
    "NATIVE_WORK_UNITS",
    "NativeProgram",
    "PIPELINE_COHORT_WINDOW",
    "PIPELINE_FOLD_WINDOW",
    "PIPELINE_UNROLL_LIMIT",
    "ParallelProgram",
    "PipelinePhase",
    "PipelineProgram",
    "PipelineTemplate",
    "PipelineTemplateProgram",
    "Program",
    "ProgramFragment",
    "Recipe",
    "RecipeBook",
    "RecipeContext",
    "RecurrentScan",
    "ResourceEnvelope",
    "ReuseKey",
    "Stage",
    "TransferRecipe",
    "VectorInstructionGroup",
    "VisibilityContract",
    "coalesce_execution_program",
    "coalesce_programs",
    "fold_interior_pipeline",
    "indexed_pipeline_consumer_available",
    "interior_fold_plan",
    "is_pure_static_program",
    "merge_programs",
    "program_signature",
    "proportional_coupled_placement",
    "scope_allocations",
    "split_cohort_kernel",
    "stage_from_fragment",
    "unit_group",
]
