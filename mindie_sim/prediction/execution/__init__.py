"""Physical execution engine compiler, STARS queue, Eager D2H, and cohorts.

[Role]
  Public facade for ``mindie_sim.prediction.execution`` re-exporting all
  historical execution symbols plus device-boundary ``StarsDescriptorQueue``,
  ``EagerSplitBarrier``, and ``compute_uncovered_stars_ns``.

[Invariants]
  * Preserves 100% public and internal-test symbol compatibility with the
    pre-split ``mindie_sim.prediction.execution`` module.

[Timeouts & Locks]
  Pure re-export facade; no I/O or locks.

[Side-Effect Policy]
  Side-effect-free package facade.
"""
from __future__ import annotations

from .cohort import CollectiveParticipant
from .cohort_launch import _RankEngine, _bind_native_resource_view, launch_native_cohort
from .context import (
    INLINE_FRONTIER_TASK_CAP,
    PROGRAM_CACHE_ENTRY_LIMIT,
    ResourceRates,
    StageProfile,
    _CollectivePlanBinding,
    _LedgerBoundCost,
    _ServiceSetupProgram,
    _execution_geometry,
    _ledger_recorder,
    build_stage_profiles,
)
from .runner import HOST_LOOKAHEAD_DEFAULT, Execution
from .streams import (
    META_CACHE_LIMIT,
    _META_ID_CACHE,
    _META_KEY_CACHE,
    _META_SCAN_CACHE,
    _NO_REV_OPS_CACHE,
    _collective_native_basis,
    _declaration_of,
    _freeze_key,
    _isolated_script_key,
    _legacy_op_meta_key_from_metadata,
    _meta_diff,
    _meta_patch,
    _metadata_dict,
    _normalize_token,
    _op_meta_key,
    _op_meta_key_slow,
    _op_meta_key_slow_meta,
    _rank_token,
    inherit_meta_id_key,
    mark_no_rev_collective,
    meta_scan_flags,
    op_meta_key_from_metadata,
    seed_meta_id_key,
    stream_id_for_op,
)
from .submission import (
    EagerSplitBarrier,
    ScheduledIteration,
    SplitBarrierResult,
    SplitBarrierSpec,
    StarsDescriptorId,
    StarsDescriptorQueue,
    StarsDescriptorRecord,
    SubmissionPolicy,
    compute_uncovered_stars_ns,
    interval_union_ns,
    is_zero_kernel_view,
)

__all__ = [
    "CollectiveParticipant",
    "EagerSplitBarrier",
    "Execution",
    "HOST_LOOKAHEAD_DEFAULT",
    "INLINE_FRONTIER_TASK_CAP",
    "META_CACHE_LIMIT",
    "PROGRAM_CACHE_ENTRY_LIMIT",
    "ResourceRates",
    "ScheduledIteration",
    "SplitBarrierResult",
    "SplitBarrierSpec",
    "StageProfile",
    "StarsDescriptorId",
    "StarsDescriptorQueue",
    "StarsDescriptorRecord",
    "SubmissionPolicy",
    "_CollectivePlanBinding",
    "_LedgerBoundCost",
    "_RankEngine",
    "_ServiceSetupProgram",
    "_bind_native_resource_view",
    "_collective_native_basis",
    "_declaration_of",
    "_execution_geometry",
    "_freeze_key",
    "_isolated_script_key",
    "_ledger_recorder",
    "_legacy_op_meta_key_from_metadata",
    "_meta_diff",
    "_meta_patch",
    "_metadata_dict",
    "_normalize_token",
    "_op_meta_key",
    "_op_meta_key_slow",
    "_op_meta_key_slow_meta",
    "_rank_token",
    "build_stage_profiles",
    "compute_uncovered_stars_ns",
    "inherit_meta_id_key",
    "interval_union_ns",
    "is_zero_kernel_view",
    "launch_native_cohort",
    "mark_no_rev_collective",
    "meta_scan_flags",
    "op_meta_key_from_metadata",
    "seed_meta_id_key",
    "stream_id_for_op",
]
