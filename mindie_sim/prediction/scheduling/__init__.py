"""Shared fragment execution on the one causal Engine.

[Role] Lower ``ProgramFragment`` declarations into ordinary Engine tasks:
  ordered repeats, conditional/program/slot allocations, stream order, event
  ports, state visibility and bounded online aggregation.
[Invariants] One virtual clock and one Engine. A fragment keeps its complete
  executable semantics; unsupported executable semantics are rejected before
  work is admitted, never silently projected away. Repeats are sequential on
  the fragment stream and batch only when homogeneous. State reads bind at
  start and writes/commit/rollback apply once at completion; same-time commits
  need an explicit arbiter. Observations never feed durations back.
[Timeouts & Locks] One task per fragment unless repeats must expand, bounded by
  ``repeat_limit``. Homogeneous repeats use one aggregated task. No I/O or sleep.
[Side-Effect Policy] Mutates only the caller's Engine, the state ledger owned
  by that runtime and this conductor's bounded observation caches.
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from collections.abc import Sequence as _AbcSequence  # noqa: F401 - historical package attribute
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import AllocationEffect, ProgramFragment, StateEffect

from ..engine import Allocation, Engine, Event
from .allocations import _AllocationPlan as _AllocationPlan
from .allocations import _PlannedAllocation as _PlannedAllocation
from .conductor import FragmentConductor
from .state import (
    _MISSING as _MISSING,
)
from .state import (
    _STATE_ACTIONS_WRITE as _STATE_ACTIONS_WRITE,
)
from .state import (
    FragmentContractError,
    StateLedger,
    UnsupportedFragmentSemantics,
)
from .state import (
    _nonnegative_int as _nonnegative_int,
)
from .state import (
    _version as _version,
)
from .submission import FragmentHandle
from .submission import _LiveRecord as _LiveRecord

# Runtime consumers resolve the callable through this package so tests can
# replace the reader and its ABI marker together.
from .templates import _AbcMapping as _AbcMapping
from .templates import (
    _drop_periodic_run as _drop_periodic_run,
)
from .templates import (
    _indexed_scoped_allocations as _indexed_scoped_allocations,
)
from .templates import (
    _pipeline_scope as _pipeline_scope,
)
from .templates import (
    _scoped_allocations as _scoped_allocations,
)
from .templates import (
    start_pipeline_template,
)

__all__ = [
    "Allocation",
    "AllocationEffect",
    "Any",
    "Engine",
    "Event",
    "FragmentConductor",
    "FragmentContractError",
    "FragmentHandle",
    "Mapping",
    "ModelingError",
    "OrderedDict",
    "ProgramFragment",
    "Sequence",
    "StateEffect",
    "StateLedger",
    "UnsupportedFragmentSemantics",
    "annotations",
    "dataclass",
    "deque",
    "math",
    "replace",
    "start_pipeline_template",
]
