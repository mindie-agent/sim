"""Logical, GM, on-chip and HBM access resolution with explicit cache state.

[Role] Turn an ordered sequence of data-identity accesses into physical path
  work: logical payload, on-chip copy service, staging occupancy, cache fill,
  eviction/writeback and next-level line transactions.
[Invariants] Logical bytes are not HBM bytes. A hit removes only the next-level
  fill for the retained lines; on-chip service, staging occupancy and a declared
  ready delay remain. Bypass never fills or evicts. A written line that cannot
  be retained is materialized in the same access. A dirty line is written back
  at most once, on eviction, invalidation or explicit flush. HBM write work
  always satisfies direct_write + writeback; residency is reported per identity
  and never added to a capacity allocation here. Every non-empty access is held
  by one reservation; zero-byte accesses hold nothing.
[Timeouts & Locks] O(resident identities) per access, no I/O, threads or locks.
[Side-Effect Policy] Mutates only the CacheState owned by the caller. Program
  submission uses the caller's Engine and transient staging allocations only;
  retained residency is reported, never reserved here.
"""
from __future__ import annotations as annotations

from dataclasses import dataclass as dataclass
from dataclasses import field as field
from dataclasses import replace as replace

from mindie_sim.schema.common import ModelingError as ModelingError

from ..cache_geometry import (
    CacheAccessGeometry,
    TransactionSummary,
)
from ..cache_geometry import (
    Span as Span,
)
from ..cache_geometry import (
    byte_spans_of_line_spans as byte_spans_of_line_spans,
)
from ..cache_geometry import (
    drop_last_spans as drop_last_spans,
)
from ..cache_geometry import (
    full_line_spans as full_line_spans,
)
from ..cache_geometry import (
    intersect_spans as intersect_spans,
)
from ..cache_geometry import (
    lines_of_byte_spans as lines_of_byte_spans,
)
from ..cache_geometry import (
    normalize_spans as normalize_spans,
)
from ..cache_geometry import (
    range_count as range_count,
)
from ..cache_geometry import (
    subtract_spans as subtract_spans,
)
from ..cache_geometry import (
    take_first_spans as take_first_spans,
)
from ..cache_geometry import (
    union_spans as union_spans,
)
from ..engine import Allocation as Allocation
from ..engine import Engine as Engine
from ..engine import Event as Event
from ..program import KernelProgram as KernelProgram
from ..program import Stage as Stage
from ..transfer import FiniteTransfer as FiniteTransfer
from .contracts import (
    A5_LINE_BYTES,
    A5_TRANSFER_BYTES,
    CacheAccessResult,
    CacheGeometry,
    CachePolicy,
)
from .contracts import (
    CacheAccessJournal as CacheAccessJournal,
)
from .contracts import (
    _align_up as _align_up,
)
from .contracts import (
    _byte_spans_of_lines as _byte_spans_of_lines,
)
from .programs import (
    AccessSummary,
    CacheAccessProgram,
    CacheStage,
    CacheTransfer,
    cache_work_stages,
)
from .residency import CacheAccessReservation, CacheState, CacheWriteback
from .residency import _Resident as _Resident

__all__ = [
    "A5_LINE_BYTES", "A5_TRANSFER_BYTES", "AccessSummary",
    "CacheAccessGeometry", "CacheAccessProgram", "CacheAccessReservation",
    "CacheAccessResult", "CacheGeometry", "CachePolicy", "CacheStage",
    "CacheState", "CacheTransfer", "cache_work_stages",
    "CacheWriteback", "TransactionSummary",
]
