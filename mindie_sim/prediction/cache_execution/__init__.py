"""Event-time cache access execution on the shared Engine.

[Role] Bind a typed cache access request to real ready, source, conversion,
  fill, write, staging and completion events. State is resolved at the ready
  event and published at the actual fill/dirty completion event, never at
  graph-build time. Builder and independent execution use this one executor.
[Invariants] A request does not cache a previous result. Same-identity overlap
  and capacity that cannot be reserved are rejected. HBM read/write work,
  source reads, conversion and staging are actually submitted; counters/state
  are published only in the matching completion callbacks. A named dependency
  is a real port: a generic ready cannot substitute for it. Each launch has one
  immutable occurrence receipt; staging is never relabelled as a T08 grant.
  Outstanding transactions are finite.
[Timeouts & Locks] Virtual Engine events only; one reservation per identity.
[Side-Effect Policy] Mutates the caller's CacheState and Engine only.
"""
from __future__ import annotations as annotations

from collections import deque as deque
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from typing import Mapping as Mapping

from mindie_sim.schema.common import ModelingError as ModelingError

from ..cache_path import (
    CacheAccessGeometry as CacheAccessGeometry,
)
from ..cache_path import (
    CachePolicy as CachePolicy,
)
from ..cache_path import (
    CacheState as CacheState,
)
from ..cache_path import (
    CacheTransfer as CacheTransfer,
)
from ..cache_path import (
    cache_work_stages as cache_work_stages,
)
from ..engine import Allocation as Allocation
from ..engine import Engine as Engine
from ..engine import Event as Event
from .executor import CacheExecution
from .operation import CacheAccessOperation
from .requests import CacheAccessBinding, CacheAccessRequest, CacheTypedView

__all__ = [
    "CacheAccessBinding", "CacheAccessOperation", "CacheAccessRequest",
    "CacheExecution", "CacheTypedView",
]
