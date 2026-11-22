"""Request scheduling driven by resource completion and cache ownership.

[Role] Build bounded prefill/decode batches and commit real output events.
[Invariants] Padding never produces tokens. Speculation produces integer token
  counts only after verification. KV is reserved before execution and retained
  until consumers finish. Preemption frees bytes and causes actual recomputation.
  Committed state is versioned: a later batch observes exactly the state that
  completed, never an uncommitted candidate. A speculative verify reservation
  covers every proposed row; rejected rows are released at commit.
[Timeouts & Locks] Pure in-memory transitions, no external work or waits.
[Side-Effect Policy] Mutates per-request progress and the prediction Memory.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping

from mindie_sim.catalog.speculative_state import ONE, TWO, ZERO
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

from ..capacity import Memory
from ..execution.serving_io import TokenTimeline
from .batching import _PREEMPTED as _PREEMPTED
from .batching import Scheduler
from .batching import _live_ids as _live_ids
from .progress import DraftPolicy, Progress, running_first

__all__ = [
    "Callable",
    "CaseSpec",
    "DraftPolicy",
    "IterationPlan",
    "Mapping",
    "Memory",
    "ModelingError",
    "ONE",
    "Progress",
    "RequestSpec",
    "RequestView",
    "Scheduler",
    "SpeculativeSpec",
    "TWO",
    "TokenTimeline",
    "ZERO",
    "annotations",
    "dataclass",
    "field",
    "hashlib",
    "json",
    "math",
    "replace",
    "running_first",
]
