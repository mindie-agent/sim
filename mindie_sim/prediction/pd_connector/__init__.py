"""Physical connector service for a complete P/D cache handoff.

[Role] Bind declared connector framing, propagation and capacity to topology.
[Invariants] Endpoints 0/1 denote prefill/decode pools, not inferred model ranks.
  One handoff is one logical message; wire bytes exclude duplicated capacity
  demands. HBM work is supplied separately by each pool's representative shard.
  Optional chunk/credit/startup fields are explicit already-normalized T06
  services; absent fields keep the legacy whole-message path. A finite receiver
  credit is an Engine transport pool shared by plans on the same Engine, never
  extra wire work. Source readiness, destination grant and consumer release are
  distinct events; both transport branches consume the same common readiness
  boundary and neither branch drops a caller-provided event.
[Timeouts & Locks] Constant-sized connector graph and aggregate counters only.
[Side-Effect Policy] Submits tasks only to the caller's in-memory engine.
"""
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link

from ..communication import MessageFlow
from ..topology import Topology, TransportOptions
from .connector import PDConnector
from .handoff import PDHandoff, PDTransfer
from .handoff import _non_negative_number as _non_negative_number
from .handoff import _normalize_releases as _normalize_releases
from .transport import _AliasedTopology as _AliasedTopology

__all__ = [
    "Hardware",
    "Link",
    "Mapping",
    "MessageFlow",
    "ModelingError",
    "PDConnector",
    "PDHandoff",
    "PDTransfer",
    "Topology",
    "TransportOptions",
    "dataclass",
    "defaultdict",
    "math",
    "replace",
]
