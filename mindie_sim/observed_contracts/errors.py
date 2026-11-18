"""Shared failure type for observed structure-carrier projection.

[Role]
  Define the one exception raised when a declared structure carrier cannot be
  projected into observed targets (malformed nesting, duplicate subgraph ids,
  unknown members or cross-placement members).

[Invariants]
  * ``StructureTargetError`` is a ``ValueError`` so callers that already treat
    invalid structure payloads as data errors keep working unchanged.
  * No profiler, hardware, catalogue or simulation import appears here.

[Timeouts & Locks]
  Pure class declaration; no waits, I/O or locks.

[Side-Effect Policy]
  Import-time declaration only.
"""
from __future__ import annotations


class StructureTargetError(ValueError):
    """A declared structure carrier is not a valid target set."""
