"""Explicit failure types for public extension discovery and dispatch.

[Role] Keep discovery, compatibility and provider-resolution failures actionable,
  and report an unmodeled capability as a structured, replayable gap instead of
  a bare string or a silent approximation.
[Invariants] Extension/gap errors are ordinary modeling-input errors; callers may
  catch ``ModelingError`` and still distinguish the concrete reason string and,
  for ``CapabilityGap``, the affected results plus the minimal replay input.
[Timeouts & Locks] Pure value type; no I/O or locks.
[Side-Effect Policy] No side effects.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindie_sim.schema.common import ModelingError

GAP_SCHEMA = "mindie-sim.gap/1"
GAP_KINDS = frozenset({
    "unmodeled_capability", "insufficient_fields", "infeasible",
    "invalid_input",
})


class ExtensionError(ModelingError):
    """A declared extension source or selected provider is unusable."""


class CapabilityGap(ExtensionError):
    """An actionable gap: what is missing, what it affects and how to replay.

    The exception remains a ``ModelingError`` so existing callers keep working.
    ``gap`` is a JSON-compatible payload with a stable schema; it never reports
    a substituted integer/generic result because no result was produced.
    """

    def __init__(self, *, capability: str, entry: str, kind: str, code: str,
                 message: str, affected_results: Any = (),
                 minimal_replay_input: Any = None,
                 fallback: str = "not_attempted",
                 cause: BaseException | None = None) -> None:
        if kind not in GAP_KINDS:
            raise ValueError("capability_gap_kind_invalid:" + str(kind))
        self.capability = str(capability)
        self.entry = str(entry)
        self.kind = str(kind)
        self.code = str(code)
        self.message = str(message)
        self.affected_results = tuple(str(item) for item in affected_results)
        self.minimal_replay_input = minimal_replay_input
        self.fallback = str(fallback)
        payload: dict[str, Any] = {
            "schema": GAP_SCHEMA,
            "status": "gap",
            "capability": self.capability,
            "entry": self.entry,
            "kind": self.kind,
            "code": self.code,
            "message": self.message,
            "affected_results": list(self.affected_results),
            "minimal_replay_input": minimal_replay_input,
            "fallback": self.fallback,
            "result_available": False,
            "replayable": isinstance(minimal_replay_input, Mapping)
            and bool(minimal_replay_input),
        }
        self.gap: dict[str, Any] = payload
        super().__init__(self.code + ": " + self.message)
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        return dict(self.gap)

    def __repr__(self) -> str:
        return ("CapabilityGap(capability=%r, kind=%r, code=%r)"
                % (self.capability, self.kind, self.code))


class UnmodeledCapabilityError(CapabilityGap):
    """A timing entry refused to fabricate work for an unmodeled core.

    The exception remains a ``ModelingError`` (through ``CapabilityGap``), so
    callers that already handle modeling gaps keep working, but it carries the
    structured ``mindie-unmodeled-structure-hint/1`` payload an agent needs to
    implement the missing physical model. ``result_available`` is always false:
    no latency, throughput or efficiency number was produced.
    """

    KIND = "unmodeled_capability"
    RESULT_AVAILABLE = False
    FALLBACK = "timing_rejected"

    def __init__(self, *, capability: str, entry: str, code: str, message: str,
                 blockers: Any = (), hints: Any = (),
                 affected_results: Any = (), minimal_replay_input: Any = None,
                 cause: BaseException | None = None) -> None:
        hint_list = tuple(dict(hint) for hint in hints)
        super().__init__(
            capability=capability, entry=entry, kind=self.KIND, code=code,
            message=message, affected_results=affected_results,
            minimal_replay_input=minimal_replay_input,
            fallback=self.FALLBACK, cause=cause)
        self.blockers = tuple(dict(item) for item in blockers)
        self.result_available = self.RESULT_AVAILABLE
        self.onboarding_hint = (dict(hint_list[0]) if hint_list else {})
        self.unmodeled_structure_onboarding_hints = hint_list
        self.gap.update({
            "result_available": self.result_available,
            "onboarding_hint": self.onboarding_hint,
            "unmodeled_structure_onboarding_hints": list(hint_list),
            "blockers": list(self.blockers),
        })

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.gap)
        payload["unmodeled_structure_onboarding_hints"] = [
            dict(hint) for hint in self.unmodeled_structure_onboarding_hints]
        return payload

    def __repr__(self) -> str:
        return ("UnmodeledCapabilityError(capability=%r, entry=%r, code=%r)"
                % (self.capability, self.entry, self.code))


__all__ = ["CapabilityGap", "ExtensionError", "GAP_KINDS", "GAP_SCHEMA",
           "UnmodeledCapabilityError"]
