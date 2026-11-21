"""Fragment contracts and versioned state.

[Role] Validate and track the state versions accessed by fragments.
[Invariants] Default reads see committed versions. Explicit reads may name live
  versions.
[Timeouts & Locks] Live versions are bounded per target. All updates are in memory.
[Side-Effect Policy] Updates only the runtime-owned ledger.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import StateEffect

_DEFAULT_STATE_VERSION_LIMIT = 64


class FragmentContractError(ModelingError):
    """The caller supplied a fragment whose executable meaning is unresolved."""


class UnsupportedFragmentSemantics(FragmentContractError):
    """A declared fragment feature is not executable by this slice."""


_MISSING = object()
_STATE_ACTIONS_WRITE = frozenset({"write", "commit", "rollback"})


def _nonnegative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FragmentContractError(f"{name}_invalid:{value!r}")
    return value


def _version(value: Any, target: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FragmentContractError(f"state_version_invalid:{target}:{value!r}")
    return value


class StateLedger:
    """Visible committed versions plus speculative live buffers.

    ``write`` adds a candidate; ``commit`` changes the visible version;
    ``rollback`` discards only an uncommitted candidate. A commit retains the
    previous committed version for already-issued readers plus versions at or
    ahead of the new commitment; older superseded candidates are reclaimed so a
    long request history cannot grow this ledger without bound.
    """

    def __init__(self, initial: Mapping[str, int] | None = None, *,
                 version_limit: int = _DEFAULT_STATE_VERSION_LIMIT):
        self._committed: dict[str, int] = {}
        self._live: dict[str, set[int]] = {}
        #: Runtime root that adopted this ledger; a second root is an error.
        self.runtime_uid: str | None = None
        self.version_limit = max(2, _nonnegative_int(
            "state_version_limit", version_limit))
        # Same-timestamp commits are grouped before selecting a winner. The
        # winner is the largest (arbiter, sequence) pair, so container order is
        # never the authority.
        self._pending: dict[tuple[str, float], dict[int, tuple[str, int]]] = {}
        self._commit_base: dict[tuple[str, float], int | None] = {}
        for target, version in dict(initial or {}).items():
            if not isinstance(target, str) or not target:
                raise FragmentContractError("state_target_invalid")
            value = _version(version, target)
            self._committed[target] = value
            self._live[target] = {value}

    def snapshot(self) -> dict[str, Any]:
        return {
            "committed": dict(sorted(self._committed.items())),
            "live": {target: tuple(sorted(versions))
                     for target, versions in sorted(self._live.items())},
        }

    def initialize(self, initial: Mapping[str, int] | None = None) -> None:
        """Seed targets that are not yet known; a different version conflicts."""
        for target, version in dict(initial or {}).items():
            if not isinstance(target, str) or not target:
                raise FragmentContractError("state_target_invalid")
            value = _version(version, target)
            current = self._committed.get(target)
            if current is None:
                self._committed[target] = value
                self._live.setdefault(target, set()).add(value)
            elif current != value:
                raise FragmentContractError(
                    f"state_ledger_conflict:{target}:{current}:{value}")

    def committed_version(self, target: str) -> int | None:
        return self._committed.get(target)

    def live_versions(self, target: str) -> tuple[int, ...]:
        return tuple(sorted(self._live.get(target, ())))

    def check_read(self, target: str, version: int | None, *, at: float) -> int:
        current = self._committed.get(target)
        if version is None:
            if current is None:
                raise FragmentContractError(f"state_target_uninitialized:{target}")
            return current
        value = _version(version, target)
        if value == current or value in self._live.get(target, ()):
            return value
        raise FragmentContractError(
            f"state_version_unavailable:{target}:{value}:committed={current}")

    def apply(self, effects: Sequence[StateEffect], *, at: float,
              sequence: int) -> dict[str, Any]:
        self._prune(at)
        for effect in effects:
            action = effect.action
            if action == "read":
                continue
            target = effect.target
            live = self._live.setdefault(target, set())
            if action == "write":
                current = self._committed.get(target)
                version = effect.version
                if version is None:
                    version = 1 if current is None else current + 1
                version = _version(version, target)
                if current is not None and version < current:
                    raise FragmentContractError(
                        f"state_write_behind_committed:{target}:{version}:{current}")
                live.add(version)
                if len(live) > self.version_limit:
                    raise FragmentContractError(
                        f"state_live_versions_exceeded:{target}:"
                        f"{len(live)}:{self.version_limit}")
                continue
            version = effect.version
            if version is None:
                raise FragmentContractError(f"state_effect_version_missing:{target}")
            version = _version(version, target)
            if version not in live:
                raise FragmentContractError(
                    f"state_version_not_live:{target}:{version}")
            if action == "rollback":
                if self._committed.get(target) == version:
                    raise FragmentContractError(
                        f"state_rollback_committed:{target}:{version}")
                live.discard(version)
                continue
            self._commit(target, version, effect.arbiter, sequence, at)
        return self.snapshot()

    def _commit(self, target: str, version: int, arbiter: str,
                sequence: int, at: float) -> None:
        key = target, at
        if key not in self._commit_base:
            self._commit_base[key] = self._committed.get(target)
        previous = self._commit_base[key]
        if previous is not None and version < previous:
            raise FragmentContractError(
                f"state_commit_behind_committed:{target}:{version}:{previous}")
        records = self._pending.setdefault(key, {})
        records.setdefault(version, (arbiter, sequence))
        if len(records) == 1:
            self._committed[target] = version
            self._retain(target, version, previous)
            return
        if any(not item[0] for item in records.values()):
            raise FragmentContractError(
                f"state_effect_conflict_requires_arbiter:{target}")
        winner = max(records.items(),
                     key=lambda item: (item[1][0], item[1][1]))
        self._committed[target] = winner[0]
        self._retain(target, winner[0], previous)

    def _retain(self, target: str, version: int, previous: int | None) -> None:
        live = self._live.setdefault(target, set())
        # Keep every version at or ahead of the pre-commit base until a later
        # timestamp. A same-time competing commit may still arrive after this
        # one, and dropping its candidate first would turn a declared arbiter
        # conflict into a spurious "not live" error.
        floor = previous if previous is not None else version
        keep = {version}
        if previous is not None:
            keep.add(previous)
        keep.update(item for item in live if item >= floor)
        if len(keep) > self.version_limit:
            raise FragmentContractError(
                f"state_live_versions_exceeded:{target}:"
                f"{len(keep)}:{self.version_limit}")
        self._live[target] = keep

    def _prune(self, at: float) -> None:
        stale = [key for key in self._pending if key[1] < at]
        for key in stale:
            del self._pending[key]
        stale = [key for key in self._commit_base if key[1] < at]
        for key in stale:
            del self._commit_base[key]
