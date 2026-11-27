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
        # Same-timestamp concurrent commits compete deterministically by
        # (arbiter, sequence, version); causally later commits at the same
        # timestamp supersede their ancestor commits cleanly.
        self._pending: dict[tuple[str, float], dict[int, tuple[int, str, int]]] = {}
        self._commit_base: dict[tuple[str, float], int | None] = {}
        self._committed_at_t: dict[tuple[str, float], set[int]] = {}
        self._commit_counter = 0
        self._causal_time: float | None = None
        self._live_tokens: set[int] = set()
        self._event_commits: dict[int, set[int]] = {}
        self._events_by_token: dict[int, set[int]] = {}
        self._event_preds: dict[int, tuple[int, ...]] = {}
        self._introduced_commits: dict[tuple[str, float, int], set[int]] = {}
        self._read_ancestors: dict[int, set[int]] = {}
        self._causal_stack: list[tuple[frozenset[int], set[int], int]] = []
        self._read_stack: list[int] = []
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

    def _retire_token(self, token: int) -> None:
        self._live_tokens.discard(token)
        holders = self._events_by_token.pop(token, None)
        if not holders:
            return
        for seq in holders:
            bucket = self._event_commits.get(seq)
            if bucket is None:
                continue
            bucket.discard(token)
            if not bucket:
                del self._event_commits[seq]

    def _note_event_predecessors(self, event_sequence: int, after: Any) -> None:
        if not after:
            return
        if hasattr(after, "sequence"):
            self._event_preds[event_sequence] = (after.sequence,)
            return
        preds = tuple(pred.sequence for pred in after if hasattr(pred, "sequence"))
        if preds:
            self._event_preds[event_sequence] = preds

    def _propagate_event_commits(self, event_sequence: int, *, at: float) -> None:
        preds = self._event_preds.pop(event_sequence, None)
        if not preds:
            return
        self._prune(at)
        if not self._event_commits:
            return
        ancestors: set[int] = set()
        for pred_seq in preds:
            tokens = self._event_commits.get(pred_seq)
            if tokens:
                ancestors.update(tokens)
        if ancestors:
            self._record_event_commits(event_sequence, ancestors, at=at)

    def _inherit_event_commits(
        self, event_sequence: int, after: Any, *, at: float,
    ) -> None:
        """Propagate active same-timestamp commits from completed predecessors."""
        if not after or (not self._event_commits and not self._causal_stack):
            return
        self._prune(at)
        if not self._event_commits and not self._causal_stack:
            return
        active_seq = self._causal_stack[-1][2] if self._causal_stack else -1
        ancestors: set[int] = set()
        if hasattr(after, "sequence"):
            seq = after.sequence
            tokens = self._event_commits.get(seq)
            if tokens:
                ancestors.update(tokens)
            elif seq == active_seq:
                anc, emitted, _ = self._causal_stack[-1]
                ancestors.update(anc)
                ancestors.update(emitted)
        else:
            for pred in after:
                seq = getattr(pred, "sequence", None)
                if seq is None:
                    continue
                tokens = self._event_commits.get(seq)
                if tokens:
                    ancestors.update(tokens)
                elif seq == active_seq:
                    anc, emitted, _ = self._causal_stack[-1]
                    ancestors.update(anc)
                    ancestors.update(emitted)
        if ancestors:
            self._record_event_commits(event_sequence, ancestors, at=at)

    def _propagate_to_waiters(
        self, event_sequence: int, waiters: Sequence[int], *, at: float,
    ) -> None:
        """Push same-timestamp commit tokens forward to awakened waiting sequences."""
        if not self._event_commits and not self._causal_stack:
            return
        self._prune(at)
        tokens: set[int] = set()
        own = self._event_commits.get(event_sequence)
        if own:
            tokens.update(own)
        if self._causal_stack:
            anc, emitted, _ = self._causal_stack[-1]
            if anc:
                tokens.update(anc)
            if emitted:
                tokens.update(emitted)
        if not tokens:
            return
        if own is None:
            self._record_event_commits(event_sequence, tokens, at=at)
        if waiters:
            for wseq in waiters:
                self._record_event_commits(wseq, tokens, at=at)

    def _collect_causal_ancestors(
        self, predecessors: Sequence[Any], *, at: float,
    ) -> frozenset[int]:
        self._prune(at)
        if not predecessors or not self._event_commits:
            return frozenset()
        ancestors: set[int] = set()
        for pred in predecessors:
            tokens = self._event_commits.get(pred.sequence)
            if tokens:
                ancestors.update(tokens)
        return frozenset(ancestors)

    def _record_event_commits(
        self, event_sequence: int, tokens: frozenset[int] | set[int], *, at: float,
    ) -> None:
        self._prune(at)
        live = {tok for tok in tokens if tok in self._live_tokens}
        if not live:
            return
        bucket = self._event_commits.setdefault(event_sequence, set())
        for tok in live:
            if tok not in bucket:
                bucket.add(tok)
                self._events_by_token.setdefault(tok, set()).add(event_sequence)

    def _link_active_event_commits(
        self, event_sequences: Sequence[int], *, at: float,
    ) -> None:
        if not self._causal_stack:
            return
        ancestors, emitted, _ = self._causal_stack[-1]
        combined = ancestors | emitted
        if not combined:
            return
        for seq in event_sequences:
            self._record_event_commits(seq, combined, at=at)

    def _run_read_scope(self, event_sequence: int, callback, event) -> None:
        self._read_stack.append(event_sequence)
        try:
            callback(event)
        finally:
            self._read_stack.pop()

    def _run_causal_scope(
        self, ancestors: frozenset[int], callback, event, *, at: float,
    ) -> frozenset[int]:
        self._prune(at)
        emitted: set[int] = set()
        self._causal_stack.append((ancestors, emitted, int(event.sequence)))
        try:
            callback(event)
        finally:
            self._causal_stack.pop()
        return frozenset(emitted)

    def check_read(self, target: str, version: int | None, *, at: float) -> int:
        self._prune(at)
        current = self._committed.get(target)
        if version is None:
            if current is None:
                raise FragmentContractError(f"state_target_uninitialized:{target}")
            return current
        value = _version(version, target)
        if value == current or value in self._live.get(target, ()):
            if self._read_stack:
                intro = self._introduced_commits.get((target, at, value))
                if intro:
                    live_intro = intro & self._live_tokens
                    if live_intro:
                        self._read_ancestors.setdefault(
                            self._read_stack[-1], set()
                        ).update(live_intro)
            return value
        raise FragmentContractError(
            f"state_version_unavailable:{target}:{value}:committed={current}")

    def apply(
        self, effects: Sequence[StateEffect], *, at: float,
        sequence: int, causal_ancestors: Sequence[int] = (),
    ) -> dict[str, Any]:
        self._prune(at)
        incoming: set[int] = {tok for tok in causal_ancestors if tok in self._live_tokens}
        if self._causal_stack:
            incoming.update(tok for tok in self._causal_stack[-1][0] if tok in self._live_tokens)
        read_tokens = self._read_ancestors.pop(sequence, None)
        if read_tokens:
            incoming.update(tok for tok in read_tokens if tok in self._live_tokens)
        # Sibling commits in the same fragment share the fragment's incoming
        # ancestry and arbitrate symmetrically instead of superseding by order.
        incoming_frozen = frozenset(incoming)
        newly_written: dict[str, set[int]] = {}
        emitted_tokens: list[tuple[str, int, int]] = []
        for effect in effects:
            action = effect.action
            if action == "read":
                continue
            target = effect.target
            live = self._live.setdefault(target, set())
            if action == "write":
                key = target, at
                base = (self._commit_base[key]
                        if key in self._commit_base
                        else self._committed.get(target))
                records = self._pending.get(key)
                if records and incoming_frozen:
                    sup_versions = [
                        rec[0] for tok, rec in records.items() if tok in incoming_frozen
                    ]
                    if sup_versions:
                        sup_max = max(sup_versions)
                        base = sup_max if base is None else max(base, sup_max)
                current = self._committed.get(target)
                version = effect.version
                if version is None:
                    version = 1 if current is None else current + 1
                version = _version(version, target)
                if base is not None and version < base:
                    raise FragmentContractError(
                        f"state_write_behind_committed:{target}:{version}:{base}")
                if version not in live:
                    newly_written.setdefault(target, set()).add(version)
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
            token = self._commit(target, version, effect.arbiter, sequence, at, incoming_frozen)
            emitted_tokens.append((target, version, token))
        for target, version, token in emitted_tokens:
            if self._causal_stack:
                self._causal_stack[-1][1].add(token)
            if version in newly_written.get(target, ()):
                self._introduced_commits.setdefault((target, at, version), set()).add(token)
        return self.snapshot()

    def _commit(
        self, target: str, version: int, arbiter: str,
        sequence: int, at: float, ancestors: frozenset[int],
    ) -> int:
        key = target, at
        existing_records = self._pending.get(key, {})
        surviving = {
            tok: rec for tok, rec in existing_records.items() if tok not in ancestors
        }
        superseded_toks = [tok for tok in existing_records if tok in ancestors]
        superseded_versions = [existing_records[tok][0] for tok in superseded_toks]
        base = (
            self._commit_base[key]
            if key in self._commit_base
            else self._committed.get(target)
        )
        effective_prev = base
        if superseded_versions:
            sup_max = max(superseded_versions)
            effective_prev = (
                sup_max if effective_prev is None else max(effective_prev, sup_max)
            )
        if effective_prev is not None and version < effective_prev:
            raise FragmentContractError(
                f"state_commit_behind_committed:{target}:{version}:{effective_prev}")
        distinct_versions = {rec[0] for rec in surviving.values()} | {version}
        competing = len(distinct_versions) > 1
        has_empty_arbiter = (not arbiter) or any(
            not rec[1] for rec in surviving.values()
        )
        if competing and has_empty_arbiter:
            raise FragmentContractError(
                f"state_effect_conflict_requires_arbiter:{target}")
        if key not in self._commit_base:
            self._commit_base[key] = base
        for tok in superseded_toks:
            self._retire_token(tok)
        self._commit_counter += 1
        token = self._commit_counter
        self._live_tokens.add(token)
        surviving[token] = (version, arbiter, sequence)
        self._pending[key] = surviving
        hist = self._committed_at_t.setdefault(key, set())
        if base is not None:
            hist.add(base)
        hist.add(version)
        if len(surviving) == 1:
            self._committed[target] = version
            if superseded_versions:
                # Reclaim strictly older superseded committed versions while
                # preserving the immediate predecessor and uncommitted peers.
                keep_floor = max(superseded_versions)
                reclaimed = [v for v in hist if v < keep_floor]
                live = self._live.setdefault(target, set())
                for old_v in reclaimed:
                    live.discard(old_v)
                    hist.discard(old_v)
                    self._introduced_commits.pop((target, at, old_v), None)
            self._retain(target, version, self._commit_base[key])
            return token
        winner = max(surviving.values(),
                     key=lambda rec: (rec[1], rec[2], rec[0]))
        self._committed[target] = winner[0]
        self._retain(target, winner[0], self._commit_base[key])
        return token

    def _retain(self, target: str, version: int, previous: int | None,
                *, closing: bool = False) -> None:
        live = self._live.setdefault(target, set())
        keep = {version}
        if previous is not None and (closing or previous in live):
            keep.add(previous)
        if closing:
            keep.update(item for item in live if item >= version)
        elif previous is not None:
            # Keep every live version at or ahead of the pre-commit base until a
            # later timestamp so same-time competing commits stay live.
            keep.update(item for item in live if item >= previous)
        else:
            # Uninitialized target: defer pruning lower candidates until this
            # timestamp closes so a same-time higher-arbiter commit to a lower
            # version cannot hit a spurious "not live" error.
            keep.update(live)
        if len(keep) > self.version_limit:
            raise FragmentContractError(
                f"state_live_versions_exceeded:{target}:"
                f"{len(keep)}:{self.version_limit}")
        self._live[target] = keep

    def _prune(self, at: float) -> None:
        if (self._causal_time == at and not self._pending
                and not self._committed_at_t and not self._commit_base):
            return
        if self._pending:
            stale = [key for key in self._pending if key[1] < at]
            for key in stale:
                for tok in self._pending.pop(key):
                    self._retire_token(tok)
        if self._committed_at_t:
            stale_hist = [key for key in self._committed_at_t if key[1] < at]
            for key in stale_hist:
                del self._committed_at_t[key]
        if self._commit_base:
            stale_base = [key for key in self._commit_base if key[1] < at]
            for key in stale_base:
                target, _ = key
                previous = self._commit_base.pop(key)
                committed = self._committed.get(target)
                if committed is not None:
                    self._retain(target, committed, previous, closing=True)
        if self._causal_time is None or at > self._causal_time:
            if self._live_tokens:
                self._live_tokens.clear()
            if self._event_commits:
                self._event_commits.clear()
            if self._events_by_token:
                self._events_by_token.clear()
            if self._introduced_commits:
                self._introduced_commits.clear()
            if self._read_ancestors:
                self._read_ancestors.clear()
            self._causal_time = at
