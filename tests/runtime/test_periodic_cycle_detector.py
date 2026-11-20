"""Exact-period ``CycleDetector`` contract and its real-Engine equivalence.

[Role] Pin the private exact-period helper (``engine_periodic``): settled
  frontiers, structural identity, cursor/phase bounds, template interning and
  history budgets, then prove that a detector-driven skip over a real 32-owner,
  two-slot, persistent-accumulator Engine pipeline equals the detailed run.
[Invariants] A skip needs two (or ``evidence_periods``) structurally equal
  measured intervals; identity is deep, type-exact and never a digest or repr;
  cursor progress is not part of the key but phase class, slots, epoch, release
  offsets, placements and wait order are; cycles are bounded by the tightest
  run phase and by ``stop_at``; external/opaque/unmeasured input is rejected,
  never approximated; interner and history memory stay within their budgets;
  skipped totals scale the measured interval exactly.
[Timeouts & Locks] In-memory only; the 2,720-tile real-Engine control runs
  in well under a second.
[Side-Effect Policy] ``reset_state_caches()`` clears process-local detector
  metrics before tests that count them; no filesystem or environment access.
"""
from __future__ import annotations

from collections import UserDict
import math

import pytest

from mindie_sim.prediction.engine import Allocation, Engine, ExecutionError, Resource
from mindie_sim.prediction.engine_periodic import (
    CanonicalState, Cursor, CycleDetector, IntervalDelta, PeriodicContractError, SettledFrontier,
    StateInterner, TemplateRef, metrics, reset_state_caches, structural_difference, structural_equal,
)

pytestmark = pytest.mark.tier0

INF = math.inf
STATE = (
    ("schema", "periodic-test-v1"),
    ("resources", (("mte2", 100.0, "work", "fair", 1), ("cube", 100.0, "work", "fair", 1))),
    ("pools", (("l1", 128.0, 32), ("l0c", 256.0, 32))),
    ("tasks", (
        ("run", 1, "tile@0", "load", (("mte2", 8.0),), 0, ()),
        ("run", 1, "tile@0", "compute", (("cube", 16.0),), 1, ("tile@0:load",)),
        ("run", 1, "tile@1", "load", (("mte2", 8.0),), 0, ("tile@0:load",)),
    )),
    ("live_allocations", (("run", 1, "tile@0", "panel", "l1", 4.0, (0,)),
                          ("run", 1, None, "accumulator0", "l0c", 8.0, (0,)))),
    ("pool_occupied", (("l1", 4.0), ("l0c", 8.0))),
    ("unit_occupied", ((("l1", 0), 4.0), (("l0c", 0), 8.0))),
    ("heaps", (("ready", ((0, 1),)), ("release", ()), ("completable", ()))),
    ("promises", (("complete",),)),
)


def cursor(run_id=1, phase_ordinal=0, phase_end=67, next_issue=3, retired=1, slots=(0, 1)):
    return Cursor(run_id, phase_ordinal, phase_end, next_issue, retired, slots)


def frame(now, *, state=STATE, cursors=None, sequence=100, epoch=7):
    return SettledFrontier(now=now, epoch=epoch, normalized_state=state, sequence=sequence,
                           cursors=tuple(cursors or (cursor(next_issue=2, retired=0),)))


def delta(dt=1.0, *, issued=None, sequence_count=4, service=None, busy=None, peaks=None,
          crossed=False, logical_completions=1, allocation_grants=0, allocation_last_uses=0):
    return IntervalDelta(
        dt=dt, issued_retired_by_run=issued or {1: (1, 1)}, sequence_count=sequence_count,
        service_by_resource={"mte2": 8.0, "cube": 16.0} if service is None else service,
        busy_by_resource={"mte2": 1.0, "cube": 1.0} if busy is None else busy,
        allocation_event_count=2, logical_completions=logical_completions,
        per_unit_peaks={("l1", 0): 4.0, ("l0c", 0): 8.0} if peaks is None else peaks,
        crossed_external_or_opaque=crossed, allocation_grants=allocation_grants,
        allocation_last_uses=allocation_last_uses)


def engine_state(*, remaining=8.0, caps=(("cube", 1.0),), tasks=1, placement=()):
    """Engine 5-tuple form: tasks, runs, allocations, pools, releases."""
    rows = tuple(((("periodic", 1000, 0, index, "stage"), (("cube", remaining),), 0, 0.0, 0.0, 0, caps,
                   placement if index == 0 else (), False, "owner")) for index in range(tasks))
    return rows, ((1000, 0, 67, (0, 1), (("l0c", 8.0, (0,)),)),), (("l1", 4.0, (0,), 1),), \
        (("l1", 4.0, ((0, 4.0),)),), ()


def run_frames(states, *, run_id=1000, slots=(0, 1), detector=None, phase_end=67, **interval):
    """Feed one frontier per state with one issue+retire per period; return (detector, frames, results)."""
    detector = detector or CycleDetector()
    frames, results = [], []
    for index, state in enumerate(states):
        current = frame(float(index), state=state, sequence=100 + 4 * index,
                        cursors=(cursor(run_id, 0, phase_end, 2 + index, index, slots),))
        frames.append(current)
        results.append(detector.observe(
            current, None if index == 0 else delta(issued={run_id: (1, 1)}, **interval), stop_at=INF))
    return detector, frames, results


def test_two_equal_intervals_return_a_bounded_exact_skip_with_scaled_totals():
    _, _, results = run_frames([STATE] * 3, run_id=1, allocation_grants=2, allocation_last_uses=1)
    assert results[:2] == [None, None]
    skip = results[2]
    assert (skip.cycles, skip.period, skip.seq_stride) == (63, 1.0, 4)  # 67 - next_issue(4)
    assert dict(skip.tile_strides_by_run) == {1: 1}
    assert (skip.delta.dt, skip.delta.logical_completions, skip.delta.allocation_grants) == (1.0, 1, 2)
    run = skip.correspondence.run(1)
    assert (run.next_issue_before, run.next_issue_after, run.retired_before, run.retired_after) == (4, 67, 2, 65)
    assert run.tile_offset == run.finished_offset == 63
    assert [(tile.old_tile, tile.new_tile) for tile in skip.correspondence.window] == [(2, 65), (3, 66)]
    assert skip.correspondence.event_sequence_offset == 252
    total = skip.total_delta()
    assert (total.dt, total.sequence_count, total.allocation_event_count) == (63.0, 252, 126)
    assert dict(total.issued_retired_by_run) == {1: (63, 63)}
    assert total.service_by_resource["mte2"] == 504.0 and total.busy_by_resource["cube"] == 63.0
    assert total.per_unit_peaks == skip.delta.per_unit_peaks
    assert (total.allocation_grants, total.allocation_last_uses) == (126, 63)


DEEP = STATE[:3] + (("tasks", (("run", 1, "tile@0", "load", (("mte2", 8.0000001),), 0, ()),)
                    + STATE[3][1][1:]),) + STATE[4:]


@pytest.mark.parametrize("left,right,equal", [
    (STATE, tuple(STATE), True),
    (STATE, DEEP, False),
    (STATE, STATE[:5] + (tuple(reversed(STATE[5])),) + STATE[6:], False),
    ((("m", {"a": 1.0, "b": 2.0}),), (("m", {"b": 2.0, "a": 1.0}),), True),
    ((("flag", True),), (("flag", 1),), False),
    ((("flag", 1),), (("flag", 1.0),), False),
    ((("flag", True),), (("flag", True),), True),
], ids=["copy", "deep-value", "sequence-order", "mapping-order", "bool-int", "int-float", "same-bool"])
def test_structural_identity_is_deep_ordered_and_type_exact(left, right, equal):
    assert structural_equal(left, right) is equal


def test_deep_difference_is_located_and_a_stable_changed_shape_becomes_its_own_cycle():
    assert structural_difference(STATE, DEEP) == "state[3][1][0][4][0][1]:8.0!=8.0000001"
    detector, _, results = run_frames([STATE, DEEP, DEEP, DEEP], run_id=1)
    assert results[:3] == [None, None, None] and results[3].cycles == 62
    assert detector.last_rejection == "skip"


SHARED = StateInterner(limit=8)
LEFT, RIGHT = (SHARED.intern(((name,),), generation=7, fingerprint=123) for name in ("left", "right"))
LOCAL_LEFT, LOCAL_RIGHT = (StateInterner(limit=2).intern(((name,),), generation=7, fingerprint=5)
                           for name in ("left", "right"))
FIXED = StateInterner(limit=8).intern((("fixed",),), generation=7)
FLAG_BOOL = SHARED.intern((("flag", True),), generation=7)
FLAG_INT = SHARED.intern((("flag", 1),), generation=7)
D = (("d", 1.0),)
BASE, NEXT = (0, 67, 2, 0), (0, 67, 3, 1)  # cursor (phase_ordinal, phase_end, next_issue, retired)

# id: (f0 state, f0 cursor, f1 state, f1 cursor, interval, stop_at, skip cycles or rejection[, difference])
PAIRS = {
    "cursor-progress-not-key": (STATE, (2, 67, 3, 1), STATE, (2, 67, 4, 2), {}, INF, 63),
    "phase-class-changed": (STATE, (2, 67, 3, 1), STATE, (3, 68, 4, 2), {}, INF, "phase_changed"),
    "phase-remaining-bounds": (STATE, (0, 8, 2, 0), STATE, (0, 8, 3, 1), {}, INF, 5),
    "phase-boundary": (STATE, (0, 8, 7, 5), STATE, (0, 8, 8, 6), {}, INF, "boundary"),
    "slot-count-past-phase": (STATE, (0, 4, 5, 3, 2), STATE, (0, 4, 6, 4, 2), {}, INF, "boundary"),
    "stop-at-inside": (STATE, (0, 100, 2, 0), STATE, (0, 100, 3, 1), {}, 4.5, 3),
    "stop-at-exact-tie": (STATE, (0, 100, 2, 0), STATE, (0, 100, 3, 1), {}, 5.0, 3),
    "stop-at-before-period": (STATE, (0, 100, 2, 0), STATE, (0, 100, 3, 1), {}, 0.5, "boundary"),
    "external-or-opaque": (STATE, BASE, STATE, NEXT, {"crossed": True}, INF, "external_or_opaque"),
    "same-shape-other-value": (STATE, BASE, STATE[:-1] + (("promises", (("other",),)),), NEXT, {}, INF,
                               "no_repeated_span", "raw"),
    "engine-static-caps": (engine_state(), BASE, engine_state(caps=(("cube", 2.0),)), NEXT, {}, INF,
                           "no_repeated_span", "canonical.template"),
    "engine-dynamic-remaining": (engine_state(), BASE, engine_state(remaining=7.5), NEXT, {}, INF,
                                 "no_repeated_span", "canonical.dynamic"),
    "engine-placement": (engine_state(tasks=2, placement=((0,),)), BASE, engine_state(tasks=2, placement=((1,),)),
                         NEXT, {}, INF, "no_repeated_span", "canonical.template"),
    "mutable-external-caps": (engine_state(caps=[("cube", 1.0)]), BASE, engine_state(caps=[("cube", 2.0)]), NEXT,
                              {}, INF, "no_repeated_span", "raw"),
    "fingerprint-collision": (CanonicalState(LEFT, D), BASE, CanonicalState(RIGHT, D), NEXT, {}, INF,
                              "no_repeated_span", "canonical.template"),
    "interner-local-ids": (CanonicalState(LOCAL_LEFT, D), BASE, CanonicalState(LOCAL_RIGHT, D), NEXT, {}, INF,
                           "no_repeated_span", "canonical.template"),
    "typed-bool-vs-int": (CanonicalState(FLAG_BOOL, (("flag", True),)), BASE,
                          CanonicalState(FLAG_INT, (("flag", 1),)), NEXT, {}, INF,
                          "no_repeated_span", "canonical.template"),
    "dynamic-wait-order": (CanonicalState(FIXED, (("waiters", (1, 2)),), origin="external"), BASE,
                           CanonicalState(FIXED, (("waiters", (2, 1)),), origin="external"), NEXT, {}, INF,
                           "no_repeated_span", "canonical.external"),
}


@pytest.mark.parametrize("first,first_cursor,second,second_cursor,interval,stop_at,expected",
                         [row[:6] + (row[6] if len(row) == 7 else row[6:],) for row in PAIRS.values()],
                         ids=list(PAIRS))
def test_single_evidence_period_skips_only_exact_bounded_repeats(
        first, first_cursor, second, second_cursor, interval, stop_at, expected):
    reset_state_caches()
    detector = CycleDetector(evidence_periods=1)
    assert detector.observe(frame(0.0, state=first, cursors=(cursor(1, *first_cursor),)), None, stop_at=INF) is None
    skip = detector.observe(frame(1.0, state=second, sequence=104, cursors=(cursor(1, *second_cursor),)),
                            delta(**interval), stop_at=stop_at)
    if isinstance(expected, int):
        assert skip is not None and skip.cycles == expected
        return
    rejection, difference = expected if isinstance(expected, tuple) else (expected, "")
    assert skip is None and detector.last_rejection == rejection
    assert (detector.last_difference or "").startswith(difference)


def test_epoch_interval_and_release_offset_changes_reset_baseline_until_stable():
    # Epoch change forces a new baseline immediately.
    detector = CycleDetector(evidence_periods=1)
    detector.observe(frame(0.0, epoch=7), None, stop_at=INF)
    assert detector.observe(frame(1.0, epoch=8, sequence=104), delta(), stop_at=INF) is None
    assert detector.last_rejection == "epoch_changed"
    # Interval mismatch resets candidate period; two repeats of the new delta skip.
    detector, other = CycleDetector(evidence_periods=2), delta(service={"mte2": 9.0, "cube": 16.0})
    obs = [detector.observe(frame(float(i), cursors=(cursor(next_issue=2 + i, retired=i),), sequence=100 + 4 * i),
                            [None, delta(), other, other][i], stop_at=INF) for i in range(4)]
    assert obs[:3] == [None, None, None] and dict(obs[3].delta.service_by_resource)["mte2"] == 9.0
    # Future release offset drift breaks identity until the offset stabilizes.
    detector, later = CycleDetector(evidence_periods=1), STATE + (("release_offset", 1.0),)
    assert detector.observe(frame(0.0, state=STATE + (("release_offset", 2.0),)), None, stop_at=3.0) is None
    assert detector.observe(frame(1.0, state=later, cursors=(cursor(),), sequence=104), delta(), stop_at=3.0) is None
    assert detector.last_rejection in {"baseline", "no_repeated_span"}
    assert detector.observe(frame(2.0, state=later, cursors=(cursor(next_issue=4, retired=2),), sequence=108),
                            delta(), stop_at=INF) is not None


def test_joint_cohort_and_per_run_strides_bound_the_skip():
    detector = CycleDetector(evidence_periods=1)
    runs = [cursor(r, 1, 10 if r == 1 else 68, 4, 2) for r in range(1, 33)]
    shared = STATE + (("shared_units", ((("l1", 0), 64.0),)),)
    detector.observe(frame(0.0, state=shared, cursors=runs, sequence=1000), None, stop_at=INF)
    advanced = [cursor(c.run_id, 1, c.phase_end_exclusive, 5, 3) for c in runs]
    skip = detector.observe(frame(1.0, state=shared, cursors=advanced, sequence=1128),
                            delta(issued={c.run_id: (1, 1) for c in runs}, sequence_count=128,
                                  logical_completions=32), stop_at=INF)
    assert skip.cycles == 5 and set(skip.tile_strides_by_run) == set(range(1, 33))
    offsets = {run.run_id: run.tile_offset for run in skip.correspondence.runs}
    assert offsets[1] == offsets[2] == 5

    # Per-run strides: run 2 allows (12-6)//2 = 3; run 1 allows (10-4)//1 = 6.
    detector = CycleDetector(evidence_periods=1)
    detector.observe(frame(0.0, cursors=(cursor(1, 0, 10, 3, 1), cursor(2, 0, 12, 4, 0))), None, stop_at=INF)
    skip = detector.observe(frame(2.0, sequence=106, cursors=(cursor(1, 0, 10, 4, 2), cursor(2, 0, 12, 6, 2))),
                            delta(dt=2.0, issued={1: (1, 1), 2: (2, 2)}, sequence_count=6), stop_at=INF)
    assert dict(skip.tile_strides_by_run) == {1: 1, 2: 2} and skip.cycles == 3
    assert (skip.correspondence.run(1).tile_offset, skip.correspondence.run(2).tile_offset) == (3, 6)
    assert skip.correspondence.event_sequence_offset == 18


_Float, _Int = type("_Float", (float,), {}), type("_Int", (int,), {})


def _waiting_subclass():
    rows = list(engine_state()[0])
    rows[0] = rows[0][:2] + (_Int(0),) + rows[0][3:]
    return (tuple(rows),) + engine_state()[1:]


@pytest.mark.parametrize("build", [
    lambda: SettledFrontier(now=0.0, epoch=1, normalized_state=(object(),), cursors=(), sequence=0),
    lambda: SettledFrontier(now=float("nan"), epoch=1, normalized_state=(), cursors=(), sequence=0),
    lambda: IntervalDelta(dt=-1.0, issued_retired_by_run={1: (1, 1)}, sequence_count=1, service_by_resource={},
                          busy_by_resource={}, allocation_event_count=0, per_unit_peaks={},
                          logical_completions=0, crossed_external_or_opaque=False),
    lambda: Cursor(1, 0, 4, 1, 2, (0, 1)),  # retired > next_issue
    lambda: CanonicalState(FIXED, (object(),)),
    lambda: frame(0.0, state=_waiting_subclass(), cursors=(cursor(1000, 0, 67, 2, 0),)),
    lambda: frame(0.0, state=engine_state(remaining=_Float(8.0)), cursors=(cursor(1000, 0, 67, 2, 0),)),
    lambda: CanonicalState(FIXED, (((((("cube", _Float(1.0)),), 0, None, None),), (), (), (), ())),
                           origin="engine"),
    lambda: TemplateRef.from_validated(1, 7, 4242, ("mapping", dict, ()), "", 0),
    lambda: TemplateRef.from_validated(1, 7, 4242, ("mapping", int, ()), "", 0),
], ids=["opaque-state", "nan-time", "negative-dt", "retired-ahead", "opaque-dynamic", "int-subclass",
        "float-subclass", "engine-float-subclass", "forged-dict-tag", "forged-int-tag"])
def test_unmeasured_opaque_or_forged_input_is_rejected_not_approximated(build):
    with pytest.raises(PeriodicContractError):
        build()


def test_transition_cursor_and_real_mapping_classes_are_accepted():
    # next_issue past the phase end is a legal transition observation (bound 0).
    assert Cursor(1, 0, 4, 5, 2, 2).slots == 2
    assert TemplateRef.from_validated(1, 7, 4242, ("mapping", UserDict, ()), "", 0).retained_bytes > 0


def test_engine_shape_interns_and_reuses_fixed_template_tree():
    reset_state_caches()
    detector, frames, results = run_frames([engine_state(tasks=100) for _ in range(20)],
                                           detector=CycleDetector(evidence_periods=1))
    assert all(item.state_form == "engine" for item in frames)
    assert frames[0].canonical_state.template is frames[1].canonical_state.template
    assert results[1].cycles == 64
    stats = detector.stats()
    assert stats["interner"]["misses"] == 1 and stats["interner"]["hits"] >= 19
    assert stats["fast_state_compares"] >= 1 and stats["skips"] >= 1 and stats["rejections"].get("skip") >= 1
    state_metrics = metrics()
    assert (state_metrics["template_validations"], state_metrics["dynamic_validations"]) == (1, 20)
    assert (state_metrics["template_builds"], state_metrics["template_hits"]) == (1, 19)
    assert state_metrics["template_probes"] >= 19 and state_metrics["template_hit_nodes"] > 0
    assert state_metrics["dynamic_nodes"] > 0
    # A hit must not rebuild the canonical tree.
    assert state_metrics["template_nodes"] == state_metrics["template_build_nodes"] > 0
    reset_state_caches()
    _, frames, _ = run_frames([(engine_state(tasks=100), "not-the-engine-form") for _ in range(20)])
    assert frames[0].state_form == "raw" and metrics()["raw_frames"] == 20
    assert metrics()["raw_nodes"] > 0 and metrics()["template_validations"] == 0


def test_interner_eviction_and_type_exact_keys():
    reset_state_caches()
    bounded = StateInterner(limit=2)
    for index in range(4):
        bounded.intern((("t", index),), generation=1)
    assert bounded.templates == 2 and bounded.evictions >= 2
    bounded.intern((("t", 99),), generation=2)
    assert all(ref.generation == 2 for ref in bounded._refs.values()) and bounded.templates == 1

    interner = StateInterner(limit=8)
    plain = interner.intern(({"x": 1},), generation=7, fingerprint=4242)
    user = interner.intern((UserDict({"x": 1}),), generation=7, fingerprint=4242)
    assert repr(plain.template) != repr(user.template) and plain.template_id != user.template_id
    assert len({interner.intern((("flag", v),), generation=7).template_id for v in (True, 1, 1.0)}) == 3
    # Ordinary mapping insertion order is not structural.
    assert interner.intern(({"a": 1, "b": 2},), generation=7, fingerprint=4243).template_id == \
        interner.intern(({"b": 2, "a": 1},), generation=7, fingerprint=4243).template_id
    assert interner.intern(({"x": 1},), generation=7, fingerprint=4242) is plain
    before = metrics()
    assert interner.intern((UserDict({"x": 1}),), generation=7, fingerprint=4242) is user
    after = metrics()
    assert after["template_builds"] == before["template_builds"]
    assert after["template_hits"] == before["template_hits"] + 1
    assert after["template_hit_nodes"] > before["template_hit_nodes"]
    # A real three-frame type change is never a skip proposal.
    detector, _, results = run_frames([CanonicalState(ref, D, origin="external") for ref in (plain, user, plain)])
    assert results == [None, None, None] and detector.last_rejection != "skip"


@pytest.mark.parametrize("external", [True, False], ids=["canonical-external", "raw"])
def test_shared_mutable_remaining_is_snapshotted_per_frontier(external):
    # First three changing values must not skip; once 6.0 holds for two real
    # periods, the fifth frontier yields 67 - 6 = 61 cycles.
    reset_state_caches()
    template = StateInterner(limit=64).intern((("fixed",),), generation=7)
    remaining = [("cube", 8.0)]
    detector, frames, results, differences = CycleDetector(), [], [], []
    for index, value in enumerate((8.0, 7.0, 6.0, 6.0, 6.0)):
        remaining[0] = ("cube", value)
        state = (CanonicalState(template, (("remaining", remaining),), version=index, origin="external")
                 if external else (("remaining", remaining),))
        frames.append(frame(float(index), state=state, sequence=100 + 4 * index,
                            cursors=(cursor(1000, 0, 67, 2 + index, index),)))
        results.append(detector.observe(frames[-1], None if index == 0 else delta(issued={1000: (1, 1)}),
                                        stop_at=INF))
        differences.append(detector.last_difference)
    assert results[:4] == [None] * 4 and "8.0" in differences[1] and "7.0" in differences[1]
    skip = results[4]
    run = skip.correspondence.run(1000)
    assert skip.cycles == 61 and (run.next_issue_before, run.next_issue_after,
                                  run.retired_before, run.retired_after) == (6, 67, 4, 65)
    if external:
        assert not structural_equal(frames[0].canonical_state.dynamic, frames[2].canonical_state.dynamic)
    else:  # The public input was mutated in place; history used its frozen snapshot.
        assert frames[0].normalized_state[0][1][0][1] == 6.0
        assert frames[0]._raw_snapshot != frames[2]._raw_snapshot


def assert_small_shape_skips(state, limit, *, slots=(0, 1), **interval):
    detector, _, results = run_frames([state] * 3, slots=slots, detector=CycleDetector(history_bytes_limit=limit),
                                      **interval)
    assert results[:2] == [None, None] and results[2] is not None and detector.stats()["history_bytes"] <= limit


def assert_baseline_dropped(state, limit=4096):
    detector = CycleDetector(history_bytes_limit=limit)
    assert detector.observe(frame(0.0, state=state, cursors=(cursor(1000, 0, 67, 2, 0),)), None, stop_at=INF) is None
    assert detector.last_rejection == "history_budget" and detector.observations == 0
    assert detector.stats()["history_bytes"] <= limit


def test_interner_and_detector_byte_budgets_reject_oversized_entries():
    interner = StateInterner(limit=64, retained_bytes_limit=4096)
    small = interner.intern((("small",),), generation=7)
    for _ in range(4):
        large = interner.intern((("large", b"x" * 8192),), generation=7)
        assert large.template_id != small.template_id and interner.templates == 1
        assert interner.retained_bytes <= 4096 and all(ref is not large for ref in interner._refs.values())
    stats = interner.stats()
    assert stats["oversized"] >= 1 and stats["uncached"] >= 1 and stats["live_refs"] >= 2
    assert stats["live_bytes"] >= 8192 and stats["retained_bytes"] <= 4096
    assert_baseline_dropped(CanonicalState(large, D))
    # The public template constructor owns its cost: a supplied zero cannot
    # under-report an 8 KiB tree into the history envelope.
    spoofed = TemplateRef(1, 7, 4242, (("large", b"z" * 8192),), "", 0, 0)
    assert spoofed.retained_bytes > 8192
    assert_baseline_dropped(CanonicalState(spoofed, D))
    # A legal interval whose single resource key is 8 KiB cannot stay behind.
    small_state = CanonicalState(TemplateRef(1, 7, 1, (("small",),), "", 0, 0), ())
    detector = CycleDetector(history_bytes_limit=4096)
    run_frames([small_state], slots=1, detector=detector)
    assert detector.observations == 1
    detector.observe(frame(1.0, state=small_state, sequence=104, cursors=(cursor(1000, slots=1),)),
                     delta(service={"r" * 8192: 1.0}, issued={1000: (1, 1)}, busy={}, peaks={}), stop_at=INF)
    assert detector.last_rejection == "history_budget" and detector.observations == 0
    assert detector.stats()["history_bytes"] <= 4096
    assert_small_shape_skips(small_state, 4096, slots=1, service={}, busy={}, peaks={})
    assert_small_shape_skips(CanonicalState(small, D), 16384)


# --- Real Engine: 32 owners x 2 slots, first/middle/last phases, persistent
# L0C accumulator, shared L1 panels; 8 x 136 + 24 x 68 = 2,720 tiles.


class _Run:
    def __init__(self, owner, outputs, panels, engine):
        self.owner, self.run_id, self.total, self.panels = owner, 1000 + owner, outputs * panels, panels
        self.ready = engine.boundary(f"run{owner}:ready")
        self.complete = engine.future(f"run{owner}:complete")
        self.phases = tuple(p for w in range(outputs) for p in (
            (w * panels, w * panels + 1), (w * panels + 1, w * panels + panels - 1),
            (w * panels + panels - 1, w * panels + panels)))
        self.next_issue = self.retired = self.finished = 0
        self.stage_tail = {}


class _CohortHarness:
    def __init__(self, owners=32):
        self.engine = Engine((Resource("mte2", 320.0), Resource("cube", 320.0), Resource("mte3", 320.0)),
                             pools={"l1": 128.0, "l0c": 128.0}, pool_units={"l1": 32, "l0c": 32})
        self.runs, self.tasks, self.alloc_meta = {}, {}, {}
        for owner in range(owners):
            run = _Run(owner, 2 if owner < 8 else 1, 68, self.engine)
            self.runs[run.run_id] = run
            begin = self.engine.submit(f"run{owner}:begin", after=(run.ready,), owner=f"run{owner}",
                                       on_complete=lambda _e, run=run: [self._issue(run) for _ in range(2)])
            self.tasks[run.ready.sequence] = (run.run_id, -1, 0, "ready")
            self.tasks[begin.sequence] = (run.run_id, -1, 0, "begin")

    def _issue(self, run):
        index = run.next_issue
        if index >= run.total:
            return
        wave, panel = divmod(index, run.panels)
        tile = Allocation(f"run{run.owner}:tile{index}:panel", "l1", 4.0, (run.owner,))
        acc = Allocation(f"run{run.owner}:wave{wave}:accumulator", "l0c", 8.0, (run.owner,))
        self.alloc_meta[tile.name] = ("tile", run.run_id, index, "panel", "l1", 4.0, (run.owner,))
        self.alloc_meta[acc.name] = ("persistent", run.run_id, wave, "accumulator", "l0c", 8.0, (run.owner,))
        stages = (("load", {"mte2": 8.0}, (tile,), (), ()),
                  ("compute", {"cube": 16.0}, (acc,) if panel == 0 else (), (tile,), ("load",)),
                  ("store", {"mte3": 4.0}, (), (acc,) if panel == run.panels - 1 else (), ("compute",)))
        local = {}
        for ordinal, (name, work, acquire, free, after) in enumerate(stages):
            deps = [local[k] for k in after] + ([run.stage_tail[name]] if name in run.stage_tail else [])
            ev = self.engine.submit(f"run{run.owner}:tile{index}:{name}", work, after=tuple(deps) or (run.ready,),
                                    acquire=acquire, free=free, owner=f"run{run.owner}")
            self.tasks[ev.sequence] = (run.run_id, index, ordinal, "stage")
            local[name] = run.stage_tail[name] = ev
        retire = self.engine.submit(f"run{run.owner}:tile{index}:retire", after=tuple(local.values()),
                                    on_complete=lambda _e, run=run: self._retire(run), owner=f"run{run.owner}")
        self.tasks[retire.sequence] = (run.run_id, index, len(local), "retire")
        run.next_issue += 1

    def _retire(self, run):
        run.retired += 1
        run.finished += 1
        self.engine.resolve(run.complete) if run.finished == run.total else self._issue(run)

    def _alloc(self, item):
        scope, run_id, anchor, *rest = self.alloc_meta[item.name]
        return (scope, run_id, anchor - (self.runs[run_id].retired if scope == "tile" else 0), *rest)

    def frame(self):
        e = self.engine

        def key(seq):
            run_id, tile, ordinal, kind = self.tasks[seq]
            return (run_id, tile - self.runs[run_id].retired, ordinal, kind)

        cursors = []
        for run in sorted(self.runs.values(), key=lambda item: item.run_id):
            if not run.complete.done:
                anchor = run.next_issue - (run.next_issue > run.retired)
                ordinal = next((i for i, (s, end) in enumerate(run.phases) if s <= anchor < end), len(run.phases) - 1)
                cursors.append(Cursor(run.run_id, ordinal, run.phases[ordinal][1], run.next_issue, run.retired,
                                      tuple(range(run.next_issue - run.retired))))
        records = sorted((
            key(s), tuple(sorted(t.remaining.items())), tuple(sorted(t.demand.items())), t.waiting,
            max(0.0, t.release - e.now), t.latency, None if t.start is None else max(0.0, t.start + t.latency - e.now),
            tuple(sorted(t.caps.items())), tuple(map(self._alloc, t.acquire)), tuple(map(self._alloc, t.free)),
            tuple(sorted((n, tuple(u)) for n, u in t.placement.items())), tuple(sorted(t.allocation_demands)),
            t.priority, t.coupled, tuple(key(w) for w in t.event._waiters), s in e._active, s in e._ready_seen,
            s in e._completable_seen) for s, t in e._tasks.items())
        state = (
            ("resources", tuple(sorted((n, r.rate, r.unit, r.policy, r.units) for n, r in e.resources.items()))),
            ("pools", tuple(sorted((n, c, e.pool_units[n]) for n, c in e.pools.items()))),
            ("records", tuple(records)),
            ("ready_heap", tuple((e._tasks[s].priority, key(s)) for _p, s in e._ready_heap if s in e._tasks)),
            ("releases", tuple((t - e.now, key(s)) for t, s in e._releases if s in e._tasks)),
            ("completable", tuple(key(s) for s in e._completable if s in e._tasks)),
            ("live_allocations", tuple(sorted(self._alloc(e.allocations[n]) for n in e.allocations))),
            ("allocated", tuple(sorted(e.allocated.items()))),
            ("unit_allocated", tuple(sorted(e.unit_allocated.items()))),
            ("promises", tuple(sorted((ev.name, ev.done) for ev in e._promises.values()))),
        )
        return SettledFrontier(now=e.now, epoch=1, normalized_state=state, cursors=tuple(cursors), sequence=e._sequence)

    def snapshot(self):
        e = self.engine
        return (e.now, e._sequence, dict(e.serviced), dict(e.busy_ms), e.allocation_event_count, e.completed_count,
                {r: (run.next_issue, run.retired) for r, run in self.runs.items()})

    def interval(self, previous):
        now, seq, serviced, busy, allocations, completed, cursors = previous
        e = self.engine
        return IntervalDelta(
            dt=e.now - now, sequence_count=e._sequence - seq,
            issued_retired_by_run={r: (run.next_issue - cursors[r][0], run.retired - cursors[r][1])
                                   for r, run in self.runs.items() if not run.complete.done},
            service_by_resource={n: v - serviced[n] for n, v in e.serviced.items()},
            busy_by_resource={n: v - busy[n] for n, v in e.busy_ms.items()},
            allocation_event_count=e.allocation_event_count - allocations,
            per_unit_peaks=dict(e.peak_unit_allocated), logical_completions=e.completed_count - completed,
            crossed_external_or_opaque=False)

    def apply_skip(self, skip):
        e, total = self.engine, skip.total_delta()
        e.now += skip.period * skip.cycles
        e._sequence += skip.correspondence.event_sequence_offset
        for attr, mapping in ((e.serviced, total.service_by_resource), (e.busy_ms, total.busy_by_resource)):
            for name, amount in mapping.items():
                attr[name] += amount
        e.allocation_event_count += total.allocation_event_count
        e.completed_count += total.logical_completions
        for pair, value in total.per_unit_peaks.items():
            e.peak_unit_allocated[pair] = max(e.peak_unit_allocated.get(pair, 0.0), value)
        for pool in e.pools:
            e.peak_allocated[pool] = max([e.peak_allocated[pool],
                                          *(v for (p, _u), v in e.peak_unit_allocated.items() if p == pool)])
        for run_id, shift in skip.correspondence.tile_offset_by_run.items():
            if not shift:
                continue
            corr = skip.correspondence.run(run_id)
            for seq, (tr, tile, ordinal, kind) in list(self.tasks.items()):
                if tr == run_id and tile < corr.next_issue_before:
                    self.tasks[seq] = (tr, tile + shift, ordinal, kind)
            for name, (scope, tr, anchor, *rest) in list(self.alloc_meta.items()):
                if scope == "tile" and tr == run_id and anchor < corr.next_issue_before:
                    self.alloc_meta[name] = (scope, tr, anchor + shift, *rest)
            run = self.runs[run_id]
            run.next_issue, run.retired, run.finished = (
                corr.next_issue_after, corr.retired_after, run.finished + corr.finished_offset)

    def run(self, detector=None):
        skips, cycles, previous = 0, 0, self.snapshot()
        while not all(run.complete.done for run in self.runs.values()):
            if detector is not None:
                skip = detector.observe(self.frame(), self.interval(previous), stop_at=INF)
                previous = self.snapshot()
                if skip is not None:
                    assert all(run.next_issue_after <= run.phase_end_exclusive for run in skip.correspondence.runs)
                    self.apply_skip(skip)
                    previous, skips, cycles = self.snapshot(), skips + 1, cycles + skip.cycles
                    continue
            try:
                self.engine.advance(max_steps=1)
            except ExecutionError as exc:
                if str(exc) != "execution_step_budget_exhausted":
                    raise
        return skips, cycles


def test_real_engine_periodic_cohort_skip_matches_detailed_reference():
    detailed, accelerated = _CohortHarness(), _CohortHarness()
    detailed.run()
    skips, cycles = accelerated.run(CycleDetector())
    d, a = detailed.engine, accelerated.engine
    assert sum(run.total for run in detailed.runs.values()) == 2720
    assert (d.completed_count, d.allocation_event_count) == (10_944, 5_520)
    assert d.peak_allocated == {"l1": 8.0, "l0c": 16.0}
    # The two middle phases are really skipped, not stepped.
    assert skips >= 2 and cycles >= 125 and a.advance_steps < d.advance_steps
    assert (a.completed_count, a._sequence, a.allocation_event_count) == \
        (d.completed_count, d._sequence, d.allocation_event_count)
    assert a.now == pytest.approx(d.now, rel=1e-12)
    for name in d.serviced:
        assert a.serviced[name] == pytest.approx(d.serviced[name], rel=1e-12)
        assert a.busy_ms[name] == pytest.approx(d.busy_ms[name], rel=1e-12)
    assert a.peak_unit_allocated == d.peak_unit_allocated
    assert a.allocated == {"l1": 0.0, "l0c": 0.0} and a.unit_allocated == {}
    assert all(run.complete.done and run.finished == run.total for run in accelerated.runs.values())
