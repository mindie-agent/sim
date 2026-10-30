# Independent service-regime boundary review

Review target: shared API `2f48346b`, exercised on the MC2 branch after the same
runtime patch. No new body parameters or capture-derived timing changes are
part of this review.

Two public JSON contract problems were reproduced and repaired:

- Physical shape selectors arrived as JSON arrays while immutable `OpSpec`
  shapes (and source metadata) used tuples. Direct equality silently missed an
  otherwise matching rule. A real 4000-byte copy with `local_shape: [2000]`
  returned 4 ms instead of the declared 9 ms completion response. The matcher
  now compares nested lists/tuples structurally without copying or mutating the
  operation. The same control now returns 9 ms and records one response query.
- An explicitly supplied `domain.max: null` was accepted at construction and
  later caused `TypeError` in selection. Supplied bounds now require finite
  nonnegative numbers. Omitting `max` continues to mean no upper bound; null
  does not introduce another implicit domain convention.

The initial suspicion of a missing duplicate-ID check was withdrawn: the
original API already adds IDs to its set. The independent test confirms two
nonoverlapping physical domains keep separate receipts and duplicate IDs remain
rejected. No production change to that logic is included.

`tests/test_service_regime_boundaries.py` supplies 16 actual Engine controls.
Together with the existing regime/group/buffer checks, **44 tests passed in
0.17 seconds**. These are physical boundary controls, not accuracy fits:

| Control | Independent expected result |
|---|---|
| JSON shape + nested metadata, Host3 ms, startup2 ms, minimum12 ms, release11 ms | Response9/15 ms finishes26/29 ms; physical work starts16 ms |
| Two native groups, late input11 ms, startup3 ms, physical6 ms, response10 ms | Finish31 ms; 2 native queries; eager2 Host enqueues or1 captured replay |
| UB occupied until5 ms, group startup2 ms, physical3 ms, response12 ms | Finish17 ms; response starts after actual admission; all UB released |
| Two DP ranks, zero sender, late sender9 ms, startup3 ms,4-byte message | Link0.5/1/2 B/ms gives20/19/19 ms; exactly4 wire bytes and2 native queries |
| Whole-group link response with startup3 ms | Link0.5/1/2 gives16/8/5 ms; the fastest arm is bounded by actual startup+wire |
| Mixed BF16/FP32 matrix, only BF16 rate scaled | 0.5/1/2 rate gives8/4/2.5 ms; unscaled FP32 service exceeds the fast response |
| Directional MTE1 mode rate, fixed base pipe | Mode0.5/1/2 gives2/1/0.5 ms; service remains on the one real MTE1 pipe |
| Existing20 ms primitive, response12 ms, startup3 ms+physical4 ms | Finish27 ms from release7 ms; none of these inclusive constraints are added |
| Three concurrent resource-service curves, each1 ms startup+2000 B | Physical shared service finishes14 ms from release7 ms despite5 ms responses |
| Reused group IDs in two iterations | Independent12 ms responses finish17/29 ms, with2 queries and zero live state |

Domain misses retain the analytic program and an `analytic_outside_domain`
receipt. Missing coordinates retain analytic execution and a
`missing_coordinate` receipt. All exercised graphs end with zero live tasks,
programs and allocations. Served bytes and arithmetic work retain their physical
meaning; a response does not create synthetic resource occupancy.

Reproduce locally:

```sh
.venv/bin/python -m pytest -q tests/test_service_regime_boundaries.py tests/test_service_regimes.py tests/test_kernel_group_buffers.py tests/test_device_kernel_groups.py
```
