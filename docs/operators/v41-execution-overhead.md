# V4.1 physical-program execution overhead

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

This change reduces Python bookkeeping while keeping the same physical work,
event ordering, finite-buffer dependencies and resource arbitration. It does not
change a source program, hardware rate or effective-service parameter.

The disjoint owner allocator now retains at most512 value-only templates, each
with at most256 contenders. Keys include the rate, physical-unit count, ordered
caps and ordered unit tuples. Actual task IDs are rebound after lookup. Overlap
arbitration keeps its original task IDs and filling algorithm. Larger cohorts
use the original arithmetic without retention. No Event, task or RecipeContext
is retained in the cache.

The ordinary allocation pass checks for coupled work during its existing active
task traversal. Work retirement computes `4*ulp(remaining)` directly: because
`used=min(remaining,dt*rate)` is nonnegative and no larger than remaining, this
is exactly the previous `4*max(ulp(remaining),ulp(used))`, including subnormals.
No empty Stage is skipped and no event-settling rule changes.

## Acceptance

The producer
compares the literal Engine/owner allocator from commit101dbbe2 against the new
implementation, with all other runtime files and inputs held fixed. It clears
matrix compilation caches before each arm. The3-layer control preserves the
original native geometry, DP8 symmetric condition and source programs; it is a
software equality check, not reduced-layer model acceptance.

The control receipt
confirms the entire output dictionary is equal, including bounded trace,
parameters, allocations, work, service and all202,704 completed tasks. Both
predict1.6929672374301328ms. CPU changed4.629665→4.398581s; wall time changed
4.704160→4.789520s. The wall result does not establish an end-user speedup under
concurrent host load. Earlier diagnostic cProfile timings amplify the benefit
of removing Python calls and are not used as acceptance wall times.

The complete40-layer receipt
uses an independent child per arm and the same60s limit as the original report:

| Conditional DP8 input | Status | Wall / CPU | Peak RSS | Body |
|---|---|---:|---:|---:|
| c64,8rows per DP | completed |59.181 /58.916s|453.86MiB|21.460178ms|
| c128,16rows per DP | timeout |60.010s / unavailable|unavailable|unavailable|

The c64 result completes2,737,312 tasks with57,432 peak live tasks and zero
remaining tasks/programs. Its owner cache has634,061 hits and61,658 misses.
The c128 result was not retried with a longer budget. This work **does not meet
the<30s target**. The retained native Body comparison for c64 remains conditional
on the declared peer/expert geometry and is-18.49%; software optimization has
not improved that prediction error.

## Remaining high-value work

The3-layer task census identifies event-producing programs rather than another
uncached geometry ledger:

| Program roles | Completed tasks | Share |
|---|---:|---:|
| expert up + down |52,656|25.98%|
| attention + FFN HCPre |40,992|20.22%|
| grouped output A |10,200|5.03%|
| MQSMLA |8,920|4.40%|
| MC2 prefix ready + weighted reduction + send/read/quantize |18,624|9.19%|

HC source geometry already has a bounded template cache. Its per-owner
AIV-read/cast/transpose/publish and MTE1/Cube stages still produce real Engine
events. Further large improvements require an explicit finite-period program
compression with input, buffer-release and competing-resource controls. They
cannot be obtained by dropping zero-work events or replacing dependencies with
an isolated duration. Existing compact matrix contention limits still apply.

The focused acceptance ran59 tests covering ownership, Engine, coupled service,
HC, MQSMLA and ordinary MC2. New controls include500 disjoint geometries,
overlapping/reordered placement, an uncached1,024-contender cohort, finite-slot
Engine schedules with callbacks, subnormal-to-large work and mixed coupled
tasks. Full maintained-suite integration remains the parent task's check.
