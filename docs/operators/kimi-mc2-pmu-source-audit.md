# Kimi MC2: full-population PMU and the remaining source boundary

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The retained audit
reads twelve compact profiler partition reports, covering 618 accepted bodies,
56,856 DispatchV3 calls and 56,856 CombineV3 calls. Integer duration histograms
reconcile with weighted means. No large CSV, new capture or device context was
opened. The report records input, retained source and current simulator hashes.
It adds no timing parameters or runtime changes.

The two c2 captures contain a repeatable **Combine device-activity anomaly**.
The complete captured tensor ABI is identical for c1/c2/c4/c8: one output row,
width 3584, top-k 8, BF16 return capacity 16, one-dimensional BOOL mask `[1]`.
BlockNum is 64 throughout. Capacity and mask shape do not reveal valid peer
rows, mask values, routed slots or arrival times. Model IDs 53/54/55/56 are
recorded graph identities, not proof of distinct kernel algorithms.

| Family/capture | Task mean µs | AIV µs | Scalar µs | MTE2 µs | Wait µs |
|---|---:|---:|---:|---:|---:|
| Dispatch, all twelve ranges | 31.319–34.207 | 14.690–15.280 | 3.651–3.883 | 1.550–1.866 | 0 |
| Combine, c1 p4k | 14.544 | 14.271 | 7.545 | 3.279 | 0.08575 |
| Combine, c2 p4k | 34.497 | 34.230 | 17.920 | 8.343 | 0.08590 |
| Combine, c2 p16k | 34.585 | 34.319 | 17.972 | 8.358 | 0.08662 |
| Combine, c4 p4k | 14.448 | 14.177 | 7.409 | 3.315 | 0.08729 |

Combine is AI_VECTOR_CORE; Dispatch is MIX_AIV. Dispatch's 16–20 µs difference
between task duration and AIV time remains an unidentified measurement/execution
boundary. It cannot be assigned to host submission, a launch floor, network or
scalar service by subtraction. PMU pipe activities overlap; none are added.
The c2 Combine excess is about 1.84 ms of native activity over 92 calls per body;
this is not an additive critical-path correction to the modeled body.

## Fixed source facts and current omissions

Source is `cann/ops-transformer` revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. SHA256 identities are in the audit.
A capture-binary match and the runtime tiling values have not been established.

- `moe_distribute_combine_v2_a5_mte.h:959–993` implements flag DMA, vector
  initialization/comparison and scalar bit-mask traversal. Lines 1009–1036
  choose one batch flag read versus per-slot checks by expert-mask/special-
  expert flags, **not client concurrency**. Lines 1115–1130 and 1565–1593 retry
  unfinished output tokens. A scheduler `Wait Time` near zero does not count
  these in-kernel polling iterations.
- For the declared unquantized H3584/K8 branch, one slot has 15 packets; one
  output-token successful batch check reads 3,840 flag bytes, initializes and
  compares 960 FP32 elements each, and examines 15 uint64 masks with
  `GetValue`/`ScalarGetSFFValue`. Current `receive_flags` models the DMA once,
  but no predicate/mask instructions. Failed iterations repeat physical work;
  their number and early scalar exit positions are not identified by these
  PMU aggregates. The 1,920 elementary vector operations alone are too small
  to establish the cause of a 20 µs excess.
- Combine lines 1388–1405 serially select eight router weights and expert
  slots. Lines 1298–1352 perform each slot's read, cast, multiply and accumulation
  with queue/pipeline dependencies. The present whole-phase read then reduction
  envelope preserves payload but loses per-command/scalar issue and transaction
  behavior. Lines 1512–1533 correctly constrain output reception to output-token
  owners; one output row cannot borrow all 64 AIVs. This owner restriction is
  already present in `mc2_phase.py`.
- Dispatch `moe_distribute_dispatch_v2_a5_full_mesh.h:1034–1058`, 1221–1247 and
  1299–1316 contain separate count-ready/prefix/data polling loops. Lines
  1267–1295 perform scalar local-expert counts and interval filtering. At E16,
  the source has one count owner and 63 eligible payload owners; source/expert
  intervals, not total rows alone, determine active receive owners.
  `mc2_phase.py` explicitly approximates received assignment ownership by an
  expected row grid. Exact owners require routed source/expert cells.
- The source V3 entry always selects this arch35 MTE implementation for its
  supported BF16/FP16 branch. The captured optional performance tensor is empty;
  the retained host sets `isPerformance` from its presence
  (`moe_distribute_combine_v2_tiling.cpp:1814–1815`). A performance-record branch
  is therefore not supported by this ABI. Mask values, peer slot readiness,
  binary identity and ordinary tiling fields remain unknown. No c2 branch was
  identified that is justified by the captured shape alone.

## Highest-value next mechanisms and discriminating evidence

1. Add the **source instruction ledger and ordering** of one successful flag
   check, scalar metadata traversals and per-top-k read/reduction. Keep scalar
   issue, cache-line accesses and DMA command service distinct from vector
   arithmetic; use independently documented hardware service parameters when
   available. Do not divide PMU scalar time by an invented instruction count.
2. Tie exact output-token flags to actual peer/slot arrivals when explicit routed
   geometry is available. A finite polling model can consume local resources
   while waiting for those arrivals; it must not insert both a fitted wait and
   the same remote transfer. Retain the current expected-count candidate where
   source/expert owner mapping is unknown rather than manufacturing token IDs.
3. To distinguish c2 causes, the next useful *existing* evidence is actual
   `xActiveMask`, `epSendCount`/`assistInfoForCombine`, selected tiling/binary
   identity, and peer timestamps from the same captures if retained. Compare
   identical output rows under identical masks/peer counts before interpreting
   a client-concurrency label. Per-owner PMU counters or source flag-poll counts
   would separate repeated polling from scalar instruction/cache/issue stalls;
   this audit does not request or start a new hardware experiment.

The current evidence supports prioritizing missing control/predicate and
transaction dependencies. It does **not** select a polling multiplier, bandwidth
adjustment, c2 special case or a native-kernel fixed floor. Most MC2 means remain
below 50 µs; the c2 absolute error is still material under the small-call
15 µs criterion, without reclassifying it as a large-call relative-error pass.
