# MoE routing alignment (`moe_align_block_size`)

Status: source-derived operator planner for the current vLLM checkout, not a
historical capture image. No device measurement, capture dispatch or timing
comparison is claimed. Structural work conservation is not accuracy acceptance.

## Selected implementation

`make_op()` selects exactly one implementation:

```
implementation = "cuda_moe_align_block_size"
symbol (small) = vllm::moe::moe_align_block_size_small_batch_expert_kernel
symbol (large) = vllm::moe::moe_align_block_size_kernel
sort symbol    = vllm::moe::count_and_sort_expert_tokens_kernel
```

Any other `implementation` value is rejected. The selected path and every
launch button are retained in `OpSpec.metadata["dispatch"]`, including
`pad_sorted_ids`, `ignore_invalid_experts`, expert-map presence, the
kernel-map argument flag, the explicit `expert_ids_post_gather`/native-layout
support flags, integer width, launch geometry, `kernel_phases` and
`host_submissions`.

The wrapper's optional post-gather path (`expert_map` present with
`ignore_invalid_experts=False`) is not selected. It is an ATen
advanced-indexing launch outside the supplied native source; `make_op` rejects
it with `moe_align_post_gather_not_supported:requires_native_layout` rather
than leaving an unowned global stage. The filtering branch
(`ignore_invalid_experts=True`) remains supported because it is implemented by
the supplied CUDA kernels.

## Source identity

| Path | SHA-256 | Bytes | Selected lines |
| --- | --- | ---: | --- |
| `/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/fused_moe/moe_align_block_size.py` | `c3f7fc2087836f0160a32ab99ceb1d8f6e87c793679da34dd672e225cb7fa31b` | 8339 | 74-103 |
| `/Users/maoxx241/code/vllm-v0.25.1/csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` | `e2bdb00824895cf7f43569159910ca369db3d7ad1da08c51cff2325d03d2e68b` | 40399 | 85-185, 187-293, 295-324, 326-351, 453-466, 582-680 |

These hashes match the coordinator-owned wave4 manifest at
`tests/fixtures/operator_sources/moe_align.json`.
The manifest describes the current checkout, not the historical capture image;
the module records
`scope = current_checkout_manifest_matched_not_historical_capture_image`. The
selected CUDA lines add the small-fill barriers (205-216) and the first worker
barrier (240) to the earlier 187-293 span.

The wrapper imports `round_up()` from
`/Users/maoxx241/code/vllm-v0.25.1/vllm/utils/math_utils.py` (lines 20-22),
which is outside the manifest. That helper is `((x + y - 1) // y) * y` for
positive integers, SHA-256
`23d9d5af62ee43dc55fd647822265f5e20b813bf89e5510130065932c7868fa6`.

## Dispatch and host kernel boundaries

The Python wrapper sizes the buffers and the CUDA host selects the launch
branch (`moe_align_sum_kernels.cu:582-680`):

```
padded_num_experts = ceil(num_experts / 32) * 32
require padded_num_experts < 1024                      # host line 599
small_batch_expert_mode = numel < 1024 and num_experts <= 64   # lines 613-614
```

`numel` is `topk_ids.numel()`; the `< 1024` comparison is strict. Both output
index tensors and `num_tokens_post_pad` are int32; the kernel reinterprets the
optional expert map as int32. `topk_ids` width is an explicit dispatch axis
because the source dispatch macro accepts integral tensors of several widths.

| Source boundary | Grid / threads | Shared memory | Symbol |
| --- | --- | --- | --- |
| small align (`one launch`) | `<<<1, 256 + max(E,32)>>>` | `((max(E,32)+1)*E + E+1) * 4` | `moe_align_block_size_small_batch_expert_kernel` |
| large align (`first launch`) | `<<<2, 1024>>>` | `ceil(E/32)*32 * 4` | `moe_align_block_size_kernel` |
| large sort (`second launch`) | `<<<(1, min(ceil(numel/256), 65535)), 256>>>` | 0 | `count_and_sort_expert_tokens_kernel` |
| wrapper post-gather | ATen advanced indexing, outside supplied source | n/a | wrapper lines 100-103; rejected at `make_op` |

`kernels(op)` exports the real host submissions. The small path returns one
child; the large path returns two children with the sort dependent on the
align child. Expansion preserves the caller dependency, placement and terminal
join. `host_submissions` in `dispatch` is 1 or 2 accordingly. Eager execution
therefore charges one host submission per real launch, not one per wrapper
call. A filtered expert map is a kernel argument inside the same launch, not a
separate host submission.

Boundary cases are exercised: `(numel=1023, E=64)` is small,
`(1024, 64)` and `(1023, 65)` are large, `(1, 64)` is small and
`(1, 65)` is large. `E=992` is accepted on the large path; `E=993` is rejected
because `padded_num_experts` becomes 1024 and the host check fails.

## Wrapper sizes and padding

Let `N = numel`, `E = num_experts`, `B = block_size`, `T = tokens`,
`K = top_k`, and let `V` be the number of counted routing entries after an
optional filtering map (normally `V = N`).

```
max_tokens_padded = N + E * (B - 1)
if pad_sorted_ids: max_tokens_padded = round_up(max_tokens_padded, B)
if N < E:          max_tokens_padded = min(N * B, max_tokens_padded)
max_m_blocks      = ceil(max_tokens_padded / B)
sentinel          = N
```

For each expert `e`:

```
counts[e]   = exact number of counted routing entries assigned to e
padded[e]   = ceil(counts[e] / B) * B
cumsum[0]   = 0
cumsum[e+1] = cumsum[e] + padded[e]
P = cumsum[E]                  # num_tokens_post_pad
U = P / B                      # used blocks
```

`P` is a multiple of `B` and is at most `max_tokens_padded`; the wrapper
allocation shrink for `N < E` remains safe. `P` is exposed as
`OpSpec.padded_shape[0]`, `max_tokens_padded` as `launch_shape[0]`, and
`counts`/`padded_per_expert`/`cumsum` in metadata.

Routing ids must be in `[0, E)`, and expert-map entries must be `-1` or in
`[0, E)`. The large sort kernel guards `expert_id >= num_experts` (line 307),
but the small-batch histogram/scatter path indexes shared counters without a
range check (lines 229-238, 278-291); negative or out-of-range ids are rejected
as invalid input instead of being modeled as memory corruption. The planner
applies the same `[0, E)` precondition to both launch branches rather than
modeling the large-path skip differently.

The sentinel is `numel`, not a mean or bucket value. A skewed routing with
`E=8, B=16, N=32` produces `P=32`; the balanced routing produces `P=128`.
Neither is replaced by `E * ceil(N/E/B) * B`.

Campaign cases (all five select the small path because `N < 1024` and
`E <= 64`):

| Case | Split | N | E | B | counts | M | MB | P |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `moe_align-0` | development | 2 | 4 | 16 | `(1,1,0,0)` | 32 | 2 | 32 |
| `moe_align-1` | development | 6 | 4 | 16 | `(2,1,3,0)` | 80 | 5 | 48 |
| `moe_align-2` | development | 3 | 2 | 32 | `(3,0)` | 96 | 3 | 32 |
| `moe_align-3` | heldout_geometry | 4 | 8 | 16 | `(1,1,0,2,0,0,0,0)` | 128 | 8 | 48 |
| `moe_align-4` | heldout_geometry | 8 | 4 | 32 | `(1,2,3,2)` | 160 | 5 | 128 |

All five cases satisfy `N < 1024`, `E <= 64`, positive `B` and ids in `[0,E)`;
none is invalid for the selected source and none is skipped or substituted.

## Small-batch kernel (one CTA, one launch)

Source: `moe_align_sum_kernels.cu:187-293`, entry at lines 453-466, launched at
lines 616-637.

1. Fill threads `0..255` initialize `sorted_token_ids[0:M]` with `numel`, then
   execute the three block-wide `__syncthreads()` calls at lines 212-214 and
   return.
2. Worker threads `tid = threadIdx.x - 256`, stride `s = max(E, 32)` zero
   `s*E` shared counters.
3. Histogram: each worker reads its routing ids and atomically increments its
   private row of shared counters.
4. The worker group's first `__syncthreads()` (line 240) rendezvous with the
   fill group's first barrier (line 212). Therefore the prefix reduction over
   threads for each expert (`E*s` additions) starts only after BOTH the
   histogram and `init_sorted` have completed.
5. Prefix stage: per-thread reduction (`E*s` additions), then the second
   barrier (line 250), then tid 0's serial exclusive prefix over experts
   (`E` iterations) writes the shared cumsum and `num_tokens_post_pad`. The
   third barrier (line 264) follows.
6. `E` threads fill used expert blocks, then all `s` worker threads fill the
   residual tail.
7. Workers scatter `V` counted ids to `cumsum[expert] + rank`.

The planner encodes the first rendezvous on the `prefix` stage
(`after=("histogram","init_sorted")`), not on `expert_ids`. The services are
not changed to mask the dependency: on the review's one-SM control
(`N=2,E=4,B=16`, `vector=1.36e6 ops/s`, `write=128000 B/s`) the source
dependency produces 1.225 ms, while deferring it to `expert_ids` produces
1.15625 ms because `prefix` starts about 0.936 ms before `init_sorted` ends.

The small-path scatter order is deterministic but is not flat token order:
thread `tid` owns ids `tid, tid+s, tid+2s, ...`; within one expert, all writes
from thread `tid` precede all writes from thread `tid+1`. The CPU oracle
therefore sorts each expert's indices by `(i mod s, i // s)`. For 70 ids all on
expert 0, the first scattered positions are `0, 32, 64, 1, 33, 65, ...`.

### Residual `expert_ids` fill: every stride thread writes

The `if (tid < num_experts)` guard closes at source line 270. The residual
loop at lines 272-276 executes for every worker thread:

```
fill_start_idx = cumsum[num_experts] / block_size + tid
for (i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[i] = inactive_expert_id;
}
```

Because `tid` ranges over `0 .. stride-1`, the union of the residue classes is
every integer from `cumsum[E]/B` to `max_m_blocks-1`. The entire tail is
initialized. This is also true for the large-kernel loop at lines 179-184,
where `threadIdx.x` ranges over `0 .. 1023`; a filtered large batch can make
the tail longer than 1024, but the residue classes still cover every block.

Counterexample (also in the tests): `E=4, B=32, N=1000` with
`expert_map=[0,-1,-1,-1]` filtering 990 remote ids, leaving 10 local ids:
`M=1152`, `MB=36`, `P=32`, `U=1`, `tail=35`, and all 35 residual blocks are
written. `expert_ids` has 36 initialized int32 blocks and the ledger charges
`36 * 4 = 144` write bytes, not 8 blocks / 32 bytes. There is no uninitialized
tail hole in the model.

## Large-batch path (two launches)

Source: `moe_align_sum_kernels.cu:85-185`, `295-351`, launched at lines
638-678.

Align kernel, grid 2:

* block 1 initializes `sorted_token_ids[0:M]` with `numel` and returns
  (lines 103-112);
* block 0 zeroes `padded_num_experts` shared counters, histograms `N` ids,
  computes per-expert `ceil(count/B)*B`, runs
  `cub::BlockScan<int32_t,1024>::ExclusiveSum`, writes `cumsum[0:E+1]` and
  `num_tokens_post_pad`, fills used `expert_ids` blocks, then fills the
  residual tail from all 1024 threads.

Sort kernel: re-reads the ids, `atomicAdd`s each counted token's
`cumsum[expert]` slot, and stores the returned rank into `sorted_token_ids`.
Its grid is `(1, min(ceil(N/256), 65535))` with 256 threads per block. The sort
kernel's within-expert order is atomic and is not claimed to be deterministic.
The oracle checks the per-expert index sets, the exact padded prefix, the block
map and the fixed rank windows, not a made-up atomic order.

## Physical CTA ownership

A source launch is a physical CTA boundary, and a CTA owns one physical unit
share of each declared pipe. The planner never gives separate stage names an
independent full-device budget:

* **Small kernel:** one CTA. All six stages (`init_sorted`, `clear_counts`,
  `histogram`, `prefix`, `expert_ids`, `scatter`) are pinned to owner slot 0
  with `context.owned(0, core_count, ("vector","mte2","mte3"))`. The fill group
  and worker group therefore share the same per-unit vector/copy capacity;
  HBM remains globally shared. `init_sorted` and `histogram` run
  concurrently; `prefix` waits for BOTH at the first source barrier, and
  `expert_ids`/`scatter` follow the third barrier in source order.
* **Large align kernel:** two CTAs. `init_sorted` is pinned to owner slot 1;
  `clear_counts`, `histogram`, `scan` and `expert_ids` are pinned to owner slot
  0. On a device with one physical unit both CTAs map to the same owner and
  time-share; on a larger device they use distinct unit shares.
* **Large sort kernel:** the grid has `G = min(ceil(N/256), 65535)` CTAs. The
  planner creates one lazy `PipelineProgram` per physical unit; unit `u`
  processes CTAs `u, u+core_count, u+2*core_count, ...` serially, so a unit
  never hosts two source CTAs at once. Each per-CTA work share is derived from
  the source thread mapping: `(i // 256) % G`, with exact per-CTA id and valid
  entry counts. The streams are generated lazily and bounded by `core_count`.
* **Wrapper post-gather:** the ATen advanced-indexing kernel is outside the
  supplied source, so its CUDA launch, grid, CTA layout and resource ownership
  are not invented. `make_op` rejects that input with
  `requires_native_layout`; no unowned stage is executable. The native
  filtered map (`ignore_invalid_experts=True`) stays inside the align launch.

No stage uses `caps` as a substitute for ownership; the caps mapping is empty.
The tests inspect the actual built stages and the actual execution trace
placements. In the static placement test with 64 vector units and 32 copy
units over 32 owners, the small CTA has vector placement `(0,1)` and
mte2/mte3 placement `(0,)`, and the large align CTA1 placement is disjoint
from CTA0.

## Dynamic shared memory

When the hardware declares a `shared` buffer pool (for example
`"buffers": {"shared": 548}`, `"buffer_units": {"shared": 32}`), the planner
creates an `Allocation` of the source dynamic shared-memory size, binds it to
the same owner slot as the CTA, and frees it after the CTA program. The
allocation is per unit and the engine enforces per-unit capacity. Without a
declared `shared` pool, the size is still reported in `analysis` and
`dispatch["shared_bytes"]`, but no capacity check is claimed. Global output
tensors are not on-chip allocations and are charged as traffic only.

## Work ledger

`stage_work(op)` exposes the exact declarative ledger used by `build`.  For
the parent wrapper it reports every source launch stage; for an expanded
`kernels(op)` child it reports only that child's phase ledger.
All amounts are physical bytes/operations; there is no matrix work and no
latency floor. `t` is the declared `topk_ids` element width (`int32` is the
default and the campaign uses four bytes); `m = 4` bytes for int32 indices.

### Small path

| Stage | HBM / copy bytes | Vector ops |
| --- | --- | --- |
| `init_sorted` | `4M` write (`hbm`, `mte3`) | 0 |
| `clear_counts` | 0 | `sE` |
| `histogram` | `Nt + map` read (`hbm`, `mte2`) | `N` |
| `prefix` | `4` total write (`hbm`, `mte3`) | `sE + 2E` |
| `expert_ids` | `4MB` write (`hbm`, `mte3`) | `MB` |
| `scatter` | `Nt+map` read (`hbm`,`mte2`) + `4V` write (`hbm`,`mte3`) | `N` |

`map = 4N` only when the kernel receives `expert_map` (the native filtered
mode). Routing ids and the map are read once by the histogram and once by
scatter; each stage is charged one pass. The rejected wrapper post-gather has
no ledger entry and no executable stage.

Vector entries are elementary source loop iterations for phases that update
shared or indexed state (clear, histogram, prefix, block fill, scatter/sort).
The linear sentinel fill is charged as memory only. These entries are not a
claim of one device instruction per iteration, and the CUB scan is explicitly
excluded.

### Large path

| Stage | HBM / copy bytes | Vector ops |
| --- | --- | --- |
| `init_sorted` | `4M` write (`hbm`, `mte3`) | 0 |
| `clear_counts` | 0 | `padded_num_experts` |
| `histogram` | `Nt + map` read (`hbm`, `mte2`) | `N` |
| `scan` | `4(E+1)+4` write (`hbm`, `mte3`) | **0, CUB internals unsupported** |
| `expert_ids` | `4MB + 4(2E+1024)` HBM; `4(2E+1024)` `mte2`; `4MB` `mte3` | `MB` |
| `sort` | `Nt+map+12V` HBM; `Nt+map+4V` `mte2`; `8V` `mte3` | `N` |

The expert-ids fill re-reads cumsum: two loads per expert thread and one
fill-start load in each of the 1024 align threads (`2E+1024` logical loads,
no cache reuse assumed). The sort bytes are one cumsum atomic read (`4V`),
one cumsum atomic write (`4V`) and one sorted-id store (`4V`). The scan's CUB
shuffle, instruction and synchronization cost is not declared; no per-element
scan-library rate is invented. Shared-memory atomics, barriers and general
integer issue are explicit unresolved items.

### Dependencies

Small path:

```
init_sorted ───────────────────────────────┐
clear_counts -> histogram ─────────────────┴-> prefix -> expert_ids -> scatter
```

Large path:

```
init_sorted (CTA1) ─────────────────────────┐
clear_counts (CTA0) -> histogram -> scan -> expert_ids
both align CTAs ----------------------------> join
join -> sort CTAs (per-unit lazy streams)
```

The first small barrier sits between the concurrent `init_sorted`/`histogram`
pair and `prefix`; this dependency is declared on `prefix` itself. The second
and third barriers are represented by the serial `prefix -> expert_ids ->
scatter` order, because those stages are executed by the same worker threads.
The large align join is the end of the first host launch; the sort child is the
second host launch. A filtered map is an argument to those launches, and the
wrapper post-gather launch is rejected rather than placed after them.

## Buffers and workspace

The wrapper allocates global int32 tensors:

```
sorted_token_ids:     4M bytes
expert_ids:           4MB bytes
num_tokens_post_pad:  4 bytes
cumsum_buffer:        4(E+1) bytes on the large path only
```

These sizes are reported in `analysis["workspace_bytes"]`. The engine has no
global-workspace capacity pool, so they are not represented as `Allocation`
objects and no capacity check is claimed. Writes and reads are charged in the
stage ledger. Dynamic shared memory is represented only when the hardware
declares a `shared` pool, as described above.

## Independent oracles and checks

`tests/operators/test_moe_align.py` contains its own NumPy routing interpreter
and does not reuse the planner's formulas:

* exact counts vs `np.bincount`, per-expert padding, exclusive prefix and
  `num_tokens_post_pad`;
* small-path deterministic scatter order (`i mod s`, then `i // s`), sentinel
  holes and the block-to-expert map;
* large-path fixed rank windows and sentinel holes (atomic order is not
  claimed), plus the exact residual-thread coverage for both kernels,
  including a filtered large tail longer than the 1024-thread align cycle;
* the 1000-id/10-valid/E4/B32 counterexample: 35 residual writes, 36
  initialized `expert_ids` blocks, 144 write bytes;
* wrapper sizes for `pad_sorted_ids` on/off and the `N < E` shrink;
* dispatch boundaries `(1023,64)`, `(1024,64)`, `(1023,65)`, `(1,64)`,
  `(1,65)`, `(1024,992)` and the rejected `E=993`;
* native filtered-map counting and exact bytes; the post-gather wrapper input
  is rejected at `make_op` with `requires_native_layout` and has no positive
  execution test;
* `kernels(op)` child ids/dependencies and 1/2 actual host submissions via
  `Execution`, not metadata alone;
* the small `init_sorted`/`histogram` -> `prefix` barrier: on the one-SM
  control the trace must satisfy `prefix.start >= histogram.end` and
  `prefix.start >= init_sorted.end`; a negative control that keeps only the
  histogram dependency reproduces 1.15625 ms and fails the second inequality,
  while the source dependency gives 1.225 ms;
* the large path's two eager submissions (50 us x 2 = 0.10 ms) versus one
  full-capture submission (50 us x 1 = 0.05 ms) at `N=1024`, no post-gather;
* physical placement for the one-CTA small kernel, the two-CTA align kernel and
  the lazy per-unit sort streams; fixed per-unit rates show the singleton CTA
  does not scale with device size and the two align CTAs use distinct owners
  for 64 vector units and 32 copy units over 32 owners;
* optional `shared` pool `Allocation` lifetime, per-CTA owner units and
  capacity rejection;
* independent stage bytes/operations, plan totals and `verify_operator`
  work conservation on unified-memory and split-copy synthetic controls;
* bounded engine state for a 160,000-id large batch;
* source file hashes and the wave4 manifest identity.

Commands run in this worktree:

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  tools/run_operator_cases.py mindie_sim.prediction.operators.moe_align \
  tests/fixtures/operator_cases/moe_align.json \
  -o data/verification/operators/moe_align.json

PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest -q tests/operators/test_moe_align.py
```

Results:

* coordinator case runner: `passed`, 10 rows (five cases times unified and
  split-copy synthetic controls), development and `heldout_geometry` both
  preserved, no dispatch mismatches, identity invariant true;
* focused tests: 29 passed.

The synthetic rates produce finite durations; they are structural controls
only and are not hardware latency evidence. No physical experiment, capture
match or accuracy acceptance is claimed. The case runner's 10 rows are five
immutable wave4 cases on unified-memory and split-copy controls, not ten
distinct workloads.

## Unresolved parameters and limitations

1. `cub::BlockScan<int32_t,1024>::ExclusiveSum` internal instruction,
   shuffle and synchronization rate is unsupported; only cumsum traffic is
   charged. No scan-library instruction rate is invented.
2. Shared-memory atomic, bank-conflict, barrier and general integer issue costs
   are unpriced. The histogram/sort counts are loop iterations, not a claimed
   device instruction schedule.
3. The wrapper post-gather (`expert_map` with
   `ignore_invalid_experts=False`) is rejected, not priced: its ATen CUDA
   launch, grid, CTA layout and copy/vector ownership are outside the supplied
   source. Supporting it requires independent native-source evidence for that
   launch; a metadata note would not be coverage.
4. `topk_ids` runtime width is not fixed by the supplied source. `int32` is the
   default; `topk_ids_dtype` is an explicit dispatch/read-byte axis.
5. `round_up()` lives outside the manifest; the current-checkout helper is
   recorded above.
6. The sort grid is modeled as one CTA per physical unit in serial waves. This
   occupancy-1 assumption is structural, not a measured occupancy or
   efficiency value. A device with more CTAs than units therefore shows wave
   serialization rather than an unmeasured co-residency.
7. No historical `moe_align` capture row, dispatch or timing was supplied.
8. There is no global-workspace capacity pool in the engine. A proposed
   interface is a declared `hbm_workspace` pool (capacity supplied by
   `Geometry.buffers`/the hardware record) so `sorted_token_ids`,
   `expert_ids` and `cumsum_buffer` could be `Allocation`s without pretending
   they are on-chip. This task does not require or change that interface.
9. A calibrated scan primitive keyed by integer type, width and element count
   would price CUB-style scans without inventing a vector multiplier. That is
   a separate primitive-calibration task, not a planner parameter.
10. Dynamic shared memory is bound only when the hardware declares a `shared`
    pool. Without that declaration the source size is reported but not checked
    against capacity; this is an explicit missing-capacity report, not a
    silently removed check.
11. The large align kernel skips topk ids at or above `num_experts`, while the
    small path does not guard them. The planner applies the `[0, E)` input
    precondition to both branches; it does not model the large-path skip as an
    alternate input domain.

## Integration

The coordinator can bind
`PlannedRecipe(moe_align.build, moe_align.kernels)` under recipe id
`cuda_moe_align_block_size`; no registry or frontend changes were made here.
`make_op` accepts the campaign parameters plus optional
`ignore_invalid_experts`, `topk_ids_dtype` and `phase`. An `expert_map` is
accepted only with `ignore_invalid_experts=True`; the wrapper post-gather form
is rejected explicitly. It is O(`N`) once on the explicit routing list to build
the histogram and per-CTA counters, and `build` is O(`E`) plus the bounded
`kernels` phase table. The returned `OpSpec` declares no matrix work, so
BF16/FP32 peaks are irrelevant to this operator.
