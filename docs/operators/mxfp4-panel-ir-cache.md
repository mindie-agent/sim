# MXA8W4 compiled-panel IR cache: bounded full-model verification

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

Runtime package SHA256: `add6c1bfd38098ada25e579fb4598cd071082a0b9a36ded14aa6b294f309ad52`. The verification tree at `2ccedd6d` has the same complete `mindie_sim/` contents as main `86c1ee8f` (`git diff 86c1ee8f -- mindie_sim` was empty). No runtime file was modified during this campaign.

## Exact compilation change

`mxfp4_panels.compiled_panels` caches only deeply immutable `KernelProgram`/`Stage` definitions. Its key contains scalar physical resource bindings, rates, units, compute peaks, service modes, tile dimensions, owner, actual repetitions and shared-client counts. It retains no operator, request metadata, caller context, event, callback, allocation or execution state. Output/Fixpipe and buffer reservations remain in the matrix owner program. The same source `panel_parts` builder serves both detailed execution and compact compilation.

The cache has a 1024-entry LRU bound. Independent pre-integration controls compared the old source builder with the cached IR, including exact task traces, service, completion span, repeated reuse, concurrent reuse, nonzero owners, directional MTE1, K tails, input readiness and cleanup. Fifty associated tests passed. The stable uncached oracle then replaced interception of the removed matrix-local helper; its eleven checks passed after integration.

## Same-runtime cache ablation

The ablation producer runs two fresh processes against the same runtime and c1 input. It uses three layers solely as a diagnostic. The uncached test harness invokes the exact compiler through `__wrapped__`; it does not select another program, matrix policy or resource model. The raw result retains both complete execution summaries and operator timelines.

| Arm | Wall s | CPU s | Span ms | Actual Engine tasks | All submissions, including nominal controls | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| uncached | 2.914779 | 2.821828 | 0.600419245803 | 44,611 | 44,776 | 53.62 |
| cached | 1.738818 | 1.718374 | 0.600419245803 | 44,611 | 44,776 | 65.92 |

Package identity, input hash, geometry, span, all task counts, complete execution/resource dictionaries and operator timelines are **exactly equal** across arms. Both finish with zero live tasks/programs. Cached execution has 768 hits / 384 misses. In this diagnostic, CPU falls by 39.10%; this is a compilation result, not a change to modeled latency.

The older 36.39 s full-model receipt also predates the automatic matrix-subtile domain correction. It is therefore not a single-variable cache speed comparison.

## Fourteen full 40-layer conditions

The campaign producer starts one cold independent process per row, with a 60 s simulation deadline and 10 s receipt/cleanup allowance. The raw campaign records all completed and timed-out rows, package/input hashes, task counts, CPU/wall time, RSS, resource service and cache statistics. No case is shortened or retried with a larger budget.

All seven source cases retain their original `scope="body"`, q1, explicit context 2048, full 40 layers and source programs. `single` means query input is present only on rank 0; other EP ranks can still receive and compute expert rows. `symmetric` supplies the declared rank-0 query count to every DP rank. These are explicit peer-workload alternatives, not two reconstructions of known measured group-list values. Graph capacity does not establish valid token count or routing identity. Client concurrency labels c1/c4/c8 all have the same measured rank-0 B1 geometry.

| Client label | Rank-0 B | Condition | Status | Body span ms | Wall / CPU s | Actual task submissions | IR hit / miss | RSS MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| c1 | 1 | single | completed | 6.984277 | 18.55 / 17.81 | 587,962 | 14,976 / 384 | 145.72 |
| c1 | 1 | symmetric | completed | 9.722434 | 58.85 / 52.59 | 1,407,048 | 19,968 / 512 | 308.62 |
| c4 | 1 | single | completed | 6.984277 | 29.43 / 26.07 | 587,962 | 14,976 / 384 | 144.72 |
| c4 | 1 | symmetric | completed | 9.722434 | 53.23 / 50.18 | 1,407,048 | 19,968 / 512 | 334.58 |
| c8 | 1 | single | completed | 6.984277 | 25.48 / 24.09 | 587,962 | 14,976 / 384 | 145.00 |
| c8 | 1 | symmetric | completed | 9.722434 | 50.93 / 48.04 | 1,407,048 | 19,968 / 512 | 315.22 |
| c16 | 2 | single | timeout | — | 60.01 / 57.86 | 543,399 | 14,080 / 512 | 164.48 |
| c16 | 2 | symmetric | completed | 13.029748 | 49.71 / 47.83 | 1,468,744 | 19,968 / 512 | 348.38 |
| c32 | 4 | single | completed | 8.145057 | 27.08 / 25.97 | 764,538 | 19,968 / 512 | 166.12 |
| c32 | 4 | symmetric | completed | 19.644375 | 52.44 / 49.81 | 1,523,016 | 19,968 / 512 | 327.28 |
| c64 | 8 | single | completed | 9.887671 | 44.37 / 32.84 | 778,145 | 19,968 / 512 | 144.69 |
| c64 | 8 | symmetric | timeout | — | 60.01 / 49.63 | 1,297,232 | 15,616 / 512 | 312.64 |
| c128 | 16 | single | completed | 13.408987 | 25.69 / 24.58 | 805,359 | 19,968 / 512 | 163.75 |
| c128 | 16 | symmetric | timeout | — | 60.01 / 56.66 | 1,783,912 | 18,944 / 512 | 372.95 |

**11 completed, 3 timed out.** Every completed row has zero live tasks/programs; raw remaining allocation values retain floating-point roundoff. All fourteen workers have one identical clean package identity before and after execution. Maximum peak RSS is 372.95 MiB. Each completed full case has only 384 or 512 unique compiled panel programs, reused across forty layers.

Timeout receipts preserve partial Engine state and never publish a partial simulated span as a completed body latency. The c16 single-active case is a distinct performance counterexample: its 12 routed rows produce unequal 2/1 rank work, and it reaches only 543,399 task submissions in 57.86 CPU seconds; the symmetric c16 condition completes 1,468,744 tasks in 47.83 CPU seconds. Cache hit rates are high in both, so remaining runtime cost cannot be explained by compilation or task count alone. Shared-resource allocation is a separate follow-up; this baseline does not change Engine arbitration.

All body spans remain conditional source/ABI/workload predictions. The records do not assert capture binary identity, independent validation, or whole-body accuracy. Family elapsed sums include contention and can overlap across ranks; they are not additive isolated costs.

## Exact allocator indexing: retry only previous timeouts

After the frozen fourteen-row baseline, `c2261d63` replaces an active-flow × saturated-resource membership scan with the union of the same saturated-resource consumer sets. This verification tree differs from the previous runtime in `prediction/coupled.py` only; it does not change rates, model inputs, arbitration mathematics or matrix programs. The independent allocator review and same-graph profile are retained in `coupled-consumer-union-review-20261030.json` and `coupled-consumer-union-profile-20261030.json`.

The retry producer selects exactly the three original timeout rows and invokes the same worker once per row with the original 60 s limit. The new receipt references the immutable baseline SHA. Original successful rows are not rerun or relabeled as runs of the new version.

New runtime package SHA256: `1c11a530e9451bc4b797144cbec02b348e6bbafc3b3e7cc6a414225e6722f6f0`. All three retain identical input hashes and geometry, full forty layers, clean/stable package identity, and finish with zero live tasks/programs.

| Client label / condition | Previous result | New body span ms | New wall / CPU s | Actual task submissions | Peak RSS MiB |
|---|---|---:|---:|---:|---:|
| c64 symmetric | timeout at 60 s | 32.886306659 | 39.744 / 38.875 | 1,631,872 | 339.64 |
| c16 single | timeout at 60 s | 7.553434102 | 43.290 / 42.369 | 757,754 | 164.91 |
| c128 symmetric | timeout at 60 s | 33.236938502 | 45.414 / 44.884 | 1,849,584 | 340.25 |

Thus all fourteen distinct conditions have completed within the fixed budget across the preserved baseline and three follow-up runs. This is not a new fourteen-row replay and not measured body-accuracy acceptance. In particular, no speedup ratio is computed against an unfinished baseline. The original timeout evidence remains intact.
