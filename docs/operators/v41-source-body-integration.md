# V4.1 conditional source programs through complete body graphs

At runtime `dd4933b8` (base `ffa61916`), seven A5 V4.1 inputs mount the previously
implemented MQSMLA, MXA8W4 NZNK grouped matrix and native HC programs through the
ordinary frontend. All retain the complete 40-layer model, TP1/DP8/EP8, original
quantization and resource declarations. This establishes executable source
selection and exposes a simulation CPU bottleneck; it does not establish captured
binary identity or complete-body timing accuracy.

The inputs live in
`examples/cases/high-concurrency-20261030/v41-source-conditional/`.
`data/verification/v41-body-source-inputs-20261030.json` retains each original
case hash, compact profiler geometry hash and the full source-program identities.
No raw capture CSV, weights, remote device or timing fit is used.

| Client concurrency label | Observed rank-0 query rows | Declared single-active cohort |
|---:|---:|---|
| 1 | 1 | `[1,0,0,0,0,0,0,0]` |
| 4 | 1 | `[1,0,0,0,0,0,0,0]` |
| 8 | 1 | `[1,0,0,0,0,0,0,0]` |
| 16 | 2 | `[2,0,0,0,0,0,0,0]` |
| 32 | 4 | `[4,0,0,0,0,0,0,0]` |
| 64 | 8 | `[8,0,0,0,0,0,0,0]` |
| 128 | 16 | `[16,0,0,0,0,0,0,0]` |

Each is decode q1 at the explicitly declared context 2048. The captured rank-0
row count is not divided from client concurrency. Other DP ranks and their live
contexts are not observable in these compact inputs. A second condition declares
eight identical per-rank row counts, but it is an alternative workload, not a
reconstruction of the capture. No condition is selected by proximity to a measured
time. Original graph mode and graph buckets remain unchanged.

The source variant declares FP8/group32/BF16-scale window cache and
FP4/group16/BF16-scale compressed cache, matching the selected MQSMLA source
producer/consumer representation. MQSMLA uses `metadata_support_fd=false`, scalar
addressing and no LSE output. The original sparse-index/AICPU metadata producer
and captured binary ABI remain unverified and explicit in partial coverage.
Inherited capture-audit metadata remains unchanged; `source_scenario` describes
this new conditional implementation. Body probing bypasses serving admission, so
this run is not a PD capacity or placement acceptance result.

## Integration counterexample and correction

Selecting HC on a receive-only EP rank previously raised
`native_fusion_selected_roles_unmatched:hc_post,hc_pre`: there is no local body
row and therefore no local HC invocation, but the rank can still receive and
execute expert work. Commit `dd4933b8` lets the frontend explicitly report zero
local rows. Only the HC required-match check is omitted for that case. Nonempty
ranks and all unrelated selected roles retain strict matching. Empty senders
remain in the EP communication graph.

The focused HC/per-DP/native-resource checks passed 21 tests. The seven new source
inputs additionally pass real full-graph construction and MXA8W4 preparation
checks in 1.71 seconds. Rank 0 mounts 40 MQSMLA calls, 80 HC-pre and 80 HC-post
calls; all seven idle body replicas retain their 40 expert-combine boundaries.
These graph checks do not replace complete Engine execution.

## Complete graph runs and CPU bottleneck

`data/verification/v41-body-source-20261030.json` preserves three serial public
`predict(scope="body")` attempts. c1, c8 and c64 single-active conditions each
reached their 60-second budget. None produced a completed body time. Worker peak
RSS was unavailable because termination occurred before its final receipt.
The c64 symmetric condition was not started after these failures. Package hashes
before and after the campaign are identical.

The bounded diagnostic in
`data/verification/v41-source-engine-profile-20261030.json` keeps the exact physical
shapes and all eight EP participants, but explicitly limits the model to its first
three layers for CPU profiling. It completed in 14.924 seconds with cProfile,
57.813 MiB peak RSS and 314,672 Engine submissions:

| Actual source/family | Tasks | Share of all tasks |
|---|---:|---:|
| MXA8W4 | 306,756 | 97.48% |
| MQSMLA | 157 | 0.05% |
| HC pre + post | 996 | 0.32% |

The diagnostic graph span is 0.599445 ms, with tasks, programs and allocations
released. This is a three-layer diagnostic, not a full-body estimate. Refill
retirement/prefetch and Engine submission dominate Python CPU time. Existing
MXA8W4 `compact=true` merges equal output tiles but still expands every inner
K-panel. The next fix belongs in shared periodic tile-pipeline execution; rates,
observed medians and source shape declarations need no adjustment to address
this computational bottleneck.

Reproduction producers are stored beside their reports. Preserve these timeout
reports when rerunning the same full inputs after the shared pipeline fix; record
the new runtime/package hash separately.

## Periodic inner-panel execution: first complete body

A subsequent explicit local runtime from `sim-mxfp4-steady-panels-20261030`
(commit `a4a0f662` plus the recorded compact-panel patch) was tested against the
identical seven-case input contract. The package SHA was
`d1eb06ee22b8a11e4749e412cc25bc3f6e8f6a0c873ecc8aa89eecc253b2e164` before and
after every run. `data/verification/v41-source-compact-20261030.json` retains the
four receipts, complete physical service and bounded profiler summaries.

| Execution | Wall seconds | Engine task submissions | Device span ms | Peak RSS MiB | Status |
|---|---:|---:|---:|---:|---|
| Three-layer, detailed panels, cProfile | 17.584 | 318,364 | 0.537320 | 60.34 | completed |
| Three-layer, compact panels, cProfile | 9.671 | 47,794 | 0.537061 | 54.50 | completed |
| Complete 40-layer c1, compact | 44.791 | 619,764 | 6.303705 | 122.20 | completed |
| Complete 40-layer c64, compact | 60.050 | 741,063 before deadline | unavailable | 145.42 | timeout |

The paired three-layer runs use the same current runtime and both count Fixpipe
BF16 output as 2MN payload bytes. The periodic-panel switch is their only input
difference. Actual serviced work is conserved within 1e-12 relative roundoff;
the largest absolute accumulation difference is 1.96e-5 bytes. Tasks fall 84.99%,
and the compact device span differs by -0.0482%. Both runs release all tasks,
programs and allocations. Original `dd4933b8` evidence had 4MN Fixpipe and earlier
non-grouped programs, so its span must not be used as a compression-only control.

The complete c1 run has zero live tasks/programs/allocations, 619,614 completed
main-Engine tasks and peak 12,270 live tasks. The submission instrumentation also
counts 150 tasks in bounded nominal producer controls; this explains the slightly
larger 619,764 submission total. Per-family elapsed sums remain available, but
include waiting/overlap and cannot be added as isolated body costs.

The c64 run is still too slow for the 60-second budget; its measured peak RSS is
available because the local deadline unwinds to write a receipt. The symmetric
c64 alternative was not started, and c8 repeats the same q1/B1 geometry as c1,
so no duplicate full-body run was made. Further compiler cost reduction remains
required. The 6.30 ms c1 output is a conditional prediction, not a comparison
against a measured rank-cohort body or an assertion of 15% accuracy.

## Shared-owner nominal service and bounded geometry reuse

Runtime `5fe2be3b` adds nominal HBM sharing within identical owner-pipeline
classes and caches up to 128 pure geometry ledgers. The same complete c1 and c64
inputs were rerun once each with a 60-second budget. Both have identical start/end
package SHA `0cb8d4b47798c30c38fc3caabfdf7163507d2d53caa979e3abf408b31bc9aa0c`.
Receipts are in `data/verification/v41-source-shared-owner-20261030.json`.

| Full 40-layer condition | Wall seconds | CPU seconds | Predicted body ms | Peak RSS MiB | Result |
|---|---:|---:|---:|---:|---|
| c1 single active rank0 | 36.390 | 35.925 | 6.306813 | 131.77 | completed |
| c64 single active rank0 | 60.056 | 57.081 | unavailable | 147.58 | timeout |

The c1 pure geometry cache has 478 hits and 2 misses; nominal producer controls
have 199,669 hits and 11 misses. The caches retain 2/128 and 11/512 entries.
The completed main Engine still retires 619,614 tasks and releases every task,
program and allocation. Its physical service matches the preceding compact c1
run within floating summation tolerance; this optimization changes the timing
approximation and construction cost, not the modeled byte/FLOP demand.

For c64, geometry reuse reaches 638 hits/2 misses and nominal controls reach
263,669 hits/11 misses. The deadline interrupts 5,264 live tasks and 319 active
programs after 799,117 tasks complete. Those partial counts and buffers are
reported as unfinished execution, not successful cleanup or a completed body
span. The symmetric `[8,8,8,8,8,8,8,8]` workload remains unexecuted because its
required single-active precursor did not finish; no repeated or extended-budget
c64 attempt was made. Shared compilation still needs further cost reduction.
