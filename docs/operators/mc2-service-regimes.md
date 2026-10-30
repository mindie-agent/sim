# Observable MC2 native-call service regimes

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The retained A5 captures support an inclusive completion response for a declared
MC2 fullmesh kernel domain. It is applied once per DispatchV3 or CombineV3 group
as `max(analytic completion, response)`. The count, payload, flags, local copies,
quantization, weighted combine and physical peer dependencies remain in the
normal Engine. The response is neither an added body residual nor occupancy of
an invented vector/network resource.

The reproducible parameter and validation ledger is
`mc2-service-regimes-20261030.json`,
produced by the adjacent Python file. Its four explicit case variants retain
their source-compatible parent declarations and add only these response rules
plus provenance. Original cases are unchanged.

## Physical domains and units

All four responses are restricted to `Ascend950DT_9582`, EP8, BF16 input/output,
and the stated expert count, width, top-k and source implementation. They do
not inspect the model name. `M` is the native top-k/source-row count per rank,
not client concurrency or padded receiver capacity.

| Domain | Native kernel | Domain M | Inclusive response (μs) |
|---|---|---:|---:|
| E16, width3584, top8, BF16 transfer | DispatchV3 | 1–4 | 30.380747 + 0.954065 M |
| E16, width3584, top8, BF16 return | CombineV3 | 1–4 | 14.543737 |
| E384, width5120, top6, MX32 transfer | DispatchV3 | 2–16 | 17.448562 + 0.224795 M |
| E384, width5120, top6, BF16 return | CombineV3 | 2–16 | 11.297089 + 0.256561 M |

The implementation expresses the row term as external framed payload bytes:
53,760 B/M for the first domain, 32,256 B/M for MX dispatch and 59,136 B/M for
its BF16 return. These use the existing 480-data-byte / 512-byte framing and
7/8 external fraction. Count/status bytes remain separately charged by the
source program. Fractional expected assignments are never individually ceiled.

The slope is an observable response at this retained hardware configuration.
There is no independent hardware sweep identifying an HBM, AIV, or network
primitive, so no guessed `scale_with` is attached. Existing analytic resources
still respond to hardware changes and may exceed this completion constraint;
the response component's bandwidth sensitivity is **unidentified**, not
established as invariant hardware performance. Outside the M domain, analytic
simulation remains available without this response.

## Calibration and withheld checks

Kimi BF16 dispatch uses c1/p4k and c32/p4k. Its combine uses only c1/p4k because
the endpoint difference is just 0.040 μs, insufficient evidence for a meaningful
throughput slope. All other native-call samples remain withheld, including
c16's distinct M2 geometry and every p16k capture. All 12 compact native ABI
partitions independently confirm the source M values (1, 2, or 4).

The MX domain uses V4.1 M2 and M16 native calls. M4 and M8 remain withheld. The
fit uses native-call means consistently; retained medians and outliers remain
in the ledger. These are complete-forward call observations, not isolated
hardware benchmarks. No body timing enters the parameter calculation.

| Withheld control | Dispatch response vs observed | Combine response vs observed |
|---|---:|---:|
| Kimi c32/p16k | 34.197 vs 34.207 μs | 14.544 vs 14.615 μs |
| Kimi c16/p4k | 32.289 vs 33.414 μs | 14.544 vs 14.726 μs |
| V4.1 M4 | 18.348 vs 18.548 μs | 12.323 vs 12.381 μs |
| V4.1 M8 | 19.247 vs 19.944 μs | 13.350 vs 19.057 μs |

The latter Combine remains **−29.95%**, although its 5.71 μs absolute error is
within the agreed small-call tolerance. Kimi c2 Combine is a stronger failed
validation: both contexts are approximately 34.5 μs at the same observed M1
ABI, leaving approximately 20 μs error. The rules deliberately contain no
batch-two exception. Peer readiness, route values and private dispatch state
are not observed sufficiently to identify that regime.

PMU vector/scalar/MTE times overlap. Elapsed minus AIV time is not separately
measured network wait, and neither difference is added to the source phases.
Capture binary identity and actual peer assignment values remain independent
evidence questions; their uncertainty does not disable analytic simulation.

These are coefficient-assignment holdouts, not a blind experiment: the retained
observations were already available during diagnosis. Calibration and validation
roles here are specific to this response experiment; original raw observations
and historical role declarations are unchanged.

## Actual Engine execution

The paired control receipt is
`mc2-service-execution-20261030.json`.
On base `ab37ffe5` plus shared API `2f48346b`, all four candidate pairs completed
with 7 ms delayed inputs and zero live tasks/programs. Every serviced resource
is unchanged between analytic and response arms (largest floating summation
difference: 9.31e−10 bytes). Kimi queries each response once; the complete V4.1
EP8 cohort queries each response eight times, once per native call/rank.

| Full Kimi c32 body | Prediction | Local execution | Peak RSS | Per-direction response queries |
|---|---:|---:|---:|---:|
| p4k / actual context4104 | 32.357471 ms | 18.24 s | 145.1 MiB | 92 |
| p16k / actual context16392 | 37.544225 ms | 31.78 s | 141.7 MiB | 92 |

Both are the complete 93-layer body and stay below the 60-second / 512-MiB
budgets. These numbers are predictions, not new body accuracy claims. V4.1's
full c64/c128 executions are coordinated with the grouped-response lane as a
combined experiment; its complete-body result must be labeled with both rule
sets. The MC2-only cases remain available for a separate controlled comparison.
