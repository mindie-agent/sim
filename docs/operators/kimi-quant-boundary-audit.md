# Kimi Decode quantization boundary audit

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The `65.155426 us` SituMxQuant body cohort and the earlier `65.134110 us`
task / `3.824011 us` AIV result are from the **same capture**. The CSV SHA256
is `ab1d44d3613ecb92d71af6d790ab9dfc72be5aabfe92bf4f99d584fbd0f9e648`.
The newer canonical path aliases the retained 2026-10-23 long-context capture;
it is not independent hardware evidence. The former uses 5,796 calls from 63
verified complete body forwards; the latter uses all 6,898 matching tiny calls.

Reproducer and exact source identities
and compact report
stream the 110 MB CSV, use Decimal-to-integer-nanosecond timestamps, select
individual complete windows, and reproduce all four canonical observed means.
Every row below contains 92 calls per body. Runtime control is pinned to
`5935c87b55576390f236b136ff80550ecb7d00c1`; canonical predictions retain their
own source report identity.

| Complete-body cohort | Task mean us | Exported AIV / AIC mean us | Observed activity ms/body | Predicted activity ms/body | Signed activity gap ms/body |
|---|---:|---:|---:|---:|---:|
| SituMxQuant, BF16 `[1,1536]` | 65.155 | 3.826 / — | 5.994 | 0.121 | -5.874 |
| GroupedMatmulSituQuant, allocated M512 | 74.250 | 65.199 / 70.677 | 6.831 | 0.712 | -6.119 |
| Shared QuantBatchMatmulV4 gate/up | 30.427 | 6.840 / 9.961 | 2.799 | 0.365 | -2.435 |
| Shared QuantBatchMatmulV4 down | 13.143 | 4.963 / 9.215 | 1.209 | 0.145 | -1.064 |

These are activity sums, **not additive body critical-path savings**. The two
MXFP4 matrix cohorts and grouped fusion are explicitly `reference_only` with
`mxfp4_native_execution_unmodeled`; they do not establish validated physical
prediction errors. In particular, GroupedMatmulSituQuant's 512 rows describe
allocated buffers. Captured group-list values are absent. The candidate uses
an independently declared probability workload with two expected local rows,
not 512 observed active rows. Replacing expected rows with allocation capacity
would invent routing work.

The grouped fusion is nevertheless the stronger next physical investigation:
both exported AIC and AIV activity are on the scale of its 74 us task interval,
with substantial MTE1, MTE2 and vector activity. Its missing native MXFP4 matrix,
scale application, fused SiTU/quantization, and actual routed work need their
own contracts. Those overlapping counters cannot be added into a duration.

Situ's 65.155 us is the retained primary device task interval. Within the same
5,796 events its AIV mean is 3.825694 us, with BlockNum=3. The historical exporter
and PMU endpoints are not independently identified, as established by the
profiler `situ-mx-pmu-boundary.json` source audit. Task duration includes device
scheduling, execution and completion reporting, but **61.33 us of subtraction
is not a measured STARS wrapper or kernel dispatch latency**. Neither that
difference nor the full 65 us can be installed as an arithmetic/launch floor.
The available evidence does not identify a physical-versus-wrapper split.

An actual local Engine control with `situ_mx_quant_arch35` prices the complete
source call: 3 AIV owners, 47,904 vector lanes, 3,072 input bytes, 792 output
bytes and 252,864 UB bytes per owner. It finishes at the declared inclusive
1.312 us vector minimum, with exact work conservation and no live tasks.
Thus selecting this program repairs lane/owner/pipe semantics but does not
explain the task interval. Its source revision is not the capture binary.

No new floor, utilization multiplier, PMU-based timing replacement, or
whole-body accuracy claim is introduced by this audit. The existing unknown
historical checkpoint precision, routing and exporter semantics remain visible.
