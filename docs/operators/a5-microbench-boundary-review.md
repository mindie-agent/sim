# A5 useful-kernel timing boundaries, independently reviewed 2026-10-30

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The new capture establishes complete device durations for 20 concrete eager
programs. It does **not** identify a universal A5 Vector/Cube startup constant
or an isolated Host submission time. No runtime/profile parameter was changed.
The observations remain `unassigned`.

Source directory:
`/Volumes/data/mindie-sim-data/a5/a5_kernel_floor_1tok_high_conc_decode_20261030/`.
The independent compact report
retains hashes of the producer, original summary/report/log, every kernel CSV,
and prior empty/tiny summaries. The [analyzer](../../tools/analyze_a5_microbench_boundaries.py)
only reads local data; it never imports or executes the hardware producer.

## What was measured

The producer makes 20 warmup calls outside profiling, synchronizes, then runs
50 eager calls in one NPU-only `torch_npu.profiler` window and synchronizes at
the end. Inputs and weights exist before the window. Every call returns newly
allocated output tensors; there is no `out=` reuse, graph replay, CPU activity
trace, or Host API timer. RMSNorm returns both normalized output and reciprocal
standard deviation. All 1,000 profiled rows are retained, one kernel per call,
with one device/stream in each window. The directory's `high_conc_decode` label
does not make this a concurrent serving experiment: tokens are tensor rows.

Reported hardware is one Ascend950DT_9582 die, 32 AIC / 64 AIV, with Torch
2.10.0+cpu, torch_npu 2.10.0.post7 and vLLM 0.30.0. The producer records
`vllm_ascend: n/a`; the exact plugin version cannot be inferred from that field.
The supplied report states the image/CANN version; this review did not inspect
a container or collect independent remote version evidence.

## Corrected one-row results

All values below are microseconds, from the unmodified CSV. Min and P50 are
complete device task durations, including useful work. Cadence is same-stream
start-to-start spacing; gap is next start minus previous end.

| Program, BF16 | Actual blocks/type | Device min | Device P50 | Cadence P50 | Signed gap P50 |
| --- | --- | ---: | ---: | ---: | ---: |
| Add `[1,4096]` | 8 AIV | 1.312 | 1.5195 | 5.056 | 3.600 |
| Muls `[1,4096]`, scalar 1.5 | 2 AIV | 1.429 | 1.5845 | 5.438 | 3.921 |
| RMSNorm `[1,4096]`, epsilon 1e-6 | 1 AIV | 1.802 | 1.9925 | 18.536 | 16.624 |
| MatMul `1×4096×4096` | 32 AIC + 64 mixed AIV | 11.471 | 11.993 | 12.011 | 0.022 |
| MatMul `1×12288×4096` | 32 AIC | 25.049 | 25.514 | 25.535 | 0.022 |

These Add/Muls/RMSNorm implementations use different block counts. They cannot
be treated as three measurements of the same launch-only program. The 20-case
report also contains rows 4/16/64, exact shapes/layouts, core labels and PMU
means; it does not average these domains into one hardware floor.

## Timestamp precision correction

`Start Time(us)` is approximately `1793294410022176.168`. Converting that epoch
stamp to a binary float before subtraction limits resolution to 0.25 us. The
producer did that conversion, producing 5.25/5.50 us Add/Muls median cadences
and negative MatMul gaps. Recomputing with Decimal-to-integer-nanosecond
conversion before subtraction yields **zero negative gaps in 980 pairs**;
the old arithmetic creates **161 negative gaps**. All eight MatMul conditions
have a 0.022 us median gap. Thus the reported MatMul overlap does not establish
a dispatch-vs-execution timestamp distinction.

The original report remains preserved. This review retains its float-derived
statistics alongside corrected values rather than rewriting the source.
Two fast regression tests use the first four original Add and MatMul CSV rows,
including a concrete MatMul false-negative-gap counterexample. The analyzer
also keeps the first transition: Add's first gap is 85.347 us, which must not
be silently removed to claim a lower average. A separately named diagnostic
excludes only that transition; primary statistics retain all 50 calls.

`duration_sum_over_window_span` uses exactly
`sum(all kernel durations) / (last end - first start)`. It is different from
the producer's `mean(50 durations) / mean(49 start intervals)`. It is a
timeline occupancy statistic, not a Host-time measurement or general device
utilization measurement.

## Parameter absorption and support domain

| Boundary | Supported interpretation | Simulator use |
| --- | --- | --- |
| Empty AscendC AIV/AIC task | Hardware/toolchain/core/block-specific empty-program observation | Keep existing explicit empty-program invocation classes |
| New Add/Muls/RMSNorm task duration | Inclusive duration of the named implementation, shape, dtype and blocks | A concrete measured program scenario may use `startup_ms=0` and `minimum_ms=P50/1000`; do not install a generic A5 vector class |
| MatMul task duration | Useful matrix operation with substantial operand work | Model matrix work, transfer and scheduling; never install 11.471 us as a Cube invocation floor |
| New eager start cadence / signed gap | Composite spacing of this Python/framework/allocation/profiler/device-queue path | Descriptive observation only; not `launch_us`, device startup, or guaranteed idle time |
| Prior direct ACL asynchronous submit timer | Host call cost for that launcher/run | Explicit direct-ACL Host submission scenario only; not this eager wrapper or graph replay |

The existing invocation mechanism takes
`max(minimum_ms, startup_ms + modeled_body_ms)` once per complete invocation.
An inclusive measured whole-kernel duration must not also be added as startup,
or attached to every tile/phase. Existing named measured-program classes are
opt-in and require the caller to assert their support domain; their selector
does not itself verify shape/compiler provenance. The new measurements do not
justify broadening that selector to all Vector or Cube programs.

The [accepted empty-kernel run](a5-invocation-measurement.md) gave AIV steady
P50 approximately 0.482/0.591/0.986 us and AIC 0.477/0.583/1.159 us at blocks
1/8/64. It measured no Host submit-only timer. Its approximately 4 us ACL event
span per launch includes queue spacing and is a third boundary.

The [prior custom FP32 Add control](a5-tiny-control-measurement.md) gave
0.970/0.989/1.009 us steady P50 for 64/256/1024 floats at one AIV block.
That profiler run separately timed direct ACL asynchronous submission at
1.834–2.102 us/call; its unprofiled run gave 1.607–1.655 us/call. The runs remain
distinct. No successful tiny useful-work Cube control was obtained there.
The new useful MatMul captures fill an observation domain, but are not a
substitute for an empty or genuinely tiny useful-work Cube measurement.

For `M=1,K=N=4096`, BF16 B alone is 32 MiB; N=12288 raises it to 96 MiB. The
minimum duration rises from 11.471 to 25.049 us. MTE2 occupies much of reported
AIC time, consistent with operand-transfer pressure. The same B is reused for
20 warmups and 50 calls: logical bytes do **not** prove that each call fetched
all 32/96 MiB from HBM rather than cache. This capture contains no measured
HBM/L2 byte-volume evidence to make that attribution.

The source does not validate output numerics, sweep clock/power/concurrency,
or establish a representative high-batch production Forward. These limits do
not prevent analytic simulation; they constrain which parameter or accuracy
claim this specific experiment supports. The existing micro-operator
acceptance tolerance remains an evaluation policy, not measured body work.

## Reproduction

```sh
.venv/bin/python tools/analyze_a5_microbench_boundaries.py \
  /Volumes/data/mindie-sim-data/a5/a5_kernel_floor_1tok_high_conc_decode_20261030 \
  -o data/verification/a5-microbench-boundary-review-20261030.json
.venv/bin/python -m pytest -q tests/test_a5_microbench_boundaries.py
```

This review ran locally, changed no source capture, launched no device work,
and changed no model/runtime/profile parameter.
