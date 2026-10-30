# Qwen sampler-window activity audit

The Qwen3.5 and Qwen3.6 decode differences are larger than their observed
GDN activity sums. The retained summaries place Qwen3.5 conv/scan/gating/norm
near 1.107 ms per step and Qwen3.6 near 2.339 ms; their complete sampler-window
medians are 23.718750 and 34.479625 ms. These family statistics are useful
priorities, not additive critical-path components.

An independent two-pass CSV audit recovered the original sampler timestamps
using decimal-to-integer nanoseconds. It retained only the first window and
the window closest to the marker-interval median in each capture. All intervals
are clipped to their own window before computing unions and intersections.
No raw row was removed; no complete-row identity duplicates were found in
these selected windows.

| Selected window | Observed span ms | Compute/helper union ms | Native communication union ms | AICPU union ms | All-domain union ms | Uncovered interval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 window 27 | 23.717073 | 12.627029 | 1.573651 | 11.753892 | 16.685351 | 7.031722 |
| Qwen3.6 window 9 | 34.592349 | 18.469403 | 2.438775 | 18.513429 | 22.054955 | 12.537394 |

The domain unions overlap. AICPU is a control/activity domain, not additional
physical link work: Qwen3.5 also has 40 `Cumsum` activities in this domain.
The native/AICPU intersections are 1.077105 and 1.980323 ms respectively.
Missing intervals in this export do not identify host launch latency, device
startup, network latency or global device idle. Wait columns are recorded
separately and never added to event duration.

The Qwen3.6 historical `stream.all_stream_idle_ms=12.843875` is not a consistent
single-window measurement. Its reducer selected the first clean window's
activities but subtracted their union from the median duration of all windows.
The first window is actually 30.274435 ms. Clipping its crossing AICPU activities
gives an all-domain union of 21.544265 ms and uncovered interval of 8.730170 ms.
The historical summary remains preserved; this audit supplies the qualified
replacement instead of using its residual as a startup parameter.

The dominant compute-side observations also differ by model. In the selected
Qwen3.5 window, native matmul activity sums to 2.652010 ms and the two leading
grouped kernels to 2.009098 and 1.787838 ms. In Qwen3.6, the quantized matrix
family sums to 6.517687 ms and the other main matmul to 3.853977 ms.
This supports prioritizing matrix/quantization mechanisms and complete caller
submission/synchronization evidence alongside communication, rather than
attributing the full difference to GDN.

`data/verification/qwen-window-activity-audit-20261029.json` preserves original
source identities, exact timestamps, event counts, clipping, per-domain sums
and unions, wait coverage and the first-window controls. Exact execution context,
speculative acceptance and native-to-wrapper operation correlation remain
unrecovered. This audit does not add new simulation inputs or certify a
model-level prediction.
