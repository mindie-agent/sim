# Remaining grouped-call differences

This is a descriptive analysis of the retained INT8/FRACTAL_NZ grouped probe
against runtime revision `475a0b71a8a62744ef89ee48572476b769483e1c`. The remaining
differences require work on the actual call boundary and grouped dispatch.
The observations do not support adding a universal latency constant, a timing
multiplier, or one stable average cost per grouped call.

The source and prediction records are retained in:

- `data/observations/protocols/grouped-call-protocol.json`
- `data/verification/quantized-grouped-calls-20261029.json`
- `data/verification/grouped-gap-analysis-20261029.json`

The last file records source hashes, all ten shape comparisons, original
complete-event statistics, declared predictions and controlled shape deltas.
All 1,200 observations remain `unassigned`. The device/submitted predictions
and NPU event measurements retain different boundaries; there are zero strict
validation pairs. No observation was used to change predictions or calibrate
parameters.

## Complete-event evidence

Each table value is the median of 40 original event samples. `M` is the total
routed row count and `G` is the number of nonempty experts; the count vector
always has 256 entries. All nonempty experts have equal integer row counts.
Times are microseconds.

| Projection | M | G | Eager event | Graph1 event | Graph8 event | Graph8 / 8 | Device call prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_up | 32 | 8 | 50.828 | 40.576 | 213.128 | 26.641 | 2.111 |
| gate_up | 128 | 8 | 50.128 | 40.499 | 215.460 | 26.933 | 2.357 |
| gate_up | 32 | 32 | 53.771 | 43.451 | 240.169 | 30.021 | 4.266 |
| gate_up | 128 | 32 | 54.101 | 43.814 | 240.284 | 30.035 | 4.340 |
| gate_up | 256 | 256 | 88.767 | 92.419 | 709.031 | 88.629 | 33.827 |
| down | 32 | 8 | 49.601 | 39.828 | 207.186 | 25.898 | 0.630 |
| down | 128 | 8 | 49.520 | 39.611 | 207.193 | 25.899 | 0.779 |
| down | 32 | 32 | 55.013 | 45.317 | 256.107 | 32.013 | 2.240 |
| down | 128 | 32 | 55.043 | 45.569 | 256.896 | 32.112 | 2.353 |
| down | 256 | 256 | 98.627 | 103.880 | 795.575 | 99.447 | 17.615 |

Graph8 contains eight complete sequential calls in one replay. Dividing its
event median by eight is a descriptive normalization, not eight independent
measurements of one call. For the eight small-G conditions, this normalization
is only 65.0%-70.6% of the corresponding Graph1 event. At G=256 it rises to
95.7%-95.9%. Consequently neither Graph1 nor Graph8/8 can be promoted to a
boundary-independent per-call constant.

Within this capture, repeatability is much better than those boundary
differences: the largest difference between the two 20-sample trial medians,
relative to the pooled median, is 0.81% for Graph1 and 0.22% for Graph8.
This supports repeatability of these particular complete events. Two replay
lengths and two trials do not establish an asymptotic steady-call latency.

## A one-time replay addition is insufficient

At M=32/G=8, the observed Graph8-minus-Graph1 median difference is 172.552 us
for gate_up and 167.357 us for down. The independently predicted work for seven
additional device calls is 14.778 us and 4.407 us respectively. A hypothetical
single fixed replay cost would cancel in this subtraction, so that mechanism
alone cannot explain the difference.

This does not identify a constant cost for every added call. Graph length can
change dispatch and scheduling; source-level kernel startup, small transfers,
group control and cache behavior are also unresolved. The producer separately
captures Graph1 and Graph8. Their samples are not paired, and no fit or
intercept is inferred from their two medians.

The submitted scenario already includes the supplied nominal 20 us launch
service. For M=32/G=8 gate_up, its Graph8 prediction is 36.889 us, compared with
the 213.128 us observed event. Reusing that launch value as another correction
would both double count a declared mechanism and leave most of the event
difference unexplained.

## Shape trends rank the mechanism work

1. **Recover the executed call and event boundaries.** The producer executes
   `start.record(); call(); end.record()` in eager mode and surrounds one graph
   replay in graph mode. Input quantization, weight conversion, warmup and
   capture are outside these timed regions. Eager outputs are discarded;
   captured calls use static allocations. Event intervals can therefore expose
   submission gaps and different allocation/dispatch paths as well as kernel
   work. Their contribution cannot be recovered from this capture alone.
   For small-G shapes, eager exceeds Graph1 by 9.47-10.32 us; at G=256 the
   sign reverses, with Graph1 exceeding eager by 3.65 us for gate_up and
   5.25 us for down. This prevents treating the eager/graph difference as a
   universal overhead. Loaded binary/tiling identities and device timelines
   are the highest-value missing evidence.

2. **Model active-group scheduling and matrix/output geometry.** At fixed
   G=8, increasing M from 32 to 128 multiplies logical FLOPs by four but changes
   the full Graph8 event by only +2.332 us for gate_up and +0.007 us for down.
   At fixed M=32, increasing G from 8 to 32 changes those events by +27.040 us
   and +48.921 us. The latter change is not just arithmetic: active weight
   traffic, per-channel scales and group scheduling all change. The complete
   count vector remains 256 entries, so merely increasing the bytes charged
   for scanning `group_list` would contradict the input contract.

3. **Investigate the orientation-sensitive output and control path.** Gate_up
   has K=2048/N=256; down has K=128/N=2048. At equal M/G, down has half the
   active weight bytes but eight times the per-channel scales and output
   elements. At M=32/G=32 the observed Graph8 events are 240.169 us for gate_up
   and 256.107 us for down, while the device predictions are 34.130 us and
   17.918 us. At M=256/G=256 the observed order remains reversed relative to
   the device predictions. This is a useful counterexample for a
   weight-bandwidth-only explanation. It prioritizes inspecting actual
   grouped tiling, NZ accesses, per-group control and final epilogue execution;
   it does not prove that scale multiplication alone explains the gap.

The G=256 points also change M, so they are not a controlled experiment of G
alone. With one row per active expert, moving from M=32/G=32 to M=256/G=256
multiplies active tensor work by eight, but Graph8 grows by about 2.95 times
for gate_up and 3.11 times for down. This further rules out a single latency
multiplier across the observed shapes.

## Current physical model and remaining limits

The final predictions include matrix input reads on existing AIC resources,
final scale reads on existing AIV resources, MTE1 operand service, shared HBM,
matrix/vector work, output writes and the argument-count fetch. The report's
read-role diagnostics are complete for the supplied hardware. These resource
models remain useful before exact native dispatch is identified.

The retained producer's before/after clock fields are all unavailable, so
clock drift cannot be asserted as an explanation. The current data also do
not identify the historical compiled grouped kernel, graph-internal gaps,
instruction startup, cache residency or allocation service.

The next useful implementation should expose a source-supported dispatch or
control mechanism and test its trends across M, G, orientation and replay
length. These observations remain descriptive counterexamples during that
work; they are not fitted completion parameters, formal accuracy evidence or
whole-model convergence evidence.
