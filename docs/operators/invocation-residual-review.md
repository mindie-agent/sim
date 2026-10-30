# Invocation residual evidence review

The retained primitive measurements support protocol-specific complete response
comparisons. They do not independently identify a universal device startup or
40 us completion minimum. No timing parameter, calibration assignment or runtime
code is changed by this review.

`data/verification/invocation-residual-review-20261029.json` records exact source
paths and SHA-256 hashes. Both original Qwen CSV hashes were recomputed by bounded
streaming; no large capture was loaded into memory.

## Independent primitive counterexamples

The A3 activation producer uses one NPU event pair around a Python callback, or
one pair around 100 consecutive callbacks. Inputs, warmup and synchronization
precede these regions. The table uses tokens=1 and intermediate=17408; it does
not claim that fused and separate variants have identical tensor interfaces.

| Run | Variant | Single event us | Burst event / 100 us | Difference us |
| --- | --- | ---: | ---: | ---: |
| a | SiLU | 49.160 | 10.981 | 38.179 |
| b | SiLU | 56.920 | 10.544 | 46.376 |
| a | Fused SwiGLU | 69.370 | 29.785 | 39.585 |
| b | Fused SwiGLU | 60.680 | 27.971 | 32.709 |
| a | Separate SwiGLU | 64.230 | 25.318 | 38.912 |
| b | Separate SwiGLU | 73.980 | 25.624 | 48.356 |

The difference changes with run and protocol. It mixes enqueue/event gaps,
allocation and device execution; burst averaging does not isolate device startup.
Separate SwiGLU uses two native operations yet is faster than the fused API in
run a, further preventing a fixed per-kernel interpretation of these differences.
All original observations remain `unassigned`.

The A5 grouped gate/up M32/G8 event is 40.576499 us for Graph1 and 213.128500 us
for eight sequential calls, or 26.641062 us per call descriptively. Thus 40 us
cannot be a universal serial per-call minimum. Graph8 minus Graph1 is 172.552001
us: a single replay addition alone cannot explain the result either. These are
separate event conditions, not isolated startup measurements.

The retained GPU copy curve has a fitted nonnegative startup of zero over
64/96/128 MiB source sizes and 24-launch event batches. This is one response-model
domain, not independently measured zero startup or evidence for an A3 parameter.

## Same-window Qwen controls

| Window | Span ms | Compute/helper union | Native comm union | AICPU union | All union | Uncovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 / 27 | 23.717073 | 12.627029 | 1.573651 | 11.753892 | 16.685351 | 7.031722 |
| Qwen3.6 / 9 | 34.592349 | 18.469403 | 2.438775 | 18.513429 | 22.054955 | 12.537394 |

Native/AICPU intersections are 1.077105 and 1.980323 ms. Domains overlap and
cannot be added; AICPU also contains Qwen3.5 Cumsum work, not just communication.
No complete-row duplicates were found in these windows. Wait columns are separate.
Uncovered export intervals are not identified host launch, device startup or
global idle. The retained directories contain device CSVs, without host API or
CPU enqueue timelines that could resolve this attribution.

Qwen3.5 has 269 matrix events totaling 2.652010 ms (9.859 us mean), plus grouped
gate/up and down activity of 2.009098 and 1.787838 ms. Qwen3.6 has 220 quantized
matrix events totaling 6.517687 ms, 3.853977 ms of other main matrix work, and
220 DynamicQuant events totaling 0.904141 ms (4.110 us mean). These native
durations themselves reject a universal 40 us kernel minimum.

The next useful work is matrix/quantization, grouped output and routing/control
mechanisms, followed by matched caller submission/synchronization evidence.
The remaining whole-step gap cannot be assigned to GDN or invocation startup by
subtracting overlapping activity totals. A future empirical response must retain
its actual API/event protocol; a device invocation parameter needs independent
device-boundary evidence. Neither requires changing analytical simulation.
