# GDN update-convolution serial-cycle compaction

The default source convolution program now compacts consecutive equal resource
cycles on each physical AIV. It retains a detailed oracle through
`conv_plan(op, geometry, compact=False)`. This reduces simulation work; it does
not add a timing multiplier or claim that the remaining measured convolution
gap has been explained.

## Source boundary and implementation

The retained `native-causal-conv-a4a7958/op_kernel/causal_conv1d.h` source assigns
tasks as `blockIdx + n * blockNum`. A task is a sequence/channel-tile pair.
`ProcessDefaultByWindowMode` (1216–1288) drains MTE3 at the end of each task.
There is no source basis for overlapping different tasks on that owner.

Two consecutive tasks can share a compact cycle only when query length and
valid channel width match. Their weights, history, inputs and destinations
remain distinct: every phase's physical demand is multiplied by the **number
of real tasks**, with the same owner and dependency graph. The source UB
allocation stays 188,416 bytes per active owner and is reused across the run.
No combined resident tensor or larger buffer is invented. Ragged lengths and
channel tails break runs. Iteration retains one pending run; it does not build
a list of all expanded tasks.

Without external competitors, a repeated serial resource cycle is positively
homogeneous in its work. Time-dilating one cycle preserves its serial duration.
With competing owners or other kernels, compaction can rephase shared service.
That approximation is reported as `serial_cycle_fluid_rephasing`, and the
detailed oracle remains available. It is not a claim of instruction-accurate
cross-kernel scheduling. The existing per-task token and speculative-state
phase envelopes remain unchanged.

The planner's `conv_schedule` diagnostics expose logical task count, compact resource-cycle
count, physical AIV count, channel tile and source DMA descriptor counts. The
descriptor counts are informational; no guessed per-descriptor latency is
introduced.

## Retained control results

`data/verification/gdn-convolution-compaction-20261030.json` and its producer
use the unchanged `examples/cases/qwen38-a5.json` hardware declaration. These
are isolated detailed/compact implementation controls, not measurements or
body-forward accuracy claims.

| Geometry | Detailed Engine tasks | Compact Engine tasks | Detailed / compact prediction, µs |
| --- | ---: | ---: | ---: |
| Qwen3.6 c4, dim 1280 | 1,014 | 1,014 | 0.262 / 0.262 |
| Qwen3.6 c40, dim 1280 | 2,758 | 2,310 | 2.792 / 2.792 |
| Qwen3.6 c128, dim 1280 | 4,102 | 2,310 | 3.490 / 3.490 |
| Qwen3.8 c4, dim 2560 | 726 | 726 | 0.698 / 0.698 |
| Qwen3.8 c32, dim 2560 | 4,998 | 2,310 | 2.094 / 2.094 |
| Qwen3.8 c128, dim 2560 | 18,438 | 2,310 | 6.980 / 6.980 |

All six isolated durations agree to numerical precision. In three additional
controls with a concurrent 10 MB HBM transfer, compact/detailed duration
differences are −0.38%, −0.51% and −0.40%; physical service agrees apart from
floating summation roundoff (largest retained difference 0.0000031 bytes).
These controls characterize specific shared-bandwidth schedules, not a global
bound on contention error. One local Qwen3.8 c128 evaluation fell from 0.118 s
to 0.015 s; task-count reduction is the reproducible result, while wall time
depends on local load.

Focused tests cover single-owner timing equality under read, write and vector
bottlenecks; ragged round-robin ownership; tails; widths 2/3/4; bias;
speculative/non-speculative state work; late input; fixed UB reuse; shared HBM
contention; and AIV read-bandwidth response. No hardware rate or invocation
parameter changed.

## Noncontiguous state slots and the unresolved measured gap

Source state addresses use
`cacheIdx * convStateStride0 + position * convStateStride1 + channelStart`
(377–405 and 1008–1097). Different cache slots therefore remain separate
transfers. They do not make all cached states active, and equal transfer sizes
do not justify weight/state reuse across sequences.

For the six checked geometries, each channel tile and tail is a multiple of
16 BF16 elements, and dim is 1280 or 2560. Payloads and contiguous state-row
strides are already multiples of 32 bytes. There is no supported reason to
inflate every transfer to recover the remaining Qwen3.8 c128 difference from
the retained approximately 39.37 µs mean. That difference remains open; the
6.98 µs resource prediction has intentionally not changed through compaction.
Capture tensor extents do not reveal the actual sequence-to-cache-slot values
or cache residency, so slot-dependent HBM behavior is not identifiable from
those extents alone.

A width-4/q4 speculative task issues 17 tensor read descriptors and 10 tensor
write descriptors in this source path. Qwen3.8 c128 has 640 logical tasks:
10,880 reads and 6,400 writes, retained even when reduced to 64 resource
cycles. `ResolveSeqTaskWindowByMode` reads two prefix values in varlen mode;
`ResolveSeqCacheIndex` reads the cache index; speculative mode additionally
reads accepted length (1115–1189 and 1249–1267). These are scalar `GetValue`
accesses, not tensor MTE2 descriptors. Their issue/cache/response time and
event-fence costs need independently scoped physical parameters before adding
latency. PMU scalar time cannot be added as a disjoint residual floor.

The caller's strided cache-index first-column materialization is a separate
invocation owned by the caller adapter. It must not be counted once per
channel task or hidden inside this convolution program. This change leaves
that interface and the native recurrence program untouched.
