# A5 empty AIV/AIC invocation measurement (2026-10-29)

Status: **new hardware collection succeeded** on `rack06os1` (`177.25.21.1`),
physical NPU 0, Ascend950DT 9582, CANN 9.2.0, `npu-smi` 25.1.rc3.1. This is an
independent run, not a historical observation. Raw evidence is
`data/verification/a5-invocation-20261029/`.

## Question and answer

What is the actual device duration of a launched kernel whose body returns
immediately, separately for AIV-only and AIC-only?

| Kernel | Profiled task type | Blocks | Rows | min | p10 | p50 | p90 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `empty_aiv` | `AI_VECTOR_CORE` | 1 | 500 | 0.443 | 0.457 | 0.482 | 0.519 | 0.816 |
| `empty_aiv` | `AI_VECTOR_CORE` | 8 | 500 | 0.552 | 0.566 | 0.591 | 0.628 | 0.941 |
| `empty_aiv` | `AI_VECTOR_CORE` | 64 | 500 | 0.951 | 0.965 | 0.986 | 1.025 | 1.339 |
| `empty_aic` | `AI_CORE` | 1 | 500 | 0.439 | 0.453 | 0.477 | 0.514 | 0.813 |
| `empty_aic` | `AI_CORE` | 8 | 500 | 0.545 | 0.559 | 0.583 | 0.620 | 0.996 |
| `empty_aic` | `AI_CORE` | 64 | 500 | 1.115 | 1.132 | 1.159 | 1.193 | 1.478 |

All units are microseconds. Each condition has 550 original rows: 50 warmups,
then five batches of 100 launches. The table excludes those first 50 chronological
launches separately for each kernel/block condition. All raw rows and original
all-row statistics remain retained. The first row of some conditions is longer;
its cause is not identified as a hardware startup parameter.

`tools/probes/a5_invocation_20261029/analyze.py` independently recomputes the
distributions, five batch medians, and core-counter checks from the CSV.
This is one process run on one device, not 500 independently restarted experiments.

## Launch was not eliminated

The kernel bodies contain only `KERNEL_TASK_TYPE_DEFAULT` metadata: no compute,
no copy, no global-memory access. The compiler still emitted callable symbols
(`T empty_aiv`, `T empty_aic`), and `msprof` recorded **3300 distinct kernel
rows** (1650 per kernel), with task IDs and block counts. Loading the AIC object
with cube-core ELF magic changed its profiled type from `AI_VECTOR_CORE` to
`AI_CORE`; the first vector-magic attempt is retained as a rejected control.
This establishes a real launched device kernel, not a host-only call.
All accepted rows also have positive cycles on the matching core type and zero
cycles on the other core type. Symbol-section names alone are not the type oracle.

## Method

- Preflight read-only `npu-smi` showed NPU 0 explicitly idle: 0% utilization,
  no running NPU process. The subsequent ACL precheck reported 97.78 GB free
  within its exposed allocation capacity. No other die was used.
- `empty_aiv.cpp` compiled with `bisheng --cce-aicore-arch=dav-c310-vec`;
  `empty_aic.cpp` with `--cce-aicore-arch=dav-c310-cube`, both
  `-D__DAV_C310__ --cce-aicore-only -O2`.
- `empty_launch.cpp` loads each ELF with ACL (`aclrtBinaryLoadFromData`), gets
  the function, and calls `aclrtLaunchKernelWithHostArgs` for block counts 1, 8,
  64. Exact commands and hashes are in `compiler-command.txt`.
- `msprof --ai-core=on --aicpu=on --task-time=on` produced `op_summary` CSV.
  Primary duration is `Task Duration(us)` from that CSV, one row per kernel.

## Boundaries

- ACL stream event batches measured **3.991--4.279 us per launch** in the
  unprofiled corrected run, and **4.002--4.162 us** in the profiler run.
  Each is an event span over 100 launches divided by 100. They include
  launch/queue spacing and remain separate by run in `summary.json`;
  they are not kernel durations or host submit-only measurements.
- Empty-kernel duration is **not** a universal real-op floor and must not be
  used to fit operator residuals or replace body work.
- This empty-kernel run collected no nonempty Vector/Cube controls, graph replay
  boundaries, clock/thermal sweep, or numeric correctness checks.
  Empty kernels have no output to verify.
- The AIV/AIC object-task-type mechanics on this CANN build required separate
  ELF magic selection; a vector-magic AIC load is invalid evidence for cube.

## Independent acceptance

The reviewer verified 3,300 unique physical profiler identities, the six exact
kernel/block conditions, all core counters, and local/remote source hashes.
The final executable hash is `fc2e2fde8d2c47cd7a08cc59f2c57cc811fcb6c184df4eecbeadcf360074d567`;
the previously recorded `ad4d...` hash belongs to the rejected vector-magic
launcher. The retained readback includes both corrected-run event logs and the
profiler log. The post-run NPU inventory lists no active NPU processes.

These data support an empty-body observation indexed by hardware, compiler/CANN,
core type and block count. They do not identify a universal 3/5 us startup term,
a 40 us operator floor, or the start/stop cost of a nonempty vector/cube pipeline.
No simulator parameter or calibration split was changed; the samples remain
`unassigned`.

## Optional measured execution parameters

`examples/invocations/a5-measured-programs.json` is a device-parameter fragment
for explicit matching programs. Merge its `machine.device.kernel_invocations`
map into the selected case. It contains nine named classes: empty AIV/AIC at
1, 8 and 64 blocks, plus the three FP32 tiny Add shapes from
`a5-tiny-control-measurement.md`. Each declares `startup_ms=0` and an inclusive
`minimum_ms` equal to the independently reviewed steady P50 divided by 1,000.
The body and minimum combine by maximum; the measured complete-kernel duration
is not added to its modeled body.

The physical `OpSpec` must explicitly set `metadata.invocation_class`, for
example `empty_aiv_blocks1` or `tiny_aiv_add_fp32_n64_blocks1`. These names
assert the recorded implementation, hardware/toolchain, shape and block count;
the selector does not automatically validate geometry against provenance.
An empty body also needs a real source-bound kernel boundary. The file contains
no generic `vector` or `cube` key, so importing it leaves ordinary kernels
unaffected. Existing explicit runtime invocation rules retain their precedence.

Every entry retains the source summary and raw capture hashes, run ID,
measurement boundary, sample/warmup counts, binary/source identity and support
limits. Selection is an explicit measured scenario, not a validation result.
Original observations remain `unassigned`; the file supplies neither an
unmeasured Cube-work duration nor a universal A5 floor.
