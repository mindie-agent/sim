# A5 tiny real-work control measurement (2026-10-29)

Status: **partial hardware collection**. The AIV vector control succeeded and is
profiled; the AIC cube control is **not measured**. Independent run
`tiny-a5-1793278251` on `rack06os1` (`177.25.21.1`), physical NPU 0,
Ascend950DT 9582, CANN 9.2.0. Raw evidence:
`data/verification/a5-invocation-20261029/tiny/`.

## AIV vector control (succeeded)

Custom AscendC `tiny_aiv_add` (real float Add, one block, no copy-only body),
compiled `dav-c310-vec`, loaded with vector-core ELF magic. `msprof`
`op_summary` reports task type **`AI_VECTOR_CORE`**, `Block Num=1`, 510 kernel
rows. Numeric checks passed for every shape with zero max error.

| Shape (floats) | Rows | min | p50 | max |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 150 | 0.876 | **0.970** | 1.329 |
| 256 | 150 | 0.916 | **0.989** | 1.362 |
| 1024 | 150 | 0.923 | **1.009** | 1.346 |

Units: microseconds, `Task Duration(us)`. The table excludes 20 warmups from each
shape, leaving three batches of 50 launches. All 510 original rows remain.
The first row of the 64 shape is longer (5.308 us); its cause is not identified.
Shape assignment follows the fixed producer order, with exactly 170 chronological
rows per shape; raw `op_summary` shape fields are unknown.

**Launch was real:** the ELF contains `T tiny_aiv_add`, and profiler recorded
510 distinct kernel rows. This is not a host-only timing and not the prior empty
kernel (empty kernels were not re-run).

## AIC cube control (blocked, not fabricated)

No suitable pure-cube AscendC sample was found in the time box.
The installed CANN `aclnnMatmul` API was tried with tiny FP16
ND tensors. `aclnnMatmulGetWorkspaceSize` returned **561103** before any kernel
launch, so there is no cube kernel row, task type, block count, or numeric result
to report. The failed attempt is retained in `run-identity-and-post.log`, with
its distinct run ID `tiny-a5-1793278239`.

Independent inspection of the actual error log found `MatMulV3` had no registered
infer-shape function (`SpaceRegistry is nullptr`). The failed command did not
source CANN's environment setup, so an OPP path problem is plausible; the evidence
does not establish a tensor-shape rejection. A proposed retry stopped before
device initialization because the fresh read-only `npu-smi` preflight returned
`dcmi module initialize failed`, code `-8001`, at 12:53:45 UTC. No device retry,
system change, or restart was performed. That later preflight error does not
establish causation by this probe.

Accordingly this report does **not** claim a pure AIC, MIX_AIC, or any cube
duration. A future run needs an available device and a working pure-cube template
or operator registration; the AIV row above is not a substitute.

## Three timing boundaries

These are reported separately and are not mixed:

- **Profiler device kernel rows:** the AIV table above (`Task Duration(us)`).
- **Device event batches:** the profiler run's batches were 4.119–4.314 us per launch over 50
  launches; includes inter-launch stream spacing, not kernel duration.
- **Host submit:** 1.834–2.102 us per `aclrtLaunchKernelWithHostArgs` call,
  measured around the asynchronous submit loop only.

All three are bound to profiler run `tiny-a5-1793278251` and NPU 0. The separate
unprofiled AIV run `tiny-a5-1793278233` had event intervals 4.113–4.532 us and
host-submit means 1.607–1.655 us; its timings are not substituted into the
profiler run. The exact compiler
command and real remote SHA-256 values are in `compiler-command.txt`/
`run-identity-and-post.log`; hashes were read from the remote build, not copied
from the earlier empty-kernel run.

## Cleanup and limits

Post-run `npu-smi` showed no running NPU processes and 0% AICore/AIVector
utilization on NPU 0 at 12:51:14 UTC; the later fresh preflight failed as recorded
above. The `pgrep` output is the inspection shell itself. No other process was
killed, no package installed, no weights loaded. This tiny AIV control is a
timing-resolution datapoint, not a universal operator floor and not a basis for
fitting a fixed overhead. The measured finite harness relies on reset/process
exit for some small allocations and events; it is not a production cleanup
template.

The independent analyzer `tools/probes/a5_invocation_20261029/analyze_tiny.py`
checks the exact producer order, all three numerical checks, 510 unique physical
rows, actual AIV task types, positive AIV cycles, zero AIC cycles, and per-shape
warmup removal. Raw source/executable hashes and run IDs accompany the result.
All samples remain `unassigned`; no runtime default or fitted floor was changed.
