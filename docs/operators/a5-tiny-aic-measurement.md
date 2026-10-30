# A5 tiny AIC control attempt (2026-10-29, blocked gate)

Status: **blocked before measurement**. This round did not collect a valid AIC
duration, numeric result, profiler task type, block count or cycle count. The
device gate itself failed, so there is no latency value to import.

Endpoint: `rack06os1` (`177.25.21.1`, port 22), physical NPU 0, Ascend950DT
9582, CANN 9.2.0, no container. Fresh remote directory:
`/tmp/sim-a5-tiny-aic-20261029T131333Z`. Evidence:
`data/verification/a5-tiny-aic-20261029/`.

## Gate sequence

At **13:12:35Z** a read-only fresh check was clean: hostname `rack06os1`,
device 0 had no processes, NPU/AICore/AIVector utilization 0%, HBM used
4517/98304 MiB (free >= 1 GiB), and 2.17 TiB host memory available.

The existing `tiny_controls` aclnn Cube branch was then run after
`source /usr/local/Ascend/ascend-toolkit/set_env.sh` to test whether OPP
registration was an environment problem. A later read-only check confirmed
`set_env_rc=0`, `ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.2.0/opp`,
`LD_LIBRARY_PATH` length 334, and the expected `lib64`/`python` prefixes, so
the source step itself was effective. The process created a device context
under that clean gate, but aborted at **13:13:54Z** with
`free(): double free detected in tcache 2` (exit 134). Its stdout was empty:
no run ID, numeric check or profiler row was retained, so this is not an AIC
measurement and is not used as one.

A pure-AIC candidate was prepared while no device work was running:

- `tiny_aic_matmul16/32/64`: AscendC ND FP16 x FP16 -> FP32, compile-time
  `MatmulApiStaticTiling`, `KERNEL_TASK_TYPE_AIC_ONLY`;
- compiled `bisheng --cce-aicore-arch=dav-c310-cube --cce-aicore-only`
  with `-DASCENDC_CUBE_ONLY`;
- loader uses `ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE`;
- one block, 170 planned launches per shape.

Fresh artifact hashes were read from that remote directory, not copied from an
older round: source `9b4378c1...a049b`, object `77d05891...95a2`, launcher
source `c981eccb...f0f8`, launcher binary `e26e2be5...fe24`. The exact compiler
commands and symbol table are in
`data/verification/a5-tiny-aic-20261029/compiler-command.txt`.

## Blocker

The mandatory fresh preflight immediately before launching the candidate
returned at **13:16:58Z**:

```text
dcmi module initialize failed. ret is -8001
```

`npu-smi info`, `npu-smi info -t usages -i 0` and
`npu-smi info -t proc-mem -i 0` each returned exit 191. A separate confirmation
at **13:17:13Z** returned the same `-8001` for all three. Per the gate, hardware
work stopped there: no candidate load, no device context, no `msprof`, no
numeric run, no restart, no driver change and no interference with other work.

The cause of the `-8001` condition is not established by this round. It is
recorded as a blocker, not as a device result, and the earlier aclnn abort is
not asserted to have caused it.

## What is and is not claimed

- Not claimed: pure-AIC TaskType, `Block Num`, cycles, kernel duration, MIX
  duration, numeric correctness, or any host/device boundary value.
- Not claimed: that the prepared cube candidate is numerically or functionally
  valid on this device; it compiled, but was never executed.
- The `aclnn` environment diagnostic is retained only as a failure transcript.
- The compiler emits `.AIV_Kernel_Type_*` metadata names even for the earlier
  empty AIC object; this naming is not treated as a profiler task type.
- `blocked` is not a measured latency, and no floor or regression was fitted.

Post-run process inspection at **13:18:02Z** found no `tiny_aic`,
`tiny_controls`, `empty_launch` or `msprof` process. The remote directory was
left in place as evidence.

A future attempt must start with a new external read-only hostname + `npu-smi`
+ free-HBM + process gate; only a normal gate may be followed by a candidate
run under `msprof` with a new run ID and fresh hashes.

## Independent acceptance

The reviewer checked all nine locally retained source/evidence hashes against
the DSH summary. They match. A read-only follow-up at remote time
**13:39:36Z** still returned DCMI -8001 / exit191 for info, utilization and
process-memory queries. No kernel, profiler or device context was started by
this follow-up. Its small raw result is retained in
`data/verification/a5-tiny-aic-20261029/raw/independent-header-and-gate-20261029T133936Z.json`.

The failed older aclnn launcher reuses the same `aclOpExecutor*` for all
20 warmups and 150 repetitions without marking the executor repeatable.
Read-only inspection confirms that installed CANN9.2.0 provides
`aclSetAclOpExecutorRepeatable` and `aclDestroyAclOpExecutor` in
`include/aclnn/acl_meta.h:73` (header SHA256
`08643f86f9f60a22f68d54fdd46aae9e5af618c53bfa905c82eedb473e16db9d`).
This is a concrete executor-lifecycle issue to resolve before reusing that
launcher, but it does not establish the cause of the observed double free
or the separate DCMI condition. The historical measured source is unchanged.
The prepared direct AscendC candidate does not use an aclnn executor.

Acceptance is limited to the blocked-attempt record and compiled candidate
provenance. It supplies no effective-AIC latency parameter and does not change
the accepted empty-AIC or tiny-AIV measurements.
