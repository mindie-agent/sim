/*
[Role] Body-immediate AIC-only AscendC kernel for A5 launch/duration probe.
[Invariants] No compute, no copy, no global-memory access; empty body only.
  Compiled for dav-c310-cube with AIC-only task metadata.
[Timeouts & Locks] Device execution is bounded by the host probe (<=120s).
[Side-Effect Policy] None: no memory is read or written; binary is loaded by host.
*/
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void empty_aic() {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
}
