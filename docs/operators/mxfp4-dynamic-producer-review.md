# Independent review of the finite MXFP4 producer

Reviewed runtime: `12e2073b`, explicit `producer_mode=dynamic_buffer4`.
The review found no blocking ring, dependency or physical-work discrepancy.
This conclusion covers the source dependency model; it does not establish
captured dispatch identity or complete-model accuracy.

The retained source separates four input UB slots, four widened-output UB
slots, two scale-transpose UB slots, and double L1/L0 buffering. Scale and
weight reads share the input-slot sequence. The AIV wait is on `PIPE_MTE3`;
the AIC releases the L1 slot on `PIPE_MTE1`, so independent MTE2/vector work
may run ahead and the final Cube operation may continue from L0. The new
dependencies model these boundaries instead of draining every producer panel.
Existing physical bytes and register-service counts are retained, including
the independent 4096-wide scale-refill period.

Independent checks comprised 66 focused tests, 160 finite FIFO/ring timing
oracles, 24 real Engine executions against those oracles, and 18 source
geometry/rate pairs. All source pairs conserved physical service and respected
late inputs and buffer release. Their largest compact-versus-detailed duration
difference was 5.324%. This finite set is not an unconditional error bound.

Four additional controls apply a 20 us external task on one resource, starting
1 us after the source input. Only the source completion below is compared;
the external task is not counted as source work.

| M/N/K | External shared resource | Detailed us | Compact us | Signed difference |
|---|---|---:|---:|---:|
| 1/128/5120 | HBM | 3.8748 | 3.8748 | approximately 0% |
| 1/160/5120 | AIV0 MTE2 | 6.0206 | 10.3826 | +72.45% |
| 17/160/4160 | AIV0 UB-store service | 3.7273 | 6.8826 | +84.66% |
| 1/128/5120 | MTE1 | 4.4299 | 5.4079 | +22.08% |

The large differences come from compressing independent source FIFOs into a
coupled interior under asynchronous contention. The service totals remain
equal; those totals alone cannot prove timing equivalence. Keep
`compact_panels=false` as the detailed counterexample control. Do not use a
multiplier, timing clamp or observed residual to hide these differences, or
describe the compact path as exact under arbitrary external competition.

The compact review receipt is
`data/verification/mxfp4-dynamic-independent-review-20261030.json`.
Its `.producer.py` reproduces the finite-oracle/source controls and its
`.contention.py` reproduces the four contention controls. These are independent
local mechanism tests; no hardware experiment or captured duration enters the
program construction. The source full-body campaign is separate evidence and
must retain its own runtime and input identities.
