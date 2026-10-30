# Periodic MXFP4 refill execution

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The standard `ascend_mxa8w4_nznk` source program now compresses its regular
K-panel interior. `compact_panels=false` retains the detailed oracle. Its two
L1 slots, two consumer slots, source owners, physical work and reserved buffers
are unchanged. The separate `compact` option controls equal output-tile runs.

Activation refills occur every `ka` elements; E8M0 scale refills occur every
4096. The schedule repeats at their least common multiple, with a real final
partial period. Repeating the first panel would wrongly multiply scale and
activation reads. A bounded integer-tick recurrence preserves dependencies
across periods without draining them; it can jump millions of panels after
witnessing repeated complete state. The finite producer DAG is evaluated by
the existing Engine and cached, rather than replaced by an empirical latency.

One producer prologue and final Cube operation remain explicit. The interior
retains all resources in a coupled phase. Equal owner schedules share nominal
HBM supply; different N-resplit classes still contend on the actual shared
resource. This distinction matters: whole-die supply for each owner lost finite
slot backpressure, while permanent equal shares across unequal 128/160-column
pipelines overconstrained the slower class after other work retired. No device
rate or measured residual is used to choose those classes.

The nominal periodic recurrence is exact for its declared scalar dependency
schedule. Arbitrary external contention is a fluid approximation. Independent
review of 48 actual A5 c1/c64 source controls across HBM, AIC and AIV 0.5x/1x/2x
rates found a maximum difference of 7.47% from detailed execution and no inverse
response. Baseline down is numerically equal, gate differs by about 0.093%.
These compare two simulation abstractions, not measurements.

91 additional mechanism controls preserve physical service, input readiness,
owners and buffer cleanup. Extreme heterogeneous HBM supply and asymmetric
external AIV contention still expose differences of -26.13% to +33.54%.
Keep the detailed control for those questions; this work does not establish
all-contention fidelity or whole-model measured accuracy.

Fixpipe work is also corrected from FP32 source width to BF16 destination
payload, `2*M*N` bytes. The retained 3510 instruction table declares 64 BF16
output elements per cycle at 128 B/cycle, versus 32 FP32 outputs at the same
byte rate. This is an independent unit correction; both comparison arms use
it. The table describes the L0C-to-L1 path; applying the declared nominal pipe
to the source's GM destination remains an explicit profile approximation.

Sources and reproducible controls:

- [Source program](../../mindie_sim/prediction/mxfp4_grouped.py),
  [period recurrence](../../mindie_sim/prediction/matrix_cycle.py), and
  [producer/interior composition](../../mindie_sim/prediction/refill_steady.py).
- Independent A5 scans
  and mechanism/pressure controls.
- The retained official table is
  `refs/cannbot-knowledge/knowledge/ops/ascendc/apis/appendix/cube_instruction_theoretical_perf_summary.md`,
  table 8, F322BF16 row. No per-model parameter was added.
