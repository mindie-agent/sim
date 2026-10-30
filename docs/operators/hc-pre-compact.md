# Opt-in arch35 HCPre finite-pipeline compression

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

Select `hc_pre_arch35_mk_compact_175ef4d4` for the `hc_pre` traffic profile in
the existing ordinary `native_fusion` binding. The original
`hc_pre_arch35_mk_175ef4d4` remains detailed. The source and ABI restrictions
are unchanged; this does not prove which binary produced a capture.

Compression applies only when each Cube owner has one L1 panel. Larger local
K ranges retain the original detailed program and expose
`program_execution.compact_pipeline.applied=false`. No intermediate L1 release
is hidden across multiple panels.

Within the supported panel, each AIV branch and the MTE1/Cube branch have a
dominating first read and a terminal publication. Each becomes its real
prologue, a resource-conserving coupled interior, and its real final store or
Cube stage. The original source DAG, including its finite queue dependencies,
is evaluated using the common Engine to derive nominal supply. No observation,
operator span or fitted constant participates.

Weight DMA stays independent. The matrix branch waits for the weight and both
AIV L1 publications. RMS writes remain on the producer branch and may finish
before Cube projection; projection still waits for the final Cube work. The
original stage1 and stage2 buffers, ownership, allocation lifetime and the
inter-stage barrier remain intact. Stage2 is unchanged.

The interior uses the same fluid contention abstraction as other steady
programs. It is not an exact arbitrary-contention schedule and has no global
error guarantee. Its tested scope is explicit in the
control receipt
and producer.
There are36 paired native-operator controls: M1/3/8/16/32, five resource axes
at0.5/1/2,15 persistent/transient external-copy or compute contenders, and a
multiple-L1-panel case that keeps the detailed program. Maximum duration
difference was5.92e-11 relative; the largest service summation difference was
floating-point rounding, with identical fixed-buffer peaks and zero live
tasks/programs. These results are not a whole-Body accuracy claim.

For M8, tasks fall853→512; for M16,973→632. A separate hand-computed two-chunk
control has read3/vector4/write2 service per chunk: overlapping reads and
serial vector work finish in13ms on one actual owner, identically before and
after compression. The ordinary frontend selection, late7ms input, early RMS
publication, odd-row AIV ownership and detailed default are covered by26 focused
tests. No new full40-layer run was performed for this candidate before handoff;
the previous c128 timeout remains recorded in the execution-overhead report.
