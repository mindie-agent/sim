# MC2 successful flag checks and token-owned slot reduction

This change follows the [full-population PMU audit](kimi-mc2-pmu-source-audit.md)
without using any PMU duration as a model input. It changes the existing explicit
`arch35_fullmesh` candidate, not the default generic collective.

The unquantized Combine branch has enough source geometry to count one
successful `CheckPackedTokenArriveBatch`: for K slots and P480-byte payload
blocks per slot, read `K*P*32` bytes through the output owner's MTE2, initialize
and compare `ceil(K*P*8/64)*64` FP32 elements each. Initialization precedes DMA;
comparison waits for DMA. The scalar mask traversal records one uint64 UB read,
SFF call and successful-mask branch test per64 compared elements. These counts
are carried as `source_service_details.scalar_instruction_counts`, with
`scalar_timing=unpriced; no_scalar_instruction_service_declaration`. They are
not converted into vector operations, an invented cycle rate, or a zero-time
accuracy claim. Failed polling passes remain unmodeled.

The source is `moe_distribute_combine_v2_a5_mte.h:959–1036` at revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. The predicate domain excludes
2D expert masks and special experts; this is the existing ordinary candidate's
unmasked expert-slot interpretation. A 1D token mask changes the number of
valid output tokens, which must be declared by the caller, not read from its
capacity. This change does not infer mask values from the recent captures.

For H3584 BF16/K8, one successful output-token check has3,840bytes of DMA,
1,920vector element operations and15mask reads/SFF calls. This cannot explain
the c2 residual by itself, and is not presented as doing so.

`ProcessMoeExpertsLoop` and `ProcessMoeExpert` (lines1298–1405) own each slot's
payload read, cast, multiply and accumulator add. The previous `receive_slots`
payload is moved into `weighted_reduce`; the former remains a dependency-only
boundary. Weights load once per owner; each output token initializes its own
accumulator and visits K slots. Integer tokens have explicit per-token chains;
a fractional expected tail scales only that final token's framed work. No token
ID, sampled expert routing or whole-token padding is invented.

The source's `bufferNum_` is a tiling input, not observed in these captures.
The candidate retains a **declared conservative serial-slot schedule**. It does
not claim recovered command issue overlap, finite native UB capacity or exact
scalar pipeline timing. Dispatch data-poll batching additionally depends on
actual source/expert intervals and UB windows; it retains its explicit gap.
Existing count/prefix services are not charged again. All source phase details
are now visible in planner metadata.

Verification:72 focused tests passed, including actual Engine controls for
single/65-token owner grids, shared copy pipes, late7ms input, resource cleanup,
fractional expected tails, independently calculated successful flag work,
per-slot and per-token ordering, ordinary cross-rank native groups, exact wire
bytes and one invocation per native group. The existing slot ledger assertions
were updated for the payload's new phase ownership. No full-body accuracy claim
or new measurement is part of this change.


Token chains are submitted through the existing `PipelineProgram(slots=1)` per
owner after its single router-weight read. Only the active token's K-slot chain
is materialized. Independent real-Engine comparisons with the preceding eager
DAG cover1/1.25/3/8.75/65 rows and2/3/4/64 owners. Span and physical service are
unchanged. A two-owner257-row control has31 peak live tasks versus3,345 in the
eager DAG, with the same22.528374999998096ms completion (input ready at7ms) and
373,164bytes HBM/MTE2 plus616,800vector operations. The five-row lazy control
also has31 peak live tasks. These use declared artificial rates to test program
semantics; they are not device timing measurements.
