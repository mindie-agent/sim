# RecurrentKda arch22 and accepted-state boundaries

`vllm_ascend_recurrent_kda_arch22` is a separate explicit implementation
selection. It follows the non-310 header selected by `recurrent_kda.cpp`,
from the same pinned source revision documented in
[KDA Decode](kda-recurrent-source.md). The kernel header SHA256 is
`043a02e8cbbdffd0f6794793e0837e2bed75f6fc2a764ccfe026201943a949de`.
This does not establish that a retained A3 capture was compiled from that
revision or with the same flags.

The two source headers share host UB selection, V-first state residency,
query checkpoint traffic, AIV task ownership and finite MTE3 output queues.
Those mechanisms use one shared program. Two real source differences are
declared explicitly:

1. The non-310 `Process` visits all sequences on each block, validates each
   sequence and filters heads using `(sequence*HV+head)%block_count`. It
   calls `CopyInBeta` only for the first owned head of each sequence, then
   reuses that resident beta slice. The arch35 header loads the beta slice
   for every owned head. A compact run therefore has independent integer
   **head repetitions** and **beta-load repetitions**; no payload is divided
   by a fitted factor.
2. The non-310 update explicitly broadcasts delta to a K×V UB scratch,
   then uses `MulAddDst` and a separate query matrix-vector product.
   Arch35's MicroAPI `ProcessKQ` fuses the rank update and query multiply
   without this materialized broadcast. The former includes K×V additional
   vector broadcast elements per query. Both retain all arithmetic and
   checkpoint payload. The default compile branch has
   `RKDA_ENABLE_ADD_FOLD_REDUCE=1`; K=128 uses add-fold plus WholeReduceSum.

The scalar traversal differs too: arch22 validates all B sequences on
every launched block, even when it owns no head in a sequence. Diagnostics
count these prefix/state-index loads separately. Unknown cache/issue and
queue-flag delays remain unpriced. Neither observed kernel duration nor
the difference between measured and predicted spans becomes an added floor.

## Accepted-token ABI

The retained source host accepts a B+1 prefix vector, state indices either
packed `[T]` or speculative `[B,max_step]`, and optional INT32/INT64 accepted
counts `[B]`. In the kernel, valid accepted counts must lie in `[1,q_b]`.
For speculative indices, the initial state address is:

```text
initial_slot[b] = state_indices[b, accepted[b] - 1]
query_checkpoint[b,t] = state_indices[b,t]    for all t in 0..q_b-1
```

Every query executes and writes its checkpoint, regardless of how many
previous proposals were accepted. Acceptance does not scale recurrence
FLOPs or checkpoint bytes. An explicit `state_selection` now enables this
branch in either source architecture. Exact selection carries prefix values,
index layout/stride and values, accepted values and state capacity. Symbolic
selection instead requires explicit caller assumptions that the prefix,
indices and accepted values are valid and sequence state pools do not alias.
Neither mode infers values from a model's statistical acceptance rate or
`RequestView.draft_accepted_tokens` for the current output. A bare
`has_accepted_tokens=True` still fails: it does not declare either domain.
See [the executable state selection contract](kda-accepted-state.md).

The compact profiler evidence
`profiler/examples/structure/acceptance-20261030-priority-operators/glm53flash-a3-b4-native-evidence.json`
contains real RecurrentKda calls at CSV rows 11236 and 11380 of source SHA256
`35c1ff6c9570a472ca26f1bfa293cd09709a1e4d0f2984b039de9e48d961c1e2`.
Their durations are 45.661 and 40.940 us. Gate/beta describe 24 active rows,
HV=4 and K=128; prefix extent is 5, state-index shape `[4,6]`, accepted
extent 4. The accepted/index **values** and gate/kernel flags are not in
this evidence. Q/K/V and state expose backing extents, not complete logical
strides. The 11-input/3-output descriptor must therefore remain distinct
from the unaccepted source control; it is not silently promoted to a
matched accuracy row.

The source wrapper passes initial-state aliases as state outputs and writes
one selected state destination. Three profiler output descriptors do not
by themselves prove three full state copies. Conversely, resolving that
alias needs the actual wrapper/binary contract; this report does not delete
observed work to obtain a match.

## Validation

The arch22 tests execute q1/q4/q6/q8, ragged heads, actual value tails, both
state precisions and real four-axis bandwidth changes. Source task-ID
enumeration independently verifies beta loads per `(owner,sequence)`.
The work difference against arch35 is precisely removed duplicate beta
loads/converts plus the explicit delta broadcast. All other state/input/
output work and buffers are conserved. A complete ordinary frontend graph
also reaches the registered arch22 program and retires with no live work.

The shared producer accepts `arch22` as its final argument and retains
`data/verification/kda-recurrent-arch22-20261030.json`. It uses the declared
A3 hardware case and includes B4/q6/HV4 **without accepted input** as a
geometry control, not a measurement comparison. Compaction may rephase
beta loads and external contention; the report preserves that difference.

The actual A3 controls caught a consequential compression counterexample:
48 heads on 40 AIVs were initially collapsed as one/two head cycles per
owner, changing isolated timing by 6–10% through shared HBM arbitration.
The final implementation separates complete launch waves from their partial
tails. All eleven isolated controls now match the detailed duration within
floating arithmetic. B128/q4/HV12 tasks fall from 89,417 to 3,577, with both
programs taking 588.263312 us under the declared A3 parameters. External
competition controls retain a maximum 0.967% difference, explicitly reported
as fluid rephasing. When HV exceeds launched AIVs, per-owner beta load/reuse
cycles differ; this branch retains detailed head issuance instead of
pretending the cycles are identical. Its service and task counts are visible.
