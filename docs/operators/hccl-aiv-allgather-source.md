# AIV Mesh1D AllGather source program

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The explicit selection is `expansion="AIV", collective="allgather",
template="mesh", block_limit=64` (or another declared physical grid).
`hccl_aiv_gather.contract(selection, available, rank)` returns the existing
`HcclPlan`; `build(op, geometry)` uses recipe `hccl_aiv_gather_source_phase`.
The central frontend binding is a separate parent integration. This change
does not edit central HCCL registration, cases or hardware rates.

The source is [hicann/hccl at c2d679](https://github.com/hicann/hccl/tree/c2d6790464654ccf58aca39448de4b65fdc7b184).
`aiv_temp_all_gather_mesh_1D.cc:119–126` preserves `numBlocksLimit`; it does not
use the AllReduce `R+1` launch rule. The kernel's two actual branches are:

- `RunCtrlCore` (`aiv_all_gather_mesh_1d.h:151–194`): at most `min(blocks,R)`
  owners partition input, copy it into CCL, and participate in `SyncAll`.
  Each owner then handles an integer range of peers, recording readiness,
  waiting, copying each peer input into output, recording an acknowledgement,
  and waiting for acknowledgements. The last producer receives the entire
  division remainder; all launched blocks participate in `SyncAll`.
- Producer/consumer (`:24–145`): when input bytes are at least 512 KiB and
  blocks are at least `2*R`, `R` producers partition input into `R` pieces and
  then `cut=(floor(blocks/R)*R-R)/R` pieces. Separate consumers occupy owners
  `R .. floor(blocks/R)*R-1`. Each consumer waits for one source fragment,
  copies it, and advances in the source's rank-rotated order. Idle launched
  owners cannot supply either copy pipe.

The equal-contribution normal-kernel slice is supported. Input/output aliasing,
superkernel segmentation, executor repeats and first-operator initialization
are outside this declaration. The actual selected capture kernel remains
unknown unless independent dispatch evidence establishes it.

## Traffic and finite resources

With `N` input bytes per rank and `R` ranks, useful local work is:

- Input to CCL: `N` HBM reads and `N` HBM writes.
- All CCL contributions to output: `R*N` HBM reads and `R*N` HBM writes.
- Combined payload: HBM `2*(R+1)*N`, MTE2_AIV `(R+1)*N`, MTE3 `(R+1)*N`.
- Useful wire payload per rank: `(R-1)*N`, including no self-message.

`Record` and a successful `WaitFlag` each move 128 bytes according to
`aiv_communication_base_v2.h:393–399,442–451`. They are explicit MTE3 writes
and MTE2_AIV reads, respectively. The small-message branch performs `2*R`
records and `2*R` successful waits. In the large branch, if input has `P`
nonempty producer fragments, there are `P` local records and `R*P` waits.
Remote flag bytes are included in the exchange matrix separately from useful
payload: small branch `2*128` per peer, large branch `P*128` per peer.

Ping/pong applies only when `N <= 512 KiB` and `R*N <= 8 MiB`. Otherwise
`BarrierAll` adds one block-wide sync and `2*R` records plus `R` successful
waits; one of its records per remote peer contributes 128 wire bytes.
Unknown additional polling iterations remain unmodeled. Scalar SetValue,
GetValue/compare/branch and block-wide synchronization counts are exposed;
no scalar rate or fixed delay is invented.

Every `CpGM2GM` uses the existing finite 95 KiB batch and one actual input-queue
slot. It performs MTE2 read followed by MTE3 write before reusing that slot;
`useDoubleBuffer=true` halves the batch without increasing this queue's depth.
The active 190 KiB copy queue plus 2560-byte flags share the existing HCCL
group buffer. Group execution reserves it for all launched blocks through
retirement, while isolated phases reserve it only for participating owners.
Unused Reduce64 queue compilation is unknown and is not silently added to
or used as NPU buffer capacity. The `scratch_bytes=N` contract denotes CCL
payload retained for peers, not the executor's complete communication-buffer
reservation or flag workspace.

## Accuracy boundary and controls

The current central phase interface waits for the whole preload phase before
the gather phase. This conservatively replaces source per-fragment flags and
producer/consumer overlap for the large-message branch. The local action order,
owner arithmetic, one copy slot, work and successful flag bytes are preserved;
arbitrary transport/producer overlap is not claimed exact. First-operator
`BarrierForFirstOP`/`ClearGM`, repeated polling, source descriptor/control issue
latency and packet amplification are not inferred from observed residuals.

The new tests use actual Engine execution with a 7 ms late input, cleanup,
single outer startup and inclusive minimum, TP4/Block64 idle-owner checks,
large-message tails and finite copy windows. Independent equations check
all payload and flag service. HBM, MTE2_AIV, MTE3 and network rate scans use
0.5/1/2 factors and preserve work. These are mechanism controls, not measured
collective accuracy or model body latency. The
compact receipt
records source hashes and representative synthetic results.
