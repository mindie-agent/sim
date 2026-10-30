# Native arch35 causal convolution FN

This is the native AscendC varlen `runMode=0` implementation in FLA revision
`a4a795857868729fbdbb0118c1e06c8655a7b31b`, from the historical capture image.
It is separate from the alternate Triton convolution recipe. Sources are
retained under `data/observations/sources/native-causal-conv-a4a7958/`.

`native_causal_conv_spec.py` owns parameter normalization and source host tiling.
`native_causal_conv_prefill.py` owns physical work and the bounded program.
`native_conv_registers.py` owns typed restore/rolling source inventories.
The public entrypoints are `make_op`, `host_tiling`, `build` and `source_ledger`.

## Dispatch and host geometry

The planner supports BF16/FP16, widths 2-4, no activation or SiLU, no-bias
rolling and biased generic paths, and CPU/device sequence metadata.
`CUTBS` keeps the full channel dimension when it fits; `CUTBSD` co-splits
tokens and channels subject to UB size and source divisibility adjustment.
The launch uses the declared AIV count, not the AIC count. The 16K, D=10240,
64-AIV case uses four channel tiles of 2560 elements and sixteen token tiles.

Device-visible all-false initial-state flags retain snapshot workspace and
SyncAll. CPU-visible all-false flags disable that branch. Absent convStates
clears initial-state use even if a supplied flag is true. Null/pad cache IDs
skip the corresponding output and state updates. Reserved snapshot capacity
is distinct from the payload actually copied.

The selected non-spec GDN caller uses this FN once across all actual tokens,
including decode rows when a prefill is present. Pure decode selects another
caller branch and does not claim this FN. Cache row IDs are declared distinct
non-null indices; their numeric historical values are not observed in CSV.
Speculative history slots extend storage beyond width-1, but FN touches only
the actual convolution history. The integrated state shape includes that
explicit runtime proposal capacity.

## Causal resource program

Each block reserves 188416 B UB on its owning AIV before initial-state
prefetch. The allocation remains live across snapshot, SyncAll, weights,
metadata, token work and state writeback. No DMA can run before parent input
readiness or before its reservation and source producer complete.
The snapshot reuses one temporary UB row: each history slice's read and write
finish before that row is loaded again. History initialization then precedes
the first input copy and rolling-partial restore. Restore can overlap that
copy; the first compute and first pair prefetch wait for both.

Rolling input copies load token pairs into reused ring slots. Output slots
alternate by token parity. The rolling compute waits for its old output slot,
while the other slot's store can overlap. In generic RunSeq, arithmetic may
start earlier: only the output cast waits for store(t-2), and next-input
prefetch depends on cast(t-2). Collapsing arithmetic and cast incorrectly
changes both kernel and competing-task service.

Rolling state writeback reloads retained input before the MTE2-to-MTE3
handshake and store. Generic writeback copies the existing ring directly.
The no-bias source register ledger separates instruction calls, active lanes
and UB payload. For one full 64-lane token, widths 2/3/4 execute 4/6/8 loads,
2/3/4 stores, two casts, 2/3/4 multiplies and 1/2/3 additions. SiLU adds
Muls, Exp, Adds and Div. Including one mask update and two issue slots for
each store, the SiLU path requires 18/24/30 slots; restore requires 7/15/25.
Unmasked loads keep their full payload on channel tails; masked stores retain
only the active payload. Channels 16 and 64 therefore use the same number of
UB port calls, while 80 needs two register vectors.

Issue slots, data-register writes, UB accesses/stores, ALU and special-function
lanes are separate capacity constraints. The declared default-FTZ intrinsic
math path follows the retained compiler controls; exact packaged ELF flags,
optimized instruction bundles and dependency latency remain unobserved.
No cross-token weight-load hoisting is assumed. Setup casts/zeros and the
generic bias path still use explicitly analytical vector work.

Vector and copies retain per-AIV ownership. On the selected split-read profile,
all FN input service belongs to `mte2_aiv`; it cannot borrow AIC read resources.
Register issue follows its declared physical service-mode binding even when
the physical pipe has a different name. Idle owners cannot supply that block.
Shared HBM is never divided by the AIV count. Programs issue a bounded token
window and do not allocate a complete per-token event history.

## Independent review

`tests/operators/test_native_causal_conv_prefill.py` checks source host tiling,
address coverage, state history, numerical convolution and branch work.
`test_native_causal_conv_invariants.py` retains the independent regressions for
parent readiness, UB snapshot lifetime, generic arithmetic/cast contention,
source instruction counts, split-copy ownership, remapped issue ownership and
first-token/restore overlap.

Original rejection and subsequent root-fix reports are retained together under
`data/verification/native-convolution-review/`. The original returned candidate
was not accepted merely because its tests passed. Six concrete defects were
closed before integration, then the split modules were reviewed again.

The current raw row 52 replay predicts 373.734 us versus 336.261 us observed
(+11.144%), retained in `gdn-native-20261028-reviewed-rolling.json`.
The earlier 168.916 us prediction used the obsolete vector-rate alias and
mixed instruction/lane units; it is preserved as an earlier report.
Its 64-block launch and active tensor descriptors match the selected source
case, but CSV does not observe runMode, activation, per-sequence values,
initial-state flags or cache indices. Full cache capacity rows are displayed
but are not equated with active modeled rows.

Scalar issue, barriers, instruction latency, register bundles and memory
transactions are still unpriced or capacity-envelope approximations. SyncAll
uses an explicit source-sized traffic assumption, not an observed exact wire
transaction count. The remaining timing discrepancy is unresolved; no
multiplier, minimum duration or model-specific correction is applied.
