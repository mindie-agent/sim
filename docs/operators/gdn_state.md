# Native GDN H state update

`gdn_state.py` implements pinned arch35 FP32 resident-state `chunk_fwd_h`,
BF16 K=V=128, chunk=64 and scalar g. Source lives in
`data/observations/sources/gdn-prefill-image-c3d00a196905/` and
`data/observations/sources/fla-a4a7958/`; identities belong to FLA revision
`a4a795857868729fbdbb0118c1e06c8655a7b31b` in the retained capture image.

The integrated caller selects FP32 gates, exp2, initial and final state, and
V,K state storage. Standalone explicit K,V storage and BF16 gates are also
modeled. State layout changes addresses and matrix operands, not a timing
coefficient. BF16 recurrent state and arch22 are different unimplemented paths.

## Ownership and recurrence

A core owns a contiguous interval of flattened `(sequence, value_head)` tasks,
of length `ceil(total_head_tasks / core_count)`. Its serial work units stop at
sequence boundaries and at four head slots. With 48 heads and 32 AICs, 24 AICs
are active, each with two heads; the eight unused AICs supply no service.
AIVs own alternating local heads, with independent resources and state banks.

For each head/chunk, the source performs:

1. Initial FP32 state to BF16 H0 conversion when requested.
2. P = W @ H using the materialized H snapshot.
3. Vector delta/right computation and the right workspace store.
4. D = K^T @ right, or the transposed operand order for V,K state layout.
5. FP32 resident-state update and the next BF16 H snapshot or final FP32 state.

The initial tensor is read twice: once by S-1 for H0 and once by the first
resident-state update. This is not a per-chunk state round trip. Every relevant
chunk still writes/reads its BF16 snapshot and right workspace. Unique key
heads are reused within each work unit, with duplication across owner cuts.

The finite two-chunk program exposes MTE2, MTE1, MMAD, Fixpipe, vector work and
stores separately. It retains two input/operand banks, head publication edges,
per-AIV order and state recurrence. Single-head W/K prefetch overlaps the prior
chunk only after the source release event. Actual local tails are rounded to
16 for matrix instructions; local zero-fill bytes do not become HBM traffic.

## Resource contract

The plan and program share the same physical work ledger. `mixed_core.py`
partitions AIC and AIV reads when the hardware declares separate resources.
Global memory remains shared; compute and copy stages are pinned to owners.

Each active AIC reserves L1=320 KiB, L0A=32 KiB, L0B=64 KiB and L0C=128 KiB.
Each of its two AIVs reserves UB=226 KiB. These are actual per-unit allocations;
the engine retains and releases each instance. The old representative-pool
limitation no longer applies.

`tests/operators/test_gdn_state.py` independently enumerates head/sequence
ownership, data addresses, matrix and transfer work, state layouts, flags and
source stage overlap. NumPy recurrence controls verify mathematical state
updates. Source instruction latency and device rounding remain separate claims.

## Comparison boundary

The real frontend uses this recipe through `gdn_prefill_arch35`, including the
prefill suffix of a mixed non-spec batch. A state gather runs before H and the
updated state is scattered afterward. No separate external Q/K normalization
is added to the native prefill branch.

The retained 16K comparison predicts 611.963 us versus 810.911 us observed
(-24.534%). This matches tensor ABI under declared sequence values and flags;
the CSV itself does not establish those values. It is not accuracy acceptance.
The exact report is `data/verification/gdn-native-20261028-integrated-caller.json`.

MTE1 and Fixpipe are now charged. Remaining uncertainties include actual MTE2
service, cache hits, scalar/flag work, vector instruction classes and dependency
latency. Nominal INI read-path supply is not a measured per-core instruction
ceiling. Repeated GM accesses are currently charged to shared memory without
an L2 hit model. None of these gaps is replaced by a fitted duration.
